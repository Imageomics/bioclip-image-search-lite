"""BioCLIP model service for embedding generation and taxonomy prediction.

Extracted from bioclip-vector-db model_server.py BioCLIPModelService class.
Runs in-process — no Flask wrapper.
"""

import logging
import threading
import time
import functools
from typing import List, Dict, Any, Optional

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

from ..instrument import phase

logger = logging.getLogger(__name__)


def _timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        logger.info(f"{func.__name__} completed in {dt:.4f}s")
        return result
    return wrapper


class ModelService:
    """BioCLIP-2 model for image embeddings and taxonomic classification."""

    def __init__(self, device: str = "cpu", model_str: str = "hf-hub:imageomics/bioclip-2"):
        self.device = device
        self.model_str = model_str
        self._tol_classifier = None
        self._Rank = None
        self._CustomLabelsClassifier = None
        # Per-image-hash embedding cache so concurrent upload + search handlers
        # encode a given image only once (fixes the first-search re-encode race).
        self._embed_cache: Dict[str, np.ndarray] = {}
        
        # Two locks
        # A dict of locks keyed by image hash. 
        # Each lock protects the embedding of a single image.
        self._embed_locks: Dict[str, threading.Lock] = {}
        # Global lock to protect the cache and the per-hash locks dict.
        self._embed_meta_lock = threading.Lock()
        # Limit the cache size to avoid unbounded memory growth. 
        self._embed_cache_max = 32
        self._load_model()

    def _load_model(self):
        from bioclip import TreeOfLifeClassifier, Rank, CustomLabelsClassifier

        logger.info(f"Loading BioCLIP model '{self.model_str}' on {self.device}")
        self._tol_classifier = TreeOfLifeClassifier(
            device=self.device, model_str=self.model_str
        )
        self._Rank = Rank
        self._CustomLabelsClassifier = CustomLabelsClassifier
        logger.info(f"Model loaded: {self._tol_classifier.model_str}")

    def embed(
        self,
        images: List[PIL.Image.Image],
        normalize: bool = True,
        image_hash: Optional[str] = None,
    ) -> np.ndarray:
        """Generate embeddings for a list of PIL images.

        If ``image_hash`` is given for a single image, the result is served
        from (or stored in) a per-hash cache, so concurrent upload + search
        handlers encode the same image only once. Cached embeddings are always
        L2-normalized (the ``normalize`` flag applies only to the uncached
        multi-image path; FAISS re-normalizes the query regardless).

        Returns:
            np.ndarray of shape (N, 768).
        """
        if image_hash is not None and len(images) == 1:
            return self._embed_cached(images[0], image_hash)
        with phase("embed", n=len(images)):
            rgb_images = [img.convert("RGB") for img in images]
            features = self._tol_classifier.create_image_features(
                rgb_images, normalize=normalize
            )
            return features.cpu().numpy()

    def _embed_cached(self, image: PIL.Image.Image, image_hash: str) -> np.ndarray:
        """Embed one image, computing it at most once per hash even under
        concurrent callers (per-hash lock + double-checked cache)."""
        
        # Check under the global lock
        with self._embed_meta_lock:
            # Has the embedding already been computed and cached?
            hit = self._embed_cache.get(image_hash)
            if hit is not None:
                return hit
            # If not, create or grab the existing per-hash lock for this image hash. 
            # `.setdefault(key, default)` is atomic under the global lock, so only one thread creates the lock.
            lock = self._embed_locks.setdefault(image_hash, threading.Lock())
        
        # Check under the per-hash lock
        # If thread is the first to acquire the lock for this image hash, 
        # it will compute the embedding.
        # Other threads will wait here until the lock is released.
        with lock:
            # Why still use the global lock in here? 
            # _embed_cache, _embed_locks are protected by the global lock, not by the per-hash lock. 
            # To avoid racing condition caused by concurrency, 
            # we need to protect them by the global lock when accessing them.
                    
            # Check again under the global lock
            # Another thread may have computed it while we waited for the lock.
            with self._embed_meta_lock:
                hit = self._embed_cache.get(image_hash)
            if hit is not None:
                return hit
            
            # Compute the embedding for this image
            with phase("embed", n=1):
                features = self._tol_classifier.create_image_features(
                    [image.convert("RGB")], normalize=True
                )
                emb = features.cpu().numpy()
            
            # Store the computed embedding in the cache under the global lock
            with self._embed_meta_lock:
                self._embed_cache[image_hash] = emb
                if len(self._embed_cache) > self._embed_cache_max:
                    oldest = next(iter(self._embed_cache))
                    self._embed_cache.pop(oldest, None)
                    self._embed_locks.pop(oldest, None)
            return emb

    def embed_and_classify(
        self,
        images: List[PIL.Image.Image],
        rank: str = "species",
        k: int = 5,
        image_hash: Optional[str] = None,
    ) -> tuple:
        """Encode images ONCE, return (embeddings, grouped_predictions).

        The embedding feeds both the search query and the taxonomy prediction,
        so the visual tower runs a single time. Pass ``image_hash`` to share
        that encode with a concurrent search via the per-hash cache (see
        :meth:`embed`) — this is what prevents the first-search re-encode.

        Mirrors ``TreeOfLifeClassifier.predict`` via the public-but-internal
        ``create_image_features`` + ``create_probabilities`` + ``format_*``
        methods. When pybioclip ships #169 this can call the public
        ``predict(image_features=...)`` instead.

        Returns:
            (np.ndarray (N, 768) normalized embeddings,
             list of per-image prediction lists)
        """
        emb = self.embed(images, normalize=True, image_hash=image_hash)
        preds = self.classify_from_embedding(emb, rank=rank, k=k)
        return emb, preds

    def classify_from_embedding(
        self,
        embedding,
        rank: str = "species",
        k: int = 5,
    ) -> List[List[Dict[str, Any]]]:
        """Classify from a precomputed embedding — runs NO image encoder.

        Lets a stored search embedding be reused for taxonomy prediction
        (e.g. when the user only changes the rank), so the image is never
        re-encoded. Accepts a 1-D vector or an (N, 768) array/list.
        """
        feats = torch.as_tensor(np.asarray(embedding, dtype="float32"))
        if feats.ndim == 1:
            feats = feats.unsqueeze(0)
        return self._classify_from_features(feats.to(self.device), rank, k)

    def _classify_from_features(
        self, feats: torch.Tensor, rank: str, k: int
    ) -> List[List[Dict[str, Any]]]:
        """Turn image features into ranked taxonomic predictions.

        Shared tail of :meth:`embed_and_classify` and
        :meth:`classify_from_embedding`. Scores the features against the
        BioCLIP 2 text embeddings (cosine logits -> softmax) and formats
        the top-``k`` predictions at the requested ``rank``.

        Args:
            feats: Image features of shape ``(N, 768)``. L2-normalized
                defensively here because ``create_probabilities`` computes
                cosine logits and assumes unit-norm inputs.
            rank: Taxonomic rank name (e.g. ``"species"``, ``"genus"``),
                case-insensitive; resolved to a pybioclip ``Rank`` enum.
            k: Number of top predictions to return per image.

        Returns:
            A list with one entry per input image (outer length ``N``). Each
            entry is a list of up to ``k`` prediction dicts ordered by
            descending score; each dict contains:
              - ``"file_name"``: identifier for the image (here the positional
                index as a string — the caller supplies real labels),
              - taxonomy columns down to ``rank`` (``"kingdom"`` .. the rank,
                plus ``"common_name"`` where available),
              - ``"score"``: the probability (summed over child species for
                non-species ranks).

        Note:
            The species/non-species branch mirrors
            ``TreeOfLifeClassifier.predict``. The model's text embeddings are
            leaf (species) labels, so the probability vector is per-species:
              - SPECIES -> a direct top-``k`` over those raw probabilities
                (``format_species_probs``).
              - any higher rank -> the probabilities of every species under
                each parent taxon must be **summed** before ranking
                (``format_grouped_probs``). That roll-up aggregation is the
                entire reason the two paths differ; there's no separate
                "genus head" in the model.
        """
        rank_enum = self._Rank[rank.upper()]
        clf = self._tol_classifier
        with phase("classify", n=int(feats.shape[0])):
            with torch.no_grad():
                # create_probabilities assumes L2-normalized features (cosine
                # logits). Normalize defensively — a no-op if already unit-norm.
                feats = F.normalize(feats, dim=-1)
                probs = clf.create_probabilities(
                    feats, clf.get_txt_embeddings()
                ).detach().cpu()
            grouped: List[List[Dict[str, Any]]] = []
            for i in range(probs.shape[0]):
                key = str(i)
                # SPECIES is a direct top-k over the leaf labels; higher ranks
                # must sum child-species probabilities first (see Note above).
                if rank_enum == self._Rank.SPECIES:
                    grouped.append(clf.format_species_probs(key, probs[i], k))
                else:
                    grouped.append(
                        clf.format_grouped_probs(key, probs[i], rank_enum, k=k)
                    )
        return grouped

    @_timer
    def predict(
        self,
        images: List[PIL.Image.Image],
        rank: str = "species",
        k: int = 5,
    ) -> List[List[Dict[str, Any]]]:
        """Predict taxonomy for images at the given rank.

        Returns:
            List of prediction lists, one per image.
        """
        rgb_images = [img.convert("RGB") for img in images]
        rank_enum = self._Rank[rank.upper()]
        predictions = self._tol_classifier.predict(rgb_images, rank=rank_enum, k=k)

        # Group flat prediction list by image
        results = []
        for i in range(len(images)):
            start = i * k
            results.append(predictions[start : start + k])
        return results

    @property
    def embedding_dim(self) -> int:
        return 768

    def is_ready(self) -> bool:
        return self._tol_classifier is not None
