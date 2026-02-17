"""BioCLIP model service for embedding generation and taxonomy prediction.

Extracted from bioclip-vector-db model_server.py BioCLIPModelService class.
Runs in-process — no Flask wrapper.
"""

import logging
import time
import functools
from typing import List, Dict, Any, Optional

import numpy as np
import PIL.Image

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

    @_timer
    def embed(
        self, images: List[PIL.Image.Image], normalize: bool = True
    ) -> np.ndarray:
        """Generate embeddings for a list of PIL images.

        Returns:
            np.ndarray of shape (N, 768).
        """
        rgb_images = [img.convert("RGB") for img in images]
        features = self._tol_classifier.create_image_features(
            rgb_images, normalize=normalize
        )
        return features.cpu().numpy()

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
