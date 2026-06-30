"""URL-based image fetching with iNaturalist rate-limit compliance.

iNaturalist rate limits (https://www.inaturalist.org/pages/api+recommended+practices):
  - ~1 request/sec, ~10k requests/day
  - 5 GB media/hour, 24 GB media/day  (permanent block if exceeded)

Key distinction:
  - inaturalist-open-data.s3.amazonaws.com  → AWS Open Data, no iNat rate limits
  - static.inaturalist.org                  → iNat CDN, subject to above limits
"""

import io
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from PIL import Image

logger = logging.getLogger(__name__)

# iNaturalist AWS Open Data bucket — 59% of our URLs. Serves size variants
# (original|large|medium|small|thumb) at /photos/<id>/<variant>.<ext>.
# Documented at https://github.com/inaturalist/inaturalist-open-data.
INAT_S3_HOST = "inaturalist-open-data.s3.amazonaws.com"
# Captures: (scheme+host+/photos/<id>/)(variant)(.ext)(?querystring)
_INAT_S3_PATH_RE = re.compile(r"^(.*?/photos/\d+/)(\w+)(\.\w+)(\?.*)?$")
INAT_S3_VARIANTS = ("original", "large", "medium", "small", "thumb")

# Sentinel distinguishing "url absent from cache" from "url cached as a miss".
_CACHE_MISS = object()

# Domains served via AWS Open Data (no iNat rate limiting)
S3_OPEN_DATA_DOMAINS = frozenset({
    "inaturalist-open-data.s3.amazonaws.com",
})

# Domains subject to iNat CDN rate limits
RATE_LIMITED_DOMAINS = frozenset({
    "static.inaturalist.org",
})

USER_AGENT = "BioCLIP-Lite/1.0 (academic research; imageomics.org)"


class _TokenBucket:
    """Simple token-bucket rate limiter."""

    def __init__(self, rate: float = 1.0):
        self._rate = rate  # tokens per second
        self._tokens = rate
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self._rate, self._tokens + (now - self._last) * self._rate)
            self._last = now
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._rate
                time.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


class ImageService:
    """Fetches images from source URLs with respectful rate limiting."""

    def __init__(
        self,
        timeout: int = 10,
        max_workers: int = 8,
        thumbnail_max_dim: int = 256,
        variant: Optional[str] = None,
        enable_cache: bool = False,
        cache_max: int = 512,
    ):
        self.timeout = timeout
        self.max_workers = max_workers
        self.thumbnail_max_dim = thumbnail_max_dim

        # Size variant to request for iNat S3 URLs (None = leave URL as-is,
        # i.e. whatever the DB stored, usually 'original'). See INAT_S3_VARIANTS.
        self.variant = variant

        # Optional process-level image cache (url -> PIL.Image | None).
        # Off by default to preserve current behavior; the latency benchmark
        # flips it on to measure warm-cache searches.
        self.enable_cache = enable_cache
        self._cache_max = cache_max
        self._cache: Dict[str, Optional[Image.Image]] = {}
        self._cache_lock = threading.Lock()

        # Bytes downloaded during the most recent fetch_images() call.
        self._last_call_bytes = 0

        # Rate limiter for iNat CDN domains (1 req/sec)
        self._cdn_limiter = _TokenBucket(rate=1.0)

        # Persistent HTTP session with connection pooling
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max_workers,
            pool_maxsize=max_workers,
            max_retries=1,
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers["User-Agent"] = USER_AGENT

        # Bandwidth tracking
        self._bytes_lock = threading.Lock()
        self._bytes_fetched: Dict[str, int] = {}

    def fetch_images(self, metadata_list: List[Dict]) -> List[Dict]:
        """Fetch images for search results in parallel (respecting rate limits).

        Modifies each dict in-place, adding 'image' (PIL.Image or None)
        and 'image_status' fields.
        """
        t0 = time.monotonic()
        self._last_call_bytes = 0  # reset per-call byte counter

        # Partition into rate-limited vs unrestricted
        rate_limited_indices = []
        unrestricted_indices = []
        no_url_indices = []

        for i, item in enumerate(metadata_list):
            url = item.get("identifier") or ""
            if not url:
                no_url_indices.append(i)
                item["image"] = None
                item["image_status"] = "no_url"
                continue
            domain = urlparse(url).hostname or ""
            if domain in RATE_LIMITED_DOMAINS:
                rate_limited_indices.append(i)
            else:
                unrestricted_indices.append(i)

        logger.debug(
            f"Fetch plan: {len(unrestricted_indices)} parallel, "
            f"{len(rate_limited_indices)} rate-limited, "
            f"{len(no_url_indices)} no-url"
        )

        # Fetch unrestricted URLs in parallel
        if unrestricted_indices:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {
                    pool.submit(
                        self._fetch_single,
                        self.rewrite_variant(metadata_list[i].get("identifier")),
                    ): i
                    for i in unrestricted_indices
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        img, status = future.result()
                    except Exception as e:
                        img, status = None, f"error:{e}"
                    metadata_list[idx]["image"] = img
                    metadata_list[idx]["image_status"] = status

        # Fetch rate-limited URLs sequentially with throttling
        for i in rate_limited_indices:
            self._cdn_limiter.acquire()
            try:
                img, status = self._fetch_single(
                    self.rewrite_variant(metadata_list[i].get("identifier"))
                )
            except Exception as e:
                img, status = None, f"error:{e}"
            metadata_list[i]["image"] = img
            metadata_list[i]["image_status"] = status

        dt = time.monotonic() - t0
        statuses = [m.get("image_status", "?") for m in metadata_list]
        ok = sum(1 for s in statuses if s in ("ok", "ok_cached"))
        cached = statuses.count("ok_cached")
        no_url = statuses.count("no_url")
        logger.info(
            f"Fetched {len(metadata_list)} images in {dt:.2f}s "
            f"(ok={ok}, cached={cached}, no_url={no_url}, "
            f"failed={len(metadata_list) - ok - no_url}, "
            f"{self._last_call_bytes / 1024:.0f}KB)"
        )
        return metadata_list

    def _fetch_single(self, url: Optional[str]) -> Tuple[Optional[Image.Image], str]:
        """Fetch one image. Returns (PIL Image or None, status string)."""
        if not url:
            return None, "no_url"

        # Cache lookup (cached hits cost zero bytes — this is the warm path)
        if self.enable_cache:
            hit = self._cache_get(url)
            if hit is not _CACHE_MISS:
                return hit, "ok_cached" if hit is not None else "cached_fail"

        domain = urlparse(url).hostname or "?"
        t0 = time.monotonic()
        try:
            resp = self.session.get(url, timeout=self.timeout)
            dt = time.monotonic() - t0
            if resp.status_code == 200:
                self._track_bytes(url, len(resp.content))
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                logger.debug(f"OK {domain} {len(resp.content)/1024:.0f}KB {dt:.2f}s")
                self._cache_put(url, img)
                return img, "ok"
            elif resp.status_code == 429:
                logger.warning(f"Rate limited (429) from {domain} after {dt:.2f}s")
                return None, "rate_limited"  # transient — do not cache
            else:
                logger.debug(f"HTTP {resp.status_code} from {domain} after {dt:.2f}s")
                if 400 <= resp.status_code < 500:
                    self._cache_put(url, None)  # permanent miss — cache it
                return None, f"http_{resp.status_code}"
        except requests.Timeout:
            logger.debug(f"Timeout from {domain} after {self.timeout}s")
            return None, "timeout"  # transient — do not cache
        except Exception as e:
            logger.debug(f"Error from {domain}: {e}")
            return None, f"error:{str(e)[:80]}"

    def fetch_full_resolution(self, url: Optional[str]) -> Tuple[Optional[Image.Image], str]:
        """Fetch a single image at full resolution (for on-click detail view)."""
        if not url:
            return None, "no_url"
        domain = urlparse(url).hostname or ""
        if domain in RATE_LIMITED_DOMAINS:
            self._cdn_limiter.acquire()
        return self._fetch_single(url)

    def make_thumbnail(self, img: Image.Image) -> Image.Image:
        """Resize to thumbnail for gallery display."""
        thumb = img.copy()
        thumb.thumbnail(
            (self.thumbnail_max_dim, self.thumbnail_max_dim), Image.LANCZOS
        )
        return thumb

    def rewrite_variant(self, url: Optional[str]) -> Optional[str]:
        """Rewrite an iNat S3 URL to ``self.variant`` size, if applicable.

        No-op when ``self.variant`` is None, the host isn't the iNat S3
        bucket, or the path doesn't match the documented
        ``/photos/<id>/<variant>.<ext>`` shape. Other hosts (observation.org,
        eol, etc.) are left untouched — they have their own URL grammars.
        """
        if not url or not self.variant or INAT_S3_HOST not in url:
            return url
        m = _INAT_S3_PATH_RE.match(url)
        if not m:
            return url
        head, _old_variant, ext, qs = m.groups()
        return f"{head}{self.variant}{ext}{qs or ''}"

    # ------------------------------------------------------------------
    # Image cache (process-level, bounded, thread-safe). Caches misses too.
    # ------------------------------------------------------------------
    def _cache_get(self, url: str):
        with self._cache_lock:
            return self._cache.get(url, _CACHE_MISS)

    def _cache_put(self, url: str, img: Optional[Image.Image]):
        with self._cache_lock:
            if url not in self._cache and len(self._cache) >= self._cache_max:
                # FIFO trim: drop oldest insertion (dicts preserve order)
                self._cache.pop(next(iter(self._cache)), None)
            self._cache[url] = img

    @property
    def last_call_bytes(self) -> int:
        """Bytes downloaded during the most recent fetch_images() call."""
        return self._last_call_bytes

    def _track_bytes(self, url: str, nbytes: int):
        domain = urlparse(url).hostname or "unknown"
        with self._bytes_lock:
            self._last_call_bytes += nbytes
            self._bytes_fetched[domain] = self._bytes_fetched.get(domain, 0) + nbytes
            total = self._bytes_fetched[domain]
            # Warn at 4 GB/hr for rate-limited domains
            if domain in RATE_LIMITED_DOMAINS and total > 4 * 1024**3:
                logger.warning(
                    f"High bandwidth for {domain}: {total / 1024**3:.1f} GB fetched this session"
                )

    def close(self):
        self.session.close()
