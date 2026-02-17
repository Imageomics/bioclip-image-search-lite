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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from PIL import Image

logger = logging.getLogger(__name__)

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
    ):
        self.timeout = timeout
        self.max_workers = max_workers
        self.thumbnail_max_dim = thumbnail_max_dim

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
        # Partition into rate-limited vs unrestricted
        rate_limited_indices = []
        unrestricted_indices = []

        for i, item in enumerate(metadata_list):
            url = item.get("identifier") or ""
            domain = urlparse(url).hostname or ""
            if domain in RATE_LIMITED_DOMAINS:
                rate_limited_indices.append(i)
            else:
                unrestricted_indices.append(i)

        # Fetch unrestricted URLs in parallel
        if unrestricted_indices:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {
                    pool.submit(
                        self._fetch_single, metadata_list[i].get("identifier")
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
                img, status = self._fetch_single(metadata_list[i].get("identifier"))
            except Exception as e:
                img, status = None, f"error:{e}"
            metadata_list[i]["image"] = img
            metadata_list[i]["image_status"] = status

        return metadata_list

    def _fetch_single(self, url: Optional[str]) -> Tuple[Optional[Image.Image], str]:
        """Fetch one image. Returns (PIL Image or None, status string)."""
        if not url:
            return None, "no_url"

        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                self._track_bytes(url, len(resp.content))
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                return img, "ok"
            elif resp.status_code == 429:
                logger.warning(f"Rate limited (429) for {url}")
                return None, "rate_limited"
            else:
                return None, f"http_{resp.status_code}"
        except requests.Timeout:
            return None, "timeout"
        except Exception as e:
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

    @staticmethod
    def get_thumbnail_url(url: str) -> str:
        """Transform URL to thumbnail variant if the CDN supports it."""
        if "static.inaturalist.org" in url and "/original/" in url:
            return url.replace("/original/", "/medium/")
        return url

    def _track_bytes(self, url: str, nbytes: int):
        domain = urlparse(url).hostname or "unknown"
        with self._bytes_lock:
            self._bytes_fetched[domain] = self._bytes_fetched.get(domain, 0) + nbytes
            total = self._bytes_fetched[domain]
            # Warn at 4 GB/hr for rate-limited domains
            if domain in RATE_LIMITED_DOMAINS and total > 4 * 1024**3:
                logger.warning(
                    f"High bandwidth for {domain}: {total / 1024**3:.1f} GB fetched this session"
                )

    def close(self):
        self.session.close()
