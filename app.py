"""BioCLIP Lite — Lightweight Image Search Application.

A single-process Gradio app combining BioCLIP-2 model inference, FAISS vector
search, DuckDB metadata lookup, and URL-based image retrieval.

Usage:
    python app.py \
        --faiss-index /path/to/index.index \
        --duckdb-path /path/to/metadata.duckdb \
        --device cpu \
        --scope all
"""

import hashlib
import io
import logging
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.bioclip_lite.config import LiteConfig, parse_args, resolve_data_paths, setup_logging
from src.bioclip_lite.services.image_service import ImageService
from src.bioclip_lite.services.model_service import ModelService
from src.bioclip_lite.services.search_service import SearchService

logger = logging.getLogger(__name__)

CSS = """
.custom-gallery { height: 640px !important; overflow-y: auto !important; }
"""

SCOPE_CHOICES = ["All Sources", "URL-Available Only", "iNaturalist Only"]


def _image_hash(img: Image.Image) -> str:
    """Fast perceptual hash for cache key (MD5 of downscaled bytes)."""
    small = img.copy()
    small.thumbnail((64, 64))
    return hashlib.md5(small.tobytes()).hexdigest()


def _placeholder(label: str, size: int = 256) -> Image.Image:
    """Create a gray placeholder image with a text label."""
    img = Image.new("RGB", (size, size), color=(80, 80, 80))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()
    lines = ["Image unavailable", label[:40]]
    y = size // 2 - 20
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        draw.text(((size - w) // 2, y), line, fill="white", font=font)
        y += 20
    return img


class BioCLIPLiteApp:
    """Main application orchestrating all services."""

    def __init__(self, config: LiteConfig):
        self.config = config

        logger.info("Initializing services...")
        self.model_service = ModelService(
            device=config.device, model_str=config.model_str
        )
        self.search_service = SearchService(
            faiss_index_path=config.faiss_index_path,
            duckdb_path=config.duckdb_path,
            nprobe=config.default_nprobe,
            over_fetch_factor=config.over_fetch_factor,
            metadata_columns=config.METADATA_COLUMNS,
        )
        self.image_service = ImageService(
            timeout=config.image_fetch_timeout,
            max_workers=config.image_fetch_max_workers,
            thumbnail_max_dim=config.thumbnail_max_dim,
        )
        logger.info("All services ready.")

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def on_image_upload(
        self,
        img: Optional[Image.Image],
        rank: str,
    ) -> Tuple[Optional[List[float]], str, Optional[str]]:
        """Called on image upload: embed + predict immediately.

        Returns (embedding, prediction_html, image_hash).
        """
        if img is None:
            return None, _prediction_placeholder(), None

        img_h = _image_hash(img)
        logger.info(f"Image uploaded (hash={img_h[:8]}, size={img.size}), embedding + predicting at rank={rank}")
        embedding = self.model_service.embed([img], normalize=False)[0].tolist()
        pred_html = self._predict_html(img, rank)
        return embedding, pred_html, img_h

    def search(
        self,
        img: Optional[Image.Image],
        cached_embedding: Optional[List[float]],
        cached_hash: Optional[str],
        top_n: int,
        nprobe: int,
        scope: str,
    ) -> Tuple[List[Image.Image], str, List[Dict], Optional[List[float]], Optional[str]]:
        """Run FAISS search, fetch images, return gallery + metadata.

        Returns (gallery_images, tree_summary, metadata_list,
                 updated_embedding, updated_hash).
        """
        if img is None:
            return [], "No results.", [], None, None

        # Reuse cached embedding if image hasn't changed
        current_hash = _image_hash(img)
        if cached_embedding and cached_hash == current_hash:
            query_vector = np.array(cached_embedding, dtype="float32")
            logger.info("Reusing cached embedding")
        else:
            query_vector = self.model_service.embed([img], normalize=False)[0]
            cached_embedding = query_vector.tolist()
            cached_hash = current_hash

        # FAISS + DuckDB search
        results = self.search_service.search(
            query_vector=query_vector,
            top_n=int(top_n),
            nprobe=int(nprobe),
            scope=scope,
        )

        if not results:
            logger.info("Search returned 0 results")
            return [], "No results found.", [], cached_embedding, cached_hash

        logger.info(f"FAISS+DuckDB returned {len(results)} results, fetching images...")

        # Fetch images from URLs
        results = self.image_service.fetch_images(results)

        ok = sum(1 for r in results if r.get("image_status") == "ok")
        logger.info(f"Image fetch complete: {ok}/{len(results)} succeeded")

        # Build gallery — pass full-res images so Gradio's lightbox popup
        # shows high quality. Gradio handles grid thumbnail scaling via CSS.
        gallery_images = []
        display_metadata = []
        for r in results:
            pil_img = r.get("image")
            if pil_img:
                gallery_images.append(pil_img)
            else:
                label = r.get("species") or r.get("common_name") or "Unknown"
                gallery_images.append(_placeholder(label))
            display_metadata.append(r)

        tree = self._generate_tree_summary(display_metadata)
        return gallery_images, tree, display_metadata, cached_embedding, cached_hash

    def on_gallery_select(
        self, evt: gr.SelectData, metadata_list: List[Dict]
    ) -> Tuple[Optional[Image.Image], str, str, str]:
        """Handle gallery click — show full-res image + metadata.

        Returns (image, header_md, taxonomy_md, source_md).
        """
        empty = (None, "*No image selected*", "", "")
        if not metadata_list or evt.index >= len(metadata_list):
            return empty

        meta = metadata_list[evt.index]

        # Try to show the already-fetched image (or fetch full-res)
        img = meta.get("image")
        if img is None:
            url = meta.get("identifier")
            if url:
                img, status = self.image_service.fetch_full_resolution(url)
                if img:
                    meta["image"] = img

        header, taxonomy, source = self._format_metadata(meta, evt.index + 1)
        return img, header, taxonomy, source

    def predict_on_rank_change(
        self, img: Optional[Image.Image], rank: str
    ) -> str:
        """Re-predict when user changes taxonomic rank."""
        if img is None:
            return _prediction_placeholder()
        return self._predict_html(img, rank)

    def export_results(self, metadata_list: List[Dict]) -> Optional[str]:
        """Export current results as a zip file."""
        images = [m.get("image") for m in metadata_list if m.get("image")]
        if not images:
            return None
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            tmp = tempfile.NamedTemporaryFile(
                suffix=".zip", prefix=f"bioclip_lite_{ts}_", delete=False
            )
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, img in enumerate(images, 1):
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    zf.writestr(f"result_{i}.png", buf.getvalue())
            tmp.close()
            return tmp.name
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _predict_html(self, img: Image.Image, rank: str, k: int = 5) -> str:
        """Generate prediction HTML."""
        try:
            preds = self.model_service.predict([img], rank=rank.lower(), k=k)
            if preds and preds[0]:
                return _format_predictions(preds[0], rank)
            return "<p style='color:#f88;'>No predictions returned.</p>"
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return f"<p style='color:#f88;'>Prediction error: {e}</p>"

    @staticmethod
    def _format_metadata(meta: Dict, rank: int) -> Tuple[str, str, str]:
        """Format metadata into three sections: header, taxonomy, source.

        Returns:
            (header_md, taxonomy_md, source_md)
        """
        common = meta.get("common_name") or "Unknown"
        species = meta.get("species") or meta.get("scientific_name") or "Unknown"
        dist = meta.get("distance", 0)
        img_status = meta.get("image_status", "")

        # ── Header (always visible above the tabs) ──
        header_lines = [
            f"### #{rank} {common}",
            f"*{species}*",
            "",
            f"**Distance:** `{dist:.4f}`",
        ]
        if img_status and img_status != "ok":
            header_lines.append(f"  \n> Image: `{img_status}`")

        # ── Taxonomy tab ──
        tax_lines = [
            "| Rank | Name |",
            "| :--- | :--- |",
        ]
        for label, key in [
            ("Kingdom", "kingdom"), ("Phylum", "phylum"), ("Class", "class"),
            ("Order", "order"), ("Family", "family"), ("Genus", "genus"),
            ("Species", "species"),
        ]:
            val = meta.get(key) or "—"
            tax_lines.append(f"| {label} | {val} |")

        # ── Source tab ──
        source = meta.get("source_dataset", "Unknown")
        source_id = meta.get("source_id", "")
        if source and source.lower() == "gbif" and source_id:
            source_display = f"[GBIF](https://gbif.org/occurrence/{source_id})"
        else:
            source_display = source or "Unknown"

        url = meta.get("identifier", "")
        src_lines = [
            "| | |",
            "| :--- | :--- |",
            f"| **Dataset** | {source_display} |",
            f"| **Publisher** | {meta.get('publisher') or '—'} |",
            f"| **Type** | {meta.get('img_type') or '—'} |",
        ]
        if url:
            src_lines.append(f"| **URL** | [View Original]({url}) |")

        return "\n".join(header_lines), "\n".join(tax_lines), "\n".join(src_lines)

    @staticmethod
    def _generate_tree_summary(metadata_list: List[Dict]) -> str:
        if not metadata_list:
            return "No results to summarize."

        # Build nested tree: kingdom > phylum > class > order > family > genus > species
        RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")

        def _nested():
            return defaultdict(_nested)

        root = _nested()
        for m in metadata_list:
            node = root
            for r in RANKS:
                node = node[m.get(r) or "Unknown"]

        def _count(node) -> int:
            if not node:
                return 1  # leaf
            return sum(_count(child) for child in node.values())

        def _render(node, prefix="", is_last_list=None):
            """Recursively render the tree with box-drawing characters."""
            if is_last_list is None:
                is_last_list = []
            lines = []
            items = sorted(node.items())
            for i, (name, children) in enumerate(items):
                is_last = i == len(items) - 1
                count = _count(children)

                # Build the branch prefix
                if not is_last_list:
                    connector = "├── " if not is_last else "└── "
                else:
                    connector = "├── " if not is_last else "└── "

                line_prefix = ""
                for prev_last in is_last_list:
                    line_prefix += "    " if prev_last else "│   "

                lines.append(f"{line_prefix}{connector}{name} ({count})")

                if children:
                    lines.extend(
                        _render(children, prefix, is_last_list + [is_last])
                    )
            return lines

        lines = [f"Search Results: {len(metadata_list)} images", ""]
        lines.extend(_render(root))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Gradio interface
    # ------------------------------------------------------------------

    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(title="BioCLIP Image Search Lite", css=CSS) as demo:
            gr.Markdown("# BioCLIP Image Search Lite")

            # Session state
            embedding_state = gr.State(value=None)
            hash_state = gr.State(value=None)
            metadata_state = gr.State(value=[])

            with gr.Row():
                # --- Left column: controls ---
                with gr.Column(scale=1, min_width=280):
                    img = gr.Image(type="pil", label="Upload Image", height=300)

                    scope_dropdown = gr.Dropdown(
                        choices=SCOPE_CHOICES,
                        value=SCOPE_CHOICES[0],
                        label="Image Source Scope",
                        info="Filter results by image source availability.",
                    )
                    nprobe = gr.Slider(
                        1, 128, value=self.config.default_nprobe, step=1,
                        label="Search Depth (nprobe)",
                        info="IVF partitions to search. Higher = more accurate, slower.",
                    )
                    top_n = gr.Slider(
                        1, 128, value=self.config.default_top_n, step=1,
                        label="Top N Results",
                    )
                    run_btn = gr.Button("Search", variant="primary")
                    export_btn = gr.Button(
                        "Export Results", variant="secondary",
                        visible=self.config.enable_export,
                    )
                    download_file = gr.File(
                        label="Export", visible=self.config.enable_export
                    )

                # --- Middle column: gallery + prediction ---
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.TabItem("Search Results"):
                            gallery = gr.Gallery(
                                label="Results",
                                columns=4,
                                height=580,
                                elem_classes="custom-gallery",
                            )
                        with gr.TabItem("Prediction"):
                            rank_dropdown = gr.Dropdown(
                                choices=[
                                    "kingdom", "phylum", "class", "order",
                                    "family", "genus", "species",
                                ],
                                value="species",
                                label="Taxonomic Rank",
                            )
                            prediction_output = gr.HTML(
                                value=_prediction_placeholder()
                            )

                # --- Right column: detail view ---
                with gr.Column(scale=1, min_width=300):
                    with gr.Tabs():
                        with gr.TabItem("Selected"):
                            selected_image = gr.Image(
                                label="Selected Image", height=280,
                                show_label=False,
                            )
                            metadata_header = gr.Markdown(
                                value="*Click an image to see details*"
                            )
                            with gr.Tabs():
                                with gr.TabItem("Taxonomy"):
                                    taxonomy_display = gr.Markdown()
                                with gr.TabItem("Source"):
                                    source_display = gr.Markdown()
                        with gr.TabItem("Summary"):
                            tree_summary = gr.Code(
                                label="Taxonomy Tree", language=None, lines=25,
                                value="Run a search to see summary.",
                            )

            # --- Event wiring ---

            # Embed + predict on upload
            img.change(
                self.on_image_upload,
                inputs=[img, rank_dropdown],
                outputs=[embedding_state, prediction_output, hash_state],
            )

            # Search
            run_btn.click(
                self.search,
                inputs=[
                    img, embedding_state, hash_state,
                    top_n, nprobe, scope_dropdown,
                ],
                outputs=[
                    gallery, tree_summary, metadata_state,
                    embedding_state, hash_state,
                ],
            )

            # Re-predict on rank change
            rank_dropdown.change(
                self.predict_on_rank_change,
                inputs=[img, rank_dropdown],
                outputs=[prediction_output],
            )

            # Gallery click
            gallery.select(
                self.on_gallery_select,
                inputs=[metadata_state],
                outputs=[selected_image, metadata_header, taxonomy_display, source_display],
            )

            # Export
            if self.config.enable_export:
                export_btn.click(
                    self.export_results,
                    inputs=[metadata_state],
                    outputs=[download_file],
                )

        return demo


# ------------------------------------------------------------------
# Prediction formatting (shared helper)
# ------------------------------------------------------------------

def _prediction_placeholder() -> str:
    return "<p style='color:#888;'>Upload an image to get predictions.</p>"


def _format_predictions(predictions: List[Dict], rank: str) -> str:
    if not predictions:
        return "<p style='color:#888;'>No predictions available.</p>"

    top = predictions[0]
    parts = []
    for key in ("kingdom", "phylum", "class", "order", "family", "genus"):
        v = top.get(key)
        if v:
            parts.append(v)
    epithet = top.get("species_epithet", "")
    common = top.get("common_name", "")
    header = " ".join(parts)
    if epithet:
        header += f" {epithet}"
    if common:
        header += f" ({common})"

    html = f"""<div style="font-family:system-ui,sans-serif;">
<h3 style="color:#ff9500;margin-bottom:16px;font-size:18px;line-height:1.4;">{header}</h3>
<hr style="border:none;border-top:2px solid #ff9500;margin-bottom:16px;">"""

    for pred in predictions:
        score = pred.get("score", 0)
        pct = score * 100
        label_parts = []
        for key in ("kingdom", "phylum", "class", "order", "family", "genus"):
            v = pred.get(key)
            if v:
                label_parts.append(v)
        epithet = pred.get("species_epithet", "")
        cn = pred.get("common_name", "")
        if epithet:
            label_parts.append(epithet)
        label = " ".join(label_parts)
        if cn:
            label += f" ({cn})"
        bar_color = "#ff9500" if pct >= 50 else "#ffb347" if pct >= 10 else "#666"
        html += f"""<div style="margin-bottom:12px;">
<div style="display:flex;justify-content:space-between;margin-bottom:4px;">
<span style="color:#ddd;font-size:14px;">{label}</span>
<span style="color:#888;font-size:14px;">{pct:.0f}%</span></div>
<div style="background:#333;border-radius:4px;height:6px;overflow:hidden;">
<div style="background:{bar_color};width:{pct}%;height:100%;border-radius:4px;"></div>
</div></div>"""

    html += "</div>"
    return html


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    config = parse_args()
    setup_logging(config)
    resolve_data_paths(config)

    logger.info("=" * 60)
    logger.info("BioCLIP Lite — starting up")
    logger.info(f"  FAISS index: {config.faiss_index_path}")
    logger.info(f"  DuckDB:      {config.duckdb_path}")
    logger.info(f"  Device:      {config.device}")
    logger.info(f"  Scope:       {config.scope}")
    logger.info(f"  Log dir:     {config.log_dir or '(console only)'}")
    logger.info("=" * 60)

    app = BioCLIPLiteApp(config)
    demo = app.create_interface()
    logger.info(f"Launching Gradio on {config.host}:{config.port}")
    demo.launch(server_name=config.host, server_port=config.port)


if __name__ == "__main__":
    main()
