import atexit
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from threading import Lock

# Allow `python app.py` to find the package whether or not it is installed.
_src = Path(__file__).resolve().parent / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import gradio as gr
import torch

from src.clips import trim_video
from src.indexing import load_index, search_records
from src.logging_config import configure_logging
from src.vector_store import load_faiss_index, search_faiss_records


ROOT = Path(__file__).resolve().parent
INDEX_FILE = Path(os.getenv("VIDEO_INDEX", ROOT / "video_embeddings.json"))
MODEL_NAME = os.getenv("VIDEO_MODEL", "jinaai/jina-embeddings-v5-omni-nano-retrieval")
_model = None
_model_lock = Lock()
_index_cache: tuple | None = None  # (records, faiss_index_or_None, mtime)
_index_lock = Lock()
_preview_dir = Path(tempfile.mkdtemp(prefix="yvcs_preview_"))
atexit.register(shutil.rmtree, _preview_dir, True)
logger = logging.getLogger(__name__)
configure_logging()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info("Loading search model %s on %s", MODEL_NAME, device)
                _model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device, model_kwargs={"default_task": "retrieval"})
                logger.info("Search model loaded")
    return _model


def get_index():
    """Return (records, faiss_index_or_None), reloading only when the file changes."""
    global _index_cache, MODEL_NAME, _model
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"Index not found: {INDEX_FILE}")
    mtime = INDEX_FILE.stat().st_mtime
    with _index_lock:
        if _index_cache is None or _index_cache[2] != mtime:
            if INDEX_FILE.suffix == ".faiss":
                recs, faiss_idx = load_faiss_index(INDEX_FILE)
                _index_cache = (recs, faiss_idx, mtime)
            else:
                import json as _json
                raw = _json.loads(INDEX_FILE.read_text(encoding="utf-8"))
                stored_model = raw.get("model_name")
                if stored_model and stored_model != MODEL_NAME:
                    logger.info("Index model %s differs from %s — switching", stored_model, MODEL_NAME)
                    MODEL_NAME = stored_model
                    _model = None  # force reload with correct model
                _index_cache = (load_index(INDEX_FILE), None, mtime)
            logger.info("Index loaded/refreshed from %s", INDEX_FILE)
        return _index_cache[0], _index_cache[1]


def clip_path(clip: dict[str, object]) -> str | None:
    """Return path to stored clip file, or None if not saved."""
    raw = str(clip["clip_path"])
    if not raw:
        return None
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [ROOT / path, ROOT / "v_clips" / path, ROOT / "a_clips" / path]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def source_video_path(clip: dict[str, object]) -> str | None:
    """Return path to the source video file the clip was cut from."""
    raw = str(clip.get("source_video") or "")
    if not raw:
        return None
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def extract_preview(clip: dict[str, object]) -> str | None:
    """Extract a clip segment from the source video into a temp file for preview."""
    src = source_video_path(clip)
    if src is None:
        return None
    start = float(clip.get("start_seconds") or 0.0)
    end = float(clip.get("end_seconds") or 0.0)
    if end <= start:
        return None
    out = _preview_dir / f"preview_{clip['clip_id']}.mp4"
    try:
        trim_video(src, out, start, end)
        logger.info("Extracted preview %s → %s", clip["clip_id"], out)
        return str(out)
    except Exception:
        logger.exception("Failed to extract preview for clip %s", clip["clip_id"])
        return None


def format_time(seconds: object) -> str:
    total = max(0, int(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def search_clips(query: str, top_k: int, threshold: float | None, suppress_neighbors: bool):
    if not query.strip():
        return "Enter a description to search.", [], gr.Dropdown(choices=[]), []

    logger.info("Search request: query=%r top_k=%d", query, top_k)
    try:
        model = get_model()
        stored_records, faiss_index = get_index()
        if faiss_index is not None:
            results = search_faiss_records(stored_records, faiss_index, query, model, top_k, threshold, suppress_neighbors)
        else:
            results = search_records(stored_records, query, model, top_k, threshold, suppress_neighbors)
    except Exception as error:
        logger.exception("Search failed for query %r", query)
        return f"Search failed: {error}", [], gr.Dropdown(choices=[]), []

    rows = [
        [
            result["clip_id"],
            f"{float(result['similarity']) * 100:.1f}%",
            f"{format_time(result['start_seconds'])} - {format_time(result['end_seconds'])}",
            Path(str(result["source_video"] or result["clip_path"])).name,
        ]
        for result in results
    ]
    choices = [(f"{row[0]} | {row[1]} | {row[2]}", result["clip_id"]) for row, result in zip(rows, results)]
    status = f"{len(results)} matching scene{'s' if len(results) != 1 else ''}"
    return status, rows, gr.Dropdown(choices=choices, value=choices[0][1] if choices else None), results


def render_timeline(clip_id: str | None, results: list[dict[str, object]]) -> str:
    if not results:
        return ""
    selected = next((r for r in results if r["clip_id"] == clip_id), None)
    source = str(selected["source_video"]) if selected else None
    peers = [r for r in results if str(r.get("source_video")) == source] if source else results
    if not peers:
        peers = results
    total = max(float(r["end_seconds"]) for r in peers)
    if total <= 0:
        return "<p style='color:#888;font-size:13px;'>Timeline unavailable — re-index with timestamps.</p>"
    span = total

    def pct(s: float) -> str:
        return f"{max(0.0, min(100.0, s / span * 100)):.2f}%"

    segments = ""
    for r in peers:
        selected_flag = r["clip_id"] == clip_id
        left = pct(float(r["start_seconds"]))
        width = pct(float(r["end_seconds"]) - float(r["start_seconds"]))
        color = "#f97316" if selected_flag else "#3b82f6"
        opacity = "1" if selected_flag else "0.55"
        tip = f"{format_time(r['start_seconds'])} – {format_time(r['end_seconds'])}  {float(r['similarity'])*100:.1f}%"
        segments += (
            f'<div title="{tip}" style="position:absolute;left:{left};width:{width};'
            f'top:0;bottom:0;background:{color};opacity:{opacity};border-radius:3px;min-width:4px;"></div>'
        )

    n = 6
    markers = ""
    for i in range(n + 1):
        t = span * i / n
        markers += (
            f'<div style="position:absolute;left:{i/n*100:.1f}%;transform:translateX(-50%);'
            f'font-size:11px;color:#9ca3af;">{format_time(t)}</div>'
        )

    return f"""
<div style="padding:12px 4px 4px;">
  <div style="font-size:12px;font-weight:600;color:#6b7280;letter-spacing:.05em;margin-bottom:6px;">TIMELINE</div>
  <div style="position:relative;height:26px;background:#e5e7eb;border-radius:4px;margin-bottom:18px;overflow:visible;">
    {segments}
  </div>
  <div style="position:relative;height:14px;">{markers}</div>
  <div style="margin-top:10px;font-size:11px;color:#9ca3af;">
    <span style="display:inline-block;width:10px;height:10px;background:#f97316;border-radius:2px;vertical-align:middle;margin-right:3px;"></span>selected&ensp;
    <span style="display:inline-block;width:10px;height:10px;background:#3b82f6;border-radius:2px;vertical-align:middle;margin-right:3px;opacity:.55;"></span>other matches
  </div>
</div>"""


def preview_clip(clip_id: str | None, results: list[dict[str, object]]):
    result = next((item for item in results if item["clip_id"] == clip_id), None)
    timeline = render_timeline(clip_id, results)
    if result is None:
        return None, None, "Select a search result to preview it.", timeline
    path = clip_path(result) or extract_preview(result)
    if path is None:
        src = source_video_path(result)
        msg = "Source video not found." if src is None else f"Could not extract preview from {Path(src).name}."
        return None, None, msg, timeline
    source_name = Path(str(result["source_video"] or result["clip_path"])).name
    details = (
        f"**{source_name}**  \n"
        f"Match: **{float(result['similarity']) * 100:.1f}%**  \n"
        f"Timestamp: **{format_time(result['start_seconds'])} – {format_time(result['end_seconds'])}**"
    )
    return path, path, details, timeline


with gr.Blocks(title="Video Scene Search") as demo:
    gr.Markdown("# Find the moment\nDescribe a scene, object, or action in your indexed videos.")
    with gr.Row():
        query = gr.Textbox(label="Search description", placeholder="a man holding a stick", scale=5)
        top_k = gr.Slider(1, 20, value=8, step=1, label="Results", scale=1)
        search_button = gr.Button("Search", variant="primary", scale=1)
    with gr.Row():
        threshold = gr.Slider(-1, 1, value=None, step=0.01, label="Minimum similarity", scale=1)
        suppress_neighbors = gr.Checkbox(value=True, label="Suppress neighboring clips", scale=1)
    status = gr.Markdown("Ready to search.")
    result_table = gr.Dataframe(
        headers=["Clip", "Match", "Timestamp", "Source"],
        datatype=["str", "str", "str", "str"],
        label="Matching scenes",
        interactive=False,
    )
    result_state = gr.State([])
    clip_selector = gr.Dropdown(label="Preview result", choices=[], interactive=True)
    timeline = gr.HTML()
    with gr.Row():
        video = gr.Video(label="Clip preview", interactive=False)
        with gr.Column():
            details = gr.Markdown("Select a search result to preview it.")
            download = gr.File(label="Download clip", interactive=False)

    search_button.click(
        search_clips,
        inputs=[query, top_k, threshold, suppress_neighbors],
        outputs=[status, result_table, clip_selector, result_state],
    )
    query.submit(
        search_clips,
        inputs=[query, top_k, threshold, suppress_neighbors],
        outputs=[status, result_table, clip_selector, result_state],
    )
    clip_selector.change(
        preview_clip,
        inputs=[clip_selector, result_state],
        outputs=[video, download, details, timeline],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
