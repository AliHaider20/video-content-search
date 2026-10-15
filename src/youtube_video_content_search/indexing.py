import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Iterable

import ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from .models import ClipRecord


INDEX_VERSION = 1
logger = logging.getLogger(__name__)


def save_index(records: Iterable[ClipRecord], output_file: str | Path, model_name: str | None = None) -> None:
    payload: dict = {
        "version": INDEX_VERSION,
        "records": [record.to_dict() for record in records],
    }
    if model_name:
        payload["model_name"] = model_name
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved JSON index with %d records to %s", len(payload["records"]), output_path)


def load_index(index_file: str | Path) -> list[ClipRecord]:
    index_path = Path(index_file)
    data = json.loads(index_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "records" in data:
        records = [ClipRecord.from_dict(item) for item in data["records"]]
        logger.info("Loaded JSON index with %d records from %s", len(records), index_path)
        return records

    # Read the original path -> vector format so existing indexes remain usable.
    clip_length = _detect_clip_length(index_path.parent, data)
    records = []
    for path, vector in data.items():
        stem = Path(path).stem
        try:
            idx = int(stem)
            start = idx * clip_length
            end = start + clip_length
        except ValueError:
            start, end = 0.0, 0.0
        records.append(ClipRecord(
            clip_id=stem, source_video="", clip_path=path,
            start_seconds=start, end_seconds=end, embedding=vector,
        ))
    logger.info("Loaded legacy index with %d records from %s", len(records), index_path)
    return records


def _detect_clip_length(base_dir: Path, data: dict) -> float:
    try:
        import av  # noqa: PLC0415
        subdirs = [base_dir, base_dir / "v_clips", base_dir / "clips"]
        for path in data:
            p = Path(path)
            candidates = [p] if p.is_absolute() else [d / p.name for d in subdirs]
            for candidate in candidates:
                if candidate.exists():
                    with av.open(str(candidate)) as container:
                        if container.duration:
                            return float(container.duration) / 1_000_000
                    break
    except Exception:
        pass
    return 0.0


def make_records(
    clip_paths: Iterable[str | Path],
    source_video: str | Path = "",
    clip_length: float = 0.0,
    overlap: float = 0.0,
) -> list[ClipRecord]:
    records = []
    for index, clip_path in enumerate(sorted(map(Path, clip_paths), key=lambda path: path.name)):
        # Window runs from (start_time - overlap) to (start_time + clip_length),
        # matching the clip_window() call in create_clips.
        start = max(0.0, index * clip_length - overlap)
        end = (index + 1) * clip_length if clip_length else 0.0
        records.append(
            ClipRecord(
                clip_id=clip_path.stem,
                source_video=str(source_video),
                clip_path=str(clip_path),
                start_seconds=start,
                end_seconds=end,
            )
        )
    return records


def _trim_for_embedding(input_file: Path, output_file: Path, start: float, end: float, fps: int = 2) -> None:
    """Trim a clip re-encoded at low fps to keep frame count manageable for embedding."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    stream = ffmpeg.input(str(input_file), ss=start, to=end).video
    ffmpeg.run(
        ffmpeg.output(stream, str(output_file), vf=f"fps={fps}", vcodec="libx264", preset="ultrafast", crf=28),
        overwrite_output=True, quiet=True,
    )


def _load_embedding_model(model_name: str, device: str | None = None) -> SentenceTransformer:
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading embedding model %s on %s", model_name, selected_device)
    model_kwargs: dict = {"default_task": "retrieval", "modality": "vision"}
    if selected_device == "cuda":
        model_kwargs["dtype"] = torch.bfloat16
    return SentenceTransformer(
        model_name,
        trust_remote_code=True,
        device=selected_device,
        model_kwargs=model_kwargs,
    ).to(selected_device)


def embed_videos(
    records: list[ClipRecord],
    model_name: str = "jinaai/jina-embeddings-v5-omni-nano",
    batch_size: int = 2,
    device: str | None = None,
) -> list[ClipRecord]:
    if not records:
        raise ValueError("Cannot index an empty clip collection")
    model = _load_embedding_model(model_name, device)
    paths = [record.clip_path for record in records]
    vectors = model.encode_document(paths, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    for record, vector in zip(records, vectors):
        record.embedding = np.asarray(vector).tolist()
    logger.info("Embedded %d clips", len(records))
    return records


def embed_video_windows(
    source_video: str | Path,
    clip_length: float = 10.0,
    overlap: float = 3.0,
    model_name: str = "jinaai/jina-embeddings-v5-omni-nano",
    batch_size: int = 2,
    device: str | None = None,
) -> list[ClipRecord]:
    """Embed a video by sliding window without saving clips to disk.

    Temporary clip files are created per batch, embedded, then immediately deleted.
    The returned records hold start/end timestamps and embeddings; no clip files persist.
    """
    from .clips import clip_window, get_video_duration  # avoid circular at module level

    source_path = Path(source_video)
    duration = get_video_duration(source_path)
    logger.info("Windowing %s (%.1fs, clip=%.1fs, overlap=%.1fs)", source_path.name, duration, clip_length, overlap)

    windows: list[tuple[int, float, float]] = []
    start = 0.0
    idx = 0
    while start < duration:
        w_start, w_end = clip_window(start, clip_length, overlap, duration)
        windows.append((idx, w_start, w_end))
        idx += 1
        start += clip_length

    model = _load_embedding_model(model_name, device)

    records: list[ClipRecord] = []
    total_batches = (len(windows) + batch_size - 1) // batch_size
    for batch_num, batch_start in enumerate(range(0, len(windows), batch_size), 1):
        batch = windows[batch_start : batch_start + batch_size]
        logger.info("Embedding batch %d/%d (%d windows)", batch_num, total_batches, len(batch))
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_paths: list[Path] = []
            for win_idx, w_start, w_end in batch:
                tmp = Path(tmpdir) / f"{win_idx}.mp4"
                _trim_for_embedding(source_path, tmp, w_start, w_end)
                tmp_paths.append(tmp)
            vectors = model.encode_document(
                [str(p) for p in tmp_paths],
                batch_size=len(tmp_paths),
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        for (win_idx, w_start, w_end), vector in zip(batch, vectors):
            records.append(ClipRecord(
                clip_id=str(win_idx),
                source_video=str(source_path),
                clip_path="",
                start_seconds=w_start,
                end_seconds=w_end,
                embedding=np.asarray(vector).tolist(),
            ))

    logger.info("Embedded %d windows from %s", len(records), source_path.name)
    return records


def search_records(
    records: Iterable[ClipRecord],
    query: str,
    model: SentenceTransformer,
    top_k: int = 5,
    threshold: float | None = None,
    suppress_neighbors: bool = True,
    visual_weight: float = 0.8,
) -> list[dict[str, object]]:
    available = [record for record in records if record.embedding]
    if not available:
        logger.warning("Search requested with no records containing embeddings")
        return []
    query_vector = model.encode_query(query, convert_to_tensor=True)
    vectors = torch.tensor([record.embedding for record in available], device=query_vector.device)
    similarities = F.cosine_similarity(F.normalize(vectors, dim=-1), F.normalize(query_vector, dim=-1), dim=-1)
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    ranked = sorted(
        (
            (record, visual_score * visual_weight + transcript_score(record, query_terms) * (1 - visual_weight))
            for record, visual_score in zip(available, similarities)
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    results: list[dict[str, object]] = []
    for record, score in ranked:
        if threshold is not None and score < threshold:
            continue
        if suppress_neighbors and any(
            result["source_video"] == record.source_video
            and abs(float(result["start_seconds"]) - record.start_seconds) < record.duration_seconds
            for result in results
        ):
            continue
        result = record.to_dict()
        result["similarity"] = float(score)
        results.append(result)
        if len(results) >= top_k:
            break
    return results


def transcript_score(record: ClipRecord, query_terms: set[str]) -> float:
    if not query_terms or not record.transcript:
        return 0.0
    transcript_terms = set(re.findall(r"[a-z0-9]+", record.transcript.lower()))
    return len(query_terms & transcript_terms) / len(query_terms)
