import json
import logging
import re
from pathlib import Path

import numpy as np

from .indexing import transcript_score
from .models import ClipRecord


logger = logging.getLogger(__name__)


def _faiss():
    try:
        import faiss
    except ImportError as error:
        raise RuntimeError("FAISS is required for this backend. Run `uv sync` first.") from error
    return faiss


def metadata_path(index_file: str | Path) -> Path:
    path = Path(index_file)
    return path.with_name(f"{path.stem}.metadata.json")


def save_faiss_index(records: list[ClipRecord], index_file: str | Path) -> None:
    if not records or any(record.embedding is None for record in records):
        raise ValueError("FAISS indexes require non-empty embeddings for every record")
    faiss = _faiss()
    vectors = np.asarray([record.embedding for record in records], dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    index_path = Path(index_file)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    metadata_path(index_path).write_text(
        json.dumps({"version": 1, "records": [record.to_dict() for record in records]}, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved FAISS index with %d vectors to %s", len(records), index_path)


def load_faiss_index(index_file: str | Path) -> tuple[list[ClipRecord], object]:
    faiss = _faiss()
    index_path = Path(index_file)
    records = [ClipRecord.from_dict(item) for item in json.loads(metadata_path(index_path).read_text(encoding="utf-8"))["records"]]
    index = faiss.read_index(str(index_path))
    logger.info("Loaded FAISS index with %d vectors from %s", index.ntotal, index_path)
    return records, index


def search_faiss_records(
    records: list[ClipRecord],
    index: object,
    query: str,
    model: object,
    top_k: int = 5,
    threshold: float | None = None,
    suppress_neighbors: bool = True,
    visual_weight: float = 0.8,
) -> list[dict[str, object]]:
    query_vector = model.encode_query(query, convert_to_numpy=True).astype("float32").reshape(1, -1)
    _faiss().normalize_L2(query_vector)
    scores, ids = index.search(query_vector, min(max(top_k * 10, top_k), len(records)))
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    ranked = []
    for visual_score, record_id in zip(scores[0], ids[0]):
        if record_id < 0:
            continue
        record = records[int(record_id)]
        score = float(visual_score) * visual_weight + transcript_score(record, query_terms) * (1 - visual_weight)
        ranked.append((record, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    results = []
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
        result["similarity"] = score
        results.append(result)
        if len(results) >= top_k:
            break
    return results
