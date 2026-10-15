from pathlib import Path

import numpy as np
import pytest

from youtube_video_content_search.models import ClipRecord
from youtube_video_content_search.vector_store import (
    load_faiss_index,
    metadata_path,
    save_faiss_index,
    search_faiss_records,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_record(clip_id, start, end, embedding, source="src.mp4"):
    return ClipRecord(
        clip_id=str(clip_id),
        source_video=source,
        clip_path=f"v_clips/{clip_id}.mp4",
        start_seconds=start,
        end_seconds=end,
        embedding=embedding,
    )


class FakeNumpyModel:
    def encode_query(self, query: str, convert_to_numpy: bool = True):
        return np.array([1.0, 0.0], dtype="float32")


# ---------------------------------------------------------------------------
# metadata_path
# ---------------------------------------------------------------------------

def test_metadata_path_adds_metadata_json_suffix():
    assert metadata_path("clips.faiss") == Path("clips.metadata.json")


def test_metadata_path_preserves_parent_directory():
    result = metadata_path("output/clips.faiss")
    assert result == Path("output/clips.metadata.json")


# ---------------------------------------------------------------------------
# save_faiss_index / load_faiss_index
# ---------------------------------------------------------------------------

def test_faiss_round_trip_preserves_record_count(tmp_path):
    index_path = tmp_path / "test.faiss"
    records = [
        make_record("0", 0, 10, [1.0, 0.0]),
        make_record("1", 20, 30, [0.0, 1.0]),
        make_record("2", 40, 50, [0.5, 0.5]),
    ]
    save_faiss_index(records, index_path)
    loaded, faiss_index = load_faiss_index(index_path)
    assert len(loaded) == 3
    assert faiss_index.ntotal == 3


def test_faiss_round_trip_preserves_timestamps(tmp_path):
    index_path = tmp_path / "test.faiss"
    records = [make_record("0", 5.5, 15.5, [1.0, 0.0])]
    save_faiss_index(records, index_path)
    loaded, _ = load_faiss_index(index_path)
    assert loaded[0].start_seconds == 5.5
    assert loaded[0].end_seconds == 15.5


def test_faiss_save_raises_on_missing_embeddings(tmp_path):
    index_path = tmp_path / "test.faiss"
    records = [ClipRecord("0", "src", "clip.mp4", embedding=None)]
    with pytest.raises(ValueError, match="embeddings"):
        save_faiss_index(records, index_path)


def test_faiss_save_raises_on_empty_records(tmp_path):
    index_path = tmp_path / "test.faiss"
    with pytest.raises(ValueError):
        save_faiss_index([], index_path)


def test_faiss_creates_metadata_sidecar(tmp_path):
    index_path = tmp_path / "test.faiss"
    save_faiss_index([make_record("0", 0, 10, [1.0, 0.0])], index_path)
    assert metadata_path(index_path).exists()


# ---------------------------------------------------------------------------
# search_faiss_records
# ---------------------------------------------------------------------------

def test_search_faiss_returns_best_match_first(tmp_path):
    index_path = tmp_path / "test.faiss"
    records = [
        make_record("match", 0, 10, [1.0, 0.0]),    # aligns with query
        make_record("no-match", 30, 40, [0.0, 1.0]),
    ]
    save_faiss_index(records, index_path)
    loaded, faiss_index = load_faiss_index(index_path)
    results = search_faiss_records(loaded, faiss_index, "q", FakeNumpyModel(), top_k=2)
    assert results[0]["clip_id"] == "match"


def test_search_faiss_threshold_filters_low_scores(tmp_path):
    index_path = tmp_path / "test.faiss"
    records = [
        make_record("good", 0, 10, [1.0, 0.0]),
        make_record("bad", 30, 40, [0.0, 1.0]),
    ]
    save_faiss_index(records, index_path)
    loaded, faiss_index = load_faiss_index(index_path)
    results = search_faiss_records(loaded, faiss_index, "q", FakeNumpyModel(), top_k=5, threshold=0.5)
    ids = [r["clip_id"] for r in results]
    assert "good" in ids
    assert "bad" not in ids


def test_search_faiss_suppress_neighbors_removes_adjacent(tmp_path):
    index_path = tmp_path / "test.faiss"
    records = [
        make_record("0", 0, 10, [1.0, 0.0]),
        make_record("1", 7, 17, [0.99, 0.01]),   # overlaps with 0
        make_record("2", 60, 70, [0.5, 0.5]),
    ]
    save_faiss_index(records, index_path)
    loaded, faiss_index = load_faiss_index(index_path)
    results = search_faiss_records(loaded, faiss_index, "q", FakeNumpyModel(), top_k=5, suppress_neighbors=True)
    ids = [r["clip_id"] for r in results]
    assert "0" in ids
    assert "1" not in ids


def test_search_faiss_top_k_limits_results(tmp_path):
    index_path = tmp_path / "test.faiss"
    records = [make_record(i, i * 30, i * 30 + 10, [1.0, 0.0]) for i in range(10)]
    save_faiss_index(records, index_path)
    loaded, faiss_index = load_faiss_index(index_path)
    results = search_faiss_records(loaded, faiss_index, "q", FakeNumpyModel(), top_k=3)
    assert len(results) <= 3
