from pathlib import Path

import torch
import numpy as np

from youtube_video_content_search.clips import clip_window
from youtube_video_content_search.indexing import load_index, save_index, search_records
from youtube_video_content_search.models import ClipRecord
from youtube_video_content_search.vector_store import load_faiss_index, save_faiss_index, search_faiss_records


class FakeModel:
    def encode_query(self, query: str, convert_to_tensor: bool = True):
        return torch.tensor([1.0, 0.0])


class FakeNumpyModel:
    def encode_query(self, query: str, convert_to_numpy: bool = True):
        return np.array([1.0, 0.0], dtype="float32")


def test_clip_window_clamps_first_and_last_boundaries():
    assert clip_window(0, 10, 3, 25) == (0.0, 10.0)
    assert clip_window(20, 10, 3, 25) == (17.0, 25.0)


def test_clip_window_rejects_invalid_overlap():
    try:
        clip_window(0, 10, 10, 25)
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("Expected invalid overlap to raise ValueError")


def test_metadata_index_round_trip(tmp_path: Path):
    index_path = tmp_path / "index.json"
    records = [ClipRecord("1", "source.webm", "v_clips/1.mp4", start_seconds=2, end_seconds=12, embedding=[1.0, 0.0])]
    save_index(records, index_path)
    loaded = load_index(index_path)
    assert loaded[0].clip_id == "1"
    assert loaded[0].start_seconds == 2


def test_search_ranks_and_suppresses_neighboring_clips():
    records = [
        ClipRecord("0", "source.webm", "v_clips/0.mp4", start_seconds=0, end_seconds=10, embedding=[1.0, 0.0]),
        ClipRecord("1", "source.webm", "v_clips/1.mp4", start_seconds=7, end_seconds=17, embedding=[0.99, 0.01]),
        ClipRecord("2", "source.webm", "v_clips/2.mp4", start_seconds=30, end_seconds=40, embedding=[0.0, 1.0]),
    ]
    results = search_records(records, "a scene", FakeModel(), top_k=5)
    assert [result["clip_id"] for result in results] == ["0", "2"]
    assert isinstance(results[0]["similarity"], float)
    assert results[0]["similarity"] > results[1]["similarity"]


def test_faiss_round_trip_preserves_metadata(tmp_path: Path):
    index_path = tmp_path / "clips.faiss"
    records = [
        ClipRecord("0", "source.webm", "v_clips/0.mp4", start_seconds=0, end_seconds=10, embedding=[1.0, 0.0]),
        ClipRecord("1", "source.webm", "v_clips/1.mp4", start_seconds=20, end_seconds=30, embedding=[0.0, 1.0]),
    ]
    save_faiss_index(records, index_path)
    loaded, faiss_index = load_faiss_index(index_path)
    results = search_faiss_records(loaded, faiss_index, "scene", FakeNumpyModel(), top_k=1)
    assert results[0]["clip_id"] == "0"
    assert loaded[0].start_seconds == 0
