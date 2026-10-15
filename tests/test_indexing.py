import json
from pathlib import Path

import numpy as np
import pytest
import torch

from youtube_video_content_search.indexing import (
    _detect_clip_length,
    load_index,
    make_records,
    save_index,
    search_records,
    transcript_score,
)
from youtube_video_content_search.models import ClipRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_record(clip_id, start, end, embedding=None, source="src.mp4", transcript=""):
    return ClipRecord(
        clip_id=str(clip_id),
        source_video=source,
        clip_path=f"v_clips/{clip_id}.mp4",
        start_seconds=start,
        end_seconds=end,
        transcript=transcript,
        embedding=embedding or [1.0, 0.0],
    )


class FakeModel:
    def encode_query(self, query: str, convert_to_tensor: bool = True):
        return torch.tensor([1.0, 0.0])


# ---------------------------------------------------------------------------
# make_records — regression test for the overlap end_seconds fix
# ---------------------------------------------------------------------------

def test_make_records_no_overlap_contiguous(tmp_path):
    paths = [tmp_path / f"{i}.mp4" for i in range(3)]
    for p in paths:
        p.touch()
    records = make_records(paths, clip_length=10.0, overlap=0.0)
    assert [(r.start_seconds, r.end_seconds) for r in records] == [
        (0.0, 10.0),
        (10.0, 20.0),
        (20.0, 30.0),
    ]


def test_make_records_with_overlap_end_includes_full_clip_length(tmp_path):
    # Before the fix: clip 1 end was 7+10=17 instead of 20.
    paths = [tmp_path / f"{i}.mp4" for i in range(3)]
    for p in paths:
        p.touch()
    records = make_records(paths, clip_length=10.0, overlap=3.0)
    assert records[0].start_seconds == 0.0
    assert records[0].end_seconds == 10.0
    assert records[1].start_seconds == 7.0
    assert records[1].end_seconds == 20.0   # was 17 before fix
    assert records[2].start_seconds == 17.0
    assert records[2].end_seconds == 30.0   # was 27 before fix


def test_make_records_sorted_by_filename(tmp_path):
    # Files named so that lexicographic != numeric order
    names = ["10.mp4", "2.mp4", "1.mp4"]
    paths = [tmp_path / n for n in names]
    for p in paths:
        p.touch()
    records = make_records(paths, clip_length=10.0)
    # Sorted alphabetically: 1.mp4, 10.mp4, 2.mp4
    assert records[0].clip_id == "1"
    assert records[1].clip_id == "10"
    assert records[2].clip_id == "2"


def test_make_records_zero_clip_length_leaves_zero_timestamps(tmp_path):
    p = tmp_path / "0.mp4"
    p.touch()
    records = make_records([p], clip_length=0.0, overlap=0.0)
    assert records[0].start_seconds == 0.0
    assert records[0].end_seconds == 0.0


def test_make_records_source_video_stored(tmp_path):
    p = tmp_path / "0.mp4"
    p.touch()
    records = make_records([p], source_video="my_video.webm", clip_length=10.0)
    assert records[0].source_video == "my_video.webm"


# ---------------------------------------------------------------------------
# save_index / load_index
# ---------------------------------------------------------------------------

def test_save_load_round_trip_preserves_all_fields(tmp_path):
    index_path = tmp_path / "idx.json"
    original = [
        ClipRecord("a", "src.mp4", "v/a.mp4", start_seconds=5.0, end_seconds=15.0,
                   transcript="hello", embedding=[0.5, 0.5]),
    ]
    save_index(original, index_path)
    loaded = load_index(index_path)
    assert loaded[0].clip_id == "a"
    assert loaded[0].start_seconds == 5.0
    assert loaded[0].end_seconds == 15.0
    assert loaded[0].transcript == "hello"
    assert loaded[0].embedding == [0.5, 0.5]


def test_load_index_legacy_format_produces_records(tmp_path):
    # Old format: {"0.mp4": [vector]}
    index_path = tmp_path / "old.json"
    index_path.write_text(json.dumps({"0.mp4": [1.0, 0.0], "1.mp4": [0.0, 1.0]}))
    records = load_index(index_path)
    assert len(records) == 2
    assert all(r.embedding is not None for r in records)


def test_load_index_legacy_format_zero_timestamps_when_no_clips(tmp_path):
    index_path = tmp_path / "old.json"
    index_path.write_text(json.dumps({"clip.mp4": [1.0, 0.0]}))
    records = load_index(index_path)
    # Files don't exist so clip_length stays 0
    assert records[0].start_seconds == 0.0
    assert records[0].end_seconds == 0.0


def test_save_index_creates_parent_directories(tmp_path):
    deep_path = tmp_path / "a" / "b" / "index.json"
    save_index([make_record("0", 0, 10)], deep_path)
    assert deep_path.exists()


# ---------------------------------------------------------------------------
# transcript_score
# ---------------------------------------------------------------------------

def test_transcript_score_exact_match():
    record = make_record("0", 0, 10, transcript="the cat sat on the mat")
    score = transcript_score(record, {"cat", "mat"})
    assert score == 1.0


def test_transcript_score_partial_match():
    record = make_record("0", 0, 10, transcript="the cat sat")
    score = transcript_score(record, {"cat", "mat"})
    assert score == 0.5


def test_transcript_score_no_match():
    record = make_record("0", 0, 10, transcript="completely different words")
    score = transcript_score(record, {"cat", "mat"})
    assert score == 0.0


def test_transcript_score_empty_transcript_returns_zero():
    record = make_record("0", 0, 10, transcript="")
    assert transcript_score(record, {"cat"}) == 0.0


def test_transcript_score_empty_query_terms_returns_zero():
    record = make_record("0", 0, 10, transcript="cat")
    assert transcript_score(record, set()) == 0.0


# ---------------------------------------------------------------------------
# search_records
# ---------------------------------------------------------------------------

def test_search_records_returns_top_k():
    records = [make_record(i, i * 30, i * 30 + 10, embedding=[1.0, 0.0]) for i in range(10)]
    results = search_records(records, "query", FakeModel(), top_k=3)
    assert len(results) <= 3


def test_search_records_threshold_filters_low_scores():
    # query vector [1,0], record [0,1] → cosine similarity = 0
    low_match = make_record("low", 0, 10, embedding=[0.0, 1.0])
    high_match = make_record("high", 30, 40, embedding=[1.0, 0.0])
    results = search_records([low_match, high_match], "query", FakeModel(), top_k=5, threshold=0.5)
    ids = [r["clip_id"] for r in results]
    assert "high" in ids
    assert "low" not in ids


def test_search_records_suppress_neighbors_deduplicates_adjacent():
    records = [
        make_record("0", 0, 10, embedding=[1.0, 0.0]),
        make_record("1", 7, 17, embedding=[0.99, 0.01]),   # overlaps with 0
        make_record("2", 60, 70, embedding=[0.0, 1.0]),    # far away
    ]
    results = search_records(records, "q", FakeModel(), top_k=5, suppress_neighbors=True)
    ids = [r["clip_id"] for r in results]
    assert "0" in ids
    assert "1" not in ids   # suppressed — too close to "0"
    assert "2" in ids


def test_search_records_suppress_false_returns_all():
    records = [
        make_record("0", 0, 10, embedding=[1.0, 0.0]),
        make_record("1", 7, 17, embedding=[0.99, 0.01]),
    ]
    results = search_records(records, "q", FakeModel(), top_k=5, suppress_neighbors=False)
    assert len(results) == 2


def test_search_records_skips_records_without_embeddings():
    records = [
        ClipRecord("no-emb", "src", "clip.mp4", start_seconds=0, end_seconds=10, embedding=None),
        make_record("has-emb", 20, 30),
    ]
    results = search_records(records, "q", FakeModel(), top_k=5)
    ids = [r["clip_id"] for r in results]
    assert "has-emb" in ids
    assert "no-emb" not in ids


def test_search_records_empty_list_returns_empty():
    results = search_records([], "q", FakeModel())
    assert results == []


def test_search_result_contains_similarity_key():
    results = search_records([make_record("0", 0, 10)], "q", FakeModel())
    assert "similarity" in results[0]
    assert isinstance(results[0]["similarity"], float)
