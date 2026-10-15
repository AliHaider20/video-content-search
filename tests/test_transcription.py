import json
from pathlib import Path

from youtube_video_content_search.models import ClipRecord
from youtube_video_content_search.transcription import save_transcripts


def _make_record(clip_id, transcript="", segments=None):
    r = ClipRecord(
        clip_id=str(clip_id),
        source_video="src.mp4",
        clip_path=f"v_clips/{clip_id}.mp4",
        start_seconds=float(clip_id) * 10,
        end_seconds=float(clip_id) * 10 + 10,
        transcript=transcript,
        transcript_segments=segments or [],
    )
    return r


def test_save_transcripts_writes_valid_json(tmp_path):
    out = tmp_path / "index.json"
    save_transcripts([_make_record("0")], out)
    data = json.loads(out.read_text())
    assert "records" in data
    assert data["version"] == 1


def test_save_transcripts_preserves_transcript_text(tmp_path):
    out = tmp_path / "index.json"
    save_transcripts([_make_record("0", transcript="a man holding a stick")], out)
    data = json.loads(out.read_text())
    assert data["records"][0]["transcript"] == "a man holding a stick"


def test_save_transcripts_preserves_segments(tmp_path):
    segments = [{"start": 0.0, "end": 2.5, "text": "hello"}]
    out = tmp_path / "index.json"
    save_transcripts([_make_record("0", segments=segments)], out)
    data = json.loads(out.read_text())
    assert data["records"][0]["transcript_segments"] == segments


def test_save_transcripts_preserves_timestamps(tmp_path):
    out = tmp_path / "index.json"
    r = _make_record("3")
    save_transcripts([r], out)
    data = json.loads(out.read_text())
    assert data["records"][0]["start_seconds"] == 30.0
    assert data["records"][0]["end_seconds"] == 40.0


def test_save_transcripts_multiple_records(tmp_path):
    out = tmp_path / "index.json"
    records = [_make_record(str(i), transcript=f"clip {i}") for i in range(5)]
    save_transcripts(records, out)
    data = json.loads(out.read_text())
    assert len(data["records"]) == 5
    assert data["records"][2]["transcript"] == "clip 2"


def test_save_transcripts_output_is_loadable_by_load_index(tmp_path):
    from youtube_video_content_search.indexing import load_index
    out = tmp_path / "index.json"
    records = [_make_record("0", transcript="hello", segments=[{"start": 0.0, "end": 1.0, "text": "hello"}])]
    save_transcripts(records, out)
    loaded = load_index(out)
    assert loaded[0].transcript == "hello"
    assert loaded[0].transcript_segments == [{"start": 0.0, "end": 1.0, "text": "hello"}]
