"""Tests for pure-Python helpers in app.py.

Imports are done at function level to avoid the Gradio network calls that
happen at module import time.
"""
from pathlib import Path


# ---------------------------------------------------------------------------
# format_time
# ---------------------------------------------------------------------------

def test_format_time_seconds_only():
    from app import format_time
    assert format_time(45) == "00:00:45"


def test_format_time_minutes_and_seconds():
    from app import format_time
    assert format_time(90) == "00:01:30"


def test_format_time_hours():
    from app import format_time
    assert format_time(3661) == "01:01:01"


def test_format_time_zero():
    from app import format_time
    assert format_time(0) == "00:00:00"


def test_format_time_negative_clamps_to_zero():
    from app import format_time
    assert format_time(-5) == "00:00:00"


def test_format_time_float_input():
    from app import format_time
    assert format_time(10.9) == "00:00:10"


# ---------------------------------------------------------------------------
# render_timeline
# ---------------------------------------------------------------------------

def test_render_timeline_empty_results_returns_empty_string():
    from app import render_timeline
    assert render_timeline("0", []) == ""


def test_render_timeline_zero_end_seconds_shows_unavailable():
    from app import render_timeline
    results = [{"clip_id": "0", "start_seconds": 0.0, "end_seconds": 0.0,
                "similarity": 0.8, "source_video": "v.mp4"}]
    html = render_timeline("0", results)
    assert "unavailable" in html.lower()


def test_render_timeline_contains_selected_color():
    from app import render_timeline
    results = [
        {"clip_id": "0", "start_seconds": 0.0, "end_seconds": 10.0,
         "similarity": 0.9, "source_video": "v.mp4"},
        {"clip_id": "1", "start_seconds": 30.0, "end_seconds": 40.0,
         "similarity": 0.7, "source_video": "v.mp4"},
    ]
    html = render_timeline("0", results)
    # selected clip uses orange
    assert "#f97316" in html
    # other clips use blue
    assert "#3b82f6" in html


def test_render_timeline_contains_time_markers():
    from app import render_timeline
    results = [{"clip_id": "0", "start_seconds": 0.0, "end_seconds": 60.0,
                "similarity": 0.9, "source_video": "v.mp4"}]
    html = render_timeline("0", results)
    assert "00:00:00" in html
    assert "00:01:00" in html


def test_render_timeline_no_selected_still_renders():
    from app import render_timeline
    results = [{"clip_id": "0", "start_seconds": 0.0, "end_seconds": 10.0,
                "similarity": 0.9, "source_video": "v.mp4"}]
    # clip_id None means nothing selected — should not crash
    html = render_timeline(None, results)
    assert isinstance(html, str)


def test_render_timeline_only_shows_peers_from_same_source():
    from app import render_timeline
    results = [
        {"clip_id": "0", "start_seconds": 0.0, "end_seconds": 10.0,
         "similarity": 0.9, "source_video": "video_a.mp4"},
        {"clip_id": "1", "start_seconds": 20.0, "end_seconds": 30.0,
         "similarity": 0.7, "source_video": "video_b.mp4"},
    ]
    html = render_timeline("0", results)
    # clip 1 is from a different source and should not widen the span
    # the bar should end at 10s (from clip 0), so 00:00:10 appears but 00:00:30 doesn't
    assert "00:00:10" in html


# ---------------------------------------------------------------------------
# clip_path
# ---------------------------------------------------------------------------

def test_clip_path_returns_none_for_empty_string(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(app, "ROOT", tmp_path)
    assert app.clip_path({"clip_path": ""}) is None


def test_clip_path_finds_file_in_root(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(app, "ROOT", tmp_path)
    f = tmp_path / "clip.mp4"
    f.touch()
    result = app.clip_path({"clip_path": "clip.mp4"})
    assert result == str(f)


def test_clip_path_finds_file_in_v_clips(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(app, "ROOT", tmp_path)
    v = tmp_path / "v_clips"
    v.mkdir()
    f = v / "42.mp4"
    f.touch()
    result = app.clip_path({"clip_path": "42.mp4"})
    assert result == str(f)


def test_clip_path_returns_none_when_not_found(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(app, "ROOT", tmp_path)
    assert app.clip_path({"clip_path": "missing.mp4"}) is None


def test_clip_path_absolute_path_found(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(app, "ROOT", tmp_path)
    f = tmp_path / "abs.mp4"
    f.touch()
    result = app.clip_path({"clip_path": str(f)})
    assert result == str(f)


# ---------------------------------------------------------------------------
# source_video_path
# ---------------------------------------------------------------------------

def test_source_video_path_returns_none_for_empty(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(app, "ROOT", tmp_path)
    assert app.source_video_path({"source_video": ""}) is None


def test_source_video_path_finds_absolute(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(app, "ROOT", tmp_path)
    f = tmp_path / "source.mp4"
    f.touch()
    result = app.source_video_path({"source_video": str(f)})
    assert result == str(f)


def test_source_video_path_finds_relative_to_root(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(app, "ROOT", tmp_path)
    f = tmp_path / "video.mp4"
    f.touch()
    result = app.source_video_path({"source_video": "video.mp4"})
    assert result == str(f)


# ---------------------------------------------------------------------------
# extract_preview
# ---------------------------------------------------------------------------

def test_extract_preview_returns_none_when_source_missing(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(app, "ROOT", tmp_path)
    clip = {"clip_id": "0", "source_video": "nonexistent.mp4",
            "start_seconds": 0.0, "end_seconds": 10.0}
    assert app.extract_preview(clip) is None


def test_extract_preview_returns_none_when_zero_duration(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(app, "ROOT", tmp_path)
    f = tmp_path / "video.mp4"
    f.touch()
    clip = {"clip_id": "0", "source_video": str(f),
            "start_seconds": 5.0, "end_seconds": 5.0}
    assert app.extract_preview(clip) is None


def test_extract_preview_uses_real_clip_when_available(tmp_path, monkeypatch):
    """extract_preview with a real source video file should create an output file."""
    import app
    monkeypatch.setattr(app, "ROOT", tmp_path)
    monkeypatch.setattr(app, "_preview_dir", tmp_path / "previews")
    (tmp_path / "previews").mkdir()

    # Use an actual clip from the project's v_clips as source
    src = Path("v_clips/0.mp4")
    if not src.exists():
        import pytest
        pytest.skip("v_clips/0.mp4 not present")

    clip = {"clip_id": "test_preview", "source_video": str(src.resolve()),
            "start_seconds": 0.0, "end_seconds": 5.0}
    result = app.extract_preview(clip)
    assert result is not None
    assert Path(result).exists()
    assert Path(result).stat().st_size > 0
