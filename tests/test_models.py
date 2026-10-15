from youtube_video_content_search.models import ClipRecord


def test_round_trip_preserves_all_fields():
    record = ClipRecord(
        clip_id="42",
        source_video="video.mp4",
        clip_path="v_clips/42.mp4",
        audio_path="a_clips/42.mp3",
        start_seconds=10.5,
        end_seconds=20.5,
        transcript="hello world",
        transcript_segments=[{"start": 10.5, "end": 12.0, "text": "hello"}],
        embedding=[0.1, 0.2, 0.3],
    )
    loaded = ClipRecord.from_dict(record.to_dict())
    assert loaded.clip_id == "42"
    assert loaded.source_video == "video.mp4"
    assert loaded.clip_path == "v_clips/42.mp4"
    assert loaded.audio_path == "a_clips/42.mp3"
    assert loaded.start_seconds == 10.5
    assert loaded.end_seconds == 20.5
    assert loaded.transcript == "hello world"
    assert loaded.transcript_segments == [{"start": 10.5, "end": 12.0, "text": "hello"}]
    assert loaded.embedding == [0.1, 0.2, 0.3]


def test_from_dict_missing_optional_fields_use_defaults():
    loaded = ClipRecord.from_dict({"clip_id": "1", "clip_path": "clip.mp4"})
    assert loaded.source_video == ""
    assert loaded.audio_path == ""
    assert loaded.start_seconds == 0.0
    assert loaded.end_seconds == 0.0
    assert loaded.transcript == ""
    assert loaded.transcript_segments is None
    assert loaded.embedding is None


def test_duration_seconds_positive():
    record = ClipRecord("0", "", "", start_seconds=5.0, end_seconds=15.0)
    assert record.duration_seconds == 10.0


def test_duration_seconds_zero_when_inverted():
    record = ClipRecord("0", "", "", start_seconds=15.0, end_seconds=5.0)
    assert record.duration_seconds == 0.0


def test_duration_seconds_zero_when_equal():
    record = ClipRecord("0", "", "", start_seconds=10.0, end_seconds=10.0)
    assert record.duration_seconds == 0.0


def test_clip_name_returns_filename_only():
    record = ClipRecord("0", "", "v_clips/subdir/42.mp4")
    assert record.clip_name == "42.mp4"


def test_clip_name_handles_plain_filename():
    record = ClipRecord("0", "", "clip.mp4")
    assert record.clip_name == "clip.mp4"


def test_to_dict_contains_expected_keys():
    record = ClipRecord("1", "src", "clip.mp4")
    d = record.to_dict()
    for key in ("clip_id", "source_video", "clip_path", "start_seconds", "end_seconds",
                "transcript", "transcript_segments", "embedding"):
        assert key in d


def test_numeric_clip_id_stored_as_string():
    loaded = ClipRecord.from_dict({"clip_id": 5, "clip_path": "c.mp4"})
    assert loaded.clip_id == "5"
    assert isinstance(loaded.clip_id, str)
