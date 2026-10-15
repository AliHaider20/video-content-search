import pytest

from youtube_video_content_search.clips import clip_window


# ---------------------------------------------------------------------------
# clip_window
# ---------------------------------------------------------------------------

def test_first_window_clamps_start_to_zero():
    start, end = clip_window(0, 10, 3, 100)
    assert start == 0.0
    assert end == 10.0


def test_mid_window_extends_backward_by_overlap():
    start, end = clip_window(10, 10, 3, 100)
    assert start == 7.0
    assert end == 20.0


def test_last_window_clamps_end_to_duration():
    start, end = clip_window(20, 10, 3, 25)
    assert start == 17.0
    assert end == 25.0


def test_no_overlap_windows_are_contiguous():
    s0, e0 = clip_window(0, 10, 0, 100)
    s1, e1 = clip_window(10, 10, 0, 100)
    assert e0 == 10.0
    assert s1 == 10.0


def test_zero_clip_length_raises():
    with pytest.raises(ValueError, match="clip_length"):
        clip_window(0, 0, 0, 100)


def test_negative_clip_length_raises():
    with pytest.raises(ValueError, match="clip_length"):
        clip_window(0, -5, 0, 100)


def test_overlap_equal_to_clip_length_raises():
    with pytest.raises(ValueError, match="overlap"):
        clip_window(0, 10, 10, 100)


def test_overlap_larger_than_clip_length_raises():
    with pytest.raises(ValueError, match="overlap"):
        clip_window(0, 10, 11, 100)


def test_negative_overlap_raises():
    with pytest.raises(ValueError, match="overlap"):
        clip_window(0, 10, -1, 100)


def test_zero_duration_raises():
    with pytest.raises(ValueError, match="duration"):
        clip_window(0, 10, 3, 0)


def test_window_that_would_produce_empty_range_raises():
    # start_time past duration means start >= end after clamping
    with pytest.raises(ValueError):
        clip_window(100, 10, 3, 50)
