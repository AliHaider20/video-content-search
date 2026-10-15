import argparse
import logging
from pathlib import Path

import ffmpeg


logger = logging.getLogger(__name__)


def get_video_duration(file_path: str | Path) -> float:
    try:
        probe = ffmpeg.probe(str(file_path))
        return float(probe["format"]["duration"])
    except (ffmpeg.Error, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Unable to read duration for {file_path}: {error}") from error


def clip_window(start_time: float, clip_length: float, overlap: float, duration: float) -> tuple[float, float]:
    if clip_length <= 0:
        raise ValueError("clip_length must be greater than zero")
    if overlap < 0 or overlap >= clip_length:
        raise ValueError("overlap must be non-negative and smaller than clip_length")
    if duration <= 0:
        raise ValueError("duration must be greater than zero")
    start = max(0.0, start_time - overlap)
    end = min(duration, start_time + clip_length)
    if end <= start:
        raise ValueError("end_time must be greater than start_time")
    return start, end


def trim_video(input_file: str | Path, output_file: str | Path, start_time: float, end_time: float) -> None:
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")
    try:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        stream = ffmpeg.input(str(input_file), ss=start_time, to=end_time)
        ffmpeg.run(ffmpeg.output(stream, str(output_file), c="copy"), overwrite_output=True)
    except ffmpeg.Error as error:
        details = error.stderr.decode(errors="replace") if error.stderr else str(error)
        raise RuntimeError(f"Error trimming video: {details}") from error


def extract_audio(input_file: str | Path, output_file: str | Path, start_time: float, end_time: float) -> None:
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")
    try:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        stream = ffmpeg.input(str(input_file), ss=start_time, to=end_time)
        ffmpeg.run(ffmpeg.output(stream.audio, str(output_file), acodec="libmp3lame"), overwrite_output=True)
    except ffmpeg.Error as error:
        details = error.stderr.decode(errors="replace") if error.stderr else str(error)
        raise RuntimeError(f"Error extracting audio: {details}") from error


def create_clips(
    input_file: str | Path,
    video_dir: str | Path = "v_clips",
    audio_dir: str | Path = "a_clips",
    clip_length: float = 10.0,
    overlap: float = 3.0,
    include_audio: bool = True,
) -> list[dict[str, str | float]]:
    input_path = Path(input_file)
    duration = get_video_duration(input_path)
    logger.info("Creating clips from %s (%.2fs, length=%.2fs, overlap=%.2fs)", input_path, duration, clip_length, overlap)
    video_dir = Path(video_dir)
    audio_dir = Path(audio_dir)
    metadata = []
    start_time = 0.0
    index = 0
    while start_time < duration:
        window_start, window_end = clip_window(start_time, clip_length, overlap, duration)
        video_path = video_dir / f"{index}.mp4"
        trim_video(input_path, video_path, window_start, window_end)
        audio_path = audio_dir / f"{index}.mp3"
        if include_audio:
            extract_audio(input_path, audio_path, window_start, window_end)
        metadata.append({
            "clip_id": str(index),
            "source_video": str(input_path),
            "clip_path": str(video_path),
            "audio_path": str(audio_path) if include_audio else "",
            "start_seconds": window_start,
            "end_seconds": window_end,
        })
        index += 1
        start_time += clip_length
    logger.info("Created %d clips from %s", len(metadata), input_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a video into searchable overlapping clips.")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--video-dir", type=Path, default=Path("v_clips"))
    parser.add_argument("--audio-dir", type=Path, default=Path("a_clips"))
    parser.add_argument("--clip-length", type=float, default=10.0)
    parser.add_argument("--overlap", type=float, default=3.0)
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()
    clips = create_clips(args.input_file, args.video_dir, args.audio_dir, args.clip_length, args.overlap, not args.no_audio)
    logger.info("Clip command complete")
    print(f"Created {len(clips)} clips")


if __name__ == "__main__":
    main()
