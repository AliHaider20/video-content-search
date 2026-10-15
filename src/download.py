from pathlib import Path
import logging

import yt_dlp


logger = logging.getLogger(__name__)


def download_youtube_video(url: str, output_dir: str | Path = ".") -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading YouTube video to %s", output_path)
    options = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": str(output_path / "%(title)s.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=True)
        result = Path(downloader.prepare_filename(info)).with_suffix(".mp4")
        logger.info("Download complete: %s", result)
        return result
