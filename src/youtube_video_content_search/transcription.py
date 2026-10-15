import json
import logging
from pathlib import Path
from typing import Any

from .models import ClipRecord


logger = logging.getLogger(__name__)


def transcribe_records(
    records: list[ClipRecord],
    model_name: str = "base",
    device: str | None = None,
) -> list[ClipRecord]:
    """Add Whisper transcript text and timestamped segments to records."""
    import whisper

    logger.info("Loading Whisper model %s", model_name)
    model = whisper.load_model(model_name, device=device)
    logger.info("Transcribing %d clips", len(records))
    for record in records:
        result = model.transcribe(record.clip_path, word_timestamps=False)
        record.transcript = result.get("text", "").strip()
        record.transcript_segments = result.get("segments", [])
        logger.debug("Transcribed %s (%d segments)", record.clip_path, len(record.transcript_segments or []))
    logger.info("Transcript extraction complete")
    return records


def save_transcripts(records: list[ClipRecord], output_file: str | Path) -> None:
    """Persist transcript segments alongside the regular index records."""
    data: list[dict[str, Any]] = [record.to_dict() for record in records]
    Path(output_file).write_text(json.dumps({"version": 1, "records": data}, indent=2), encoding="utf-8")
    logger.info("Saved transcripts for %d records to %s", len(data), output_file)
