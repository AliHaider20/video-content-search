from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ClipRecord:
    clip_id: str
    source_video: str
    clip_path: str
    audio_path: str = ""
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    transcript: str = ""
    transcript_segments: list[dict[str, Any]] | None = None
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClipRecord":
        return cls(
            clip_id=str(data["clip_id"]),
            source_video=str(data.get("source_video", "")),
            clip_path=str(data["clip_path"]),
            audio_path=str(data.get("audio_path", "")),
            start_seconds=float(data.get("start_seconds", 0.0)),
            end_seconds=float(data.get("end_seconds", 0.0)),
            transcript=str(data.get("transcript", "")),
            transcript_segments=data.get("transcript_segments"),
            embedding=data.get("embedding"),
        )

    @property
    def clip_name(self) -> str:
        return Path(self.clip_path).name

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)
