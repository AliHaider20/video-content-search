import argparse
import logging
from pathlib import Path

from .clips import create_clips
from .download import download_youtube_video
from .indexing import embed_video_windows, embed_videos, load_index, make_records, save_index, search_records
from .models import ClipRecord
from .transcription import save_transcripts, transcribe_records
from .vector_store import load_faiss_index, save_faiss_index, search_faiss_records
from .logging_config import configure_logging


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="youtube-video-content-search")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    commands = parser.add_subparsers(dest="command", required=True)

    trim = commands.add_parser("trim", help="Create overlapping video and audio clips")
    trim.add_argument("input_file", type=Path)
    trim.add_argument("--video-dir", type=Path, default=Path("v_clips"))
    trim.add_argument("--audio-dir", type=Path, default=Path("a_clips"))
    trim.add_argument("--clip-length", type=float, default=10.0)
    trim.add_argument("--overlap", type=float, default=3.0)
    trim.add_argument("--no-audio", action="store_true")

    download = commands.add_parser("download", help="Download one YouTube video")
    download.add_argument("url")
    download.add_argument("--output-dir", type=Path, default=Path("downloads"))

    index = commands.add_parser(
        "index",
        help="Embed a video (or pre-cut clips) and save a searchable index",
    )
    index.add_argument(
        "--source-video", type=Path, default=Path(),
        help="Video file to embed directly via sliding window (no clips needed)",
    )
    index.add_argument("--clip-length", type=float, default=10.0)
    index.add_argument("--overlap", type=float, default=3.0)
    index.add_argument("--clips-dir", type=Path, default=Path("v_clips"),
                       help="Pre-cut clips folder (used when --source-video is not a file)")
    index.add_argument("--metadata", type=Path, default=Path("clips.json"))
    index.add_argument("--output", type=Path, default=Path("video_embeddings.json"))
    index.add_argument("--backend", choices=("json", "faiss"), default="json")
    index.add_argument("--batch-size", type=int, default=2)
    index.add_argument("--model", default="jinaai/jina-embeddings-v5-omni-nano")

    transcribe = commands.add_parser("transcribe", help="Add Whisper transcripts to an index")
    transcribe.add_argument("--index", type=Path, default=Path("video_embeddings.json"))
    transcribe.add_argument("--output", type=Path, default=Path("video_embeddings.json"))
    transcribe.add_argument("--model", default="base")

    search = commands.add_parser("search", help="Search the saved clip index")
    search.add_argument("query")
    search.add_argument("--index", type=Path, default=Path("video_embeddings.json"))
    search.add_argument("--backend", choices=("json", "faiss"), default=None)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--threshold", type=float)
    search.add_argument("--model", default="jinaai/jina-embeddings-v5-omni-retrieval")
    search.add_argument("--visual-weight", type=float, default=0.8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.verbose, force=True)
    logger.info("Starting command: %s", args.command)
    if args.command == "download":
        output = download_youtube_video(args.url, args.output_dir)
        print(output)
    elif args.command == "trim":
        records = create_clips(args.input_file, args.video_dir, args.audio_dir, args.clip_length, args.overlap, not args.no_audio)
        save_index([ClipRecord.from_dict(record) for record in records], "clips.json")
        logger.info("Created %d clips", len(records))
        print(f"Created {len(records)} clips")
    elif args.command == "index":
        if args.source_video.is_file():
            logger.info("Direct video indexing: %s", args.source_video)
            records = embed_video_windows(
                args.source_video,
                clip_length=args.clip_length,
                overlap=args.overlap,
                model_name=args.model,
                batch_size=args.batch_size,
            )
        elif args.metadata.exists():
            records = [record for record in load_index(args.metadata) if Path(record.clip_path).exists()]
            records = embed_videos(records, args.model, args.batch_size)
        else:
            records = make_records(args.clips_dir.glob("*.mp4"), args.source_video, args.clip_length, args.overlap)
            records = embed_videos(records, args.model, args.batch_size)
        logger.info("Saving %d records with %s backend", len(records), args.backend)
        if args.backend == "faiss":
            save_faiss_index(records, args.output)
        else:
            save_index(records, args.output, model_name=args.model)
        print(f"Indexed {len(records)} windows → {args.output}")
    elif args.command == "search":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.model, trust_remote_code=True)
        backend = args.backend or ("faiss" if args.index.suffix == ".faiss" else "json")
        if backend == "faiss":
            records, faiss_index = load_faiss_index(args.index)
            results = search_faiss_records(records, faiss_index, args.query, model, args.top_k, args.threshold, visual_weight=args.visual_weight)
        else:
            results = search_records(load_index(args.index), args.query, model, args.top_k, args.threshold, visual_weight=args.visual_weight)
        for result in results:
            print(f"{result['similarity']:.4f}  {result['clip_path']}  {result['start_seconds']:.1f}-{result['end_seconds']:.1f}s")
        logger.info("Search returned %d results", len(results))
    elif args.command == "transcribe":
        records = transcribe_records(load_index(args.index), args.model)
        save_transcripts(records, args.output)
        logger.info("Transcribed %d clips", len(records))
        print(f"Saved transcripts for {len(records)} clips to {args.output}")
