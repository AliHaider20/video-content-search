# YouTube Video Content Search

Search video clips with natural-language descriptions such as `a man holding a stick`.

## Workflow

```powershell
uv sync
uv run youtube-video-content-search download "https://www.youtube.com/watch?v=..."
uv run youtube-video-content-search index --source-video .\video.webm
uv run youtube-video-content-search search "three people in a kitchen" --top-k 5
uv run gradio app.py
```

No clip files are saved. The `index` command slices the video into windows in memory, embeds each one, then discards the temp files. The only outputs are the source video and `video_embeddings.json`. When you preview a result in the UI, the matching segment is extracted on-the-fly from the source video.

Open the local Gradio URL shown in the terminal. Search results include similarity score, timestamps, a visual timeline showing where matches fall in the video, clip preview playback, and a download button.

### Options

```powershell
# Larger FAISS index for faster search on big collections
uv run youtube-video-content-search index --source-video .\video.webm --backend faiss --output video_embeddings.faiss

# Adjust window size and overlap
uv run youtube-video-content-search index --source-video .\video.webm --clip-length 10 --overlap 3

# Add speech transcripts for hybrid text+visual search
uv run youtube-video-content-search transcribe --model base

# Search from the CLI
uv run youtube-video-content-search search "three people in a kitchen" --top-k 5
```

## Index format

`video_embeddings.json` stores a versioned `records` list. Each record contains the source video path, start/end timestamps, optional transcript, and its embedding. No clip files are referenced or required. The source video must be present at search time for preview playback.

For larger collections `--backend faiss` writes vectors to a FAISS inner-product index with a `.metadata.json` sidecar. Search automatically selects the FAISS backend when the index path ends in `.faiss`.

### Pre-cut clips (legacy)

If you already have a `v_clips/` folder from `trim`, the old workflow still works:

```powershell
uv run youtube-video-content-search index --clips-dir v_clips --source-video .\video.webm
```

## Configuration

- `VIDEO_INDEX`: index file used by the API; defaults to `video_embeddings.json`.
- `VIDEO_MODEL`: Sentence Transformers model used by the API.
- `--clip-length`, `--overlap`, `--top-k`, and `--threshold` control the retrieval workflow.
- `--visual-weight` controls hybrid visual/transcript ranking; `0.8` favors visual similarity.

Transcript search is optional. The `transcribe` command uses OpenAI Whisper, stores the full transcript and timestamped segments, and keeps working for clips with no speech.

The embedding model loads lazily when the first API search is made. The server can therefore start without downloading model weights.

## Logging

The CLI logs lifecycle events, model loading, clip counts, index reads/writes, transcript progress, and failures to stderr. Use `--verbose` before a command for debug logging:

```powershell
uv run youtube-video-content-search --verbose search "a man holding a stick"
```

The Gradio app logs model loading, requests, result counts, and failures.

## Development

```powershell
uv run pytest
uv run python -m compileall app.py src video_trimmer.py
```

FFmpeg must be installed and available to `ffmpeg-python`. GPU acceleration is used when CUDA is available; CPU remains supported.
