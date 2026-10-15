# YouTube Video Content Search

Search a YouTube video using natural-language descriptions like *"a man holding a stick"* or *"three people in a kitchen"*. The tool downloads the video, builds a semantic embedding index in memory, and lets you query it from a web UI or the CLI.

No clip files are written to disk. The `index` command slices the video into overlapping windows in memory, embeds each one, then discards the temp data. The only outputs are the source video and `video_embeddings.json`. When you preview a result in the UI, the matching segment is extracted on-the-fly.

---

## Prerequisites

- Python 3.10+ with [uv](https://github.com/astral-sh/uv)
- [FFmpeg](https://ffmpeg.org/download.html) installed and on your PATH

---

## Quick start

```powershell
# 1. Install dependencies
uv sync

# 2. Download a video
uv run youtube-video-content-search download "https://www.youtube.com/watch?v=..."
# Video saved to .\downloads\<title>.mp4

# 3. Build the index
uv run youtube-video-content-search index --source-video .\downloads\<title>.mp4

# 4. Launch the UI
uv run gradio app.py
```

Open the local URL printed in the terminal. Results show similarity scores, timestamps, a visual timeline, clip preview playback, and a download button.

---

## Commands

### `download`

Downloads a YouTube video (best quality, merged to MP4).

```powershell
uv run youtube-video-content-search download "https://www.youtube.com/watch?v=..."

# Save to a specific folder
uv run youtube-video-content-search download "https://..." --output-dir .\videos
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `downloads` | Folder to save the video |

---

### `index`

Embeds the video using a sliding window and writes a searchable index.

```powershell
uv run youtube-video-content-search index --source-video .\downloads\<title>.mp4

# Tune window size and overlap
uv run youtube-video-content-search index --source-video .\video.mp4 --clip-length 10 --overlap 3

# Use FAISS for faster search on longer videos
uv run youtube-video-content-search index --source-video .\video.mp4 --backend faiss --output video_embeddings.faiss
```

| Flag | Default | Description |
|------|---------|-------------|
| `--source-video` | *(required)* | Path to the video file |
| `--clip-length` | `10.0` | Window length in seconds |
| `--overlap` | `3.0` | Overlap between windows in seconds |
| `--output` | `video_embeddings.json` | Output index file path |
| `--backend` | `json` | `json` (default) or `faiss` |
| `--model` | `jinaai/jina-embeddings-v5-omni-nano` | Embedding model |
| `--batch-size` | `2` | Videos processed per batch |

---

### `transcribe` *(optional)*

Adds speech transcripts to an existing index using OpenAI Whisper. Enables hybrid text + visual search when you query from the UI or CLI.

```powershell
uv run youtube-video-content-search transcribe

# Use a more accurate Whisper model
uv run youtube-video-content-search transcribe --model small
```

| Flag | Default | Description |
|------|---------|-------------|
| `--index` | `video_embeddings.json` | Index file to read |
| `--output` | `video_embeddings.json` | Index file to write (can be same) |
| `--model` | `base` | Whisper model: `tiny`, `base`, `small`, `medium`, `large` |

---

### `search`

Query the index from the CLI without launching the UI.

```powershell
uv run youtube-video-content-search search "a man holding a stick"

# More results, higher threshold, FAISS index
uv run youtube-video-content-search search "three people in a kitchen" --top-k 10 --threshold 0.25 --index video_embeddings.faiss
```

| Flag | Default | Description |
|------|---------|-------------|
| `--index` | `video_embeddings.json` | Index file to search |
| `--top-k` | `5` | Number of results to return |
| `--threshold` | *(none)* | Minimum similarity score (0–1) to include |
| `--visual-weight` | `0.8` | Visual vs transcript balance (1.0 = visual only) |
| `--backend` | *(auto)* | `json` or `faiss`; auto-detected from file extension |
| `--model` | `jinaai/jina-embeddings-v5-omni-retrieval` | Query embedding model |

---

### `trim` *(advanced)*

Pre-cuts a video into overlapping `.mp4` and `.mp3` clip files on disk, then saves a `clips.json` manifest. Useful if you want to inspect individual clips. The `index` command does not require this step.

```powershell
uv run youtube-video-content-search trim .\video.mp4 --clip-length 10 --overlap 3
```

| Flag | Default | Description |
|------|---------|-------------|
| `--video-dir` | `v_clips` | Output folder for video clips |
| `--audio-dir` | `a_clips` | Output folder for audio clips |
| `--clip-length` | `10.0` | Clip length in seconds |
| `--overlap` | `3.0` | Overlap between clips in seconds |
| `--no-audio` | off | Skip audio extraction |

---

## Examples

### Find a specific moment in a cooking video

```powershell
uv run youtube-video-content-search download "https://www.youtube.com/watch?v=..."
uv run youtube-video-content-search index --source-video ".\downloads\Gordon Ramsay Beef Wellington.mp4"
uv run youtube-video-content-search search "wrapping beef in pastry" --top-k 3
```

Output:
```
0.8241  (source video)  142.0-152.0s
0.7903  (source video)  128.0-138.0s
0.7651  (source video)  155.0-165.0s
```

---

### Search a lecture with speech (hybrid mode)

```powershell
uv run youtube-video-content-search download "https://www.youtube.com/watch?v=..."
uv run youtube-video-content-search index --source-video ".\downloads\MIT 6.006 Lecture 1.mp4" --clip-length 30 --overlap 5
uv run youtube-video-content-search transcribe --model small
uv run youtube-video-content-search search "dynamic programming explained" --visual-weight 0.4 --top-k 5
```

`--clip-length 30` works better for spoken content where context spans longer windows. `--visual-weight 0.4` shifts the ranking to favour the transcript over visuals.

---

### Large video with FAISS index

```powershell
uv run youtube-video-content-search index --source-video ".\downloads\3hour-documentary.mp4" --backend faiss --output documentary.faiss
uv run youtube-video-content-search search "crowd cheering" --index documentary.faiss --top-k 10 --threshold 0.2
```

---

### Full workflow with the UI

```powershell
uv run youtube-video-content-search download "https://www.youtube.com/watch?v=..."
uv run youtube-video-content-search index --source-video ".\downloads\<title>.mp4"
uv run youtube-video-content-search transcribe        # optional, adds speech search
uv run gradio app.py                                  # opens at http://127.0.0.1:7860
```

---

## Debugging

Add `--verbose` before any command for debug-level logs:

```powershell
uv run youtube-video-content-search --verbose search "a man holding a stick"
uv run youtube-video-content-search --verbose index --source-video .\video.mp4
```

---

## Development

```powershell
uv run pytest
uv run python -m compileall app.py src
```

GPU (CUDA) is used automatically when available; CPU is fully supported.

---

## Stack

| Layer | Library |
|-------|---------|
| Video download | [yt-dlp](https://github.com/yt-dlp/yt-dlp) |
| Video processing | [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) |
| Visual embeddings | [Sentence Transformers](https://www.sbert.net/) + Jina CLIP |
| Vector search | [FAISS](https://github.com/facebookresearch/faiss) / JSON (flat) |
| Speech transcription | [OpenAI Whisper](https://github.com/openai/whisper) |
| Web UI | [Gradio](https://www.gradio.app/) |
| Packaging / env | [uv](https://github.com/astral-sh/uv) |
