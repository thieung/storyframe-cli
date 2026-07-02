# Storyframe CLI

Storyframe CLI extracts clean story-text frames from read-aloud videos, exports
the story audio as MP3, and packages the selected frames into a reviewable PDF.
It supports local video files, folders of videos, and YouTube URLs with a local
download cache.

The default mode is designed for strict review workflows:

- keep only frames that contain story text
- avoid duplicate transcript states
- avoid text while it is still fading in or fading out
- reject title/outro/ad/promo screens when they are not story content
- preserve OCR-only pages when audio does not cover the on-screen text
- produce `frames/*.jpg`, `<video-name>.pdf`, `<video-name>.mp3`,
  `review-index.csv`, `review-index.md`, `review-contact-sheet.jpg`, and
  `debug-local-v2.json`

## Status

This is a local/free extraction tool built around FFmpeg, local ASR, OCR, and
image scoring. It is tuned for narrated children's storybook videos where the
actual text is rendered on the video frames.

The current recommended engine is `local-v2` with `strict-complete` quality.

## How It Works

`local-v2` combines four local signals:

1. ASR timing from `faster-whisper`
2. OCR observations from `rapidocr` / ONNX Runtime
3. scene/page windows from PySceneDetect
4. frame scoring based on text coverage, extra text, ink strength, stability,
   page-edge risk, overlay detection, and duplicate/subset pruning

The selector first tries to find a clean original frame. When a complete text
state only exists under a small social overlay, it can patch the overlay from a
nearby clean frame. For bottom-left logo occlusion, it can reconstruct the small
missing transcript suffix only when the ASR/OCR alignment is confident.

## Requirements

System tools:

```bash
brew install ffmpeg tesseract
```

Python:

```bash
python3 --version
# Python 3.11 or newer
```

Install package:

```bash
cd storyframe-cli
python3 -m pip install -e ".[local-v2]"
```

For this workspace, local dependencies were installed into:

```text
/Users/thieunv/Documents/Codex/2026-07-01/pha/work/.deps/storyframe-local-v2
```

## Quick Start

YouTube URL:

```bash
storyframe run "https://www.youtube.com/watch?v=VIDEO_ID" \
  --engine local-v2 \
  --asr-backend faster-whisper \
  --asr-model small.en \
  --ocr-backend rapidocr \
  --scan-mode dense-windowed \
  --page-detection scene \
  --page-window-mode all-pages \
  --download-cache-dir outputs/storyframe-youtube-cache \
  --output-root outputs/storyframe-runs \
  --keep-work
```

Single local video:

```bash
storyframe run "/path/to/book.mp4" \
  --engine local-v2 \
  --asr-backend faster-whisper \
  --asr-model small.en \
  --ocr-backend rapidocr \
  --scan-mode dense-windowed \
  --page-detection scene \
  --page-window-mode all-pages \
  --output-root outputs/storyframe-runs \
  --keep-work
```

Folder batch:

```bash
storyframe run "/path/to/video-folder" \
  --recursive \
  --engine local-v2 \
  --asr-backend faster-whisper \
  --asr-model small.en \
  --ocr-backend rapidocr \
  --scan-mode dense-windowed \
  --page-detection scene \
  --page-window-mode all-pages \
  --output-root outputs/storyframe-runs \
  --keep-work
```

## Output Layout

```text
outputs/storyframe-runs/
├── manifest.json
├── _youtube-cache/
└── video-name/
    ├── video-name.mp3
    ├── video-name.pdf
    ├── frames/
    ├── review-index.csv
    ├── review-index.md
    ├── review-contact-sheet.jpg
    ├── debug-local-v2.json
    └── manifest.json
```

When `--keep-work` is set, raw scanned frames are kept under the configured
work root:

```text
<work-root>/engine/<video-slug>/local-v2-frames/
```

Keeping work files is useful when tuning the algorithm because later rebuilds can
reuse raw frames and OCR observations instead of starting from a fresh download.

## YouTube Cache

YouTube downloads use `yt-dlp`. The tool has no built-in download limit, but
YouTube may still rate-limit, block, or require cookies.

By default, YouTube videos are cached under `<output-root>/_youtube-cache`.
Use a shared cache path to avoid repeat downloads:

```bash
storyframe run "https://www.youtube.com/watch?v=VIDEO_ID" \
  --download-cache-dir outputs/storyframe-youtube-cache \
  --output-root outputs/storyframe-runs
```

Force a fresh download:

```bash
storyframe run "https://www.youtube.com/watch?v=VIDEO_ID" \
  --redownload \
  --output-root outputs/storyframe-runs
```

Use browser cookies if YouTube requires login:

```bash
storyframe run "https://www.youtube.com/watch?v=VIDEO_ID" \
  --cookies-from-browser chrome \
  --output-root outputs/storyframe-runs
```

## Quality Modes

```bash
# Default. Preserve transcript states and repair unresolved clean-frame cases.
storyframe run "/path/to/book.mp4" --quality strict-complete

# Only use original frames. Drop unresolved warning frames.
storyframe run "/path/to/book.mp4" --quality strict-original

# Keep more candidates for manual review.
storyframe run "/path/to/book.mp4" --quality balanced
```

`strict-complete` is the recommended default.

## Scan Modes

```bash
# Practical sampled mode.
storyframe run "/path/to/book.mp4" --scan-mode sampled --fps 4

# Safer dense mode.
storyframe run "/path/to/book.mp4" --scan-mode dense --dense-fps 8

# Recommended local-v2 mode.
storyframe run "/path/to/book.mp4" --scan-mode dense-windowed --dense-fps 8

# Exhaustive source-frame scan. Slow.
storyframe run "/path/to/book.mp4" --scan-mode native
```

For broad generality across Vooks-style videos, use:

```bash
--scan-mode dense-windowed --dense-fps 8 --page-window-mode all-pages
```

## CPU And VPS Notes

This tool intentionally runs local/free. CPU usage can be high because every
selected video may require local audio transcription plus thousands of OCR
inference calls.

For an 8-minute video at `--dense-fps 8`, expect roughly:

```text
8 minutes * 60 seconds * 8 fps = 3840 candidate frames
```

Recommended minimum for one worker:

- 6 vCPU
- 12 GB RAM
- NVMe storage

Better for sustained batch runs:

- 8 vCPU / 16 GB RAM for one comfortable worker
- 12+ vCPU / 24-32 GB RAM for two workers

Thread limiting can reduce heat at the cost of runtime:

```bash
OMP_NUM_THREADS=2 \
MKL_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=2 \
storyframe run "https://www.youtube.com/watch?v=VIDEO_ID" ...
```

## Common Commands

Run with an explicit story window:

```bash
storyframe run "/path/to/book.mp4" \
  --story-start 14 \
  --story-end 175 \
  --engine local-v2 \
  --asr-backend faster-whisper \
  --ocr-backend rapidocr
```

Use a smaller ASR model for speed:

```bash
storyframe run "/path/to/book.mp4" \
  --engine local-v2 \
  --asr-backend faster-whisper \
  --asr-model base.en
```

Run OCR-only smoke mode:

```bash
storyframe run "/path/to/book.mp4" \
  --engine local-v2 \
  --asr-backend none \
  --ocr-backend rapidocr
```

## Review Files

`review-index.csv` is the main audit file. Important columns:

- `index`: frame order
- `timestamp`: chosen source timestamp
- `unit_id`: ASR/OCR temporal unit
- `image`: extracted frame path
- `transcript`: selected raw/corrected transcript text
- `normalized_text`: cleaned text used for matching
- `score`: selector score
- `status`: `clean` or `needs_review`
- `warnings`: low coverage, low ink, low stability, extra text, page edge, etc.
- `output_source`: original, overlay-cleaned, text-reconstructed, etc.
- `page_id`: scene/page interval

`review-contact-sheet.jpg` gives a fast visual overview of the run.

## Development

Run tests:

```bash
python3 -m unittest discover -s tests
```

Run the package without installing the console script:

```bash
python3 -m storyframe_cli run "https://www.youtube.com/watch?v=VIDEO_ID" \
  --engine local-v2 \
  --asr-backend faster-whisper \
  --ocr-backend rapidocr
```

Project layout:

```text
storyframe_cli/
├── cli.py
├── extract_story_transcript_frames.py
└── local_v2/
    ├── asr.py
    ├── engine.py
    ├── media.py
    ├── models.py
    ├── ocr.py
    ├── ocr_filter.py
    ├── page_detection.py
    ├── selector.py
    └── text.py
tests/
├── test_local_v2_selector.py
└── test_storyframe_algorithm.py
```

## Troubleshooting

`No module named storyframe_cli`

Install editable package or set `PYTHONPATH`:

```bash
python3 -m pip install -e ".[local-v2]"
```

`yt-dlp` fails

Update `yt-dlp`:

```bash
python3 -m pip install -U "yt-dlp[default]"
```

YouTube asks for login:

```bash
storyframe run "https://www.youtube.com/watch?v=VIDEO_ID" \
  --cookies-from-browser chrome
```

CPU is too high:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 storyframe run ...
```

Run is too slow:

- keep `--keep-work`
- reuse `--download-cache-dir`
- lower `--dense-fps`
- use `--page-window-mode unit-pages` when ASR coverage is reliable

Too many `needs_review` rows:

- inspect `review-contact-sheet.jpg`
- inspect `review-index.csv`
- compare selected frames against raw frames in `local-v2-frames`
- rerun with `--scan-mode native-windowed` only for difficult videos

## Legal Note

Only process videos you have the right to download, transform, and store. The
tool does not grant rights to third-party video, audio, artwork, or transcript
content.
