# Storyframe CLI

[Tiếng Việt](README.vi.md)

Extract clean story-text frames from read-aloud videos, export the story audio
as MP3, and package the selected images into a PDF.

Supports:

- YouTube URLs, with local download cache
- single local video files
- folders of videos

## Platforms

| Platform | Status |
| --- | --- |
| macOS | Supported and tested. |
| Linux | Supported if system packages are installed. |
| Windows | Not directly tested; use WSL2/Linux. |

## Install

```bash
brew install ffmpeg tesseract
python3 -m pip install -e ".[local]"
```

Python 3.11+ is required.

Linux system packages:

```bash
sudo apt-get install ffmpeg tesseract-ocr
```

Python dependencies installed by `.[local]`:

- base: `liteparse`, `numpy`, `opencv-python-headless`, `pillow`, `yt-dlp`
- local pipeline: `faster-whisper`, `imagehash`, `pytesseract`, `rapidocr`,
  `rapidfuzz`, `scenedetect`, `scikit-image`

`faster-whisper` downloads the selected ASR model on first use.

## Usage

YouTube:

```bash
storyframe run "https://www.youtube.com/watch?v=VIDEO_ID"
```

Local file:

```bash
storyframe run "/path/to/book.mp4"
```

Folder:

```bash
storyframe run "/path/to/video-folder"
```

Nested folders:

```bash
storyframe run "/path/to/video-folder" --recursive
```

## Output

By default, each video writes to:

```text
outputs/storyframe-runs/<video-name>/
```

Main files:

```text
<video-name>.mp3
<video-name>.pdf
frames/*.jpg
review-index.csv
review-contact-sheet.jpg
manifest.json
```

## Useful Options

```bash
# Choose output directory.
storyframe run "https://www.youtube.com/watch?v=VIDEO_ID" --output-root runs

# Reuse a shared YouTube cache.
storyframe run "https://www.youtube.com/watch?v=VIDEO_ID" \
  --download-cache-dir outputs/storyframe-youtube-cache

# Use browser cookies if YouTube asks for login.
storyframe run "https://www.youtube.com/watch?v=VIDEO_ID" --cookies-from-browser chrome

# Keep raw scanned frames and work files for debugging.
storyframe run "/path/to/book.mp4" --keep-work
```

Basic help:

```bash
storyframe run --help
```

Advanced OCR/ASR tuning flags:

```bash
storyframe run --advanced-help
```

## Notes

- `strict-complete` is the default quality mode.
- YouTube downloads are cached under `<output-root>/_youtube-cache`.
- This is a local/free pipeline, so CPU usage can be high on long videos.
- Only process videos you have the right to download, transform, and store.

## Development

```bash
python3 -m unittest discover -s tests
python3 -m storyframe_cli run "https://www.youtube.com/watch?v=VIDEO_ID"
```
