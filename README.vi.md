# Storyframe CLI

[English](README.md)

Trích xuất frame có chữ truyện sạch từ video read-aloud, xuất audio truyện
thành MP3, và đóng gói các ảnh đã chọn thành PDF.

Hỗ trợ:

- link YouTube, có cache local để tránh tải lại
- một file video local
- folder chứa nhiều video

## Platforms

| Platform | Trạng thái |
| --- | --- |
| macOS | Hỗ trợ và đã test. |
| Linux | Hỗ trợ nếu đã cài system packages. |
| Windows | Chưa test trực tiếp; nên dùng WSL2/Linux. |

## Cài Đặt

```bash
brew install ffmpeg tesseract
python3 -m pip install -e ".[local]"
```

Cần Python 3.11+.

System packages trên Linux:

```bash
sudo apt-get install ffmpeg tesseract-ocr
```

Python dependencies được cài bởi `.[local]`:

- base: `liteparse`, `numpy`, `opencv-python-headless`, `pillow`, `yt-dlp`
- local pipeline: `faster-whisper`, `imagehash`, `pytesseract`, `rapidocr`,
  `rapidfuzz`, `scenedetect`, `scikit-image`

`faster-whisper` sẽ download ASR model được chọn ở lần chạy đầu tiên.

## Cách Dùng

YouTube:

```bash
storyframe run "https://www.youtube.com/watch?v=VIDEO_ID"
```

File local:

```bash
storyframe run "/path/to/book.mp4"
```

Folder:

```bash
storyframe run "/path/to/video-folder"
```

Folder có thư mục con:

```bash
storyframe run "/path/to/video-folder" --recursive
```

## Output

Mặc định mỗi video ghi vào:

```text
outputs/storyframe-runs/<video-name>/
```

File chính:

```text
<video-name>.mp3
<video-name>.pdf
frames/*.jpg
review-index.csv
review-contact-sheet.jpg
manifest.json
```

## Options Hay Dùng

```bash
# Chọn output directory.
storyframe run "https://www.youtube.com/watch?v=VIDEO_ID" --output-root runs

# Dùng chung YouTube cache.
storyframe run "https://www.youtube.com/watch?v=VIDEO_ID" \
  --download-cache-dir outputs/storyframe-youtube-cache

# Dùng browser cookies nếu YouTube yêu cầu login.
storyframe run "https://www.youtube.com/watch?v=VIDEO_ID" --cookies-from-browser chrome

# Giữ raw scanned frames và work files để debug.
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

## Ghi Chú

- `strict-complete` là quality mode mặc định.
- Video YouTube được cache tại `<output-root>/_youtube-cache`.
- Pipeline chạy local/free nên CPU có thể cao với video dài.
- Chỉ xử lý video mà bạn có quyền download, transform, và lưu trữ.

## Development

```bash
python3 -m unittest discover -s tests
python3 -m storyframe_cli run "https://www.youtube.com/watch?v=VIDEO_ID"
```
