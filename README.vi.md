# Storyframe CLI

[English](README.md)

Storyframe CLI trích xuất các frame có chữ truyện sạch từ video read-aloud,
xuất audio truyện thành MP3, và đóng gói các ảnh đã chọn thành PDF để review.
Tool hỗ trợ file video local, folder nhiều video, và link YouTube với cache tải
về local để tránh tải lại nhiều lần.

Chế độ mặc định được thiết kế cho workflow review nghiêm ngặt:

- chỉ giữ frame có text nội dung truyện
- tránh duplicate cùng một trạng thái transcript
- tránh text đang fade in hoặc fade out
- loại title/outro/ad/promo khi không phải nội dung truyện
- giữ page OCR-only khi audio không bao phủ hết chữ trên hình
- xuất `frames/*.jpg`, `<video-name>.pdf`, `<video-name>.mp3`,
  `review-index.csv`, `review-index.md`, `review-contact-sheet.jpg`, và
  `debug-local.json`

## Trạng Thái

Đây là tool local/free, dùng FFmpeg, ASR local, OCR, và image scoring. Tool đang
được tune cho video narrated children's storybook, nơi text truyện được render
trực tiếp trên frame video.

Lệnh mặc định đã dùng pipeline local khuyến nghị:

- engine `local`
- quality `strict-complete`
- OCR scan `dense-windowed`
- ASR `faster-whisper`
- OCR `rapidocr`
- scene/page detection với `all-pages`

Các command cũ dùng `--engine local-v2` vẫn chạy vì được map sang `local`, nhưng
script mới nên dùng `--engine local`.

## Cách Hoạt Động

Engine `local` kết hợp bốn tín hiệu local:

1. ASR timing từ `faster-whisper`
2. OCR observations từ `rapidocr` / ONNX Runtime
3. scene/page windows từ PySceneDetect
4. frame scoring dựa trên text coverage, extra text, ink strength, stability,
   page-edge risk, overlay detection, và duplicate/subset pruning

Selector ưu tiên tìm frame gốc sạch. Khi text đầy đủ chỉ tồn tại dưới overlay
nhỏ, tool có thể patch overlay từ frame sạch gần đó. Với logo che góc trái dưới,
tool có thể reconstruct phần suffix transcript nhỏ bị che khi ASR/OCR alignment
đủ tự tin.

## Yêu Cầu

System tools:

```bash
brew install ffmpeg tesseract
```

Python:

```bash
python3 --version
# Python 3.11 hoặc mới hơn
```

Cài package:

```bash
cd storyframe-cli
python3 -m pip install -e ".[local]"
```

Trong workspace này, local dependencies có thể được cài vào:

```text
/Users/thieunv/Documents/Codex/2026-07-01/pha/work/.deps/storyframe-local
```

Runtime vẫn check thêm thư mục cũ `storyframe-local-v2` để backward compatible,
nên install local cũ không cần tạo lại.

## Basic Usage

Bình thường chỉ cần truyền source.

Link YouTube:

```bash
storyframe run "https://www.youtube.com/watch?v=VIDEO_ID"
```

Một file video local:

```bash
storyframe run "/path/to/book.mp4"
```

Một folder nhiều video:

```bash
storyframe run "/path/to/video-folder"
```

Chỉ thêm `--recursive` khi folder có nested folders.

Output mặc định nằm ở:

```text
outputs/storyframe-runs/<video-name>/
```

Artifacts chính:

- `<video-name>.mp3`
- `<video-name>.pdf`
- `frames/*.jpg`
- `review-index.csv`
- `review-contact-sheet.jpg`

Một vài flags hay dùng:

```bash
# Đổi output folder.
storyframe run "https://www.youtube.com/watch?v=VIDEO_ID" --output-root runs

# Dùng Chrome cookies nếu YouTube yêu cầu login.
storyframe run "https://www.youtube.com/watch?v=VIDEO_ID" --cookies-from-browser chrome

# Giữ raw scanned frames và debug work files.
storyframe run "/path/to/book.mp4" --keep-work
```

Chạy `storyframe run --help` để xem basic options.
Chạy `storyframe run --advanced-help` để xem engine/OCR/ASR tuning flags.

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
    ├── debug-local.json
    └── manifest.json
```

Khi bật `--keep-work`, raw scanned frames được giữ trong work root:

```text
<work-root>/engine/<video-slug>/local-frames/
```

Giữ work files hữu ích khi tune thuật toán vì những lần rebuild sau có thể reuse
raw frames và OCR observations thay vì tải/scan lại từ đầu.

## YouTube Cache

Download YouTube dùng `yt-dlp`. Tool không tự giới hạn số lượt tải, nhưng
YouTube vẫn có thể rate-limit, block, hoặc yêu cầu cookies.

Mặc định video YouTube được cache tại `<output-root>/_youtube-cache`.
Dùng một cache path chung để tránh tải lại giữa nhiều output folder:

```bash
storyframe run "https://www.youtube.com/watch?v=VIDEO_ID" \
  --download-cache-dir outputs/storyframe-youtube-cache
```

Ép tải lại:

```bash
storyframe run "https://www.youtube.com/watch?v=VIDEO_ID" --redownload
```

Dùng browser cookies nếu YouTube yêu cầu login:

```bash
storyframe run "https://www.youtube.com/watch?v=VIDEO_ID" --cookies-from-browser chrome
```

## Quality Modes

```bash
# Mặc định. Giữ đủ transcript states và repair các case thiếu frame sạch.
storyframe run "/path/to/book.mp4" --quality strict-complete

# Chỉ dùng frame gốc. Drop các warning frames không resolve được.
storyframe run "/path/to/book.mp4" --quality strict-original

# Giữ nhiều candidates hơn để manual review.
storyframe run "/path/to/book.mp4" --quality balanced
```

`strict-complete` là default khuyến nghị.

## Scan Modes

```bash
# Sampled mode, nhanh hơn.
storyframe run "/path/to/book.mp4" --scan-mode sampled --fps 4

# Dense mode, an toàn hơn.
storyframe run "/path/to/book.mp4" --scan-mode dense --dense-fps 8

# Local mode khuyến nghị.
storyframe run "/path/to/book.mp4" --scan-mode dense-windowed --dense-fps 8

# Scan từng source frame. Chậm.
storyframe run "/path/to/book.mp4" --scan-mode native
```

Default đã tune cho generality với video kiểu Vooks:

```bash
--scan-mode dense-windowed --dense-fps 8 --page-window-mode all-pages
```

## CPU Và VPS

Tool chạy local/free nên CPU có thể cao. Mỗi video có thể cần local audio
transcription cộng với hàng nghìn OCR inference calls.

Với video 8 phút ở `--dense-fps 8`, số candidate frame khoảng:

```text
8 phút * 60 giây * 8 fps = 3840 candidate frames
```

Khuyến nghị tối thiểu cho một worker:

- 6 vCPU
- 12 GB RAM
- NVMe storage

Tốt hơn cho batch run lâu dài:

- 8 vCPU / 16 GB RAM cho một worker thoải mái
- 12+ vCPU / 24-32 GB RAM cho hai workers

Giới hạn thread có thể giảm nóng máy, đổi lại runtime lâu hơn:

```bash
OMP_NUM_THREADS=2 \
MKL_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=2 \
storyframe run "https://www.youtube.com/watch?v=VIDEO_ID" ...
```

## Commands Hay Dùng

Chạy với story window cụ thể:

```bash
storyframe run "/path/to/book.mp4" \
  --story-start 14 \
  --story-end 175
```

Dùng ASR model nhỏ hơn để chạy nhanh hơn:

```bash
storyframe run "/path/to/book.mp4" \
  --asr-model base.en
```

Chạy OCR-only smoke mode:

```bash
storyframe run "/path/to/book.mp4" \
  --asr-backend none
```

## Review Files

`review-index.csv` là file audit chính. Các cột quan trọng:

- `index`: thứ tự frame
- `timestamp`: timestamp source được chọn
- `unit_id`: ASR/OCR temporal unit
- `image`: đường dẫn ảnh extracted
- `transcript`: transcript raw/corrected được chọn
- `normalized_text`: text đã clean dùng để matching
- `score`: điểm selector
- `status`: `clean` hoặc `needs_review`
- `warnings`: low coverage, low ink, low stability, extra text, page edge, v.v.
- `output_source`: original, overlay-cleaned, text-reconstructed, v.v.
- `page_id`: scene/page interval

`review-contact-sheet.jpg` giúp review nhanh toàn bộ run bằng mắt.

## Development

Chạy tests:

```bash
python3 -m unittest discover -s tests
```

Chạy package khi chưa install console script:

```bash
python3 -m storyframe_cli run "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output-root outputs/storyframe-runs
```

Project layout:

```text
storyframe_cli/
├── cli.py
├── extract_story_transcript_frames.py
└── local/
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
├── test_local_selector.py
└── test_storyframe_algorithm.py
```

## Troubleshooting

`No module named storyframe_cli`

Cài editable package hoặc set `PYTHONPATH`:

```bash
python3 -m pip install -e ".[local]"
```

Tên extra cũ vẫn chạy như compatibility alias:

```bash
python3 -m pip install -e ".[local-v2]"
```

`yt-dlp` fail

Update `yt-dlp`:

```bash
python3 -m pip install -U "yt-dlp[default]"
```

YouTube yêu cầu login:

```bash
storyframe run "https://www.youtube.com/watch?v=VIDEO_ID" \
  --cookies-from-browser chrome
```

CPU quá cao:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 storyframe run ...
```

Run quá chậm:

- giữ `--keep-work`
- reuse `--download-cache-dir`
- giảm `--dense-fps`
- dùng `--page-window-mode unit-pages` khi ASR coverage đáng tin

Quá nhiều dòng `needs_review`:

- kiểm tra `review-contact-sheet.jpg`
- kiểm tra `review-index.csv`
- so sánh selected frames với raw frames trong `local-frames`
- chỉ rerun `--scan-mode native-windowed` cho video khó

## Legal Note

Chỉ xử lý video mà bạn có quyền download, transform, và lưu trữ. Tool không cấp
quyền đối với video, audio, artwork, hoặc transcript content của bên thứ ba.
