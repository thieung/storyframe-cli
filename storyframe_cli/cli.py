from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image


VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}
IGNORED_DOWNLOAD_SUFFIXES = {
    ".description",
    ".info.json",
    ".json",
    ".part",
    ".temp",
    ".tmp",
    ".webp",
    ".ytdl",
}


@dataclass
class JobResult:
    source: str
    video_path: str
    output_dir: str
    mp3_path: str
    pdf_path: str
    frame_count: int
    status: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    show_advanced = "--advanced-help" in raw_args
    if show_advanced:
        raw_args = ["--help" if arg == "--advanced-help" else arg for arg in raw_args]

    parser = build_parser(show_advanced=show_advanced)
    args = parser.parse_args(raw_args)
    args.engine = normalize_engine(args.engine)
    return args


def build_parser(show_advanced: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="storyframe",
        description=(
            "Batch extract story transcript frames, MP3 audio, and a PDF from "
            "local videos, folders, or YouTube URLs."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Process URL(s), local video file(s), and/or folder(s).",
    )
    run_parser.add_argument(
        "sources",
        nargs="+",
        help="YouTube URL, local video file, or folder containing videos.",
    )
    if not show_advanced:
        run_parser.add_argument(
            "--advanced-help",
            action="store_true",
            help="Show all engine, OCR, ASR, and tuning flags.",
        )
    add_common_args(run_parser, show_advanced=show_advanced)
    return parser


def normalize_engine(engine: str) -> str:
    if engine == "local-v2":
        return "local"
    return engine


def add_common_args(parser: argparse.ArgumentParser, show_advanced: bool = False) -> None:
    def help_for(text: str, *, advanced: bool = False) -> str:
        if advanced and not show_advanced:
            return argparse.SUPPRESS
        return text

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.cwd() / "outputs" / "storyframe-runs",
        help=help_for("Root directory for per-video outputs."),
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=None,
        help=help_for("Root directory for temporary downloads/work files.", advanced=True),
    )
    parser.add_argument(
        "--engine-script",
        type=Path,
        default=Path(__file__).with_name("extract_story_transcript_frames.py"),
        help=help_for("Path to the frame extraction engine script.", advanced=True),
    )
    parser.add_argument(
        "--engine",
        choices=["legacy", "local", "local-v2"],
        default="local",
        help=help_for(
            "Frame extraction engine. 'local-v2' is accepted as a deprecated alias for 'local'.",
            advanced=True,
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help=help_for("When a source is a folder, recursively scan videos."),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=4.0,
        help=help_for("Frame sample rate passed to the extraction engine.", advanced=True),
    )
    parser.add_argument(
        "--scan-mode",
        choices=["sampled", "dense", "native", "dense-windowed", "native-windowed"],
        default="dense-windowed",
        help=help_for(
            (
                "Frame scan strategy. sampled uses --fps; dense raises sampling "
                "to --dense-fps; native scans every source video frame."
            ),
            advanced=True,
        ),
    )
    parser.add_argument(
        "--dense-fps",
        type=float,
        default=8.0,
        help=help_for("Minimum OCR sample rate for --scan-mode dense.", advanced=True),
    )
    parser.add_argument(
        "--quality",
        choices=["strict-complete", "strict-original", "balanced"],
        default="strict-complete",
        help=help_for(
            (
                "Extraction quality mode. strict-complete is default and reconstructs "
                "unresolved fade frames to avoid missing transcript lines."
            )
        ),
    )
    parser.add_argument(
        "--story-start",
        type=float,
        default=0.0,
        help=help_for("Story start time in seconds.", advanced=True),
    )
    parser.add_argument(
        "--story-end",
        type=float,
        default=None,
        help=help_for("Story end time in seconds. Defaults to video duration.", advanced=True),
    )
    parser.add_argument(
        "--title-start",
        type=float,
        default=0.0,
        help=help_for("Optional title intro start time.", advanced=True),
    )
    parser.add_argument(
        "--title-end",
        type=float,
        default=0.0,
        help=help_for("Optional title intro end time.", advanced=True),
    )
    parser.add_argument(
        "--include-title-intro",
        action="store_true",
        help=help_for("Keep title intro frames in addition to story frames.", advanced=True),
    )
    parser.add_argument(
        "--audio-bitrate",
        default="192k",
        help=help_for("MP3 bitrate for extracted audio.", advanced=True),
    )
    parser.add_argument(
        "--youtube-format",
        default=(
            "bv*[height<=720][ext=mp4][vcodec^=avc1]+ba[ext=m4a]/"
            "bv*[height<=720][ext=mp4]+ba/"
            "best[height<=720][ext=mp4]/best[height<=720]/best"
        ),
        help=help_for(
            "yt-dlp format selector. Defaults to MP4/H.264 <=720p for faster OCR-friendly downloads.",
            advanced=True,
        ),
    )
    parser.add_argument(
        "--download-cache-dir",
        type=Path,
        default=None,
        help=help_for(
            "Directory for cached YouTube videos. Defaults to <output-root>/_youtube-cache."
        ),
    )
    parser.add_argument(
        "--redownload",
        action="store_true",
        help=help_for("Ignore cached YouTube files and download again."),
    )
    parser.add_argument(
        "--playlist",
        action="store_true",
        help=help_for("Allow YouTube playlist downloads.", advanced=True),
    )
    parser.add_argument(
        "--cookies",
        type=Path,
        default=None,
        help=help_for("Cookies file for yt-dlp when YouTube requires login.", advanced=True),
    )
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help=help_for("Browser name for yt-dlp cookies, e.g. chrome."),
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help=help_for("Keep temporary engine work directories."),
    )
    parser.add_argument(
        "--no-polish-transcripts",
        dest="polish_transcripts",
        action="store_false",
        help=help_for(
            "Disable final high-confidence OCR transcript cleanup after frame selection.",
            advanced=True,
        ),
    )
    parser.set_defaults(polish_transcripts=True)
    parser.add_argument(
        "--transcript-polish-min-conf",
        type=float,
        default=0.80,
        help=help_for(
            "Minimum Tesseract confidence for transcript cleanup, 0.0-1.0.",
            advanced=True,
        ),
    )
    parser.add_argument(
        "--asr-backend",
        choices=["none", "faster-whisper"],
        default="faster-whisper",
        help=help_for("local ASR backend.", advanced=True),
    )
    parser.add_argument(
        "--asr-model",
        default="small.en",
        help=help_for("local faster-whisper model size/name.", advanced=True),
    )
    parser.add_argument(
        "--ocr-backend",
        choices=["rapidocr"],
        default="rapidocr",
        help=help_for("local OCR backend.", advanced=True),
    )
    parser.add_argument(
        "--window-padding",
        type=float,
        default=2.0,
        help=help_for("Seconds to expand ASR windows in local.", advanced=True),
    )
    parser.add_argument(
        "--page-detection",
        choices=["none", "scene"],
        default="scene",
        help=help_for("local page detector.", advanced=True),
    )
    parser.add_argument(
        "--page-window-mode",
        choices=["unit", "unit-pages", "all-pages"],
        default="all-pages",
        help=help_for("local page scan strategy after page detection.", advanced=True),
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=12.0,
        help=help_for("PySceneDetect content threshold for local pages.", advanced=True),
    )
    parser.add_argument(
        "--scene-min-len",
        type=int,
        default=8,
        help=help_for("Minimum scene length in frames for local page detection.", advanced=True),
    )
    parser.add_argument(
        "--keep-downloaded-video",
        action="store_true",
        help=help_for(
            "Deprecated: YouTube downloads are cached by default. Kept for compatibility.",
            advanced=True,
        ),
    )


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def run_command(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nSTDOUT:\n"
            + proc.stdout[-2000:]
            + "\n\nSTDERR:\n"
            + proc.stderr[-4000:]
        )
    return proc


def ensure_system_dependencies(args: argparse.Namespace) -> None:
    required = ["ffmpeg", "ffprobe"]
    if args.engine in {"legacy", "local"}:
        required.append("tesseract")
    missing = [tool for tool in required if shutil.which(tool) is None]
    if missing:
        fail("Missing system dependency: " + ", ".join(missing))


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def youtube_video_id(value: str) -> str | None:
    parsed = urlparse(value)
    host = parsed.netloc.lower().removeprefix("www.")
    if host == "youtu.be":
        candidate = parsed.path.strip("/").split("/")[0]
        return candidate or None

    if host.endswith("youtube.com"):
        query_video_id = parse_qs(parsed.query).get("v", [None])[0]
        if query_video_id:
            return query_video_id
        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) >= 2 and path_parts[0] in {"embed", "live", "shorts"}:
            return path_parts[1]
    return None


def is_cached_video_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        return False
    return not any(str(path).lower().endswith(suffix) for suffix in IGNORED_DOWNLOAD_SUFFIXES)


def find_cached_youtube_video(cache_dir: Path, video_id: str | None) -> Path | None:
    if not video_id or not cache_dir.exists():
        return None
    candidates = [
        path
        for path in cache_dir.glob(f"*-{video_id}.*")
        if is_cached_video_file(path) and path.stat().st_size > 0
    ]
    candidates += [
        path
        for path in cache_dir.glob(f"{video_id}.*")
        if is_cached_video_file(path) and path.stat().st_size > 0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def slugify(value: str, fallback: str = "video") -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value[:80] or fallback


def unique_dir(root: Path, slug: str) -> Path:
    candidate = root / slug
    if not candidate.exists():
        return candidate
    index = 2
    while True:
        candidate = root / f"{slug}-{index}"
        if not candidate.exists():
            return candidate
        index += 1


def video_duration(video_path: Path) -> float:
    proc = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
    )
    return max(0.1, float(proc.stdout.strip()))


def iter_video_files(folder: Path, recursive: bool) -> list[Path]:
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    videos = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(videos)


def download_youtube(url: str, args: argparse.Namespace, cache_dir: Path) -> list[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    video_id = youtube_video_id(url)
    if not args.redownload:
        cached_path = find_cached_youtube_video(cache_dir, video_id)
        if cached_path:
            print(f"Using cached YouTube video: {cached_path}", flush=True)
            return [cached_path]

    outtmpl = cache_dir / "%(title).200B-%(id)s.%(ext)s"
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        args.youtube_format,
        "--merge-output-format",
        "mp4",
        "-o",
        str(outtmpl),
        "--print",
        "after_move:filepath",
    ]
    if not args.playlist:
        cmd.append("--no-playlist")
    if args.cookies:
        cmd += ["--cookies", str(args.cookies)]
    if args.cookies_from_browser:
        cmd += ["--cookies-from-browser", args.cookies_from_browser]
    cmd.append(url)

    try:
        proc = run_command(cmd)
    except RuntimeError as exc:
        raise RuntimeError(
            "yt-dlp failed. Install/update with: python3 -m pip install -U 'yt-dlp[default]'\n"
            + str(exc)
        ) from exc

    paths = [Path(line.strip()) for line in proc.stdout.splitlines() if line.strip()]
    paths = [path for path in paths if path.exists()]
    if not paths:
        cached_path = find_cached_youtube_video(cache_dir, video_id)
        if cached_path:
            paths = [cached_path]
    if not paths:
        raise RuntimeError(f"yt-dlp did not report a downloaded file for: {url}")
    return paths


def expand_sources(args: argparse.Namespace, youtube_cache_dir: Path) -> list[tuple[str, Path, bool]]:
    expanded: list[tuple[str, Path, bool]] = []
    for source in args.sources:
        if is_url(source):
            for video_path in download_youtube(source, args, youtube_cache_dir):
                expanded.append((source, video_path, True))
            continue

        path = Path(source).expanduser().resolve()
        if path.is_dir():
            videos = iter_video_files(path, args.recursive)
            if not videos:
                print(f"No videos found in folder: {path}", file=sys.stderr)
            for video_path in videos:
                expanded.append((str(path), video_path, False))
            continue
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            expanded.append((str(path), path, False))
            continue
        print(f"Skipping unsupported source: {source}", file=sys.stderr)
    return expanded


def run_frame_engine(
    video_path: Path,
    output_dir: Path,
    work_dir: Path,
    args: argparse.Namespace,
    story_end: float,
) -> None:
    if args.engine == "local":
        cmd = [
            sys.executable,
            "-m",
            "storyframe_cli.local.engine",
            str(video_path),
            "--output-dir",
            str(output_dir),
            "--work-dir",
            str(work_dir),
            "--fps",
            str(args.fps),
            "--scan-mode",
            args.scan_mode,
            "--dense-fps",
            str(args.dense_fps),
            "--story-start",
            f"{args.story_start:.3f}",
            "--story-end",
            f"{story_end:.3f}",
            "--quality",
            args.quality,
            "--asr-backend",
            args.asr_backend,
            "--asr-model",
            args.asr_model,
            "--ocr-backend",
            args.ocr_backend,
            "--window-padding",
            str(args.window_padding),
            "--page-detection",
            args.page_detection,
            "--page-window-mode",
            args.page_window_mode,
            "--scene-threshold",
            str(args.scene_threshold),
            "--scene-min-len",
            str(args.scene_min_len),
        ]
        proc = subprocess.run(cmd, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"local frame extraction failed for {video_path}")
        return

    legacy_scan_mode = args.scan_mode.replace("-windowed", "")
    cmd = [
        sys.executable,
        str(args.engine_script),
        str(video_path),
        "--output-dir",
        str(output_dir),
        "--work-dir",
        str(work_dir),
        "--fps",
        str(args.fps),
        "--scan-mode",
        legacy_scan_mode,
        "--dense-fps",
        str(args.dense_fps),
        "--story-start",
        f"{args.story_start:.3f}",
        "--story-end",
        f"{story_end:.3f}",
        "--quality",
        args.quality,
        "--transcript-polish-min-conf",
        f"{args.transcript_polish_min_conf:.3f}",
    ]
    if not args.polish_transcripts:
        cmd.append("--no-polish-transcripts")
    if args.include_title_intro:
        cmd += [
            "--include-title-intro",
            "--title-start",
            f"{args.title_start:.3f}",
            "--title-end",
            f"{args.title_end:.3f}",
        ]
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Frame extraction failed for {video_path}")


def extract_mp3(
    video_path: Path,
    mp3_path: Path,
    bitrate: str,
    start_time: float,
    end_time: float,
) -> None:
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, end_time - start_time)
    run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_time:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(video_path),
            "-vn",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            str(mp3_path),
        ]
    )


def build_pdf(frames_dir: Path, pdf_path: Path) -> int:
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No extracted frames found in {frames_dir}")
    images: list[Image.Image] = []
    for frame_path in frame_paths:
        images.append(Image.open(frame_path).convert("RGB"))
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    first, rest = images[0], images[1:]
    first.save(pdf_path, "PDF", save_all=True, append_images=rest, resolution=150.0)
    for image in images:
        image.close()
    return len(frame_paths)


def process_video(
    source: str,
    video_path: Path,
    downloaded: bool,
    args: argparse.Namespace,
    output_root: Path,
    work_root: Path,
) -> JobResult:
    job_slug = slugify(video_path.stem)
    output_dir = unique_dir(output_root, job_slug)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = work_root / "engine" / output_dir.name
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n==> Processing: {video_path}", flush=True)
    effective_story_end = args.story_end
    if effective_story_end is None:
        effective_story_end = video_duration(video_path)
    run_frame_engine(video_path, output_dir, work_dir, args, effective_story_end)

    artifact_stem = output_dir.name
    mp3_path = output_dir / f"{artifact_stem}.mp3"
    pdf_path = output_dir / f"{artifact_stem}.pdf"
    extract_mp3(video_path, mp3_path, args.audio_bitrate, args.story_start, effective_story_end)
    frame_count = build_pdf(output_dir / "frames", pdf_path)

    if not args.keep_work and work_dir.exists():
        shutil.rmtree(work_dir)
    result = JobResult(
        source=source,
        video_path=str(video_path),
        output_dir=str(output_dir),
        mp3_path=str(mp3_path),
        pdf_path=str(pdf_path),
        frame_count=frame_count,
        status="ok",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(asdict(result), indent=2),
        encoding="utf-8",
    )
    print(f"MP3: {mp3_path}")
    print(f"PDF: {pdf_path}")
    print(f"Frames: {frame_count}")
    return result


def main() -> None:
    args = parse_args()
    if args.command != "run":
        fail("Only the 'run' command is currently supported.")
    ensure_system_dependencies(args)
    if args.engine == "legacy" and not args.engine_script.exists():
        fail(f"Engine script not found: {args.engine_script}")

    output_root = args.output_root.expanduser().resolve()
    work_root = (args.work_root or (output_root / "_work")).expanduser().resolve()
    youtube_cache_dir = (
        args.download_cache_dir or (output_root / "_youtube-cache")
    ).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    youtube_cache_dir.mkdir(parents=True, exist_ok=True)

    jobs = expand_sources(args, youtube_cache_dir)
    if not jobs:
        fail("No videos to process.")

    results: list[JobResult] = []
    failures = 0
    for source, video_path, downloaded in jobs:
        try:
            results.append(
                process_video(source, video_path, downloaded, args, output_root, work_root)
            )
        except Exception as exc:
            failures += 1
            print(f"FAILED: {video_path}\n{exc}", file=sys.stderr)

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )
    print(f"\nDone: {len(results)} ok, {failures} failed")
    print(f"Manifest: {manifest_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
