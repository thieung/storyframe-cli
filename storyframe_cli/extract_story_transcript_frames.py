#!/usr/bin/env python3
"""
Extract review frames that contain stable story transcript text.

The pipeline is video-aware, but uses LiteParse only as the OCR/bounding-box
layer:
  1. sample video frames with ffmpeg
  2. OCR each sampled frame with LiteParse
  3. remove watermark/ad/title text items
  4. group overlapping transcript states
  5. choose one stable, non-fade representative frame per group

Install:
  brew install ffmpeg imagemagick
  python3 -m pip install liteparse pillow

Example:
  python3 extract-story-transcript-frames.py "book.mp4" \
    --output-dir story-frame-review \
    --include-title-intro \
    --story-start 14 --story-end 175
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

warnings.filterwarnings("ignore", category=DeprecationWarning)


SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parent.parent
LOCAL_LITEPARSE_DEPS_CANDIDATES = [
    WORKSPACE_ROOT / "work" / ".deps" / "liteparse-test",
    Path.cwd() / "work" / ".deps" / "liteparse-test",
    WORKSPACE_ROOT.parent / "work" / ".deps" / "liteparse-test",
]
for local_liteparse_deps in LOCAL_LITEPARSE_DEPS_CANDIDATES:
    if local_liteparse_deps.exists():
        sys.path.insert(0, str(local_liteparse_deps))

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover - optional reconstruction quality boost.
    cv2 = None
    np = None

try:
    from liteparse import LiteParse
except Exception:  # pragma: no cover - reported cleanly at runtime.
    LiteParse = None

try:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
except Exception:  # pragma: no cover - reported cleanly at runtime.
    Image = None
    ImageDraw = None
    ImageEnhance = None
    ImageFilter = None
    ImageFont = None
    ImageOps = None


IGNORE_EXACT_WORDS = {
    "vooks",
    "vo0ks",
    "vook",
    "vooksy",
    "wooks",
    "storybooks",
    "storybook",
    "brought",
    "life",
    "tm",
    "vie",
    "vcc",
}

VIDEO_TITLE_IGNORE_WORDS = {
    "a",
    "an",
    "and",
    "animated",
    "book",
    "books",
    "for",
    "full",
    "in",
    "kids",
    "narrated",
    "of",
    "official",
    "read",
    "story",
    "storybook",
    "storybooks",
    "stories",
    "the",
    "to",
    "vooks",
    "with",
}

FRAME_REJECT_PHRASES = {
    "app store",
    "google play",
    "download on the",
    "get it on",
    "try the app",
    "free today",
    "thank you for watching",
    "thank you for watch",
    "thank you for watc",
    "thank you for wat",
    "thanks for watching",
    "created by",
    "credits",
    "www",
    "com",
    "available on",
    "narrated storybooks",
    "storybooks brought",
    "brought to life",
    "brought to lif",
    "bscribe",
    "scribe",
    "ubscribe",
    "like subscribe",
    "thank you for",
    "executive producer",
    "executive producers",
    "music by",
    "voice talent",
    "sound design",
    "registered trademark",
    "copyright",
    "all rights reserved",
    "unicorn and horse",
}

TITLE_REJECT_PHRASES = {
    "illustrated by",
    "tllustrated",
    "tlustrated",
    "lustrated by",
    "written by",
    "by by",
    "anna kang",
    "christopher weyant",
    "created by vooks",
}

AD_ITEM_WORDS = {
    "subscribe",
    "subscribed",
    "ubscribe",
    "bscribe",
    "scribe",
    "app",
    "store",
    "google",
    "play",
    "download",
    "watching",
}

STORY_SAFE_AD_WORDS = {
    "like",
}

FADE_RELEVANT_EXTRA_WORDS = {
    "ok",
    "okay",
}

ALLOWED_SHORT_TOKENS = {
    "a",
    "i",
    "am",
    "an",
    "as",
    "at",
    "be",
    "by",
    "do",
    "go",
    "he",
    "hi",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "no",
    "of",
    "oh",
    "ok",
    "on",
    "or",
    "so",
    "to",
    "um",
    "up",
    "us",
    "we",
}

TRAILING_FRAGMENT_TOKENS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "but",
    "for",
    "if",
    "in",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "to",
    "when",
    "where",
    "which",
    "who",
    "with",
}

OCR_GLITCH_WORDS = {
    "arenot",
    "ay",
    "earen",
    "erie",
    "frends",
    "gore",
    "i'mbl",
    "imbl",
    "kr",
    "nl",
    "si",
    "wn",
}


@dataclass
class TextItem:
    text: str
    confidence: float
    x: float
    y: float
    width: float
    height: float
    page_width: float
    page_height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass
class FadeFinding:
    text: str
    ink_score: float
    source: str
    x: float
    y: float
    width: float
    height: float
    page_width: float
    page_height: float
    is_transcript_word: bool

    @property
    def warning(self) -> str:
        prefix = f"{self.source}:" if self.source else ""
        return f"{prefix}{self.text}:{self.ink_score:.2f}"


@dataclass
class FrameCandidate:
    frame_path: str
    timestamp: float
    raw_text: str
    normalized_text: str
    word_count: int
    avg_confidence: float
    contrast_score: float
    ad_overlay_score: float
    edge_crop_score: float
    visual_delta_from_previous: float = 0.0
    visual_delta_from_group_anchor: float = 0.0
    stable_neighbors: int = 0
    group_index: int = -1
    group_start_time: float = 0.0
    group_end_time: float = 0.0
    group_frames_seen: int = 0
    refined_from_timestamp: float = 0.0
    refinement_status: str = ""
    fade_warnings: str = ""
    output_source: str = "original"
    reconstruction_reason: str = ""
    reconstructed_from: str = ""
    score: float = 0.0


@dataclass
class GroupSummary:
    group_index: int
    start_time: float
    end_time: float
    frames_seen: int
    selected_timestamp: float
    selected_image: str
    transcript: str
    normalized_text: str
    avg_confidence: float
    edge_crop_score: float
    refinement_status: str
    fade_warnings: str
    output_source: str
    reconstruction_reason: str
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract non-duplicate, non-fade story transcript frames."
    )
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT / "outputs" / "story-frame-review",
        help="Where review images and indexes are written",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=WORKSPACE_ROOT / "work" / "story-frame-extract",
        help="Temporary sampled frames and OCR cache",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=4.0,
        help="Frame sample rate for OCR. Use 4-8 for stricter fade detection.",
    )
    parser.add_argument(
        "--scan-mode",
        choices=["sampled", "dense", "native"],
        default="sampled",
        help=(
            "Frame scan strategy. sampled uses --fps, dense raises sampling to "
            "--dense-fps, native scans at the source video frame rate."
        ),
    )
    parser.add_argument(
        "--dense-fps",
        type=float,
        default=8.0,
        help="Minimum OCR sample rate used by --scan-mode dense.",
    )
    parser.add_argument(
        "--quality",
        choices=["strict-complete", "strict-original", "balanced"],
        default="strict-complete",
        help=(
            "strict-complete keeps every transcript state by reconstructing unresolved "
            "fade frames; strict-original drops unresolved fade; balanced keeps more originals."
        ),
    )
    parser.add_argument(
        "--story-start",
        type=float,
        default=14.0,
        help="Start time in seconds for story content. Default tuned for this video.",
    )
    parser.add_argument(
        "--story-end",
        type=float,
        default=175.0,
        help="End time in seconds for story content. Default excludes host/ad/credits.",
    )
    parser.add_argument(
        "--include-title-intro",
        action="store_true",
        help="Also keep the opening title-page intro before story text starts.",
    )
    parser.add_argument(
        "--title-start",
        type=float,
        default=8.0,
        help="Start time in seconds for title intro when --include-title-intro is set.",
    )
    parser.add_argument(
        "--title-end",
        type=float,
        default=14.0,
        help="End time in seconds for title intro when --include-title-intro is set.",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.55,
        help="Minimum OCR confidence per text item",
    )
    parser.add_argument(
        "--no-tesseract-fallback",
        dest="tesseract_fallback",
        action="store_false",
        help="Disable fallback OCR via the local tesseract CLI.",
    )
    parser.set_defaults(tesseract_fallback=True)
    parser.add_argument(
        "--tesseract-min-conf",
        type=float,
        default=0.60,
        help="Minimum confidence for fallback tesseract OCR items, 0.0-1.0.",
    )
    parser.add_argument(
        "--no-polish-transcripts",
        dest="polish_transcripts",
        action="store_false",
        help=(
            "Disable final high-confidence Tesseract transcript cleanup after frame "
            "selection. Frame selection itself is unchanged."
        ),
    )
    parser.set_defaults(polish_transcripts=True)
    parser.add_argument(
        "--transcript-polish-min-conf",
        type=float,
        default=0.80,
        help="Minimum raw Tesseract confidence for final transcript cleanup, 0.0-1.0.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=2,
        help="Minimum meaningful words after filtering",
    )
    parser.add_argument(
        "--group-overlap",
        type=float,
        default=0.62,
        help="Min token overlap to group evolving text on the same page",
    )
    parser.add_argument(
        "--duplicate-similarity",
        type=float,
        default=0.92,
        help="Similarity threshold for global duplicate removal",
    )
    parser.add_argument(
        "--text-evolution-similarity",
        type=float,
        default=0.84,
        help=(
            "Similarity threshold for dropping near-time partial/fade text when "
            "a cleaner version appears shortly after."
        ),
    )
    parser.add_argument(
        "--subset-duplicate-window",
        type=float,
        default=8.0,
        help="Drop shorter transcript subsets when a longer version follows soon",
    )
    parser.add_argument(
        "--short-partial-window",
        type=float,
        default=2.5,
        help="Drop very short partial transcripts when a longer related line follows soon.",
    )
    parser.add_argument(
        "--partial-superset-window",
        type=float,
        default=16.0,
        help=(
            "Strict mode window for removing short fade/pan fragments when a nearby "
            "cleaner frame contains the same words plus more transcript."
        ),
    )
    parser.add_argument(
        "--visual-split-threshold",
        type=float,
        default=0.035,
        help=(
            "Split a group when consecutive frames have similar transcript text "
            "but visibly different page/art state."
        ),
    )
    parser.add_argument(
        "--visual-dedupe-threshold",
        type=float,
        default=0.025,
        help="Only remove duplicate transcripts when frames are also visually near-identical.",
    )
    parser.add_argument(
        "--noisy-visual-dedupe-threshold",
        type=float,
        default=0.16,
        help=(
            "Drop short-window visual duplicates with noisy OCR below this visual distance."
        ),
    )
    parser.add_argument(
        "--complete-word-ratio",
        type=float,
        default=1.0,
        help="Pick frames with at least this fraction of group max words",
    )
    parser.add_argument(
        "--min-group-contrast-ratio",
        type=float,
        default=0.70,
        help="Avoid group representatives that are visibly dimmer than the clearest frame.",
    )
    parser.add_argument(
        "--trim-fade-seconds",
        type=float,
        default=0.75,
        help="Ignore this much at group edges when possible",
    )
    parser.add_argument(
        "--no-refine-fade-frames",
        dest="refine_fade_frames",
        action="store_false",
        help="Disable local timestamp search for cleaner non-fade frames.",
    )
    parser.set_defaults(refine_fade_frames=True)
    parser.add_argument(
        "--refine-offsets",
        default="0,0.25,0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,-0.25,-0.5",
        help="Comma-separated second offsets to probe around selected frames.",
    )
    parser.add_argument(
        "--refine-extra-seconds",
        type=float,
        default=2.5,
        help="Allow refinement this far beyond a selected group's edge.",
    )
    parser.add_argument(
        "--refine-visual-threshold",
        type=float,
        default=0.08,
        help="Reject refinement frames whose page/art state changed too much.",
    )
    parser.add_argument(
        "--refine-max-added-words",
        type=int,
        default=2,
        help="Max OCR words a refinement frame may add to the selected transcript.",
    )
    parser.add_argument(
        "--fade-text-contrast-threshold",
        type=float,
        default=0.55,
        help="Tesseract text item contrast below this is treated as in-fade text.",
    )
    parser.add_argument(
        "--fade-text-min-conf",
        type=float,
        default=0.20,
        help="Minimum raw Tesseract confidence for fade-text detection, 0.0-1.0.",
    )
    parser.add_argument(
        "--keep-unresolved-fade",
        action="store_true",
        help="Keep a selected frame even if local refinement cannot remove fade text.",
    )
    parser.add_argument(
        "--reconstruct-fade-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In strict-complete mode, clean unresolved fade/ghost text instead of dropping.",
    )
    parser.add_argument(
        "--keep-sampled-frames",
        action="store_true",
        help="Do not delete/recreate sampled-frame directory",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def run_command(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nSTDERR:\n"
            + proc.stderr[-4000:]
        )


def parse_float_list(value: str) -> list[float]:
    offsets: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        offsets.append(float(part))
    if 0.0 not in offsets:
        offsets.insert(0, 0.0)
    return offsets


def extract_frame_at(video_path: Path, timestamp: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ]
    )


def parse_frame_rate(value: str) -> float | None:
    value = value.strip()
    if not value or value == "0/0":
        return None
    if "/" not in value:
        try:
            frame_rate = float(value)
        except ValueError:
            return None
        return frame_rate if frame_rate > 0 else None
    numerator, denominator = value.split("/", 1)
    try:
        top = float(numerator)
        bottom = float(denominator)
    except ValueError:
        return None
    if top <= 0 or bottom <= 0:
        return None
    return top / bottom


def video_frame_rate(video_path: Path) -> float | None:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        frame_rate = parse_frame_rate(line)
        if frame_rate:
            return frame_rate
    return None


def effective_scan_fps(args: argparse.Namespace) -> float:
    if args.scan_mode == "native":
        frame_rate = video_frame_rate(args.video)
        if frame_rate:
            return frame_rate
        print("WARNING: could not read native video FPS; falling back to --fps")
    if args.scan_mode == "dense":
        return max(args.fps, args.dense_fps)
    return args.fps


def ensure_dependencies(args: argparse.Namespace) -> None:
    if shutil.which("ffmpeg") is None:
        fail("ffmpeg not found. Install with: brew install ffmpeg")
    if args.scan_mode == "native" and shutil.which("ffprobe") is None:
        fail("ffprobe not found. Install with: brew install ffmpeg")
    if (args.tesseract_fallback or args.polish_transcripts) and shutil.which("tesseract") is None:
        fail("tesseract not found. Install with: brew install tesseract")
    if LiteParse is None:
        fail("liteparse not found. Install with: python3 -m pip install liteparse")
    if Image is None:
        fail("Pillow not found. Install with: python3 -m pip install pillow")


def recreate_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def sample_frames(args: argparse.Namespace) -> list[tuple[Path, float]]:
    sampled_dir = args.work_dir / "sampled-frames"
    if not args.keep_sampled_frames:
        recreate_dir(sampled_dir)
    else:
        sampled_dir.mkdir(parents=True, exist_ok=True)

    sample_start = args.story_start
    if args.include_title_intro:
        sample_start = min(args.title_start, args.story_start)
    scan_fps = float(getattr(args, "effective_fps", args.fps))
    duration = max(0.0, args.story_end - sample_start) if args.story_end else None
    output_pattern = sampled_dir / "frame-%06d.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{sample_start:.3f}",
    ]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += [
        "-i",
        str(args.video),
        "-vf",
        f"fps={scan_fps}",
        "-q:v",
        "2",
        str(output_pattern),
    ]
    run_command(cmd)

    frames = sorted(sampled_dir.glob("frame-*.jpg"))
    if not frames:
        fail("No frames sampled from video")

    return [(path, sample_start + index / scan_fps) for index, path in enumerate(frames)]


def clean_for_matching(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    tokens = []
    for token in text.split():
        token = token.strip("'")
        if not token:
            continue
        if token.isdigit():
            continue
        if re.search(r"[a-z]", token) and re.search(r"\d", token):
            continue
        if token in IGNORE_EXACT_WORDS:
            continue
        if len(token) == 1 and token not in {"a", "i"}:
            continue
        tokens.append(token)
    if tokens[:4] == ["i", "am", "not", "scared"] and tokens[-1:] == ["we"]:
        tokens = tokens[:-1]
    return " ".join(tokens)


def token_set(text: str) -> set[str]:
    return set(clean_for_matching(text).split())


def item_text(item: object) -> str:
    return str(getattr(item, "text", "") or "").strip()


def item_confidence(item: object) -> float:
    value = getattr(item, "confidence", 0.0)
    try:
        return float(value)
    except Exception:
        return 0.0


def is_lower_left_watermark(item: TextItem) -> bool:
    text = clean_for_matching(item.text)
    if not text:
        return True
    words = text.split()
    is_vooks = any(
        word.startswith("vook")
        or word.startswith("vo0k")
        or word.startswith("wook")
        for word in words
    )
    return is_vooks and item.x < item.page_width * 0.22 and item.y > item.page_height * 0.72


def is_bottom_overlay(item: TextItem) -> bool:
    return item.y > item.page_height * 0.74 or item.x > item.page_width * 0.72


def is_ad_overlay_item(item: TextItem) -> bool:
    text = clean_for_matching(item.text)
    words = set(text.split())
    if not words:
        return True
    if words & AD_ITEM_WORDS:
        return True
    return False


def is_title_intro_time(timestamp: float, args: argparse.Namespace) -> bool:
    return (
        args.include_title_intro
        and args.title_start <= timestamp < args.title_end
    )


def is_story_time(timestamp: float, args: argparse.Namespace) -> bool:
    return args.story_start <= timestamp < args.story_end


def should_reject_frame(
    normalized_text: str,
    raw_text: str,
    timestamp: float,
    args: argparse.Namespace,
) -> bool:
    if not is_story_time(timestamp, args) and not is_title_intro_time(timestamp, args):
        return True
    if is_title_intro_time(timestamp, args) and "i am not scared" not in normalized_text:
        return True
    combined = clean_for_matching(raw_text)
    padded = f" {combined} "
    for phrase in FRAME_REJECT_PHRASES:
        if phrase in padded:
            return True
    if not is_title_intro_time(timestamp, args):
        for phrase in TITLE_REJECT_PHRASES:
            if phrase in padded:
                return True
        if "i am not scared" in combined and ("illustrated" in combined or "anna" in combined):
            return True
    if not normalized_text:
        return True
    return False


def video_title_tokens(args: argparse.Namespace) -> set[str]:
    stem = clean_for_matching(Path(args.video).stem)
    tokens = set()
    for token in stem.split():
        if token in VIDEO_TITLE_IGNORE_WORDS:
            continue
        if token in IGNORE_EXACT_WORDS:
            continue
        if len(token) <= 1:
            continue
        tokens.add(token)
    return tokens


def candidate_text_blob(candidate: FrameCandidate) -> str:
    return f" {clean_for_matching(candidate.raw_text)} {candidate.normalized_text} "


def has_reject_phrase(candidate: FrameCandidate) -> bool:
    blob = candidate_text_blob(candidate)
    return any(phrase in blob for phrase in FRAME_REJECT_PHRASES)


def title_overlap_ratio(candidate: FrameCandidate, args: argparse.Namespace) -> float:
    title_tokens = video_title_tokens(args)
    candidate_tokens = {
        token
        for token in candidate.normalized_text.split()
        if token not in VIDEO_TITLE_IGNORE_WORDS and len(token) > 1
    }
    if not title_tokens or not candidate_tokens:
        return 0.0
    return len(title_tokens & candidate_tokens) / max(
        1,
        min(len(title_tokens), len(candidate_tokens)),
    )


def is_late_title_or_end_card(candidate: FrameCandidate, args: argparse.Namespace) -> bool:
    duration = max(1.0, args.story_end - args.story_start)
    late_threshold = args.story_start + duration * 0.78
    if candidate.timestamp < late_threshold:
        return False
    if candidate.word_count > max(10, len(video_title_tokens(args)) + 4):
        return False
    return title_overlap_ratio(candidate, args) >= 0.60


def is_non_story_candidate(candidate: FrameCandidate, args: argparse.Namespace) -> bool:
    if has_reject_phrase(candidate):
        return True
    return is_late_title_or_end_card(candidate, args)


def is_low_quality_partial_candidate(candidate: FrameCandidate) -> bool:
    if candidate.word_count <= 3 and candidate.avg_confidence < 0.90:
        return True
    if candidate.word_count <= 2 and has_noisy_ocr(candidate):
        return True
    return False


def is_story_anchor(candidate: FrameCandidate, args: argparse.Namespace) -> bool:
    if is_non_story_candidate(candidate, args):
        return False
    if is_low_quality_partial_candidate(candidate):
        return False
    if is_edge_cropped(candidate) and candidate.word_count <= 4:
        return False
    if has_noisy_ocr(candidate):
        return False
    if candidate.word_count >= 4 and candidate.avg_confidence >= 0.84:
        return True
    return candidate.word_count >= 2 and candidate.avg_confidence >= 0.93


def trim_non_story_edges(
    selected: list[FrameCandidate],
    args: argparse.Namespace,
) -> list[FrameCandidate]:
    if not selected:
        return selected

    start_index = 0
    for index, candidate in enumerate(selected):
        if is_story_anchor(candidate, args):
            start_index = index
            break
    trimmed = selected[start_index:]

    result: list[FrameCandidate] = []
    for candidate in trimmed:
        if is_late_title_or_end_card(candidate, args):
            break
        if has_reject_phrase(candidate) or is_low_quality_partial_candidate(candidate):
            continue
        result.append(candidate)
    return result


def extract_text_items(parser: object, frame_path: Path, min_conf: float) -> tuple[list[TextItem], str]:
    result = parser.parse(str(frame_path))
    if not getattr(result, "pages", None):
        return [], ""

    page = result.pages[0]
    page_width = float(getattr(page, "width", 1.0) or 1.0)
    page_height = float(getattr(page, "height", 1.0) or 1.0)
    text_items = []
    raw_parts = []
    for source in getattr(page, "text_items", []) or []:
        text = item_text(source)
        confidence = item_confidence(source)
        if text:
            raw_parts.append(text)
        if not text or confidence < min_conf:
            continue
        text_items.append(
            TextItem(
                text=text,
                confidence=confidence,
                x=float(getattr(source, "x", 0.0) or 0.0),
                y=float(getattr(source, "y", 0.0) or 0.0),
                width=float(getattr(source, "width", 0.0) or 0.0),
                height=float(getattr(source, "height", 0.0) or 0.0),
                page_width=page_width,
                page_height=page_height,
            )
        )
    return text_items, " ".join(raw_parts)


def is_useful_fallback_word(cleaned: str) -> bool:
    if not cleaned:
        return False
    if cleaned in IGNORE_EXACT_WORDS:
        return False
    if len(cleaned) <= 3:
        return cleaned in {
            "a",
            "i",
            "am",
            "an",
            "and",
            "are",
            "fun",
            "hot",
            "no",
            "not",
            "now",
            "no",
            "or",
            "of",
            "pan",
            "pit",
            "the",
            "tub",
            "to",
            "is",
            "we",
            "yes",
            "you",
        }
    return True


def extract_tesseract_text_items(frame_path: Path, min_conf: float) -> tuple[list[TextItem], str]:
    image = Image.open(frame_path)
    page_width, page_height = image.size
    proc = subprocess.run(
        ["tesseract", str(frame_path), "stdout", "--psm", "11", "-l", "eng", "tsv"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return [], ""

    items: list[TextItem] = []
    raw_parts: list[str] = []
    reader = csv.DictReader(proc.stdout.splitlines(), delimiter="\t")
    for row in reader:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        raw_parts.append(text)
        try:
            confidence = float(row.get("conf", "-1")) / 100.0
        except ValueError:
            confidence = -1.0
        cleaned = clean_for_matching(text)
        if confidence < min_conf:
            continue
        if not is_useful_fallback_word(cleaned):
            continue
        try:
            x = float(row.get("left", "0") or 0)
            y = float(row.get("top", "0") or 0)
            width = float(row.get("width", "0") or 0)
            height = float(row.get("height", "0") or 0)
        except ValueError:
            continue
        items.append(
            TextItem(
                text=text,
                confidence=confidence,
                x=x,
                y=y,
                width=width,
                height=height,
                page_width=float(page_width),
                page_height=float(page_height),
            )
        )
    return items, " ".join(raw_parts)


def extract_tesseract_raw_text_items(frame_path: Path) -> tuple[list[TextItem], str]:
    image = Image.open(frame_path)
    page_width, page_height = image.size
    proc = subprocess.run(
        ["tesseract", str(frame_path), "stdout", "--psm", "11", "-l", "eng", "tsv"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return [], ""

    items: list[TextItem] = []
    raw_parts: list[str] = []
    reader = csv.DictReader(proc.stdout.splitlines(), delimiter="\t")
    for row in reader:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        raw_parts.append(text)
        try:
            confidence = float(row.get("conf", "-1")) / 100.0
        except ValueError:
            confidence = -1.0
        try:
            x = float(row.get("left", "0") or 0)
            y = float(row.get("top", "0") or 0)
            width = float(row.get("width", "0") or 0)
            height = float(row.get("height", "0") or 0)
        except ValueError:
            continue
        items.append(
            TextItem(
                text=text,
                confidence=confidence,
                x=x,
                y=y,
                width=width,
                height=height,
                page_width=float(page_width),
                page_height=float(page_height),
            )
        )
    return items, " ".join(raw_parts)


def extract_boosted_tesseract_raw_text_items(frame_path: Path) -> tuple[list[TextItem], str]:
    if ImageOps is None or ImageEnhance is None:
        return [], ""
    image = Image.open(frame_path).convert("L")
    boosted = ImageOps.autocontrast(image, cutoff=0)
    boosted = ImageEnhance.Contrast(boosted).enhance(2.5)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            temp_path = Path(handle.name)
        boosted.save(temp_path)
        return extract_tesseract_raw_text_items(temp_path)
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass


def filter_story_items(items: Iterable[TextItem]) -> list[TextItem]:
    story_items = []
    for item in items:
        if is_lower_left_watermark(item):
            continue
        if is_ad_overlay_item(item) and is_bottom_overlay(item):
            continue
        cleaned = clean_for_matching(item.text)
        if not cleaned:
            continue
        if all(word in IGNORE_EXACT_WORDS for word in cleaned.split()):
            continue
        story_items.append(item)
    return story_items


def item_ink_score(frame_path: Path, item: TextItem) -> float:
    image = Image.open(frame_path).convert("L")
    image_width, image_height = image.size
    sx = image_width / max(item.page_width, 1.0)
    sy = image_height / max(item.page_height, 1.0)
    left = max(0, int(math.floor((item.x - 2) * sx)))
    top = max(0, int(math.floor((item.y - 2) * sy)))
    right = min(image_width, int(math.ceil((item.right + 2) * sx)))
    bottom = min(image_height, int(math.ceil((item.bottom + 2) * sy)))
    if right <= left or bottom <= top:
        return 0.0
    crop = image.crop((left, top, right, bottom))
    pixels = sorted(crop.getdata())
    if not pixels:
        return 0.0
    p10 = pixels[max(0, int(len(pixels) * 0.10) - 1)]
    p90 = pixels[min(len(pixels) - 1, int(len(pixels) * 0.90))]
    darkness = (255.0 - p10) / 255.0
    spread = max(0.0, (p90 - p10) / 255.0)
    return max(darkness, spread)


def image_contrast_score(frame_path: Path, items: list[TextItem]) -> float:
    if not items:
        return 0.0
    scores = []
    for item in items:
        score = item_ink_score(frame_path, item)
        if score > 0:
            scores.append(score)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def text_edge_crop_score(items: list[TextItem]) -> float:
    if not items:
        return 0.0
    page_width = max(items[0].page_width, 1.0)
    page_height = max(items[0].page_height, 1.0)
    left = min(item.x for item in items)
    right = max(item.right for item in items)
    top = min(item.y for item in items)
    bottom = max(item.bottom for item in items)
    x_margin = page_width * 0.018
    y_margin = page_height * 0.014

    score = 0.0
    if left <= x_margin:
        score += 0.55
    if right >= page_width - x_margin:
        score += 0.55
    if top <= y_margin:
        score += 0.20
    if bottom >= page_height - y_margin:
        score += 0.20

    edge_items = 0
    for item in items:
        if item.x <= x_margin or item.right >= page_width - x_margin:
            edge_items += 1
    score += min(0.35, edge_items / max(1, len(items)) * 0.35)
    return min(1.0, score)


def bottom_right_ad_overlay_score(frame_path: Path) -> float:
    image = Image.open(frame_path).convert("RGB")
    width, height = image.size
    crop = image.crop((int(width * 0.70), int(height * 0.72), width, height))
    pixels = list(crop.getdata())
    if not pixels:
        return 0.0
    saturated = 0
    for red, green, blue in pixels:
        red_button = red > 170 and green < 70 and blue < 90
        red_button = red_button and red > green * 2.2 and red > blue * 2.2
        if red_button:
            saturated += 1
    return saturated / len(pixels)


def bottom_right_social_overlay_score(frame_path: Path) -> float:
    image = Image.open(frame_path).convert("RGB")
    width, height = image.size
    crop = image.crop((int(width * 0.58), int(height * 0.70), width, int(height * 0.95)))
    pixels = list(crop.getdata())
    if not pixels:
        return 0.0

    red = 0
    blue = 0
    for red_value, green_value, blue_value in pixels:
        red_button = (
            red_value > 175
            and green_value < 105
            and blue_value < 120
            and red_value > green_value * 1.8
            and red_value > blue_value * 1.8
        )
        blue_button = (
            blue_value > 135
            and green_value > 70
            and red_value < 105
            and blue_value > red_value * 1.6
            and blue_value > green_value * 1.05
        )
        if red_button:
            red += 1
        if blue_button:
            blue += 1
    return red / len(pixels) + blue / len(pixels) * 1.4


def visual_thumbnail(frame_path: str | Path, cache: dict[str, object]) -> object:
    key = str(frame_path)
    if key not in cache:
        image = Image.open(key).convert("RGB")
        width, height = image.size
        crop = image.crop((0, 0, width, int(height * 0.88)))
        cache[key] = crop.resize((64, 36))
    return cache[key]


def visual_distance(
    left_frame_path: str | Path,
    right_frame_path: str | Path,
    cache: dict[str, object],
) -> float:
    left = visual_thumbnail(left_frame_path, cache)
    right = visual_thumbnail(right_frame_path, cache)
    left_pixels = list(left.getdata())
    right_pixels = list(right.getdata())
    if not left_pixels or len(left_pixels) != len(right_pixels):
        return 1.0

    total = 0.0
    for left_rgb, right_rgb in zip(left_pixels, right_pixels):
        total += sum(abs(left_rgb[channel] - right_rgb[channel]) for channel in range(3)) / 3.0
    return total / (len(left_pixels) * 255.0)


def build_candidate(
    parser: object,
    frame_path: Path,
    timestamp: float,
    args: argparse.Namespace,
    force_tesseract: bool = False,
) -> FrameCandidate | None:
    items, raw_text = extract_text_items(parser, frame_path, args.min_conf)
    story_items = filter_story_items(items)
    transcript = " ".join(item.text for item in story_items)
    normalized = clean_for_matching(transcript)
    words = normalized.split()
    used_fallback = False
    should_try_tesseract = args.tesseract_fallback and (
        force_tesseract or len(words) < args.min_words or len(words) <= 3
    )
    if should_try_tesseract:
        fallback_items, fallback_raw_text = extract_tesseract_text_items(
            frame_path, args.tesseract_min_conf
        )
        fallback_story_items = filter_story_items(fallback_items)
        fallback_transcript = " ".join(item.text for item in fallback_story_items)
        fallback_normalized = clean_for_matching(fallback_transcript)
        fallback_words = fallback_normalized.split()
        added_words = [word for word in fallback_words if word not in set(words)]
        added_meaningful_word = any(len(word) >= 4 for word in added_words)
        should_use_fallback = len(words) < args.min_words or added_meaningful_word
        if should_use_fallback and len(fallback_words) > len(words):
            story_items = fallback_story_items
            transcript = fallback_transcript
            normalized = fallback_normalized
            words = fallback_words
            used_fallback = True
        if fallback_raw_text:
            raw_text = f"{raw_text} {fallback_raw_text}".strip()

    if len(words) < args.min_words:
        return None
    if used_fallback and not any(len(word) >= 4 for word in words):
        return None
    if should_reject_frame(normalized, raw_text, timestamp, args):
        return None
    ad_overlay_score = max(
        bottom_right_ad_overlay_score(frame_path),
        bottom_right_social_overlay_score(frame_path),
    )
    if ad_overlay_score > 0.020 and args.quality != "strict-complete":
        return None

    avg_conf = sum(item.confidence for item in story_items) / max(1, len(story_items))
    contrast = image_contrast_score(frame_path, story_items)
    edge_crop = text_edge_crop_score(story_items)
    return FrameCandidate(
        frame_path=str(frame_path),
        timestamp=timestamp,
        raw_text=transcript,
        normalized_text=normalized,
        word_count=len(words),
        avg_confidence=avg_conf,
        contrast_score=contrast,
        ad_overlay_score=ad_overlay_score,
        edge_crop_score=edge_crop,
    )


def similarity(left: str, right: str) -> float:
    left_clean = clean_for_matching(left)
    right_clean = clean_for_matching(right)
    if not left_clean or not right_clean:
        return 0.0
    if left_clean == right_clean:
        return 1.0
    left_tokens = set(left_clean.split())
    right_tokens = set(right_clean.split())
    overlap = len(left_tokens & right_tokens)
    min_overlap = overlap / max(1, min(len(left_tokens), len(right_tokens)))
    jaccard = overlap / max(1, len(left_tokens | right_tokens))
    sequence = SequenceMatcher(None, left_clean, right_clean).ratio()
    return max(min_overlap * 0.78 + jaccard * 0.22, sequence)


def near_duplicate_similarity(left: str, right: str) -> float:
    left_clean = clean_for_matching(left)
    right_clean = clean_for_matching(right)
    if not left_clean or not right_clean:
        return 0.0
    if left_clean == right_clean:
        return 1.0
    left_tokens = set(left_clean.split())
    right_tokens = set(right_clean.split())
    overlap = len(left_tokens & right_tokens)
    jaccard = overlap / max(1, len(left_tokens | right_tokens))
    sequence = SequenceMatcher(None, left_clean, right_clean).ratio()
    return max(jaccard, sequence)


def token_containment(left: str, right: str) -> float:
    left_tokens = set(clean_for_matching(left).split())
    right_tokens = set(clean_for_matching(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))


def is_edge_cropped(candidate: FrameCandidate) -> bool:
    return candidate.edge_crop_score >= 0.55


def group_candidates(
    candidates: list[FrameCandidate], args: argparse.Namespace
) -> list[list[FrameCandidate]]:
    groups: list[list[FrameCandidate]] = []
    anchors: list[str] = []
    visual_anchors: list[str] = []
    visual_cache: dict[str, object] = {}
    scan_fps = float(getattr(args, "effective_fps", args.fps))
    max_gap = max(2.5, 3.0 / scan_fps)

    for candidate in candidates:
        if not groups:
            groups.append([candidate])
            anchors.append(candidate.normalized_text)
            visual_anchors.append(candidate.frame_path)
            continue

        previous = groups[-1][-1]
        anchor = anchors[-1]
        visual_anchor = visual_anchors[-1]
        gap = candidate.timestamp - previous.timestamp
        related_to_previous = similarity(candidate.normalized_text, previous.normalized_text)
        related_to_anchor = similarity(candidate.normalized_text, anchor)
        text_related = max(related_to_previous, related_to_anchor) >= args.group_overlap
        visual_delta = visual_distance(candidate.frame_path, previous.frame_path, visual_cache)
        visual_anchor_delta = visual_distance(candidate.frame_path, visual_anchor, visual_cache)
        candidate.visual_delta_from_previous = visual_delta
        candidate.visual_delta_from_group_anchor = visual_anchor_delta
        same_visual_state = max(visual_delta, visual_anchor_delta) < args.visual_split_threshold
        crop_pan_continuation = (
            text_related
            and (is_edge_cropped(candidate) or is_edge_cropped(previous))
            and token_containment(candidate.normalized_text, anchor) >= 0.55
        )
        if gap <= max_gap and text_related and (same_visual_state or crop_pan_continuation):
            groups[-1].append(candidate)
            if candidate.word_count >= len(anchor.split()):
                anchors[-1] = candidate.normalized_text
        else:
            groups.append([candidate])
            anchors.append(candidate.normalized_text)
            visual_anchors.append(candidate.frame_path)
            candidate.visual_delta_from_group_anchor = 0.0

    for index, group in enumerate(groups):
        for candidate in group:
            candidate.group_index = index
            candidate.group_start_time = group[0].timestamp
            candidate.group_end_time = group[-1].timestamp
            candidate.group_frames_seen = len(group)
    return groups


def count_stable_neighbors(
    candidate: FrameCandidate,
    group: list[FrameCandidate],
    max_words: int,
    args: argparse.Namespace,
) -> int:
    stable = 0
    scan_fps = float(getattr(args, "effective_fps", args.fps))
    window = max(0.75, 2.0 / scan_fps)
    for other in group:
        if other is candidate:
            continue
        if abs(other.timestamp - candidate.timestamp) > window:
            continue
        if other.word_count < max(1, math.ceil(max_words * args.complete_word_ratio)):
            continue
        if similarity(candidate.normalized_text, other.normalized_text) >= args.group_overlap:
            stable += 1
    return stable


def choose_group_representative(
    group: list[FrameCandidate], args: argparse.Namespace
) -> FrameCandidate | None:
    if not group:
        return None

    start = group[0].timestamp
    end = group[-1].timestamp
    max_words = max(candidate.word_count for candidate in group)
    if max_words <= 3:
        min_words = max(args.min_words, max_words - 1)
    else:
        min_words = max(args.min_words, math.ceil(max_words * args.complete_word_ratio))
    eligible = [candidate for candidate in group if candidate.word_count >= min_words]
    if not eligible:
        eligible = list(group)

    max_contrast = max(candidate.contrast_score for candidate in group)
    if max_contrast > 0:
        contrast_floor = max_contrast * args.min_group_contrast_ratio
        high_contrast = [
            candidate for candidate in eligible if candidate.contrast_score >= contrast_floor
        ]
        if high_contrast:
            eligible = high_contrast
        elif len(group) > 1:
            return None

    if end - start > args.trim_fade_seconds * 2.2:
        trimmed = [
            candidate
            for candidate in eligible
            if start + args.trim_fade_seconds
            <= candidate.timestamp
            <= end - args.trim_fade_seconds
        ]
        if trimmed:
            eligible = trimmed

    for candidate in eligible:
        stable_neighbors = count_stable_neighbors(candidate, group, max_words, args)
        candidate.stable_neighbors = stable_neighbors
        edge_distance = min(candidate.timestamp - start, end - candidate.timestamp)
        edge_penalty = max(0.0, args.trim_fade_seconds - edge_distance) * 14.0
        completeness = candidate.word_count / max(1, max_words)
        contrast_ratio = candidate.contrast_score / max(max_contrast, 0.001)
        candidate.score = (
            candidate.avg_confidence * 42.0
            + candidate.contrast_score * 28.0
            + contrast_ratio * 12.0
            + completeness * 10.0
            + stable_neighbors * 4.0
            + min(candidate.word_count, 18) * 0.8
            - edge_penalty
            - candidate.edge_crop_score * 36.0
            - candidate.ad_overlay_score * 24.0
        )

    return max(eligible, key=lambda candidate: candidate.score)


def remove_global_duplicates(
    representatives: list[FrameCandidate], args: argparse.Namespace
) -> list[FrameCandidate]:
    selected: list[FrameCandidate] = []
    visual_cache: dict[str, object] = {}
    for candidate in representatives:
        duplicate_index = None
        for index, existing in enumerate(selected):
            if (
                args.include_title_intro
                and is_title_intro_time(candidate.timestamp, args)
                != is_title_intro_time(existing.timestamp, args)
            ):
                continue
            candidate_tokens = set(candidate.normalized_text.split())
            existing_tokens = set(existing.normalized_text.split())
            gap = abs(candidate.timestamp - existing.timestamp)
            crop_related = (
                gap <= args.subset_duplicate_window
                and (is_edge_cropped(candidate) or is_edge_cropped(existing))
                and token_containment(
                    candidate.normalized_text,
                    existing.normalized_text,
                )
                >= 0.55
            )
            visual_gap = visual_distance(candidate.frame_path, existing.frame_path, visual_cache)
            if visual_gap > args.visual_dedupe_threshold and not crop_related:
                continue
            if (
                near_duplicate_similarity(candidate.normalized_text, existing.normalized_text)
                >= args.duplicate_similarity
            ):
                duplicate_index = index
                break
            if gap <= args.subset_duplicate_window:
                candidate_is_longer = len(candidate_tokens) > len(existing_tokens)
                existing_subset = existing_tokens <= candidate_tokens
                candidate_subset = candidate_tokens <= existing_tokens
                if candidate_is_longer and existing_subset:
                    duplicate_index = index
                    break
                if candidate_subset and not candidate_is_longer:
                    duplicate_index = index
                    break
        if duplicate_index is None:
            selected.append(candidate)
            continue
        existing = selected[duplicate_index]
        candidate_tokens = set(candidate.normalized_text.split())
        existing_tokens = set(existing.normalized_text.split())
        if candidate_tokens < existing_tokens:
            continue
        if len(candidate_tokens) > len(existing_tokens) and existing_tokens <= candidate_tokens:
            selected[duplicate_index] = candidate
        elif candidate.score > existing.score:
            selected[duplicate_index] = candidate
    return sorted(selected, key=lambda item: item.timestamp)


def is_strong_visual_state(candidate: FrameCandidate) -> bool:
    return (
        candidate.contrast_score >= 0.82
        and not is_edge_cropped(candidate)
        and (candidate.group_frames_seen >= 8 or candidate.stable_neighbors >= 2)
    )


def is_fade_like_transition(candidate: FrameCandidate) -> bool:
    if candidate.contrast_score < 0.50:
        return True
    if is_edge_cropped(candidate) and candidate.stable_neighbors <= 1:
        return True
    return (
        candidate.group_frames_seen <= 2
        and candidate.stable_neighbors <= 1
        and candidate.contrast_score < 0.70
    )


def ocr_noise_ratio(candidate: FrameCandidate) -> float:
    tokens = candidate.normalized_text.split()
    if not tokens:
        return 1.0
    noisy = 0
    for token in tokens:
        has_vowel = bool(re.search(r"[aeiou]", token))
        if token in OCR_GLITCH_WORDS:
            noisy += 1
        elif len(token) <= 2 and token not in ALLOWED_SHORT_TOKENS:
            noisy += 1
        elif len(token) >= 4 and not has_vowel:
            noisy += 1
    return noisy / len(tokens)


def has_noisy_ocr(candidate: FrameCandidate) -> bool:
    noise_ratio = ocr_noise_ratio(candidate)
    return noise_ratio >= 0.34 or (
        candidate.avg_confidence < 0.75 and noise_ratio >= 0.20
    )


def has_trailing_fragment(candidate: FrameCandidate) -> bool:
    tokens = candidate.normalized_text.split()
    return bool(tokens) and tokens[-1] in TRAILING_FRAGMENT_TOKENS


def is_preferred_duplicate(
    other: FrameCandidate,
    candidate: FrameCandidate,
) -> bool:
    score_margin = 0.35
    if other.score > candidate.score + score_margin:
        return True
    if candidate.score > other.score + score_margin:
        return False
    if other.group_frames_seen != candidate.group_frames_seen:
        return other.group_frames_seen > candidate.group_frames_seen
    if other.stable_neighbors != candidate.stable_neighbors:
        return other.stable_neighbors > candidate.stable_neighbors
    if abs(other.timestamp - candidate.timestamp) > 0.001:
        return other.timestamp < candidate.timestamp
    return False


def is_noisy_token(token: str) -> bool:
    if not token:
        return True
    has_vowel = bool(re.search(r"[aeiou]", token))
    if token in OCR_GLITCH_WORDS:
        return True
    if len(token) <= 2 and token not in ALLOWED_SHORT_TOKENS:
        return True
    if len(token) >= 4 and not has_vowel:
        return True
    return False


def token_noise_penalty(tokens: list[str]) -> float:
    if not tokens:
        return 1.0
    return sum(1 for token in tokens if is_noisy_token(token)) / len(tokens)


def trim_trailing_noisy_tokens(tokens: list[str]) -> list[str]:
    trimmed = list(tokens)
    while trimmed and is_noisy_token(trimmed[-1]):
        trimmed.pop()
    return trimmed


def trim_trailing_fragment_tokens(tokens: list[str]) -> list[str]:
    trimmed = list(tokens)
    while trimmed and trimmed[-1] in TRAILING_FRAGMENT_TOKENS:
        trimmed.pop()
    return trimmed


def has_probable_token_correction(original_tokens: list[str], polished_tokens: list[str]) -> bool:
    if len(original_tokens) != len(polished_tokens):
        return False
    changed = 0
    for original, polished in zip(original_tokens, polished_tokens):
        if original == polished:
            continue
        changed += 1
        if changed > 2:
            return False
        if is_noisy_token(original) and not is_noisy_token(polished):
            continue
        if (
            polished in ALLOWED_SHORT_TOKENS
            and original not in ALLOWED_SHORT_TOKENS
            and len(polished) <= 3
            and (
                original.startswith(polished)
                or SequenceMatcher(None, original, polished).ratio() >= 0.72
            )
        ):
            continue
        return False
    return changed > 0


def should_use_polished_transcript(original_text: str, polished_text: str) -> bool:
    original = clean_for_matching(original_text)
    polished = clean_for_matching(polished_text)
    if not original or not polished or original == polished:
        return False

    original_tokens = original.split()
    polished_tokens = polished.split()
    overlap = len(set(original_tokens) & set(polished_tokens))
    containment = overlap / max(1, min(len(original_tokens), len(polished_tokens)))
    if containment < 0.70:
        return False

    trimmed_original = trim_trailing_noisy_tokens(original_tokens)
    if (
        len(trimmed_original) < len(original_tokens)
        and polished_tokens == trimmed_original
    ):
        return True

    trimmed_fragment_original = trim_trailing_fragment_tokens(original_tokens)
    if (
        len(trimmed_fragment_original) < len(original_tokens)
        and polished_tokens == trimmed_fragment_original
    ):
        return True

    original_noise = token_noise_penalty(original_tokens)
    polished_noise = token_noise_penalty(polished_tokens)
    if (
        polished_noise + 0.01 < original_noise
        and len(polished_tokens) >= len(original_tokens) - 2
    ):
        return True

    if containment >= 0.85 and has_probable_token_correction(
        original_tokens,
        polished_tokens,
    ):
        return True

    if (
        len(polished_tokens) > len(original_tokens)
        and containment >= 0.90
        and polished_noise <= original_noise + 0.05
    ):
        return True

    return False


def should_prune_against(
    candidate: FrameCandidate,
    other: FrameCandidate,
    args: argparse.Namespace,
    visual_cache: dict[str, object],
) -> bool:
    gap = abs(candidate.timestamp - other.timestamp)
    max_prune_window = max(args.subset_duplicate_window, args.partial_superset_window)
    if gap > max_prune_window:
        return False

    same_text = candidate.normalized_text == other.normalized_text
    visual_gap = visual_distance(candidate.frame_path, other.frame_path, visual_cache)
    candidate_tokens = set(candidate.normalized_text.split())
    other_tokens = set(other.normalized_text.split())
    candidate_subset = candidate_tokens <= other_tokens
    other_subset = other_tokens <= candidate_tokens
    related = similarity(candidate.normalized_text, other.normalized_text)
    near_related = near_duplicate_similarity(candidate.normalized_text, other.normalized_text)
    containment = token_containment(candidate.normalized_text, other.normalized_text)
    other_is_later = other.timestamp > candidate.timestamp
    other_is_cleaner_superset = (
        other.word_count >= candidate.word_count + 2
        and containment >= (0.55 if candidate.word_count <= 3 else 0.75)
        and not is_edge_cropped(other)
        and not has_noisy_ocr(other)
    )
    candidate_has_fragment_or_noise = (
        has_trailing_fragment(candidate)
        or token_noise_penalty(candidate.normalized_text.split()) > 0.0
        or candidate.output_source == "reconstructed"
    )

    if (
        gap <= args.partial_superset_window
        and other_is_cleaner_superset
        and candidate_has_fragment_or_noise
        and candidate.word_count <= 18
    ):
        return True

    if (
        gap <= args.partial_superset_window
        and other_is_cleaner_superset
        and candidate.word_count <= 3
        and (candidate.group_frames_seen <= 3 or candidate.stable_neighbors <= 1)
    ):
        return True

    if (
        gap <= args.partial_superset_window
        and other_is_cleaner_superset
        and candidate.word_count <= 6
        and (
            candidate.group_frames_seen <= 6
            or candidate.stable_neighbors <= 1
            or is_fade_like_transition(candidate)
        )
    ):
        return True

    if (
        is_edge_cropped(candidate)
        and candidate.group_frames_seen <= 2
        and candidate.word_count <= 16
        and (candidate.edge_crop_score >= 0.65 or has_noisy_ocr(candidate))
    ):
        return True

    if (
        candidate_subset
        and gap <= args.short_partial_window
        and candidate.word_count >= 5
        and other.word_count > candidate.word_count
        and other.score >= candidate.score - 24.0
    ):
        return True

    if (
        other_subset
        and gap <= args.short_partial_window
        and candidate.avg_confidence + 0.04 < other.avg_confidence
        and other.score >= candidate.score - 16.0
    ):
        return True

    if is_edge_cropped(candidate) and containment >= 0.55:
        if other.word_count > candidate.word_count and other.score >= candidate.score - 24.0:
            return True
        if other.word_count >= candidate.word_count and not is_edge_cropped(other):
            return True

    if same_text:
        if (
            visual_gap > 0.08
            and is_strong_visual_state(candidate)
            and is_strong_visual_state(other)
        ):
            return False
        if is_fade_like_transition(candidate) and not is_fade_like_transition(other):
            return True
        if (
            is_title_intro_time(candidate.timestamp, args)
            and is_title_intro_time(other.timestamp, args)
            and is_preferred_duplicate(other, candidate)
        ):
            return True
        return is_preferred_duplicate(other, candidate)

    if candidate_subset or (
        related >= args.group_overlap and candidate.word_count < other.word_count
    ):
        if (
            is_strong_visual_state(candidate)
            and is_strong_visual_state(other)
            and visual_gap > args.visual_split_threshold
        ):
            return False
        if (
            visual_gap <= args.visual_split_threshold * 1.25
            and other.word_count >= candidate.word_count
            and (other.score >= candidate.score or other.word_count > candidate.word_count)
        ):
            return True
        if is_fade_like_transition(candidate) and other.word_count >= candidate.word_count:
            return True

    if other_subset and is_fade_like_transition(candidate) and related >= args.group_overlap:
        return True

    if (
        gap <= args.short_partial_window
        and visual_gap <= args.noisy_visual_dedupe_threshold
        and has_noisy_ocr(candidate)
        and has_noisy_ocr(other)
        and other.score >= candidate.score
    ):
        return True

    if (
        gap <= args.short_partial_window
        and is_edge_cropped(candidate)
        and candidate.word_count <= 4
        and other.word_count >= candidate.word_count + 2
        and containment >= 0.40
    ):
        return True

    if other_is_later and near_related >= args.text_evolution_similarity:
        if visual_gap > 0.08 and is_strong_visual_state(candidate) and is_strong_visual_state(other):
            return False
        if other.word_count >= candidate.word_count and other.score >= candidate.score - 8.0:
            return True
        if other.word_count > candidate.word_count and other.avg_confidence >= candidate.avg_confidence - 0.02:
            return True
        if is_fade_like_transition(candidate) and other.score >= candidate.score - 12.0:
            return True

    if (
        other_is_later
        and gap <= args.short_partial_window
        and candidate.word_count <= 2
        and other.word_count >= candidate.word_count + 2
        and candidate_tokens & other_tokens
        and other.avg_confidence >= candidate.avg_confidence
    ):
        return True

    return False


def prune_transition_duplicates(
    selected: list[FrameCandidate], args: argparse.Namespace
) -> list[FrameCandidate]:
    visual_cache: dict[str, object] = {}
    drop_indexes: set[int] = set()
    for index, candidate in enumerate(selected):
        for other_index, other in enumerate(selected):
            if index == other_index:
                continue
            if should_prune_against(candidate, other, args, visual_cache):
                drop_indexes.add(index)
                break
    return [
        candidate
        for index, candidate in enumerate(selected)
        if index not in drop_indexes
    ]


def fade_text_findings(
    frame_path: str | Path,
    normalized_text: str,
    args: argparse.Namespace,
) -> list[FadeFinding]:
    transcript_tokens = set(normalized_text.split())
    if not transcript_tokens:
        return []

    findings: list[FadeFinding] = []
    items, _ = extract_tesseract_raw_text_items(Path(frame_path))
    visible_transcript_items: list[TextItem] = []
    for item in items:
        if is_lower_left_watermark(item):
            continue
        if is_bottom_overlay(item) and is_ad_overlay_item(item):
            continue
        cleaned = clean_for_matching(item.text)
        words = cleaned.split()
        if not words:
            continue
        if all(word in IGNORE_EXACT_WORDS for word in words):
            continue
        is_transcript_word = any(word in transcript_tokens for word in words)
        is_extra_transcript = any(word in FADE_RELEVANT_EXTRA_WORDS for word in words)
        ink_score = item_ink_score(Path(frame_path), item)
        if is_transcript_word and ink_score >= args.fade_text_contrast_threshold:
            visible_transcript_items.append(item)
        is_low_conf_extra = (
            not is_transcript_word
            and not is_extra_transcript
            and item.confidence <= 0.10
            and item.width >= 16
            and item.height >= 10
            and item.width <= item.page_width * 0.12
            and item.height <= item.page_height * 0.08
            and item.y < item.page_height * 0.65
            and 0.03 <= ink_score <= args.fade_text_contrast_threshold
            and all(len(word) >= 2 for word in words)
        )
        if not is_low_conf_extra and item.confidence < args.fade_text_min_conf:
            continue
        if not is_transcript_word and not is_extra_transcript and not is_low_conf_extra:
            continue
        if not is_extra_transcript and all(len(word) <= 1 for word in words):
            continue
        if ink_score < args.fade_text_contrast_threshold:
            findings.append(
                FadeFinding(
                    text=cleaned,
                    ink_score=ink_score,
                    source="",
                    x=item.x,
                    y=item.y,
                    width=item.width,
                    height=item.height,
                    page_width=item.page_width,
                    page_height=item.page_height,
                    is_transcript_word=is_transcript_word,
                )
            )

    boosted_items, _ = extract_boosted_tesseract_raw_text_items(Path(frame_path))
    seen_warnings = {finding.warning for finding in findings}
    for item in boosted_items:
        if item.confidence < 0.60:
            continue
        if is_lower_left_watermark(item):
            continue
        if is_bottom_overlay(item) and is_ad_overlay_item(item):
            continue
        if item.y > item.page_height * 0.65:
            continue
        cleaned = clean_for_matching(item.text)
        words = cleaned.split()
        if not words:
            continue
        if all(word in IGNORE_EXACT_WORDS for word in words):
            continue
        if all(word in transcript_tokens for word in words):
            ink_score = item_ink_score(Path(frame_path), item)
            if ink_score >= args.fade_text_contrast_threshold:
                visible_transcript_items.append(item)
            continue
        ink_score = item_ink_score(Path(frame_path), item)
        if ink_score >= args.fade_text_contrast_threshold:
            continue
        story_column_ghost_box = multiline_story_text_column_ghost_box(
            item,
            visible_transcript_items,
        )
        is_near_story_column = story_column_ghost_box is not None
        if item.confidence < 0.80 and not is_near_story_column:
            continue
        if not is_near_story_column and not any(
            len(word) >= 3 or word in FADE_RELEVANT_EXTRA_WORDS for word in words
        ):
            continue
        warning = f"boost:{cleaned}:{ink_score:.2f}"
        if warning not in seen_warnings:
            finding_x = item.x
            finding_y = item.y
            finding_width = item.width
            finding_height = item.height
            source = "boost"
            if story_column_ghost_box is not None:
                left, top, right, bottom = story_column_ghost_box
                finding_x = left
                finding_y = top
                finding_width = right - left
                finding_height = bottom - top
                source = "boost-column"
            findings.append(
                FadeFinding(
                    text=cleaned,
                    ink_score=ink_score,
                    source=source,
                    x=finding_x,
                    y=finding_y,
                    width=finding_width,
                    height=finding_height,
                    page_width=item.page_width,
                    page_height=item.page_height,
                    is_transcript_word=False,
                )
            )
            seen_warnings.add(warning)
    return findings


def fade_text_warnings(
    frame_path: str | Path,
    normalized_text: str,
    args: argparse.Namespace,
) -> list[str]:
    return [
        finding.warning
        for finding in fade_text_findings(frame_path, normalized_text, args)
    ]


def is_near_multiline_story_text_column(
    item: TextItem,
    transcript_items: list[TextItem],
) -> bool:
    return multiline_story_text_column_ghost_box(item, transcript_items) is not None


def multiline_story_text_column_ghost_box(
    item: TextItem,
    transcript_items: list[TextItem],
) -> tuple[float, float, float, float] | None:
    if len(transcript_items) < 3:
        return None
    page_width = max(item.page_width, 1.0)
    page_height = max(item.page_height, 1.0)
    left = min(text_item.x for text_item in transcript_items)
    right = max(text_item.right for text_item in transcript_items)
    top = min(text_item.y for text_item in transcript_items)
    bottom = max(text_item.bottom for text_item in transcript_items)
    if bottom - top < page_height * 0.12:
        return None

    x_center = item.x + item.width / 2.0
    horizontal_padding = page_width * 0.04
    vertical_padding = page_height * 0.18
    if not (
        left - horizontal_padding <= x_center <= right + horizontal_padding
        and bottom <= item.y <= bottom + vertical_padding
        and item.height <= page_height * 0.08
    ):
        return None

    box_left = max(0.0, left - horizontal_padding)
    box_top = max(bottom, item.y - page_height * 0.06)
    box_right = min(page_width, right + horizontal_padding)
    box_bottom = min(page_height, item.bottom + page_height * 0.10)
    if box_bottom <= box_top or box_right <= box_left:
        return None
    return box_left, box_top, box_right, box_bottom


def bottom_right_overlay_color_box(
    frame_path: Path,
) -> tuple[float, float, float, float, float, float] | None:
    image = Image.open(frame_path).convert("RGB")
    width, height = image.size
    x0 = int(width * 0.70)
    y0 = int(height * 0.78)
    x1 = width
    y1 = int(height * 0.95)
    crop = image.crop((x0, y0, x1, y1))
    xs: list[int] = []
    ys: list[int] = []
    for y in range(crop.height):
        for x in range(crop.width):
            red, green, blue = crop.getpixel((x, y))
            red_button = (
                red > 170
                and green < 70
                and blue < 90
                and red > green * 2.2
                and red > blue * 2.2
            )
            blue_button = (
                blue > 135
                and green > 70
                and red < 105
                and blue > red * 1.6
                and blue > green * 1.05
            )
            if red_button or blue_button:
                xs.append(x0 + x)
                ys.append(y0 + y)
    if len(xs) < 80:
        return None
    left = max(0.0, float(min(xs) - 24))
    top = max(0.0, float(min(ys) - 18))
    right = min(float(width), float(max(xs) + 24))
    bottom = min(float(height), float(max(ys) + 58))
    box_width = right - left
    box_height = bottom - top
    if top < height * 0.79 or box_width < 120 or box_height > box_width * 0.75:
        return None
    return left, top, right, bottom, float(width), float(height)


def social_overlay_findings(
    frame_path: str | Path,
    args: argparse.Namespace,
) -> list[FadeFinding]:
    path = Path(frame_path)
    findings: list[FadeFinding] = []
    items, _ = extract_tesseract_raw_text_items(path)
    seen: set[str] = set()
    for item in items:
        if is_lower_left_watermark(item):
            continue
        if not is_bottom_overlay(item) or not is_ad_overlay_item(item):
            continue
        cleaned = clean_for_matching(item.text)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        page_width = max(item.page_width, 1.0)
        page_height = max(item.page_height, 1.0)
        left = max(0.0, item.x - page_width * 0.10)
        top = max(0.0, item.y - page_height * 0.055)
        right = min(page_width, item.right + page_width * 0.045)
        bottom = min(page_height, item.bottom + page_height * 0.085)
        findings.append(
            FadeFinding(
                text=f"social-overlay-{cleaned}",
                ink_score=1.0,
                source="overlay",
                x=left,
                y=top,
                width=max(1.0, right - left),
                height=max(1.0, bottom - top),
                page_width=page_width,
                page_height=page_height,
                is_transcript_word=False,
            )
        )

    if findings:
        return findings

    if bottom_right_ad_overlay_score(path) <= 0.020:
        return []
    color_box = bottom_right_overlay_color_box(path)
    if color_box is None:
        return []
    left, top, right, bottom, width, height = color_box
    return [
        FadeFinding(
            text="social-overlay-color",
            ink_score=1.0,
            source="overlay",
            x=left,
            y=top,
            width=max(1.0, right - left),
            height=max(1.0, bottom - top),
            page_width=width,
            page_height=height,
            is_transcript_word=False,
        )
    ]


def image_box_for_finding(
    finding: FadeFinding,
    image_width: int,
    image_height: int,
    padding: int = 18,
) -> tuple[int, int, int, int]:
    sx = image_width / max(finding.page_width, 1.0)
    sy = image_height / max(finding.page_height, 1.0)
    left = max(0, int(math.floor(finding.x * sx)) - padding)
    top = max(0, int(math.floor(finding.y * sy)) - padding)
    right = min(image_width, int(math.ceil((finding.x + finding.width) * sx)) + padding)
    bottom = min(image_height, int(math.ceil((finding.y + finding.height) * sy)) + padding)
    return left, top, right, bottom


def local_background_color(image: object, box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    left, top, right, bottom = box
    width, height = image.size
    ring = (
        max(0, left - 18),
        max(0, top - 18),
        min(width, right + 18),
        min(height, bottom + 18),
    )
    pixels: list[tuple[int, int, int]] = []
    region = image.crop(ring).convert("RGB")
    inner = (left - ring[0], top - ring[1], right - ring[0], bottom - ring[1])
    for y in range(region.height):
        for x in range(region.width):
            if inner[0] <= x <= inner[2] and inner[1] <= y <= inner[3]:
                continue
            red, green, blue = region.getpixel((x, y))
            if max(red, green, blue) - min(red, green, blue) < 42:
                pixels.append((red, green, blue))
    if not pixels:
        return (255, 255, 255)
    channels = []
    for channel in range(3):
        values = sorted(pixel[channel] for pixel in pixels)
        channels.append(values[len(values) // 2])
    if sum(channels) / 3 > 220:
        return (255, 255, 255)
    return tuple(channels)  # type: ignore[return-value]


def inpaint_or_fill_boxes(
    image: object,
    boxes: list[tuple[int, int, int, int]],
) -> object:
    if not boxes:
        return image
    if cv2 is not None and np is not None:
        array = np.array(image.convert("RGB"))
        mask = np.zeros(array.shape[:2], dtype=np.uint8)
        for left, top, right, bottom in boxes:
            cv2.rectangle(
                mask,
                (max(0, left), max(0, top)),
                (min(array.shape[1] - 1, right), min(array.shape[0] - 1, bottom)),
                255,
                -1,
            )
        inpainted = cv2.inpaint(array, mask, 3, cv2.INPAINT_TELEA)
        return Image.fromarray(inpainted)
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, fill=local_background_color(image, box))
    return image


def clean_temporal_patch_frame(
    candidate: FrameCandidate,
    args: argparse.Namespace,
    output_path: Path,
) -> object | None:
    for offset in (2.0, 3.0, 4.0, 5.0, -1.5, -2.5):
        timestamp = candidate.timestamp + offset
        if timestamp < args.story_start or timestamp > args.story_end:
            continue
        patch_path = output_path.with_name(
            f"{output_path.stem}-patch-{int(round(timestamp * 1000)):08d}ms.jpg"
        )
        try:
            extract_frame_at(args.video, timestamp, patch_path)
        except Exception:
            continue
        overlay_score = max(
            bottom_right_ad_overlay_score(patch_path),
            bottom_right_social_overlay_score(patch_path),
        )
        if overlay_score <= 0.0001:
            return Image.open(patch_path).convert("RGB")
    return None


def apply_temporal_overlay_patch(
    image: object,
    candidate: FrameCandidate,
    args: argparse.Namespace,
    output_path: Path,
    boxes: list[tuple[int, int, int, int]],
) -> tuple[object, list[tuple[int, int, int, int]]]:
    if not boxes:
        return image, boxes
    patch = clean_temporal_patch_frame(candidate, args, output_path)
    if patch is None:
        return image, boxes
    if patch.size != image.size:
        patch = patch.resize(image.size)
    if ImageFilter is not None:
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        for box in boxes:
            draw.rectangle(box, fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(10))
        image.paste(patch, (0, 0), mask)
        return image, []
    for box in boxes:
        image.paste(patch.crop(box), box)
    return image, []


def candidate_story_text_items(
    parser: object,
    candidate: FrameCandidate,
    frame_path: Path,
    args: argparse.Namespace,
) -> list[TextItem]:
    items, _ = extract_text_items(parser, frame_path, args.min_conf)
    story_items = filter_story_items(items)
    candidate_tokens = set(candidate.normalized_text.split())
    matched = []
    for item in story_items:
        words = set(clean_for_matching(item.text).split())
        if words and (words & candidate_tokens):
            matched.append(item)
    return matched or story_items


def union_item_box(
    items: list[TextItem],
    image_width: int,
    image_height: int,
    padding: int = 10,
) -> tuple[int, int, int, int] | None:
    if not items:
        return None
    left = min(item.x for item in items)
    top = min(item.y for item in items)
    right = max(item.right for item in items)
    bottom = max(item.bottom for item in items)
    page_width = max(items[0].page_width, 1.0)
    page_height = max(items[0].page_height, 1.0)
    sx = image_width / page_width
    sy = image_height / page_height
    return (
        max(0, int(math.floor(left * sx)) - padding),
        max(0, int(math.floor(top * sy)) - padding),
        min(image_width, int(math.ceil(right * sx)) + padding),
        min(image_height, int(math.ceil(bottom * sy)) + padding),
    )


def load_reconstruction_font(size: int) -> object:
    candidates = [
        "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        "/Library/Fonts/Georgia.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    ]
    for font_path in candidates:
        path = Path(font_path)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except Exception:
                continue
    return ImageFont.load_default()


def wrap_text_to_width(draw: object, text: str, font: object, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
            continue
        lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines or [text]


def draw_text_in_box(
    image: object,
    box: tuple[int, int, int, int],
    text: str,
) -> None:
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = box
    width = max(20, right - left)
    height = max(20, bottom - top)
    text = text.strip()
    if not text:
        return
    for size in range(min(44, max(16, height)), 11, -1):
        font = load_reconstruction_font(size)
        lines = wrap_text_to_width(draw, text, font, width)
        line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
        total_height = sum(box_[3] - box_[1] for box_ in line_boxes) + max(0, len(lines) - 1) * 4
        max_line_width = max((box_[2] - box_[0] for box_ in line_boxes), default=0)
        if total_height <= height and max_line_width <= width:
            y = top + max(0, (height - total_height) // 2)
            for line, line_box in zip(lines, line_boxes):
                line_width = line_box[2] - line_box[0]
                x = left + max(0, (width - line_width) // 2)
                draw.text((x, y), line, fill=(12, 12, 12), font=font)
                y += line_box[3] - line_box[1] + 4
            return
    font = load_reconstruction_font(12)
    draw.text((left, top), text, fill=(12, 12, 12), font=font)


def reconstruct_candidate_frame(
    candidate: FrameCandidate,
    parser: object,
    args: argparse.Namespace,
    output_path: Path,
    findings: list[FadeFinding] | None = None,
) -> FrameCandidate:
    source_path = Path(candidate.frame_path)
    findings = findings or fade_text_findings(source_path, candidate.normalized_text, args)
    image = Image.open(source_path).convert("RGB")
    width, height = image.size
    fade_boxes = [
        image_box_for_finding(finding, width, height)
        for finding in findings
        if finding.source != "overlay"
    ]
    overlay_boxes = [
        image_box_for_finding(finding, width, height)
        for finding in findings
        if finding.source == "overlay"
    ]

    redraw_text = any(finding.is_transcript_word for finding in findings)
    text_box = None
    if redraw_text:
        text_items = candidate_story_text_items(parser, candidate, source_path, args)
        text_box = union_item_box(text_items, width, height)
        if text_box is not None:
            fade_boxes.append(text_box)

    image, remaining_overlay_boxes = apply_temporal_overlay_patch(
        image,
        candidate,
        args,
        output_path,
        overlay_boxes,
    )
    boxes = fade_boxes + remaining_overlay_boxes
    image = inpaint_or_fill_boxes(image, boxes)
    if redraw_text and text_box is not None:
        draw_text_in_box(image, text_box, candidate.raw_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)

    reconstructed = FrameCandidate(**asdict(candidate))
    reconstructed.frame_path = str(output_path)
    reconstructed.output_source = "reconstructed"
    reconstructed.reconstructed_from = str(source_path)
    reconstructed.reconstruction_reason = ";".join(finding.warning for finding in findings)
    reconstructed.fade_warnings = ""
    if reconstructed.refinement_status:
        reconstructed.refinement_status += "; reconstructed"
    else:
        reconstructed.refinement_status = "reconstructed"
    return reconstructed


def refinement_match_tokens(text: str) -> set[str]:
    tokens = {
        token
        for token in clean_for_matching(text).split()
        if len(token) > 1 and token not in OCR_GLITCH_WORDS
    }
    return tokens or set(clean_for_matching(text).split())


def is_refinement_text_match(
    original: FrameCandidate,
    candidate: FrameCandidate,
    args: argparse.Namespace,
) -> bool:
    original_tokens = refinement_match_tokens(original.normalized_text)
    candidate_tokens = refinement_match_tokens(candidate.normalized_text)
    if not original_tokens or not candidate_tokens:
        return False
    if candidate.word_count > original.word_count + args.refine_max_added_words:
        return False
    overlap_ratio = len(original_tokens & candidate_tokens) / max(1, len(original_tokens))
    if overlap_ratio < 0.75:
        return False
    if original_tokens <= candidate_tokens or candidate_tokens <= original_tokens:
        return True
    return similarity(original.normalized_text, candidate.normalized_text) >= args.group_overlap


def local_refinement_score(
    original: FrameCandidate,
    candidate: FrameCandidate,
    warnings: list[str],
) -> float:
    word_gain = max(0, candidate.word_count - original.word_count)
    offset_penalty = abs(candidate.timestamp - original.timestamp) * 2.0
    warning_penalty = 120.0 if warnings else 0.0
    return (
        candidate.avg_confidence * 32.0
        + candidate.contrast_score * 36.0
        + candidate.word_count * 6.0
        + word_gain * 14.0
        - offset_penalty
        - warning_penalty
        - candidate.edge_crop_score * 32.0
        - candidate.ad_overlay_score * 18.0
    )


def refinement_time_bounds(candidate: FrameCandidate, args: argparse.Namespace) -> tuple[float, float]:
    lower = candidate.group_start_time - args.refine_extra_seconds
    upper = candidate.group_end_time + args.refine_extra_seconds
    if is_title_intro_time(candidate.timestamp, args):
        lower = max(lower, args.title_start)
        upper = min(upper, args.title_end)
    else:
        lower = max(lower, args.story_start)
        upper = min(upper, args.story_end)
    return lower, upper


def build_refinement_candidate(
    parser: object,
    original: FrameCandidate,
    timestamp: float,
    frame_path: Path,
    args: argparse.Namespace,
    visual_cache: dict[str, object],
) -> tuple[FrameCandidate, list[str], float] | None:
    candidate = build_candidate(
        parser,
        frame_path,
        timestamp,
        args,
        force_tesseract=True,
    )
    if candidate is None:
        return None
    if not is_refinement_text_match(original, candidate, args):
        return None
    visual_gap = visual_distance(original.frame_path, candidate.frame_path, visual_cache)
    if visual_gap > args.refine_visual_threshold:
        return None

    candidate.group_index = original.group_index
    candidate.group_start_time = min(original.group_start_time, timestamp)
    candidate.group_end_time = max(original.group_end_time, timestamp)
    candidate.group_frames_seen = original.group_frames_seen
    candidate.stable_neighbors = original.stable_neighbors
    candidate.refined_from_timestamp = original.timestamp
    warnings = fade_text_warnings(candidate.frame_path, candidate.normalized_text, args)
    score = local_refinement_score(original, candidate, warnings)
    candidate.score = score
    candidate.fade_warnings = ";".join(warnings)
    return candidate, warnings, score


def refine_selected_frames(
    selected: list[FrameCandidate],
    parser: object,
    args: argparse.Namespace,
) -> list[FrameCandidate]:
    if not args.refine_fade_frames:
        return selected

    offsets = parse_float_list(args.refine_offsets)
    refined_dir = args.work_dir / "refined-frames"
    recreate_dir(refined_dir)
    reconstructed_dir = args.work_dir / "reconstructed-frames"
    recreate_dir(reconstructed_dir)
    visual_cache: dict[str, object] = {}
    refined: list[FrameCandidate] = []
    shifted = 0
    dropped = 0
    reconstructed_count = 0

    for index, original in enumerate(selected, start=1):
        lower, upper = refinement_time_bounds(original, args)
        original_warnings = fade_text_warnings(
            original.frame_path,
            original.normalized_text,
            args,
        )
        best_candidate = original
        best_warnings = original_warnings
        best_score = local_refinement_score(original, original, original_warnings)
        original.score = max(original.score, best_score)
        original.fade_warnings = ";".join(original_warnings)

        seen_timestamps: set[int] = set()
        for offset in offsets:
            timestamp = round(original.timestamp + offset, 3)
            timestamp_key = int(round(timestamp * 1000))
            if timestamp_key in seen_timestamps:
                continue
            seen_timestamps.add(timestamp_key)
            if timestamp < lower or timestamp > upper:
                continue

            if abs(offset) < 0.0001:
                frame_path = Path(original.frame_path)
            else:
                frame_path = refined_dir / f"refine-{index:03d}-{timestamp_key:08d}ms.jpg"
                extract_frame_at(args.video, timestamp, frame_path)

            result = build_refinement_candidate(
                parser,
                original,
                timestamp,
                frame_path,
                args,
                visual_cache,
            )
            if result is None:
                continue
            candidate, warnings, score = result
            if warnings:
                continue
            improves_words = candidate.word_count > original.word_count
            fixes_fade = bool(original_warnings)
            if score > best_score + 2.0 or improves_words or fixes_fade:
                best_candidate = candidate
                best_warnings = warnings
                best_score = score

        best_overlay_findings = social_overlay_findings(best_candidate.frame_path, args)
        should_reconstruct = (
            (best_warnings or best_overlay_findings)
            and args.quality == "strict-complete"
            and args.reconstruct_fade_frames
        )
        if should_reconstruct:
            timestamp_key = int(round(best_candidate.timestamp * 1000))
            reconstruction_path = (
                reconstructed_dir / f"reconstruct-{index:03d}-{timestamp_key:08d}ms.jpg"
            )
            findings = []
            if best_warnings:
                findings.extend(
                    fade_text_findings(
                        best_candidate.frame_path,
                        best_candidate.normalized_text,
                        args,
                    )
                )
            findings.extend(best_overlay_findings)
            best_candidate = reconstruct_candidate_frame(
                best_candidate,
                parser,
                args,
                reconstruction_path,
                findings,
            )
            best_warnings = []
            reconstructed_count += 1
        elif best_warnings and args.quality == "balanced":
            best_candidate.fade_warnings = ";".join(best_warnings)
        elif best_warnings and not args.keep_unresolved_fade:
            dropped += 1
            print(
                "Dropped unresolved fade frame: "
                f"{original.timestamp:.2f}s {original.normalized_text} "
                f"warnings={';'.join(best_warnings)}"
            )
            continue

        if best_candidate.output_source == "reconstructed":
            if abs(best_candidate.timestamp - original.timestamp) > 0.001:
                shifted += 1
                best_candidate.refinement_status = (
                    f"shifted {original.timestamp:.2f}s -> {best_candidate.timestamp:.2f}s; "
                    "reconstructed"
                )
            elif not best_candidate.refinement_status:
                best_candidate.refinement_status = "reconstructed"
        elif abs(best_candidate.timestamp - original.timestamp) > 0.001:
            shifted += 1
            best_candidate.refinement_status = (
                f"shifted {original.timestamp:.2f}s -> {best_candidate.timestamp:.2f}s"
            )
        elif best_candidate is not original:
            best_candidate.refinement_status = "ocr-refined"
        else:
            best_candidate.refinement_status = "kept"
        best_candidate.fade_warnings = ";".join(best_warnings)
        refined.append(best_candidate)

    print(
        "Refinement "
        f"shifted={shifted} dropped={dropped} reconstructed={reconstructed_count}"
    )
    return sorted(refined, key=lambda item: item.timestamp)


def slugify(text: str, max_words: int = 8) -> str:
    tokens = clean_for_matching(text).split()[:max_words]
    slug = "-".join(tokens) or "story-text"
    slug = re.sub(r"[^a-z0-9-]+", "-", slug).strip("-")
    return slug[:80] or "story-text"


def final_transcript_text_from_tesseract(
    frame_path: str | Path,
    args: argparse.Namespace,
) -> str:
    items, _ = extract_tesseract_raw_text_items(Path(frame_path))
    parts: list[str] = []
    for item in items:
        if item.confidence < args.transcript_polish_min_conf:
            continue
        if is_lower_left_watermark(item):
            continue
        if is_bottom_overlay(item) and is_ad_overlay_item(item):
            continue
        cleaned = clean_for_matching(item.text)
        if not cleaned:
            continue
        if item.height > item.page_height * 0.10:
            continue
        if len(cleaned) <= 2 and (
            item.width > item.page_width * 0.07
            or item.height > item.page_height * 0.06
        ):
            continue
        if all(word in IGNORE_EXACT_WORDS for word in cleaned.split()):
            continue
        parts.append(item.text)
    return " ".join(parts).strip()


def polish_selected_transcripts(
    selected: list[FrameCandidate],
    args: argparse.Namespace,
) -> list[FrameCandidate]:
    if not args.polish_transcripts:
        return selected

    polished_count = 0
    for candidate in selected:
        polished_text = final_transcript_text_from_tesseract(candidate.frame_path, args)
        if not polished_text:
            continue
        if not should_use_polished_transcript(candidate.normalized_text, polished_text):
            continue
        normalized = clean_for_matching(polished_text)
        if not normalized or should_reject_frame(
            normalized,
            polished_text,
            candidate.timestamp,
            args,
        ):
            continue
        candidate.raw_text = polished_text
        candidate.normalized_text = normalized
        candidate.word_count = len(normalized.split())
        if candidate.refinement_status:
            candidate.refinement_status += "; text-polished"
        else:
            candidate.refinement_status = "text-polished"
        polished_count += 1

    if polished_count:
        print(f"Transcript polish updated={polished_count}")
    return selected


def copy_selected_frames(
    selected: list[FrameCandidate],
    groups: list[list[FrameCandidate]],
    output_dir: Path,
) -> list[GroupSummary]:
    frames_dir = output_dir / "frames"
    recreate_dir(frames_dir)
    summaries: list[GroupSummary] = []

    for output_index, candidate in enumerate(selected, start=1):
        group = groups[candidate.group_index]
        group_start = candidate.group_start_time or group[0].timestamp
        group_end = candidate.group_end_time or group[-1].timestamp
        group_frames_seen = candidate.group_frames_seen or len(group)
        timestamp_slug = f"{int(round(candidate.timestamp)):06d}s"
        filename = (
            f"{output_index:03d}-{timestamp_slug}-{slugify(candidate.normalized_text)}.jpg"
        )
        dest = frames_dir / filename
        shutil.copy2(candidate.frame_path, dest)
        summaries.append(
            GroupSummary(
                group_index=candidate.group_index,
                start_time=group_start,
                end_time=group_end,
                frames_seen=group_frames_seen,
                selected_timestamp=candidate.timestamp,
                selected_image=str(dest),
                transcript=candidate.raw_text,
                normalized_text=candidate.normalized_text,
                avg_confidence=candidate.avg_confidence,
                edge_crop_score=candidate.edge_crop_score,
                refinement_status=candidate.refinement_status,
                fade_warnings=candidate.fade_warnings,
                output_source=candidate.output_source,
                reconstruction_reason=candidate.reconstruction_reason,
                score=candidate.score,
            )
        )
    return summaries


def write_indexes(
    output_dir: Path,
    summaries: list[GroupSummary],
    candidates: list[FrameCandidate],
) -> None:
    csv_path = output_dir / "review-index.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "timestamp",
                "group_start",
                "group_end",
                "frames_seen",
                "image",
                "transcript",
                "normalized_text",
                "avg_confidence",
                "edge_crop_score",
                "refinement_status",
                "fade_warnings",
                "output_source",
                "reconstruction_reason",
                "score",
            ],
        )
        writer.writeheader()
        for index, summary in enumerate(summaries, start=1):
            writer.writerow(
                {
                    "index": index,
                    "timestamp": f"{summary.selected_timestamp:.3f}",
                    "group_start": f"{summary.start_time:.3f}",
                    "group_end": f"{summary.end_time:.3f}",
                    "frames_seen": summary.frames_seen,
                    "image": summary.selected_image,
                    "transcript": summary.transcript,
                    "normalized_text": summary.normalized_text,
                    "avg_confidence": f"{summary.avg_confidence:.3f}",
                    "edge_crop_score": f"{summary.edge_crop_score:.3f}",
                    "refinement_status": summary.refinement_status,
                    "fade_warnings": summary.fade_warnings,
                    "output_source": summary.output_source,
                    "reconstruction_reason": summary.reconstruction_reason,
                    "score": f"{summary.score:.3f}",
                }
            )

    debug_path = output_dir / "debug-candidates.json"
    debug_payload = {
        "selected": [asdict(summary) for summary in summaries],
        "candidate_count": len(candidates),
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    debug_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")

    markdown_path = output_dir / "review-index.md"
    lines = ["# Story Frame Review", ""]
    lines.append("| # | Time | Image | Transcript |")
    lines.append("|---:|---:|---|---|")
    for index, summary in enumerate(summaries, start=1):
        image_path = Path(summary.selected_image)
        rel_image = image_path.relative_to(output_dir)
        transcript = summary.transcript.replace("|", "\\|")
        source_note = (
            f" ({summary.output_source})"
            if summary.output_source != "original"
            else ""
        )
        lines.append(
            f"| {index} | {summary.selected_timestamp:.2f}s | "
            f"![]({rel_image.as_posix()}) | {transcript}{source_note} |"
        )
    lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def fit_text(draw: object, text: str, max_width: int, font: object) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
        if len(lines) >= 2:
            break
    if current and len(lines) < 3:
        lines.append(current)
    return lines[:3]


def create_contact_sheet(output_dir: Path, summaries: list[GroupSummary]) -> None:
    if not summaries:
        return
    thumb_width = 320
    thumb_height = 180
    label_height = 58
    padding = 12
    columns = 3
    rows = math.ceil(len(summaries) / columns)
    sheet_width = columns * thumb_width + (columns + 1) * padding
    sheet_height = rows * (thumb_height + label_height) + (rows + 1) * padding
    sheet = Image.new("RGB", (sheet_width, sheet_height), (244, 244, 244))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, summary in enumerate(summaries):
        row, column = divmod(index, columns)
        x = padding + column * (thumb_width + padding)
        y = padding + row * (thumb_height + label_height + padding)
        image = Image.open(summary.selected_image).convert("RGB")
        image.thumbnail((thumb_width, thumb_height))
        tile = Image.new("RGB", (thumb_width, thumb_height), (255, 255, 255))
        tile.paste(image, ((thumb_width - image.width) // 2, (thumb_height - image.height) // 2))
        sheet.paste(tile, (x, y))
        label_y = y + thumb_height + 5
        source_marker = " R" if summary.output_source == "reconstructed" else ""
        title = f"{index + 1:03d}  {summary.selected_timestamp:.2f}s{source_marker}"
        draw.text((x, label_y), title, fill=(20, 20, 20), font=font)
        for line_offset, line in enumerate(
            fit_text(draw, summary.normalized_text, thumb_width, font), start=1
        ):
            draw.text(
                (x, label_y + line_offset * 13),
                line,
                fill=(55, 55, 55),
                font=font,
            )

    sheet.save(output_dir / "review-contact-sheet.jpg", quality=92)


def main() -> None:
    args = parse_args()
    ensure_dependencies(args)

    if not args.video.exists():
        fail(f"Input video does not exist: {args.video}")
    if args.story_end is not None and args.story_end <= args.story_start:
        fail("--story-end must be greater than --story-start")
    if args.include_title_intro and args.title_end <= args.title_start:
        fail("--title-end must be greater than --title-start")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    args.effective_fps = effective_scan_fps(args)
    if args.effective_fps <= 0:
        fail("--fps/scan FPS must be greater than 0")

    sample_start = min(args.title_start, args.story_start) if args.include_title_intro else args.story_start
    print(
        "Sampling frames: "
        f"{sample_start:.2f}s -> {args.story_end:.2f}s @ "
        f"{args.effective_fps:.3f} fps ({args.scan_mode})"
    )
    sampled_frames = sample_frames(args)
    print(f"Sampled {len(sampled_frames)} frames")

    parser = LiteParse(output_format="json", ocr_enabled=True, quiet=True, num_workers=2)
    candidates: list[FrameCandidate] = []
    rejected = 0
    for index, (frame_path, timestamp) in enumerate(sampled_frames, start=1):
        candidate = build_candidate(parser, frame_path, timestamp, args)
        if candidate is None:
            rejected += 1
        else:
            candidates.append(candidate)
        if index % 50 == 0 or index == len(sampled_frames):
            print(
                f"OCR {index}/{len(sampled_frames)} frames "
                f"-> candidates={len(candidates)} rejected={rejected}"
            )

    if not candidates:
        fail("No story transcript candidates found")

    groups = group_candidates(candidates, args)
    representatives = []
    for group in groups:
        representative = choose_group_representative(group, args)
        if representative is not None:
            representatives.append(representative)
    selected = remove_global_duplicates(representatives, args)
    selected = prune_transition_duplicates(selected, args)
    selected = trim_non_story_edges(selected, args)
    selected = refine_selected_frames(selected, parser, args)
    selected = prune_transition_duplicates(selected, args)
    selected = trim_non_story_edges(selected, args)
    selected = polish_selected_transcripts(selected, args)
    summaries = copy_selected_frames(selected, groups, args.output_dir)
    write_indexes(args.output_dir, summaries, candidates)
    create_contact_sheet(args.output_dir, summaries)

    print(f"Groups: {len(groups)}")
    print(f"Selected review frames: {len(summaries)}")
    print(f"Output: {args.output_dir}")
    print(f"Contact sheet: {args.output_dir / 'review-contact-sheet.jpg'}")
    print(f"Index CSV: {args.output_dir / 'review-index.csv'}")


if __name__ == "__main__":
    main()
