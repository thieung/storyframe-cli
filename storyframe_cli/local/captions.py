from __future__ import annotations

import math
import re
import shutil
from math import ceil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .media import extract_frame
from .models import FrameObservation, SelectedFrame, TranscriptUnit
from .ocr import build_ocr_backend
from .ocr_filter import split_story_and_ad_boxes
from .text import clean_text, has_reject_phrase, similarity, target_coverage


SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*$")


def merge_transcript_units_into_sentences(
    units: list[TranscriptUnit],
    max_gap: float = 1.25,
    max_words: int = 32,
    max_duration: float = 12.0,
) -> list[TranscriptUnit]:
    merged: list[TranscriptUnit] = []
    current: list[TranscriptUnit] = []
    for unit in sorted(units, key=lambda item: (item.start, item.end, item.unit_id)):
        if has_reject_phrase(unit.normalized_text):
            continue
        if not current:
            current = [unit]
            continue
        gap = unit.start - current[-1].end
        word_count = sum(len(item.normalized_text.split()) for item in current) + len(unit.normalized_text.split())
        duration = unit.end - current[0].start
        if sentence_is_complete(current[-1].text) or gap > max_gap or word_count > max_words or duration > max_duration:
            merged.append(make_sentence_unit(current, len(merged) + 1))
            current = [unit]
        else:
            current.append(unit)
    if current:
        merged.append(make_sentence_unit(current, len(merged) + 1))
    return merged


def sentence_is_complete(text: str) -> bool:
    return bool(SENTENCE_END_RE.search(text.strip()))


def make_sentence_unit(parts: list[TranscriptUnit], index: int) -> TranscriptUnit:
    text = normalize_sentence_text(" ".join(part.text.strip() for part in parts if part.text.strip()))
    source = parts[0].source if parts else "asr"
    return TranscriptUnit(
        unit_id=f"caption-{index:04d}",
        text=text,
        normalized_text=clean_text(text),
        start=parts[0].start,
        end=parts[-1].end,
        source=f"{source}-caption",
    )


def normalize_sentence_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def select_caption_frames(
    video_path: Path,
    work_dir: Path,
    units: list[TranscriptUnit],
) -> list[SelectedFrame]:
    frame_dir = work_dir / "caption-frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    selected: list[SelectedFrame] = []
    for unit in units:
        words = unit.normalized_text.split()
        if len(words) < 2:
            continue
        timestamp = max(unit.start, (unit.start + unit.end) / 2.0)
        frame_path = frame_dir / f"frame-{int(round(timestamp * 1000)):09d}ms.jpg"
        extract_frame(video_path, timestamp, frame_path)
        selected.append(
            SelectedFrame(
                unit_id=unit.unit_id,
                timestamp=timestamp,
                frame_path=str(frame_path),
                transcript=unit.text,
                normalized_text=unit.normalized_text,
                score=100.0 + min(len(words), 30),
                status="clean",
                warnings=[],
                output_source="caption-rendered",
            )
        )
    return selected


def should_use_caption_fallback(
    units: list[TranscriptUnit],
    observations: list[FrameObservation],
    min_support_ratio: float = 0.25,
) -> bool:
    if not units:
        return False
    if not observations:
        return True

    supported = sum(1 for unit in units if transcript_has_ocr_support(unit, observations))
    if supported == 0:
        return True
    if len(units) >= 4 and supported / len(units) < min_support_ratio:
        return True
    return False


def should_skip_full_ocr_for_caption_fallback(
    video_path: Path,
    work_dir: Path,
    ocr_backend_name: str,
    units: list[TranscriptUnit],
    max_samples: int = 12,
    min_samples: int = 4,
    min_support_ratio: float = 0.25,
) -> bool:
    samples = sample_transcript_units(units, max_samples)
    if len(samples) < min_samples:
        return False

    sample_dir = work_dir / "caption-ocr-samples"
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    backend = build_ocr_backend(ocr_backend_name)
    supported = 0
    for index, unit in enumerate(samples, start=1):
        timestamp = max(unit.start, (unit.start + unit.end) / 2.0)
        frame_path = sample_dir / f"sample-{index:03d}-{int(round(timestamp * 1000)):09d}ms.jpg"
        extract_frame(video_path, timestamp, frame_path)
        boxes = backend.recognize(frame_path)
        observation = observation_from_ocr_sample(frame_path, timestamp, boxes)
        if observation is not None and transcript_has_ocr_support(unit, [observation]):
            supported += 1
    required_support = max(2, ceil(len(samples) * min_support_ratio))
    return supported < required_support


def sample_transcript_units(
    units: list[TranscriptUnit],
    max_samples: int,
) -> list[TranscriptUnit]:
    candidates = [unit for unit in units if len(unit.normalized_text.split()) >= 3]
    if len(candidates) <= max_samples:
        return candidates
    if max_samples <= 1:
        return [candidates[len(candidates) // 2]]
    indexes = {
        round(index * (len(candidates) - 1) / (max_samples - 1))
        for index in range(max_samples)
    }
    return [candidates[index] for index in sorted(indexes)]


def observation_from_ocr_sample(
    frame_path: Path,
    timestamp: float,
    boxes: list,
) -> FrameObservation | None:
    story_boxes, _ad_boxes = split_story_and_ad_boxes(boxes)
    text = " ".join(box.text for box in story_boxes)
    normalized = clean_text(text)
    words = normalized.split()
    if len(words) < 2:
        return None
    avg_conf = sum(box.confidence for box in story_boxes) / max(1, len(story_boxes))
    avg_ink = sum(box.ink_score for box in story_boxes) / max(1, len(story_boxes))
    return FrameObservation(
        frame_path=str(frame_path),
        timestamp=timestamp,
        text=text,
        normalized_text=normalized,
        boxes=story_boxes,
        avg_confidence=avg_conf,
        avg_ink_score=avg_ink,
        word_count=len(words),
    )


def transcript_has_ocr_support(
    unit: TranscriptUnit,
    observations: list[FrameObservation],
    padding: float = 2.0,
) -> bool:
    unit_text = unit.normalized_text
    if len(unit_text.split()) < 2:
        return False
    nearby = [
        observation
        for observation in observations
        if unit.start - padding <= observation.timestamp <= unit.end + padding
    ]
    candidates = nearby or observations
    for observation in candidates:
        coverage = target_coverage(unit_text, observation.normalized_text)
        reverse_coverage = target_coverage(observation.normalized_text, unit_text)
        related = similarity(unit_text, observation.normalized_text)
        if coverage >= 0.55 and reverse_coverage >= 0.55:
            return True
        if related >= 0.78 and max(coverage, reverse_coverage) >= 0.45:
            return True
    return False


def render_caption_if_needed(image_path: Path, item: SelectedFrame) -> bool:
    if item.output_source != "caption-rendered":
        return False
    image = Image.open(image_path).convert("RGB")
    caption = normalize_sentence_text(item.transcript)
    draw = ImageDraw.Draw(image)
    font = load_caption_font(image.size, caption)
    lines = wrap_caption_lines(caption, font, int(image.width * 0.84), draw)
    if not lines:
        return False

    padding_x = max(24, int(image.width * 0.035))
    padding_y = max(14, int(image.height * 0.024))
    line_gap = max(6, int(image.height * 0.010))
    line_heights = [text_bbox_size(draw, line, font)[1] for line in lines]
    text_width = max(text_bbox_size(draw, line, font)[0] for line in lines)
    text_height = sum(line_heights) + line_gap * (len(lines) - 1)
    box_width = min(image.width - padding_x * 2, text_width + padding_x * 2)
    box_height = text_height + padding_y * 2
    left = (image.width - box_width) // 2
    top = image.height - box_height - max(18, int(image.height * 0.030))
    right = left + box_width
    bottom = top + box_height

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=max(12, int(image.height * 0.018)),
        fill=(255, 255, 255, 230),
        outline=(30, 30, 30, 70),
        width=2,
    )
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)

    y = top + padding_y
    for line, line_height in zip(lines, line_heights):
        width, _height = text_bbox_size(draw, line, font)
        x = left + (box_width - width) // 2
        draw.text((x, y), line, fill=(18, 18, 18), font=font)
        y += line_height + line_gap
    image.save(image_path, quality=92)
    item.transcript = caption
    item.normalized_text = clean_text(caption)
    return True


def load_caption_font(image_size: tuple[int, int], caption: str) -> ImageFont.ImageFont:
    _width, height = image_size
    words = len(clean_text(caption).split())
    size = int(max(26, min(54, height * (0.056 if words <= 10 else 0.045))))
    for font_path in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def wrap_caption_lines(
    caption: str,
    font: ImageFont.ImageFont,
    max_width: int,
    draw: ImageDraw.ImageDraw,
) -> list[str]:
    words = caption.split()
    if not words:
        return []
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if text_bbox_size(draw, candidate, font)[0] <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    if len(lines) <= 3:
        return lines
    return rebalance_caption_lines(lines, font, max_width, draw)


def rebalance_caption_lines(
    lines: list[str],
    font: ImageFont.ImageFont,
    max_width: int,
    draw: ImageDraw.ImageDraw,
) -> list[str]:
    words = " ".join(lines).split()
    target_lines = 3
    per_line = max(1, math.ceil(len(words) / target_lines))
    balanced = [" ".join(words[index : index + per_line]) for index in range(0, len(words), per_line)]
    if all(text_bbox_size(draw, line, font)[0] <= max_width for line in balanced):
        return balanced
    return lines[:2] + [" ".join(" ".join(lines[2:]).split())]


def text_bbox_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top
