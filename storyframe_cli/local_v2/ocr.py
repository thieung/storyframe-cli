from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

from PIL import Image

from .bootstrap import add_local_v2_dependency_paths
from .models import OcrBox
from .ocr_filter import is_non_story_noise_box, is_watermark_contaminated_box
from .text import clean_text

add_local_v2_dependency_paths()


def box_ink_score(frame_path: Path, box: OcrBox) -> float:
    image = Image.open(frame_path).convert("L")
    left = max(0, int(math.floor(box.x - 2)))
    top = max(0, int(math.floor(box.y - 2)))
    right = min(image.width, int(math.ceil(box.right + 2)))
    bottom = min(image.height, int(math.ceil(box.bottom + 2)))
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


def is_watermark(box: OcrBox) -> bool:
    return is_watermark_contaminated_box(box)


def meaningful_boxes(boxes: Iterable[OcrBox]) -> list[OcrBox]:
    boxes = list(boxes)
    reference_height = sorted(box.height for box in boxes)[len(boxes) // 2] if boxes else 0.0
    kept: list[OcrBox] = []
    for box in boxes:
        cleaned = clean_text(box.text)
        if not cleaned:
            continue
        if box.confidence < 0.35:
            continue
        if is_non_story_noise_box(box, reference_height):
            continue
        kept.append(box)
    return kept


class RapidOcrBackend:
    def __init__(self) -> None:
        try:
            from rapidocr import RapidOCR
        except Exception as exc:  # pragma: no cover - runtime dependency message.
            raise RuntimeError(
                "rapidocr is missing. Install local deps into work/.deps/storyframe-local-v2."
            ) from exc
        self._engine = RapidOCR()

    def recognize(self, frame_path: Path) -> list[OcrBox]:
        image = Image.open(frame_path)
        page_width, page_height = image.size
        result = self._engine(str(frame_path))
        boxes = []
        raw_boxes = [] if result.boxes is None else result.boxes
        raw_texts = [] if result.txts is None else result.txts
        raw_scores = [] if result.scores is None else result.scores
        for points, text, confidence in zip(raw_boxes, raw_texts, raw_scores):
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            left = min(xs)
            top = min(ys)
            right = max(xs)
            bottom = max(ys)
            box = OcrBox(
                text=str(text),
                confidence=float(confidence),
                x=left,
                y=top,
                width=max(1.0, right - left),
                height=max(1.0, bottom - top),
                page_width=float(page_width),
                page_height=float(page_height),
            )
            box.ink_score = box_ink_score(frame_path, box)
            boxes.append(box)
        kept = meaningful_boxes(boxes)
        if len(clean_text(" ".join(box.text for box in kept)).split()) >= 2:
            return kept
        if not clean_text(" ".join(str(text) for text in raw_texts)).split():
            return kept
        return merge_ocr_boxes(kept, tesseract_fallback_boxes(frame_path, page_width, page_height))


def tesseract_fallback_boxes(frame_path: Path, page_width: int, page_height: int) -> list[OcrBox]:
    try:
        import pytesseract
    except Exception:
        return []

    image = Image.open(frame_path).convert("RGB")
    data = pytesseract.image_to_data(
        image,
        lang="eng",
        config="--psm 11",
        output_type=pytesseract.Output.DICT,
    )
    boxes: list[OcrBox] = []
    for index, text in enumerate(data.get("text", [])):
        cleaned = clean_text(str(text))
        if not cleaned:
            continue
        try:
            confidence = float(data["conf"][index]) / 100.0
        except (TypeError, ValueError):
            continue
        if confidence < 0.60:
            continue
        box = OcrBox(
            text=str(text),
            confidence=confidence,
            x=float(data["left"][index]),
            y=float(data["top"][index]),
            width=max(1.0, float(data["width"][index])),
            height=max(1.0, float(data["height"][index])),
            page_width=float(page_width),
            page_height=float(page_height),
        )
        box.ink_score = box_ink_score(frame_path, box)
        boxes.append(box)
    return meaningful_boxes(boxes)


def merge_ocr_boxes(primary: list[OcrBox], fallback: list[OcrBox]) -> list[OcrBox]:
    merged = list(primary)
    for box in fallback:
        if any(box_overlap_ratio(box, existing) > 0.45 for existing in merged):
            continue
        merged.append(box)
    return sorted(merged, key=lambda item: (item.y, item.x))


def box_overlap_ratio(left: OcrBox, right: OcrBox) -> float:
    x_overlap = max(0.0, min(left.right, right.right) - max(left.x, right.x))
    y_overlap = max(0.0, min(left.bottom, right.bottom) - max(left.y, right.y))
    intersection = x_overlap * y_overlap
    if intersection <= 0:
        return 0.0
    smaller_area = min(left.width * left.height, right.width * right.height)
    return intersection / max(1.0, smaller_area)


def build_ocr_backend(name: str) -> RapidOcrBackend:
    if name != "rapidocr":
        raise RuntimeError(f"Unsupported local-v2 OCR backend for MVP: {name}")
    return RapidOcrBackend()
