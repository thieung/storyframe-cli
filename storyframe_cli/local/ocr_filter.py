from __future__ import annotations

import re
from statistics import median

from .models import OcrBox
from .text import clean_text, has_reject_phrase


def split_story_and_ad_boxes(boxes: list[OcrBox]) -> tuple[list[OcrBox], list[OcrBox]]:
    story_boxes: list[OcrBox] = []
    ad_boxes: list[OcrBox] = []
    reference_height = median([box.height for box in boxes]) if boxes else 0.0
    for box in boxes:
        if is_social_overlay_box(box):
            ad_boxes.append(box)
            continue
        if is_non_story_noise_box(box, reference_height):
            continue
        story_boxes.append(box)
    return story_boxes, ad_boxes


def is_non_story_noise_box(box: OcrBox, reference_height: float = 0.0) -> bool:
    cleaned = clean_text(box.text)
    if not cleaned:
        return True
    if is_watermark_contaminated_box(box):
        return True
    if is_edge_crop_fragment_box(box):
        return True
    if is_corner_short_artifact_box(box):
        return True
    if is_low_confidence_short_box(box):
        return True
    if is_tall_short_artifact_box(box, reference_height):
        return True
    return False


def is_social_overlay_box(box: OcrBox) -> bool:
    cleaned = clean_text(box.text)
    if not cleaned or not has_reject_phrase(cleaned):
        return False
    in_overlay_zone = box.x > box.page_width * 0.52 and box.y > box.page_height * 0.50
    return in_overlay_zone


def is_watermark_contaminated_box(box: OcrBox) -> bool:
    if not raw_text_contains_watermark(box.text):
        return False
    in_watermark_zone = box.x < box.page_width * 0.30 and box.y > box.page_height * 0.62
    return in_watermark_zone


def raw_text_contains_watermark(text: str) -> bool:
    compact = re.sub(r"[^a-z0-9]+", "", text.lower())
    return any(marker in compact for marker in ("vooks", "vo0ks", "wooks", "gvooks"))


def is_edge_crop_fragment_box(box: OcrBox) -> bool:
    tokens = clean_text(box.text).split()
    if not tokens or len(tokens) > 2:
        return False
    touches_left_or_right = box.x <= 2 or box.right >= box.page_width - 2
    touches_top_or_bottom = box.y <= 2 or box.bottom >= box.page_height - 2
    is_small_fragment = box.width <= box.page_width * 0.12
    return is_small_fragment and (touches_left_or_right or touches_top_or_bottom)


def is_corner_short_artifact_box(box: OcrBox) -> bool:
    tokens = clean_text(box.text).split()
    if not tokens or len(tokens) > 1:
        return False
    in_corner_zone = (
        (box.x < box.page_width * 0.18 or box.x > box.page_width * 0.82)
        and box.y > box.page_height * 0.82
    )
    is_small = box.width < box.page_width * 0.10 and box.height < box.page_height * 0.06
    return in_corner_zone and is_small and box.confidence < 0.90


def is_low_confidence_short_box(box: OcrBox) -> bool:
    tokens = clean_text(box.text).split()
    return bool(tokens) and len(tokens) <= 2 and box.confidence < 0.72


def is_tall_short_artifact_box(box: OcrBox, reference_height: float) -> bool:
    tokens = clean_text(box.text).split()
    if not tokens or len(tokens) > 2 or reference_height <= 0:
        return False
    too_tall_for_text = box.height > max(reference_height * 2.4, 90.0)
    return too_tall_for_text and box.confidence < 0.90
