from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .media import extract_frame, scan_fps, timestamps_for_window
from .models import FrameObservation, OcrBox, PageInterval, SelectedFrame, TranscriptUnit
from .ocr import build_ocr_backend
from .ocr_filter import split_story_and_ad_boxes
from .text import clean_text, corrected_text_with_reference, extra_token_ratio, has_reject_phrase, similarity, target_coverage, token_set


def observation_from_boxes(frame_path: Path, timestamp: float, boxes: list[OcrBox]) -> FrameObservation | None:
    story_boxes, ad_boxes = split_story_and_ad_boxes(boxes)
    text = " ".join(box.text for box in story_boxes)
    normalized = clean_text(text)
    words = normalized.split()
    if len(words) < 2:
        return None
    if has_reject_phrase(normalized):
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
        ad_boxes=ad_boxes,
    )


def make_windows(
    units: list[TranscriptUnit],
    story_start: float,
    story_end: float,
    padding: float,
) -> list[tuple[float, float]]:
    if not units:
        return [(story_start, story_end)]
    windows = []
    for unit in units:
        windows.append((max(story_start, unit.start - padding), min(story_end, unit.end + padding)))
    merged: list[tuple[float, float]] = []
    for start, end in sorted(windows):
        if not merged or start > merged[-1][1] + 0.25:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def collect_observations(
    video_path: Path,
    work_dir: Path,
    ocr_backend_name: str,
    windows: list[tuple[float, float]],
    scan_mode: str,
    fps: float,
    dense_fps: float,
) -> list[FrameObservation]:
    backend = build_ocr_backend(ocr_backend_name)
    frame_dir = work_dir / "local-frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    effective_fps = scan_fps(scan_mode, fps, dense_fps, video_path)

    observations: list[FrameObservation] = []
    seen_timestamps: set[int] = set()
    total = sum(len(timestamps_for_window(start, end, effective_fps)) for start, end in windows)
    done = 0
    for start, end in windows:
        for timestamp in timestamps_for_window(start, end, effective_fps):
            key = int(round(timestamp * 1000))
            if key in seen_timestamps:
                continue
            seen_timestamps.add(key)
            frame_path = frame_dir / f"frame-{key:09d}ms.jpg"
            extract_frame(video_path, timestamp, frame_path)
            boxes = backend.recognize(frame_path)
            observation = observation_from_boxes(frame_path, timestamp, boxes)
            if observation is not None:
                observations.append(observation)
            done += 1
            if done % 25 == 0 or done == total:
                print(f"local OCR {done}/{total} observations={len(observations)}")
    return sorted(observations, key=lambda item: item.timestamp)


def derive_units_from_observations(observations: list[FrameObservation]) -> list[TranscriptUnit]:
    groups: list[list[FrameObservation]] = []
    for observation in observations:
        if not groups:
            groups.append([observation])
            continue
        previous = groups[-1][-1]
        gap = observation.timestamp - previous.timestamp
        related = strict_state_similarity(observation.normalized_text, previous.normalized_text)
        same_page = (
            not observation.page_id
            or not previous.page_id
            or observation.page_id == previous.page_id
        )
        if same_page and gap <= 3.0 and related >= 0.70:
            groups[-1].append(observation)
        else:
            groups.append([observation])

    groups = drop_transition_blend_groups(groups)
    units: list[TranscriptUnit] = []
    for index, group in enumerate(groups, start=1):
        best = max(
            group,
            key=lambda item: (
                item.word_count,
                item.avg_ink_score,
                item.avg_confidence,
                item.timestamp,
            ),
        )
        units.append(
            TranscriptUnit(
                unit_id=f"ocr-{index:04d}",
                text=best.text,
                normalized_text=best.normalized_text,
                start=group[0].timestamp,
                end=group[-1].timestamp,
                source="ocr-temporal",
            )
        )
    return units


def refine_asr_units_with_ocr(
    units: list[TranscriptUnit],
    observations: list[FrameObservation],
    padding: float = 0.50,
) -> list[TranscriptUnit]:
    refined: list[TranscriptUnit] = []
    for unit in units:
        window_observations = [
            observation
            for observation in observations
            if unit.start - padding <= observation.timestamp <= unit.end + padding
        ]
        ocr_units = derive_units_from_observations(window_observations)
        if not ocr_units:
            refined.append(unit)
            continue
        best = max(
            ocr_units,
            key=lambda candidate: (
                overlap_duration(unit.start, unit.end, candidate.start, candidate.end),
                len(candidate.normalized_text.split()),
            ),
        )
        refined_text = corrected_text_with_reference(
            best.text,
            unit.text,
            allow_insertions=has_bottom_left_occluded_suffix(window_observations),
            keep_unmatched=False,
        )
        refined_normalized = clean_text(refined_text)
        if not refined_normalized:
            refined_text = best.text
            refined_normalized = best.normalized_text
        refined.append(
            TranscriptUnit(
                unit_id=unit.unit_id,
                text=refined_text,
                normalized_text=refined_normalized,
                start=unit.start,
                end=unit.end,
                source=f"{unit.source}+ocr-text",
            )
        )
    return refined


def overlap_duration(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def assign_observation_pages(
    observations: list[FrameObservation],
    pages: list[PageInterval],
) -> list[FrameObservation]:
    for observation in observations:
        observation.page_id = ""
        for page in pages:
            if page.start <= observation.timestamp <= page.end:
                observation.page_id = page.page_id
                break
    return observations


def merge_units_with_ocr_missing(
    units: list[TranscriptUnit],
    ocr_units: list[TranscriptUnit],
) -> list[TranscriptUnit]:
    merged = list(units)
    for ocr_unit in ocr_units:
        duration = ocr_unit.end - ocr_unit.start
        word_count = len(ocr_unit.normalized_text.split())
        if duration < 0.75 or word_count < 3 or has_reject_phrase(ocr_unit.normalized_text):
            continue
        is_covered = False
        for unit in units:
            temporal_overlap = overlap_duration(
                unit.start,
                unit.end,
                ocr_unit.start,
                ocr_unit.end,
            )
            temporal_gap = max(
                0.0,
                max(unit.start, ocr_unit.start) - min(unit.end, ocr_unit.end),
            )
            related = strict_state_similarity(unit.normalized_text, ocr_unit.normalized_text)
            if temporal_overlap >= 0.35 and related >= 0.62:
                is_covered = True
                break
            if related >= 0.80 and temporal_gap <= 2.0:
                is_covered = True
                break
        if not is_covered:
            merged.append(
                TranscriptUnit(
                    unit_id=f"ocr-missing-{len(merged) + 1:04d}",
                    text=ocr_unit.text,
                    normalized_text=ocr_unit.normalized_text,
                    start=ocr_unit.start,
                    end=ocr_unit.end,
                    source=f"{ocr_unit.source}+missing",
                )
            )
    return sorted(merged, key=lambda item: (item.start, item.end, item.unit_id))


def filter_units_for_story(
    units: list[TranscriptUnit],
    story_start: float,
    story_end: float,
) -> list[TranscriptUnit]:
    sorted_units = sorted(units, key=lambda item: (item.start, item.end, item.unit_id))
    if not sorted_units:
        return []

    duration = max(0.1, story_end - story_start)
    trim_index = len(sorted_units)
    explicit_end_index = first_explicit_end_index(sorted_units, story_start, duration)
    if explicit_end_index is not None:
        trim_index = explicit_end_index + 1
    else:
        terminal_reject_index = None
        for index, unit in enumerate(sorted_units):
            if unit.start < story_start + duration * 0.60:
                continue
            if has_reject_phrase(unit.normalized_text):
                terminal_reject_index = index
                break

        if terminal_reject_index is not None:
            trim_index = terminal_reject_index
            terminal_start = sorted_units[terminal_reject_index].start
            for gap_index in range(0, terminal_reject_index):
                previous = sorted_units[gap_index]
                current = sorted_units[gap_index + 1]
                gap = current.start - previous.end
                in_terminal_search_band = (
                    current.start >= story_start + duration * 0.55
                    and current.start >= terminal_start - 60.0
                )
                if gap >= 6.0 and in_terminal_search_band:
                    keep_tail_title = is_keepable_tail_title_repeat(current) and is_title_repeat_unit(
                        current,
                        sorted_units,
                        story_start,
                        duration,
                    )
                    in_terminal_lead_in = current.start >= terminal_start - 20.0
                    if has_reject_phrase(current.normalized_text):
                        trim_index = gap_index + 1
                        break
                    if keep_tail_title:
                        trim_index = gap_index + 2
                        break
                    if in_terminal_lead_in:
                        trim_index = gap_index + 1
                        break
        else:
            trim_index = end_screen_trim_index_from_title_repeat(
                sorted_units,
                story_start,
                duration,
            )

    tail_units = sorted_units[trim_index:]
    early_title_cutoff = story_start + duration * 0.20
    kept: list[TranscriptUnit] = []
    for unit in sorted_units[:trim_index]:
        if has_reject_phrase(unit.normalized_text):
            continue
        if is_short_leading_intro_unit(unit, story_start, duration):
            continue
        if should_drop_low_support_ocr_unit(unit):
            continue
        if is_early_title_card_repeat(unit, tail_units, early_title_cutoff):
            continue
        kept.append(unit)
    return kept


def first_explicit_end_index(
    units: list[TranscriptUnit],
    story_start: float,
    duration: float,
) -> int | None:
    cutoff = story_start + duration * 0.45
    for index, unit in enumerate(units):
        if unit.start < cutoff:
            continue
        if clean_text(unit.normalized_text) == "the end":
            return index
    return None


def end_screen_trim_index_from_title_repeat(
    units: list[TranscriptUnit],
    story_start: float,
    duration: float,
) -> int:
    repeat_index = first_tail_title_repeat_index(units, story_start, duration)
    if repeat_index is None:
        return len(units)

    repeat_start = units[repeat_index].start
    keep_repeat = is_keepable_tail_title_repeat(units[repeat_index])
    trim_index = repeat_index + 1 if keep_repeat else repeat_index
    for gap_index in range(repeat_index - 1, -1, -1):
        previous = units[gap_index]
        current = units[gap_index + 1]
        in_repeat_lead_in = current.start >= repeat_start - 30.0
        if current.start - previous.end >= 6.0 and in_repeat_lead_in:
            trim_index = max(gap_index + 1, repeat_index + 1) if keep_repeat else gap_index + 1
            break
    return trim_index


def is_keepable_tail_title_repeat(unit: TranscriptUnit) -> bool:
    return (
        (unit.source.startswith("ocr-temporal") or "+ocr-text" in unit.source)
        and unit.end - unit.start >= 2.0
    )


def is_title_repeat_unit(
    unit: TranscriptUnit,
    units: list[TranscriptUnit],
    story_start: float,
    duration: float,
) -> bool:
    early_cutoff = story_start + duration * 0.20
    if not 3 <= len(unit.normalized_text.split()) <= 10:
        return False
    return any(
        early is not unit
        and early.start <= early_cutoff
        and 3 <= len(early.normalized_text.split()) <= 10
        and similarity(early.normalized_text, unit.normalized_text) >= 0.92
        for early in units
    )


def first_tail_title_repeat_index(
    units: list[TranscriptUnit],
    story_start: float,
    duration: float,
) -> int | None:
    early_cutoff = story_start + duration * 0.20
    tail_cutoff = story_start + duration * 0.75
    early_titles = [
        unit
        for unit in units
        if unit.start <= early_cutoff and 3 <= len(unit.normalized_text.split()) <= 10
    ]
    for index, unit in enumerate(units):
        if unit.start < tail_cutoff or not 3 <= len(unit.normalized_text.split()) <= 10:
            continue
        if any(
            similarity(early.normalized_text, unit.normalized_text) >= 0.92
            for early in early_titles
        ):
            return index
    return None


def tail_contains_early_title_repeat(
    head_units: list[TranscriptUnit],
    tail_units: list[TranscriptUnit],
    story_start: float,
    duration: float,
) -> bool:
    early_cutoff = story_start + duration * 0.20
    early_titles = [
        unit
        for unit in head_units
        if unit.start <= early_cutoff and 3 <= len(unit.normalized_text.split()) <= 10
    ]
    for early in early_titles:
        for tail in tail_units:
            if 3 <= len(tail.normalized_text.split()) <= 10 and similarity(
                early.normalized_text,
                tail.normalized_text,
            ) >= 0.92:
                return True
    return False


def is_short_leading_intro_unit(
    unit: TranscriptUnit,
    story_start: float,
    duration: float,
) -> bool:
    lead_cutoff = story_start + min(30.0, duration * 0.10)
    return unit.start <= lead_cutoff and len(unit.normalized_text.split()) <= 2


def should_drop_low_support_ocr_unit(unit: TranscriptUnit) -> bool:
    if not unit.source.startswith("ocr-temporal"):
        return False
    duration = unit.end - unit.start
    word_count = len(unit.normalized_text.split())
    return duration < 0.75 or word_count < 3


def is_early_title_card_repeat(
    unit: TranscriptUnit,
    tail_units: list[TranscriptUnit],
    early_title_cutoff: float,
) -> bool:
    if "asr" in unit.source:
        return False
    if unit.start > early_title_cutoff:
        return False
    if len(unit.normalized_text.split()) > 10:
        return False
    return any(
        similarity(unit.normalized_text, tail.normalized_text) >= 0.92
        for tail in tail_units
    )


def strict_state_similarity(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    jaccard = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    return max(jaccard, similarity(left, right) * 0.55)


def best_group_observation(group: list[FrameObservation]) -> FrameObservation:
    return max(
        group,
        key=lambda item: (
            item.word_count,
            item.avg_ink_score,
            item.avg_confidence,
            item.timestamp,
        ),
    )


def drop_transition_blend_groups(
    groups: list[list[FrameObservation]],
) -> list[list[FrameObservation]]:
    if len(groups) < 3:
        return groups
    kept: list[list[FrameObservation]] = []
    for index, group in enumerate(groups):
        if 0 < index < len(groups) - 1 and len(group) <= 1:
            previous_tokens = token_set(best_group_observation(groups[index - 1]).normalized_text)
            current_tokens = token_set(best_group_observation(group).normalized_text)
            next_tokens = token_set(best_group_observation(groups[index + 1]).normalized_text)
            is_blend = (
                previous_tokens
                and next_tokens
                and previous_tokens <= current_tokens
                and next_tokens <= current_tokens
                and len(current_tokens) > max(len(previous_tokens), len(next_tokens))
            )
            if is_blend:
                continue
        kept.append(group)
    return kept


def stability_score(observation: FrameObservation, observations: list[FrameObservation]) -> float:
    nearby = [
        other
        for other in observations
        if other is not observation
        and abs(other.timestamp - observation.timestamp) <= 0.75
        and (not observation.page_id or not other.page_id or observation.page_id == other.page_id)
    ]
    if not nearby:
        return 0.0
    related = [
        similarity(observation.normalized_text, other.normalized_text)
        for other in nearby
    ]
    same_state = [score for score in related if score >= 0.82]
    if not same_state:
        return 0.0
    support = min(1.0, len(same_state) / 2.0)
    return (sum(same_state) / len(same_state)) * support


def page_edge_score(
    observation: FrameObservation,
    pages_by_id: dict[str, PageInterval],
) -> tuple[float, list[str]]:
    if not observation.page_id or observation.page_id not in pages_by_id:
        return 0.5, []
    page = pages_by_id[observation.page_id]
    duration = max(0.001, page.end - page.start)
    edge_gap = min(observation.timestamp - page.start, page.end - observation.timestamp)
    stable_margin = min(1.2, max(0.25, duration * 0.22))
    score = max(0.0, min(1.0, edge_gap / stable_margin))
    warnings: list[str] = []
    if duration < 0.75:
        warnings.append(f"short-page:{duration:.2f}")
    if edge_gap < min(0.35, duration * 0.20):
        warnings.append(f"page-edge:{edge_gap:.2f}")
    return score, warnings


def candidate_observations_for_unit(
    unit: TranscriptUnit,
    observations: list[FrameObservation],
    pages: list[PageInterval],
) -> list[FrameObservation]:
    window_start = unit.start - 2.0
    window_end = unit.end + 2.0
    page_ids = {
        page.page_id
        for page in pages
        if min(window_end, page.end) - max(window_start, page.start) > 0
    }
    if page_ids:
        page_candidates = [
            observation for observation in observations if observation.page_id in page_ids
        ]
        if page_candidates:
            return page_candidates
    return [
        observation
        for observation in observations
        if window_start <= observation.timestamp <= window_end
    ]


def score_observation(
    unit: TranscriptUnit,
    observation: FrameObservation,
    observations: list[FrameObservation],
    pages_by_id: dict[str, PageInterval],
) -> tuple[float, list[str]]:
    coverage = target_coverage(unit.normalized_text, observation.normalized_text)
    extra_ratio = extra_token_ratio(unit.normalized_text, observation.normalized_text)
    stability = stability_score(observation, observations)
    page_score, page_warnings = page_edge_score(observation, pages_by_id)
    warnings: list[str] = []
    if coverage < 0.80:
        warnings.append(f"low-coverage:{coverage:.2f}")
    if observation.avg_ink_score < 0.55:
        warnings.append(f"low-ink:{observation.avg_ink_score:.2f}")
    if stability < 0.35:
        warnings.append(f"low-stability:{stability:.2f}")
    if extra_ratio > 0.35:
        warnings.append(f"extra-text:{extra_ratio:.2f}")
    warnings.extend(page_warnings)
    score = (
        coverage * 80.0
        + observation.avg_confidence * 18.0
        + observation.avg_ink_score * 34.0
        + stability * 20.0
        + page_score * 4.0
        + min(observation.word_count, 30) * 0.5
        - extra_ratio * 30.0
    )
    if any(warning.startswith("short-page") for warning in page_warnings):
        score -= 20.0
    if any(warning.startswith("page-edge") for warning in page_warnings):
        score -= 8.0
    return score, warnings


def select_frames(
    units: list[TranscriptUnit],
    observations: list[FrameObservation],
    quality: str,
    pages: list[PageInterval] | None = None,
) -> list[SelectedFrame]:
    pages = pages or []
    pages_by_id = {page.page_id: page for page in pages}
    selected: list[SelectedFrame] = []
    for unit in units:
        candidates = candidate_observations_for_unit(unit, observations, pages)
        if not candidates:
            candidates = observations
        scored = [
            (score_observation(unit, observation, observations, pages_by_id), observation)
            for observation in candidates
        ]
        if not scored:
            continue
        (score, warnings), observation = max(scored, key=lambda item: item[0][0])
        status = "clean" if not warnings else "needs_review"
        if quality == "strict-original" and warnings:
            continue
        transcript, normalized_text = selected_text_for_unit(unit, observation)
        selected.append(
            SelectedFrame(
                unit_id=unit.unit_id,
                timestamp=observation.timestamp,
                frame_path=observation.frame_path,
                transcript=transcript,
                normalized_text=normalized_text,
                score=score,
                status=status,
                warnings=warnings,
                page_id=observation.page_id,
            )
        )
    selected = coalesce_same_frame_page_selections(selected, observations)
    return prune_duplicate_selections(selected)


def selected_text_for_unit(
    unit: TranscriptUnit,
    observation: FrameObservation,
) -> tuple[str, str]:
    if not unit.unit_id.startswith("ocr-missing"):
        return unit.text, unit.normalized_text
    if has_reject_phrase(observation.normalized_text):
        return unit.text, unit.normalized_text
    unit_tokens = token_set(unit.normalized_text)
    observation_tokens = token_set(observation.normalized_text)
    if not unit_tokens or not observation_tokens:
        return unit.text, unit.normalized_text
    unit_coverage = target_coverage(unit.normalized_text, observation.normalized_text)
    observation_coverage = target_coverage(observation.normalized_text, unit.normalized_text)
    not_too_short = len(observation_tokens) >= max(2, len(unit_tokens) - 2)
    if unit_coverage >= 0.78 and observation_coverage >= 0.90 and not_too_short:
        return observation.text, observation.normalized_text
    return unit.text, unit.normalized_text


def coalesce_same_frame_page_selections(
    selected: list[SelectedFrame],
    observations: list[FrameObservation],
) -> list[SelectedFrame]:
    observations_by_frame = {observation.frame_path: observation for observation in observations}
    groups: dict[tuple[str, int, str], list[SelectedFrame]] = {}
    for item in selected:
        key = (item.frame_path, int(round(item.timestamp * 1000)), item.page_id)
        groups.setdefault(key, []).append(item)

    coalesced: list[SelectedFrame] = []
    for group in groups.values():
        if len(group) == 1:
            coalesced.append(group[0])
            continue
        base = max(group, key=selection_rank)
        observation = observations_by_frame.get(base.frame_path)
        if observation is None or has_reject_phrase(observation.normalized_text):
            coalesced.append(base)
            continue
        if observation.word_count < max(len(token_set(item.normalized_text)) for item in group):
            coalesced.append(base)
            continue
        warnings = sorted(
            {
                warning
                for item in group
                for warning in item.warnings
                if not warning.startswith("extra-text")
            }
        )
        base.transcript = observation.text
        base.normalized_text = observation.normalized_text
        base.score = max(item.score for item in group)
        base.warnings = warnings
        base.status = "clean" if not warnings else "needs_review"
        coalesced.append(base)
    return sorted(coalesced, key=lambda value: value.timestamp)


def prune_duplicate_selections(selected: list[SelectedFrame]) -> list[SelectedFrame]:
    kept: list[SelectedFrame] = []
    for item in sorted(selected, key=lambda value: (value.timestamp, -value.score)):
        duplicate_index = None
        for index, existing in enumerate(kept):
            same_unit = item.unit_id == existing.unit_id
            near_same_frame = abs(item.timestamp - existing.timestamp) <= 0.50
            near_same_state = abs(item.timestamp - existing.timestamp) <= 5.0
            same_page_near_state = (
                bool(item.page_id)
                and item.page_id == existing.page_id
                and abs(item.timestamp - existing.timestamp) <= 12.0
            )
            if (
                (same_unit or near_same_frame)
                and similarity(item.normalized_text, existing.normalized_text) >= 0.96
            ):
                duplicate_index = index
                break
            if near_same_frame and is_subset_selection_duplicate(item, existing):
                duplicate_index = index
                break
            if near_same_frame and is_occluded_ocr_selection_duplicate(item, existing):
                duplicate_index = index
                break
            if near_same_state and is_noise_tail_selection_duplicate(item, existing):
                duplicate_index = index
                break
            if (near_same_state or same_page_near_state) and is_subset_selection_duplicate(
                item,
                existing,
            ):
                duplicate_index = index
                break
        if duplicate_index is None:
            kept.append(item)
            continue
        if selection_rank(item) > selection_rank(kept[duplicate_index]):
            kept[duplicate_index] = item
    return sorted(kept, key=lambda value: value.timestamp)


def is_subset_selection_duplicate(left: SelectedFrame, right: SelectedFrame) -> bool:
    left_tokens = token_set(left.normalized_text)
    right_tokens = token_set(right.normalized_text)
    if not left_tokens or not right_tokens:
        return False
    containment = max(
        target_coverage(left.normalized_text, right.normalized_text),
        target_coverage(right.normalized_text, left.normalized_text),
    )
    return containment >= 0.85


def is_occluded_ocr_selection_duplicate(left: SelectedFrame, right: SelectedFrame) -> bool:
    ocr_item = left if left.unit_id.startswith("ocr-missing") else right
    other = right if ocr_item is left else left
    if not ocr_item.unit_id.startswith("ocr-missing"):
        return False
    if len(token_set(ocr_item.normalized_text)) > 4:
        return False
    return target_coverage(ocr_item.normalized_text, other.normalized_text) >= 0.60


def is_noise_tail_selection_duplicate(left: SelectedFrame, right: SelectedFrame) -> bool:
    left_clean = strip_likely_ocr_noise_tokens(left.normalized_text)
    right_clean = strip_likely_ocr_noise_tokens(right.normalized_text)
    if left_clean == left.normalized_text and right_clean == right.normalized_text:
        return False
    if not left_clean or not right_clean:
        return False
    return max(
        target_coverage(left_clean, right_clean),
        target_coverage(right_clean, left_clean),
    ) >= 0.90


def strip_likely_ocr_noise_tokens(text: str) -> str:
    words = clean_text(text).split()
    return " ".join(token for token in words if not is_likely_ocr_noise_token(token))


def is_likely_ocr_noise_token(token: str) -> bool:
    if len(token) > 3:
        return False
    common_short_words = {
        "a",
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
        "i",
        "if",
        "in",
        "is",
        "it",
        "me",
        "my",
        "no",
        "of",
        "on",
        "or",
        "poo",
        "she",
        "the",
        "to",
        "up",
        "we",
        "who",
        "you",
    }
    return token not in common_short_words


def selection_rank(item: SelectedFrame) -> tuple[int, int, int, float, int]:
    clean_bonus = 1 if item.status == "clean" else 0
    warning_penalty = -len(item.warnings)
    source_bonus = 0 if item.unit_id.startswith("ocr-missing") else 1
    return (
        source_bonus,
        clean_bonus,
        warning_penalty,
        item.score,
        len(token_set(item.normalized_text)),
    )


def write_outputs(
    output_dir: Path,
    selected: list[SelectedFrame],
    units: list[TranscriptUnit],
    observations: list[FrameObservation],
    pages: list[PageInterval] | None = None,
) -> None:
    pages = pages or []
    frames_dir = output_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    observations_by_frame = {observation.frame_path: observation for observation in observations}
    for index, item in enumerate(selected, start=1):
        slug = "-".join(item.normalized_text.split()[:8]) or "story-text"
        dest = frames_dir / f"{index:03d}-{int(round(item.timestamp)):06d}s-{slug[:80]}.jpg"
        source_observation = observations_by_frame.get(item.frame_path)
        copied_clean_overlay = copy_selected_frame(
            Path(item.frame_path),
            dest,
            source_observation,
            observations,
        )
        if copied_clean_overlay:
            item.output_source = "overlay-cleaned"
        if reconstruct_occluded_text_if_needed(dest, item, source_observation):
            item.output_source = (
                "text-reconstructed"
                if item.output_source == "original"
                else f"{item.output_source}+text-reconstructed"
            )
        rows.append(
            {
                "index": str(index),
                "timestamp": f"{item.timestamp:.3f}",
                "unit_id": item.unit_id,
                "image": str(dest),
                "transcript": item.transcript,
                "normalized_text": item.normalized_text,
                "score": f"{item.score:.3f}",
                "status": item.status,
                "warnings": ";".join(item.warnings),
                "output_source": item.output_source,
                "page_id": item.page_id,
            }
        )

    csv_path = output_dir / "review-index.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["index"])
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "debug-local.json").write_text(
        json.dumps(
            {
                "units": [asdict(unit) for unit in units],
                "pages": [asdict(page) for page in pages],
                "selected": [asdict(item) for item in selected],
                "observations": [
                    {
                        **asdict(observation),
                        "boxes": [asdict(box) for box in observation.boxes],
                    }
                    for observation in observations
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown_index(output_dir, rows)
    write_contact_sheet(output_dir, rows)


def copy_selected_frame(
    source_path: Path,
    dest_path: Path,
    observation: FrameObservation | None,
    observations: list[FrameObservation],
) -> bool:
    subscribe_region = red_subscribe_overlay_region(source_path)
    if subscribe_region is not None and observation is not None:
        patch_source = nearest_red_overlay_free_observation(
            observation,
            observations,
            subscribe_region,
        )
        if patch_source is not None:
            image = Image.open(source_path).convert("RGB")
            patch_image = Image.open(patch_source.frame_path).convert("RGB")
            image.paste(patch_image.crop(subscribe_region), subscribe_region)
            image.save(dest_path, quality=92)
            return True

    if observation is None or not observation.ad_boxes:
        shutil.copy2(source_path, dest_path)
        return False

    region = overlay_patch_region(observation.ad_boxes, Image.open(source_path).size)
    patch_source = nearest_overlay_free_observation(observation, observations, region)
    if patch_source is None:
        shutil.copy2(source_path, dest_path)
        return False

    image = Image.open(source_path).convert("RGB")
    patch_image = Image.open(patch_source.frame_path).convert("RGB")
    image.paste(patch_image.crop(region), region)
    image.save(dest_path, quality=92)
    return True


def reconstruct_occluded_text_if_needed(
    image_path: Path,
    item: SelectedFrame,
    observation: FrameObservation | None,
) -> bool:
    if has_reject_phrase(item.normalized_text):
        return False
    if observation is None or not observation.boxes:
        return False
    if item.normalized_text == observation.normalized_text:
        return False
    if not has_bottom_left_occluded_suffix(observation):
        return False

    image = Image.open(image_path).convert("RGB")
    region = occluded_text_patch_region(observation.boxes, image.size)
    fill = median_surrounding_color(image, region)
    draw = ImageDraw.Draw(image)
    draw.rectangle(region, fill=fill)

    line_groups = group_boxes_by_line(observation.boxes)
    token_lines = distribute_tokens_to_lines(item.normalized_text.split(), line_groups)
    font = load_reconstruction_font(line_groups)
    for line, tokens in zip(line_groups, token_lines):
        if not tokens:
            continue
        x = max(region[0] + 16, int(min(box.x for box in line)))
        y = max(region[1] + 8, int(min(box.y for box in line)))
        draw.text((x, y), format_reconstructed_line(tokens), fill=(20, 20, 20), font=font)
    image.save(image_path, quality=92)
    return True


def has_bottom_left_occluded_suffix(observations: FrameObservation | list[FrameObservation]) -> bool:
    if isinstance(observations, FrameObservation):
        observation_items = [observations]
    else:
        observation_items = observations
    return any(
        is_bottom_left_occluded_suffix_box(box)
        for observation in observation_items
        for box in observation.boxes
    )


def is_bottom_left_occluded_suffix_box(box: OcrBox) -> bool:
    tokens = clean_text(box.text).split()
    if not tokens or len(tokens) > 2:
        return False
    in_logo_zone = box.x < box.page_width * 0.32 and box.y > box.page_height * 0.82
    compact_box = box.width < box.page_width * 0.18 and box.height < box.page_height * 0.16
    likely_partial_suffix = any(len(token) <= 5 for token in tokens)
    return in_logo_zone and compact_box and likely_partial_suffix


def has_bottom_left_occluded_text(observation: FrameObservation) -> bool:
    for box in observation.boxes:
        if box.x < box.page_width * 0.30 and box.y > box.page_height * 0.70:
            return True
    return False


def occluded_text_patch_region(
    boxes: list[OcrBox],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = image_size
    left = min(box.x for box in boxes)
    top = min(box.y for box in boxes)
    right = max(box.right for box in boxes)
    bottom = max(box.bottom for box in boxes)
    return (
        max(0, int(min(left - 56, width * 0.02))),
        max(0, int(top - 22)),
        min(width, int(max(right + 42, width * 0.29))),
        min(height, int(bottom + 22)),
    )


def median_surrounding_color(
    image: Image.Image,
    region: tuple[int, int, int, int],
) -> tuple[int, int, int]:
    left, top, right, bottom = region
    pad = 34
    sample_region = (
        max(0, left - pad),
        max(0, top - pad),
        min(image.width, right + pad),
        min(image.height, bottom + pad),
    )
    pixels: list[tuple[int, int, int]] = []
    for y in range(sample_region[1], sample_region[3], 3):
        for x in range(sample_region[0], sample_region[2], 3):
            if left <= x < right and top <= y < bottom:
                continue
            pixels.append(image.getpixel((x, y)))
    if not pixels:
        return (230, 230, 220)
    channels = []
    for index in range(3):
        values = sorted(pixel[index] for pixel in pixels)
        channels.append(values[len(values) // 2])
    return tuple(channels)  # type: ignore[return-value]


def group_boxes_by_line(boxes: list[OcrBox]) -> list[list[OcrBox]]:
    groups: list[list[OcrBox]] = []
    for box in sorted(boxes, key=lambda item: (item.y, item.x)):
        if groups and abs(box.y - min(item.y for item in groups[-1])) <= max(12.0, box.height * 0.45):
            groups[-1].append(box)
        else:
            groups.append([box])
    return groups


def distribute_tokens_to_lines(tokens: list[str], line_groups: list[list[OcrBox]]) -> list[list[str]]:
    if not line_groups:
        return [tokens]
    counts = [max(1, sum(len(clean_text(box.text).split()) for box in group)) for group in line_groups]
    if len(line_groups) == 2 and len(tokens) >= 4 and counts[0] == 2:
        return [tokens[:2], tokens[2:]]
    total = sum(counts)
    if total <= 0:
        counts = [1 for _ in line_groups]
        total = len(counts)
    remaining = list(tokens)
    lines: list[list[str]] = []
    for index, count in enumerate(counts):
        if index == len(counts) - 1:
            lines.append(remaining)
            break
        take = max(1, round(len(tokens) * count / total))
        lines.append(remaining[:take])
        remaining = remaining[take:]
    return lines


def load_reconstruction_font(line_groups: list[list[OcrBox]]) -> ImageFont.ImageFont:
    heights = [box.height for group in line_groups for box in group]
    size = int(max(28, min(44, (sum(heights) / max(1, len(heights))) * 0.86)))
    for font_path in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def format_reconstructed_line(tokens: list[str]) -> str:
    words = [format_reconstructed_token(token) for token in tokens]
    if len(words) == 2 and words[0].lower() == words[1].lower():
        return f"{words[0]}, {words[1]},"
    text = " ".join(words)
    if text and text[-1] not in ".!?,":
        text += "."
    return text


def format_reconstructed_token(token: str) -> str:
    if token == "i":
        return "I"
    if token == "mommy":
        return "Mommy"
    return token


def nearest_overlay_free_observation(
    observation: FrameObservation,
    observations: list[FrameObservation],
    patch_region: tuple[int, int, int, int],
) -> FrameObservation | None:
    candidates = [
        other
        for other in observations
        if other is not observation
        and not other.ad_boxes
        and other.page_id == observation.page_id
        and abs(other.timestamp - observation.timestamp) <= 8.0
        and similarity(other.normalized_text, observation.normalized_text) >= 0.55
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            overlay_region_noise_score(Path(item.frame_path), patch_region),
            0 if item.timestamp <= observation.timestamp else 1,
            abs(item.timestamp - observation.timestamp),
        ),
    )


def nearest_red_overlay_free_observation(
    observation: FrameObservation,
    observations: list[FrameObservation],
    patch_region: tuple[int, int, int, int],
) -> FrameObservation | None:
    candidates = [
        other
        for other in observations
        if other is not observation
        and other.page_id == observation.page_id
        and abs(other.timestamp - observation.timestamp) <= 14.0
        and similarity(other.normalized_text, observation.normalized_text) >= 0.95
        and Path(other.frame_path).exists()
        and red_region_score(Path(other.frame_path), patch_region) < 0.010
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            abs(item.avg_ink_score - observation.avg_ink_score),
            abs(item.timestamp - observation.timestamp),
        ),
    )


def red_subscribe_overlay_region(frame_path: Path) -> tuple[int, int, int, int] | None:
    if not frame_path.exists():
        return None
    image = Image.open(frame_path).convert("RGB")
    left = int(image.width * 0.68)
    top = int(image.height * 0.66)
    red_pixels: list[tuple[int, int]] = []
    for y in range(top, image.height):
        for x in range(left, image.width):
            red, green, blue = image.getpixel((x, y))
            if red > 170 and green < 90 and blue < 90:
                red_pixels.append((x, y))
    region_area = max(1, (image.width - left) * (image.height - top))
    if len(red_pixels) / region_area < 0.018:
        return None
    xs = [pixel[0] for pixel in red_pixels]
    ys = [pixel[1] for pixel in red_pixels]
    return (
        max(0, min(xs) - 22),
        max(0, min(ys) - 22),
        min(image.width, max(xs) + 22),
        min(image.height, max(ys) + 22),
    )


def red_region_score(frame_path: Path, region: tuple[int, int, int, int]) -> float:
    image = Image.open(frame_path).convert("RGB")
    pixels = list(image.crop(region).getdata())
    if not pixels:
        return 0.0
    return sum(1 for red, green, blue in pixels if red > 170 and green < 90 and blue < 90) / len(pixels)


def overlay_region_noise_score(
    frame_path: Path,
    patch_region: tuple[int, int, int, int],
) -> float:
    image = Image.open(frame_path).convert("L")
    pixels = list(image.crop(patch_region).getdata())
    if not pixels:
        return 1.0
    dark_fraction = sum(1 for value in pixels if value < 80) / len(pixels)
    mid_dark_fraction = sum(1 for value in pixels if value < 130) / len(pixels)
    return dark_fraction * 2.0 + mid_dark_fraction


def overlay_patch_region(
    ad_boxes: list[OcrBox],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = image_size
    left = min(box.x for box in ad_boxes)
    top = min(box.y for box in ad_boxes)
    right = max(box.right for box in ad_boxes)
    bottom = max(box.bottom for box in ad_boxes)
    return (
        max(0, int(left - 180)),
        max(0, int(top - 80)),
        min(width, int(right + 60)),
        min(height, int(bottom + 120)),
    )


def write_markdown_index(output_dir: Path, rows: list[dict[str, str]]) -> None:
    lines = ["# Story Frame Review", "", "| # | Time | Status | Image | Transcript |", "|---:|---:|---|---|---|"]
    for row in rows:
        rel_image = Path(row["image"]).relative_to(output_dir).as_posix()
        transcript = row["transcript"].replace("|", "\\|")
        lines.append(
            f"| {row['index']} | {float(row['timestamp']):.2f}s | {row['status']} | ![]({rel_image}) | {transcript} |"
        )
    (output_dir / "review-index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_contact_sheet(output_dir: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    thumb_width = 320
    thumb_height = 180
    label_height = 56
    padding = 12
    columns = 3
    rows_count = math.ceil(len(rows) / columns)
    sheet = Image.new(
        "RGB",
        (
            columns * thumb_width + (columns + 1) * padding,
            rows_count * (thumb_height + label_height) + (rows_count + 1) * padding,
        ),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for item_index, row in enumerate(rows):
        grid_y, grid_x = divmod(item_index, columns)
        x = padding + grid_x * (thumb_width + padding)
        y = padding + grid_y * (thumb_height + label_height + padding)
        image = Image.open(row["image"]).convert("RGB")
        image.thumbnail((thumb_width, thumb_height))
        tile = Image.new("RGB", (thumb_width, thumb_height), (255, 255, 255))
        tile.paste(image, ((thumb_width - image.width) // 2, (thumb_height - image.height) // 2))
        sheet.paste(tile, (x, y))
        draw.text((x, y + thumb_height + 4), f"{row['index']} {float(row['timestamp']):.2f}s {row['status']}", fill=(20, 20, 20), font=font)
        draw.text((x, y + thumb_height + 18), row["normalized_text"][:72], fill=(50, 50, 50), font=font)
    sheet.save(output_dir / "review-contact-sheet.jpg", quality=92)
