from __future__ import annotations

import html
import re
from pathlib import Path

from .models import TranscriptUnit
from .text import clean_text, has_reject_phrase


TIMING_RE = re.compile(
    r"^\s*(?P<start>(?:\d{1,2}:)?\d{2}:\d{2}[.,]\d{3})\s+-->\s+"
    r"(?P<end>(?:\d{1,2}:)?\d{2}:\d{2}[.,]\d{3})(?:\s+.*)?$"
)
TAG_RE = re.compile(r"<[^>]+>")
VOICE_RE = re.compile(r"^<v\s+[^>]+>", re.IGNORECASE)

NON_STORY_CAPTIONS = {
    "applause",
    "music",
    "laughter",
    "foreign",
}


def load_subtitle_units(
    subtitle_path: Path,
    story_start: float,
    story_end: float,
) -> list[TranscriptUnit]:
    cues = parse_webvtt(subtitle_path.read_text(encoding="utf-8", errors="ignore"))
    units: list[TranscriptUnit] = []
    for index, (start, end, text) in enumerate(cues, start=1):
        if end < story_start or start > story_end:
            continue
        normalized = clean_text(text)
        if len(normalized.split()) < 2:
            continue
        if normalized in NON_STORY_CAPTIONS or has_reject_phrase(normalized):
            continue
        units.append(
            TranscriptUnit(
                unit_id=f"subtitle-{index:04d}",
                text=text,
                normalized_text=normalized,
                start=max(story_start, start),
                end=min(story_end, end),
                source="subtitle",
            )
        )
    return dedupe_overlapping_subtitle_units(units)


def parse_webvtt(text: str) -> list[tuple[float, float, str]]:
    lines = text.replace("\ufeff", "").splitlines()
    cues: list[tuple[float, float, str]] = []
    index = 0
    while index < len(lines):
        timing = TIMING_RE.match(lines[index])
        if timing is None:
            index += 1
            continue

        start = parse_vtt_timestamp(timing.group("start"))
        end = parse_vtt_timestamp(timing.group("end"))
        index += 1
        cue_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            if TIMING_RE.match(lines[index]):
                break
            cue_lines.append(lines[index])
            index += 1
        text_value = normalize_cue_text(" ".join(cue_lines))
        if end > start and text_value:
            cues.append((start, end, text_value))
    return cues


def parse_vtt_timestamp(value: str) -> float:
    value = value.replace(",", ".")
    parts = value.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60.0 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)
    raise ValueError(f"Unsupported subtitle timestamp: {value}")


def normalize_cue_text(text: str) -> str:
    text = VOICE_RE.sub("", text)
    text = TAG_RE.sub("", text)
    text = re.sub(r"\{[^}]+\}", " ", text)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = html.unescape(text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def dedupe_overlapping_subtitle_units(units: list[TranscriptUnit]) -> list[TranscriptUnit]:
    kept: list[TranscriptUnit] = []
    for unit in sorted(units, key=lambda item: (item.start, item.end, item.unit_id)):
        if kept and unit.normalized_text == kept[-1].normalized_text and unit.start <= kept[-1].end + 0.25:
            kept[-1].end = max(kept[-1].end, unit.end)
            continue
        kept.append(unit)
    return kept
