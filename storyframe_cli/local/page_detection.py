from __future__ import annotations

from pathlib import Path

from .bootstrap import add_local_dependency_paths
from .models import PageInterval

add_local_dependency_paths()


def detect_scene_pages(
    video_path: Path,
    story_start: float,
    story_end: float,
    threshold: float,
    min_scene_len: int,
) -> list[PageInterval]:
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector
    except Exception as exc:
        print(f"local: scene detection unavailable, using one page window: {exc}")
        return [PageInterval("page-0001", story_start, story_end, "fallback")]

    try:
        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
        )
        scene_manager.detect_scenes(video=video, show_progress=False)
        raw_scenes = scene_manager.get_scene_list()
    except Exception as exc:
        print(f"local: scene detection failed, using one page window: {exc}")
        return [PageInterval("page-0001", story_start, story_end, "fallback")]

    pages: list[PageInterval] = []
    for scene_index, (start_time, end_time) in enumerate(raw_scenes, start=1):
        start = max(story_start, float(start_time.get_seconds()))
        end = min(story_end, float(end_time.get_seconds()))
        if end <= start:
            continue
        pages.append(
            PageInterval(
                page_id=f"page-{scene_index:04d}",
                start=start,
                end=end,
                source="scene",
            )
        )
    if not pages:
        pages = [PageInterval("page-0001", story_start, story_end, "fallback")]
    return fill_page_gaps(pages, story_start, story_end)


def fill_page_gaps(
    pages: list[PageInterval],
    story_start: float,
    story_end: float,
) -> list[PageInterval]:
    filled: list[PageInterval] = []
    cursor = story_start
    gap_index = 1
    for page in sorted(pages, key=lambda item: item.start):
        if page.start > cursor + 0.05:
            filled.append(
                PageInterval(
                    page_id=f"page-gap-{gap_index:04d}",
                    start=cursor,
                    end=page.start,
                    source="gap",
                )
            )
            gap_index += 1
        filled.append(page)
        cursor = max(cursor, page.end)
    if cursor < story_end - 0.05:
        filled.append(
            PageInterval(
                page_id=f"page-gap-{gap_index:04d}",
                start=cursor,
                end=story_end,
                source="gap",
            )
        )
    return filled


def page_for_timestamp(
    pages: list[PageInterval],
    timestamp: float,
) -> PageInterval | None:
    for page in pages:
        if page.start <= timestamp <= page.end:
            return page
    return None


def merge_windows(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(windows):
        if end <= start:
            continue
        if not merged or start > merged[-1][1] + 0.05:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def expand_windows_to_pages(
    windows: list[tuple[float, float]],
    pages: list[PageInterval],
    story_start: float,
    story_end: float,
    mode: str,
) -> list[tuple[float, float]]:
    if not pages or mode == "unit":
        return merge_windows(windows)

    page_windows: list[tuple[float, float]] = []
    if mode == "all-pages":
        page_windows = [(page.start, page.end) for page in pages]
        return merge_windows(page_windows)

    for window_start, window_end in windows:
        expanded = False
        for page in pages:
            overlaps = min(window_end, page.end) - max(window_start, page.start)
            if overlaps <= 0:
                continue
            page_windows.append(
                (
                    max(story_start, page.start),
                    min(story_end, page.end),
                )
            )
            expanded = True
        if not expanded:
            page_windows.append((window_start, window_end))
    return merge_windows(page_windows)
