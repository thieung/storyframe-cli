from __future__ import annotations

import argparse
from pathlib import Path

from .asr import transcribe_units
from .captions import (
    merge_transcript_units_into_sentences,
    select_caption_frames,
    should_skip_full_ocr_for_caption_fallback,
    should_use_caption_fallback,
)
from .media import video_duration
from .page_detection import detect_scene_pages, expand_windows_to_pages
from .selector import (
    assign_observation_pages,
    collect_observations,
    derive_units_from_observations,
    filter_units_for_story,
    make_windows,
    merge_units_with_ocr_missing,
    refine_asr_units_with_ocr,
    select_frames,
    write_outputs,
)
from .subtitles import load_subtitle_units


def should_skip_asr_for_speed_auto(args: argparse.Namespace) -> bool:
    return (
        args.speed == "auto"
        and args.caption_mode == "off"
        and args.asr_backend != "none"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local/free Storyframe engine.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--story-start", type=float, default=0.0)
    parser.add_argument("--story-end", type=float, default=None)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--dense-fps", type=float, default=8.0)
    parser.add_argument(
        "--scan-mode",
        choices=["sampled", "dense", "native", "dense-windowed", "native-windowed"],
        default="dense-windowed",
    )
    parser.add_argument("--quality", choices=["strict-complete", "strict-original", "balanced"], default="strict-complete")
    parser.add_argument("--asr-backend", choices=["none", "faster-whisper"], default="none")
    parser.add_argument("--asr-model", default="small.en")
    parser.add_argument("--ocr-backend", choices=["rapidocr"], default="rapidocr")
    parser.add_argument("--window-padding", type=float, default=2.0)
    parser.add_argument("--page-detection", choices=["none", "scene"], default="scene")
    parser.add_argument(
        "--page-window-mode",
        choices=["unit", "unit-pages", "all-pages"],
        default="all-pages",
    )
    parser.add_argument("--scene-threshold", type=float, default=12.0)
    parser.add_argument("--scene-min-len", type=int, default=8)
    parser.add_argument("--caption-mode", choices=["off", "auto", "force"], default="off")
    parser.add_argument("--speed", choices=["quality", "auto"], default="quality")
    parser.add_argument("--subtitle-path", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise SystemExit(f"Input video does not exist: {args.video}")
    story_end = args.story_end if args.story_end is not None else video_duration(args.video)
    if story_end <= args.story_start:
        raise SystemExit("--story-end must be greater than --story-start")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    pages = []
    if args.page_detection == "scene":
        print("local: detecting scene/page intervals")
        pages = detect_scene_pages(
            args.video,
            args.story_start,
            story_end,
            args.scene_threshold,
            args.scene_min_len,
        )
        print(f"local: detected_pages={len(pages)}")

    print("local: building transcript units")
    units = []
    if args.speed == "auto" and args.subtitle_path and args.subtitle_path.exists():
        units = load_subtitle_units(args.subtitle_path, args.story_start, story_end)
        if units:
            print(f"local: using subtitle transcript units={len(units)} from {args.subtitle_path}")
        else:
            print(f"local: subtitle file had no usable story units: {args.subtitle_path}")
    if not units:
        asr_backend = args.asr_backend
        if should_skip_asr_for_speed_auto(args):
            asr_backend = "none"
            print(
                "local: speed=auto has no usable subtitles; "
                "skipping local ASR and deriving transcript from OCR"
            )
        units = transcribe_units(
            args.video,
            args.work_dir,
            args.story_start,
            story_end,
            asr_backend,
            args.asr_model,
        )
    windows = make_windows(units, args.story_start, story_end, args.window_padding)
    windows = expand_windows_to_pages(
        windows,
        pages,
        args.story_start,
        story_end,
        args.page_window_mode,
    )
    print(
        f"local: windows={len(windows)} asr_units={len(units)} "
        f"page_window_mode={args.page_window_mode}"
    )

    caption_units = merge_transcript_units_into_sentences(
        filter_units_for_story(units, args.story_start, story_end)
    )
    if args.caption_mode == "force":
        selected = select_caption_frames(args.video, args.work_dir, caption_units)
        if not selected:
            raise SystemExit("local found no ASR transcript units for caption rendering")
        write_outputs(args.output_dir, selected, caption_units, [], pages)
        print("local: caption-mode=force; using caption-rendered frames")
        print(f"local selected caption frames: {len(selected)}")
        print(f"Output: {args.output_dir}")
        print(f"Contact sheet: {args.output_dir / 'review-contact-sheet.jpg'}")
        print(f"Index CSV: {args.output_dir / 'review-index.csv'}")
        return

    if args.caption_mode == "auto" and should_skip_full_ocr_for_caption_fallback(
        args.video,
        args.work_dir,
        args.ocr_backend,
        caption_units,
    ):
        selected = select_caption_frames(args.video, args.work_dir, caption_units)
        if not selected:
            raise SystemExit("local found no OCR observations and no ASR transcript units")
        write_outputs(args.output_dir, selected, caption_units, [], pages)
        print("local: sampled OCR does not match transcript; using caption-rendered fallback")
        print(f"local selected caption frames: {len(selected)}")
        print(f"Output: {args.output_dir}")
        print(f"Contact sheet: {args.output_dir / 'review-contact-sheet.jpg'}")
        print(f"Index CSV: {args.output_dir / 'review-index.csv'}")
        return

    observations = collect_observations(
        args.video,
        args.work_dir,
        args.ocr_backend,
        windows,
        args.scan_mode,
        args.fps,
        args.dense_fps,
        cache_dir=args.cache_dir if args.speed == "auto" else None,
    )
    if args.caption_mode == "auto" and should_use_caption_fallback(caption_units, observations):
        selected = select_caption_frames(args.video, args.work_dir, caption_units)
        if not selected:
            raise SystemExit("local found no OCR observations and no ASR transcript units")
        write_outputs(args.output_dir, selected, caption_units, [], pages)
        print(f"local selected caption frames: {len(selected)}")
        print(f"Output: {args.output_dir}")
        print(f"Contact sheet: {args.output_dir / 'review-contact-sheet.jpg'}")
        print(f"Index CSV: {args.output_dir / 'review-index.csv'}")
        return
    observations = assign_observation_pages(observations, pages)
    ocr_units = derive_units_from_observations(observations)

    if units:
        units = refine_asr_units_with_ocr(units, observations)
        print(f"local: refined_asr_units={len(units)} with OCR plateau text")
        before_merge = len(units)
        units = merge_units_with_ocr_missing(units, ocr_units)
        print(f"local: merged_ocr_missing={len(units) - before_merge}")
    else:
        units = ocr_units
        print(f"local: derived_units={len(units)} from OCR temporal tracks")
    before_filter = len(units)
    units = filter_units_for_story(units, args.story_start, story_end)
    print(f"local: filtered_story_units={len(units)} dropped={before_filter - len(units)}")

    selected = select_frames(units, observations, args.quality, pages)
    if not selected:
        raise SystemExit("local selected no frames")

    write_outputs(args.output_dir, selected, units, observations, pages)
    print(f"local selected frames: {len(selected)}")
    print(f"Output: {args.output_dir}")
    print(f"Contact sheet: {args.output_dir / 'review-contact-sheet.jpg'}")
    print(f"Index CSV: {args.output_dir / 'review-index.csv'}")


if __name__ == "__main__":
    main()
