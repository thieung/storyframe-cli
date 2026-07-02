from __future__ import annotations

import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from storyframe_cli.local.captions import (
    merge_transcript_units_into_sentences,
    render_caption_if_needed,
    sample_transcript_units,
    should_use_caption_fallback,
)
from storyframe_cli.local.engine import should_skip_asr_for_speed_auto
from storyframe_cli.local.models import FrameObservation, OcrBox, PageInterval, SelectedFrame, TranscriptUnit
from storyframe_cli.local.selector import (
    coalesce_same_frame_page_selections,
    filter_units_for_story,
    has_bottom_left_occluded_suffix,
    load_cached_observation,
    merge_units_with_ocr_missing,
    observation_from_boxes,
    prune_duplicate_selections,
    reconstruct_occluded_text_if_needed,
    score_observation,
    selected_text_for_unit,
    write_cached_observation,
)
from storyframe_cli.local.subtitles import load_subtitle_units, parse_webvtt
from storyframe_cli.local.text import clean_text, corrected_text_with_reference, has_reject_phrase


def selected(unit_id: str, timestamp: float, text: str) -> SelectedFrame:
    normalized = text.lower().replace(".", "")
    return SelectedFrame(
        unit_id=unit_id,
        timestamp=timestamp,
        frame_path="frame.jpg",
        transcript=text,
        normalized_text=normalized,
        score=100.0,
        status="clean",
    )


def unit(unit_id: str, start: float, end: float, text: str, source: str = "asr") -> TranscriptUnit:
    normalized = text.lower().replace(".", "")
    return TranscriptUnit(
        unit_id=unit_id,
        text=text,
        normalized_text=normalized,
        start=start,
        end=end,
        source=source,
    )


def observation(timestamp: float, text: str) -> FrameObservation:
    normalized = clean_text(text)
    return FrameObservation(
        frame_path="frame.jpg",
        timestamp=timestamp,
        text=text,
        normalized_text=normalized,
        boxes=[],
        avg_confidence=0.95,
        avg_ink_score=0.90,
        word_count=len(normalized.split()),
    )


class LocalSelectorTests(unittest.TestCase):
    def test_speed_auto_default_uses_ocr_first_when_subtitles_are_unavailable(self) -> None:
        args = SimpleNamespace(
            speed="auto",
            caption_mode="off",
            asr_backend="faster-whisper",
        )

        self.assertTrue(should_skip_asr_for_speed_auto(args))

    def test_speed_auto_keeps_asr_when_caption_rendering_needs_transcript(self) -> None:
        args = SimpleNamespace(
            speed="auto",
            caption_mode="force",
            asr_backend="faster-whisper",
        )

        self.assertFalse(should_skip_asr_for_speed_auto(args))

    def test_ocr_only_selection_is_not_text_reconstructed(self) -> None:
        item = selected("ocr-0001", 12.0, "Willis: I'll slide down the chimney just like that pig!")
        item.normalized_text = "willis i'll slide down the chimney just like that pig"
        observation_item = observation(12.0, "that pig")
        observation_item.boxes = [
            OcrBox("pig", 0.99, 120, 650, 70, 32, 1280, 720, 0.9),
        ]

        changed = reconstruct_occluded_text_if_needed(
            Path("unused.jpg"),
            item,
            observation_item,
        )

        self.assertFalse(changed)

    def test_asr_fragments_are_merged_into_complete_caption_sentence(self) -> None:
        merged = merge_transcript_units_into_sentences(
            [
                unit("asr-0001", 1.0, 2.0, "Barry looked up"),
                unit("asr-0002", 2.3, 3.0, "at the huge tree"),
                unit("asr-0003", 3.2, 4.0, "and smiled."),
            ]
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "Barry looked up at the huge tree and smiled.")
        self.assertEqual(merged[0].normalized_text, "barry looked up at the huge tree and smiled")
        self.assertEqual(merged[0].source, "asr-caption")

    def test_caption_render_writes_transcript_onto_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_path = Path(temp_dir) / "frame.jpg"
            Image.new("RGB", (640, 360), (120, 150, 180)).save(frame_path, quality=92)
            before = frame_path.read_bytes()
            item = selected("caption-0001", 2.0, "Barry looked up at the huge tree and smiled.")
            item.frame_path = str(frame_path)
            item.output_source = "caption-rendered"

            changed = render_caption_if_needed(frame_path, item)

            self.assertTrue(changed)
            self.assertNotEqual(frame_path.read_bytes(), before)
            self.assertEqual(item.normalized_text, "barry looked up at the huge tree and smiled")

    def test_caption_fallback_triggers_when_ocr_does_not_match_transcript(self) -> None:
        units = [
            unit("asr-0001", 1.0, 2.0, "Barry looked up at the huge tree and smiled."),
        ]
        observations = [observation(1.5, "subscribe now")]

        self.assertTrue(should_use_caption_fallback(units, observations))

    def test_caption_fallback_stays_off_when_ocr_matches_transcript(self) -> None:
        units = [
            unit("asr-0001", 1.0, 2.0, "Barry looked up at the huge tree and smiled."),
        ]
        observations = [observation(1.5, "Barry looked up at the huge tree and smiled.")]

        self.assertFalse(should_use_caption_fallback(units, observations))

    def test_caption_ocr_sample_units_are_spread_across_transcript(self) -> None:
        units = [
            unit(f"asr-{index:04d}", float(index), float(index + 1), f"Sentence number {index}.")
            for index in range(10)
        ]

        sampled = sample_transcript_units(units, 4)

        self.assertEqual([item.unit_id for item in sampled], ["asr-0000", "asr-0003", "asr-0006", "asr-0009"])

    def test_webvtt_subtitles_are_loaded_as_story_units(self) -> None:
        vtt = """WEBVTT

00:00:01.000 --> 00:00:02.500
Where is kitty?

00:00:03.000 --> 00:00:04.000
[Music]

00:00:05.000 --> 00:00:06.000
Subscribe now.
"""
        cues = parse_webvtt(vtt)

        self.assertEqual(len(cues), 2)
        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = Path(temp_dir) / "sample.en.vtt"
            subtitle_path.write_text(vtt, encoding="utf-8")

            units = load_subtitle_units(subtitle_path, 0.0, 10.0)

        self.assertEqual([item.normalized_text for item in units], ["where is kitty"])
        self.assertEqual(units[0].source, "subtitle")

    def test_ocr_observation_cache_round_trips_without_work_dir_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "ocr.json"
            cached_frame = Path(temp_dir) / "cache-frame.jpg"
            loaded_frame = Path(temp_dir) / "loaded-frame.jpg"
            item = observation(12.5, "Where is kitty?")
            item.frame_path = str(cached_frame)

            write_cached_observation(cache_path, item)
            loaded_frame.write_bytes(b"frame")
            cached, loaded = load_cached_observation(cache_path, loaded_frame)

        self.assertTrue(cached)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.frame_path, str(loaded_frame))
        self.assertEqual(loaded.normalized_text, "where is kitty")

    def test_repeated_text_far_apart_is_preserved(self) -> None:
        items = [
            selected("asr-0001", 10.0, "I am not scared."),
            selected("asr-0007", 42.0, "I am not scared."),
        ]

        pruned = prune_duplicate_selections(items)

        self.assertEqual([item.unit_id for item in pruned], ["asr-0001", "asr-0007"])

    def test_same_text_same_moment_keeps_best_score(self) -> None:
        weak = selected("asr-0001", 10.0, "I am not scared.")
        strong = selected("asr-0002", 10.25, "I am not scared.")
        weak.score = 80.0
        strong.score = 120.0

        pruned = prune_duplicate_selections([weak, strong])

        self.assertEqual(len(pruned), 1)
        self.assertEqual(pruned[0].score, 120.0)

    def test_nearby_subset_fragment_is_pruned(self) -> None:
        full = selected("asr-0001", 82.0, "What is rain? What falls? Raindrops.")
        fragment = selected("asr-0002", 82.25, "What falls?")
        fragment.status = "needs_review"
        fragment.warnings = ["extra-text:0.60"]

        pruned = prune_duplicate_selections([full, fragment])

        self.assertEqual(len(pruned), 1)
        self.assertEqual(pruned[0].unit_id, "asr-0001")

    def test_fuzzy_ocr_missing_duplicate_prefers_asr(self) -> None:
        dirty = selected("ocr-missing-0001", 360.75, "Mommu, Mommy, hat.")
        clean = selected("asr-0001", 360.75, "mommy mommy look at that")
        dirty.score = 151.0
        clean.score = 141.0

        pruned = prune_duplicate_selections([dirty, clean])

        self.assertEqual(len(pruned), 1)
        self.assertEqual(pruned[0].unit_id, "asr-0001")

    def test_noisy_ocr_tail_duplicate_prefers_full_story_text(self) -> None:
        noisy = selected(
            "ocr-missing-0001",
            354.75,
            "The woodsman sped off quickly with a super speedy burst ry sy gr re",
        )
        full = selected(
            "asr-0001",
            353.25,
            "The woodsman sped off quickly with a super speedy burst Barry waved a friendly paw and growled his growly bye",
        )
        noisy.score = 146.0
        full.score = 155.0

        pruned = prune_duplicate_selections([noisy, full])

        self.assertEqual(len(pruned), 1)
        self.assertEqual(pruned[0].unit_id, "asr-0001")

    def test_same_page_subset_fragment_is_pruned_across_wider_gap(self) -> None:
        fragment = selected(
            "ocr-missing-0001",
            222.125,
            "one two pretend seashell store mussel shell",
        )
        full = selected(
            "asr-0001",
            229.875,
            "one two three four a pretend seashell store scallop shell oyster shell mussel shell moon snail shell",
        )
        fragment.page_id = "page-0010"
        full.page_id = "page-0010"
        fragment.score = 145.0
        full.score = 155.0

        pruned = prune_duplicate_selections([fragment, full])

        self.assertEqual(len(pruned), 1)
        self.assertEqual(pruned[0].unit_id, "asr-0001")

    def test_same_frame_segments_are_coalesced_to_full_ocr_text(self) -> None:
        first = selected("asr-0001", 43.625, "in the deepest of woodland")
        second = selected("asr-0002", 43.625, "a shy bear called Barry")
        first.frame_path = "frame.jpg"
        second.frame_path = "frame.jpg"
        first.page_id = "page-0003"
        second.page_id = "page-0003"
        first.warnings = ["extra-text:0.45"]
        second.warnings = ["extra-text:0.55"]
        observation = FrameObservation(
            "frame.jpg",
            43.625,
            "In the deepest of woodland at the start of the day, a shy bear called Barry slept in the trees.",
            "in the deepest of woodland at the start of the day a shy bear called barry slept in the trees",
            [],
            0.99,
            0.70,
            18,
            page_id="page-0003",
        )

        coalesced = coalesce_same_frame_page_selections([first, second], [observation])

        self.assertEqual(len(coalesced), 1)
        self.assertEqual(coalesced[0].normalized_text, observation.normalized_text)
        self.assertEqual(coalesced[0].status, "clean")

    def test_short_ocr_missing_unit_is_not_merged(self) -> None:
        merged = merge_units_with_ocr_missing(
            [unit("asr-0001", 10.0, 12.0, "Mommy, where is kitty?")],
            [unit("ocr-0001", 14.0, 14.25, "noisy partial text", "ocr-temporal")],
        )

        self.assertEqual([item.unit_id for item in merged], ["asr-0001"])

    def test_terminal_end_screen_is_trimmed_from_previous_large_gap(self) -> None:
        units = [
            unit("asr-0001", 100.0, 104.0, "Here is my kitty warm and dry"),
            unit("asr-0002", 112.0, 113.0, "Where Did Kitty Go"),
            unit("asr-0003", 121.0, 123.0, "Unicorn and Horse"),
            unit("asr-0004", 130.0, 131.0, "Thanks for watching"),
        ]

        filtered = filter_units_for_story(units, 0.0, 150.0)

        self.assertEqual([item.unit_id for item in filtered], ["asr-0001"])

    def test_tail_title_repeat_after_large_gap_is_trimmed(self) -> None:
        units = [
            unit("asr-0001", 30.0, 32.0, "Where Does Kitty Go In The Rain"),
            unit("asr-0002", 100.0, 104.0, "Here is my kitty warm and dry"),
            unit("asr-0003", 130.0, 132.0, "Where Does Kitty Go In The Rain"),
            unit("asr-0004", 136.0, 138.0, "Unicorn and Horse"),
        ]

        filtered = filter_units_for_story(units, 0.0, 150.0)

        self.assertEqual([item.unit_id for item in filtered], ["asr-0001", "asr-0002"])

    def test_keep_ocr_missing_tail_title_repeat_as_story_page(self) -> None:
        units = [
            unit("asr-0001", 33.0, 36.0, "Where Does Kitty Go In The Rain"),
            unit("asr-0002", 390.0, 396.0, "Here is my kitty warm and dry"),
            unit(
                "ocr-missing-0003",
                404.0,
                415.0,
                "Where Does Kitty Go In The Rain",
                "ocr-temporal+missing",
            ),
            unit("ocr-missing-0004", 421.0, 424.0, "Unicorn and Horse", "ocr-temporal+missing"),
        ]

        filtered = filter_units_for_story(units, 0.0, 472.0)

        self.assertEqual(
            [item.unit_id for item in filtered],
            ["asr-0001", "asr-0002", "ocr-missing-0003"],
        )

    def test_terminal_ad_after_tail_title_keeps_ocr_supported_title(self) -> None:
        units = [
            unit("asr-0001", 33.0, 36.0, "Where Does Kitty Go In The Rain"),
            unit("asr-0002", 390.0, 396.0, "Here is my kitty warm and dry"),
            unit(
                "asr-0003",
                404.0,
                407.0,
                "Where Does Kitty Go In The Rain",
                "asr+ocr-text",
            ),
            unit("asr-0004", 454.0, 457.0, "Thanks for watching"),
        ]

        filtered = filter_units_for_story(units, 0.0, 472.0)

        self.assertEqual(
            [item.unit_id for item in filtered],
            ["asr-0001", "asr-0002", "asr-0003"],
        )

    def test_explicit_end_keeps_fact_pages_and_drops_later_outro(self) -> None:
        units = [
            unit("asr-0001", 300.0, 304.0, "Fireflies need moist habitats"),
            unit(
                "asr-0002",
                351.0,
                354.0,
                "Ask your grown-up and start exploring more fun stories like these.",
            ),
            unit("asr-0003", 366.0, 370.0, "Scientists believe fireflies light up."),
            unit("asr-0004", 398.0, 400.0, "The End"),
            unit("asr-0005", 406.0, 410.0, "Popular titles and new releases"),
            unit(
                "ocr-missing-0006",
                446.0,
                452.0,
                "It's a Firefly Night published in the United States by Blue Apple Books",
                "ocr-temporal+missing",
            ),
        ]

        filtered = filter_units_for_story(units, 0.0, 459.0)

        self.assertEqual([item.unit_id for item in filtered], ["asr-0001", "asr-0003", "asr-0004"])

    def test_ocr_only_repeated_title_intro_is_trimmed(self) -> None:
        units = [
            unit("ocr-0001", 30.0, 32.0, "Where Does Kitty Go In The Rain", "ocr-temporal"),
            unit("asr-0002", 100.0, 104.0, "Here is my kitty warm and dry"),
            unit("asr-0003", 130.0, 132.0, "Where Does Kitty Go In The Rain"),
        ]

        filtered = filter_units_for_story(units, 0.0, 150.0)

        self.assertEqual([item.unit_id for item in filtered], ["asr-0002"])

    def test_bottom_right_subscribe_overlay_does_not_drop_story_text(self) -> None:
        boxes = [
            OcrBox("What is rain?", 0.99, 50, 180, 250, 40, 1280, 720, 0.9),
            OcrBox("The cloud becomes bigger.", 0.99, 50, 260, 430, 32, 1280, 720, 0.8),
            OcrBox("SUBSCRIBE", 0.97, 1030, 610, 150, 36, 1280, 720, 0.9),
        ]

        observation = observation_from_boxes(Path("frame.jpg"), 75.0, boxes)

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.normalized_text, "what is rain the cloud becomes bigger")
        self.assertEqual(len(observation.ad_boxes), 1)

    def test_edge_crop_fragment_is_dropped_from_story_text(self) -> None:
        boxes = [
            OcrBox("es", 0.95, 0, 176, 33, 39, 1280, 720, 0.75),
            OcrBox("Do beetles like rain?", 0.99, 818, 644, 333, 42, 1280, 720, 0.94),
        ]

        observation = observation_from_boxes(Path("frame.jpg"), 215.75, boxes)

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.normalized_text, "do beetles like rain")

    def test_tall_artifact_box_is_dropped_from_story_text(self) -> None:
        boxes = [
            OcrBox("Do worms", 0.99, 67, 135, 195, 46, 1280, 720, 0.85),
            OcrBox("like rain?", 0.99, 67, 187, 181, 51, 1280, 720, 0.84),
            OcrBox("AAAK", 0.81, 378, 560, 94, 159, 1280, 720, 0.65),
        ]

        observation = observation_from_boxes(Path("frame.jpg"), 251.25, boxes)

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.normalized_text, "do worms like rain")

    def test_low_confidence_short_artifacts_are_dropped_from_story_text(self) -> None:
        boxes = [
            OcrBox("Bird peeks out from", 0.99, 895, 206, 329, 49, 1280, 720, 0.89),
            OcrBox("a hole in a tree.", 0.94, 897, 257, 260, 43, 1280, 720, 0.42),
            OcrBox("tu", 0.54, 567, 359, 43, 29, 1280, 720, 0.51),
            OcrBox("mrm", 0.64, 519, 429, 79, 40, 1280, 720, 0.53),
        ]

        observation = observation_from_boxes(Path("frame.jpg"), 310.75, boxes)

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.normalized_text, "bird peeks out from a hole in a tree")

    def test_corner_short_artifact_is_dropped_from_story_text(self) -> None:
        boxes = [
            OcrBox("Every empty seashell on the beach", 0.98, 367, 145, 551, 87, 1280, 720, 0.53),
            OcrBox("was once a part of an animal", 0.99, 418, 216, 442, 46, 1280, 720, 0.84),
            OcrBox("from the mollusk family.", 0.98, 446, 262, 385, 52, 1280, 720, 0.84),
            OcrBox("BSCRE", 0.83, 1126, 635, 61, 19, 1280, 720, 0.68),
        ]

        observation = observation_from_boxes(Path("frame.jpg"), 274.5, boxes)

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(
            observation.normalized_text,
            "every empty seashell on the beach was once a part of an animal from the mollusk family",
        )

    def test_watermark_fused_box_is_dropped_from_story_text(self) -> None:
        boxes = [
            OcrBox("Mommu,", 0.99, 56, 601, 145, 37, 1280, 720, 0.97),
            OcrBox("Mommy,", 0.99, 193, 600, 148, 48, 1280, 720, 0.96),
            OcrBox("VOOKS.hat.", 0.94, 26, 614, 259, 91, 1280, 720, 0.77),
        ]

        observation = observation_from_boxes(Path("frame.jpg"), 361.75, boxes)

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.normalized_text, "mommu mommy")

    def test_reference_text_repairs_occluded_ocr_tokens(self) -> None:
        corrected = corrected_text_with_reference(
            "Mommu, Mommy, hat.",
            "Mommy, Mommy, what?",
        )

        self.assertEqual(corrected, "mommy mommy what")

    def test_reference_text_can_insert_logo_occluded_words(self) -> None:
        corrected = corrected_text_with_reference(
            "Mommu, Mommy, hat.",
            "Mommy, Mommy, look at that!",
            allow_insertions=True,
        )

        self.assertEqual(corrected, "mommy mommy look at that")

    def test_reference_text_preserves_plural_from_ocr(self) -> None:
        corrected = corrected_text_with_reference(
            "Do squirrels like rain?",
            "Do squirrel like rain.",
        )

        self.assertEqual(corrected.lower(), "do squirrels like rain?")

    def test_reference_text_can_drop_unmatched_ocr_noise(self) -> None:
        corrected = corrected_text_with_reference(
            "The woodsman sped off quickly with a super speedy burst ar sy gra ray ryy",
            "The woodsman sped off quickly with a super speedy burst",
            keep_unmatched=False,
        )

        self.assertEqual(corrected, "the woodsman sped off quickly with a super speedy burst")

    def test_reference_text_keeps_unmatched_story_words(self) -> None:
        corrected = corrected_text_with_reference(
            "Do squirrels like rain? If it is not raining too hard.",
            "Squirrels. If it is not raining too hard.",
            keep_unmatched=False,
        )

        self.assertEqual(
            clean_text(corrected),
            "do squirrels like rain if it is not raining too hard",
        )

    def test_reference_text_drops_vowel_heavy_leading_ocr_noise(self) -> None:
        corrected = corrected_text_with_reference(
            "AIAUS Scientists believe fireflies light up in rhythmic patterns.",
            "Scientists believe fireflies light up in rhythmic patterns.",
            keep_unmatched=False,
        )

        self.assertEqual(
            clean_text(corrected),
            "scientists believe fireflies light up in rhythmic patterns",
        )

    def test_rejects_end_screen_and_credits_text(self) -> None:
        self.assertTrue(has_reject_phrase("Popular Titles Age Elementary"))
        self.assertTrue(has_reject_phrase("Available from Buster Books Published by Buster Books"))
        self.assertTrue(has_reject_phrase("Ask your grown-up and start exploring more fun stories"))
        self.assertTrue(has_reject_phrase("Published in the United States by Blue Apple Books"))

    def test_clean_text_normalizes_accented_and_fused_ocr_title_tokens(self) -> None:
        self.assertEqual(
            clean_text("DOÉS A BEARPOO INTHE WOODS"),
            "does a bear poo in the woods",
        )
        self.assertEqual(
            clean_text("DOESA BEAR-POO IN THE WOODS"),
            "does a bear poo in the woods",
        )
        self.assertEqual(clean_text("firefliesgotcha"), "fireflies gotcha")
        self.assertEqual(clean_text("It'sa SeashelL Day"), "it's a seashell day")
        self.assertEqual(clean_text("Ihear the Ocean!"), "i hear the ocean")
        self.assertEqual(clean_text("100,000 species"), "100000 species")

    def test_ocr_missing_selection_uses_cleaner_selected_observation_text(self) -> None:
        noisy_unit = unit(
            "ocr-missing-0001",
            172.5,
            176.125,
            "sthers Each one is different. I have many more.",
            "ocr-temporal+missing",
        )
        clean_observation = FrameObservation(
            "frame.jpg",
            173.875,
            "Each one is different. I have many more.",
            "each one is different i have many more",
            [],
            0.99,
            0.85,
            8,
        )

        _, normalized = selected_text_for_unit(noisy_unit, clean_observation)

        self.assertEqual(normalized, "each one is different i have many more")

    def test_occlusion_guard_ignores_normal_lower_left_text(self) -> None:
        observation = FrameObservation(
            "frame.jpg",
            190.75,
            "Do squirrels like rain?",
            "do squirrels like rain",
            [
                OcrBox("Do squirrels", 0.99, 70, 420, 240, 42, 1280, 720, 0.95),
                OcrBox("like rain?", 0.99, 70, 472, 180, 42, 1280, 720, 0.95),
            ],
            0.99,
            0.95,
            4,
        )

        self.assertFalse(has_bottom_left_occluded_suffix(observation))

    def test_occlusion_guard_detects_logo_overlap_suffix(self) -> None:
        observation = FrameObservation(
            "frame.jpg",
            360.75,
            "Mommu, Mommy, hat.",
            "mommu mommy hat",
            [
                OcrBox("Mommu,", 0.99, 56, 601, 145, 37, 1280, 720, 0.97),
                OcrBox("Mommy,", 0.99, 193, 600, 148, 48, 1280, 720, 0.96),
                OcrBox("hat.", 0.94, 190, 647, 82, 32, 1280, 720, 0.77),
            ],
            0.97,
            0.90,
            3,
        )

        self.assertTrue(has_bottom_left_occluded_suffix(observation))

    def test_scoring_prefers_clearer_frame_over_page_center_bias(self) -> None:
        page = PageInterval("page-0001", 229.791667, 236.75, "scene")
        unit_item = unit("asr-0001", 233.34, 236.26, "Worms are squirmy on wet ground.")
        early = FrameObservation(
            "early.jpg",
            235.5,
            "Worms are squirmy on wet ground.",
            "worms are squirmy on wet ground",
            [],
            1.0,
            0.861,
            6,
            page_id=page.page_id,
        )
        later = FrameObservation(
            "later.jpg",
            236.0,
            "Worms are squirmy on wet ground.",
            "worms are squirmy on wet ground",
            [],
            1.0,
            0.937,
            6,
            page_id=page.page_id,
        )

        early_score, _ = score_observation(unit_item, early, [early, later], {page.page_id: page})
        later_score, _ = score_observation(unit_item, later, [early, later], {page.page_id: page})

        self.assertGreater(later_score, early_score)


if __name__ == "__main__":
    unittest.main()
