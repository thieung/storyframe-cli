from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from storyframe_cli.extract_story_transcript_frames import (
    FrameCandidate,
    TextItem,
    is_lower_left_watermark,
    is_near_multiline_story_text_column,
    should_prune_against,
    should_use_polished_transcript,
    trim_non_story_edges,
)


def make_args() -> SimpleNamespace:
    return SimpleNamespace(
        video=Path(
            "Where Does Kitty Go In The Rain Animated Read Aloud Kids Book "
            "Vooks Narrated Storybooks-OTOaOdCrrik.mp4"
        ),
        story_start=0.0,
        story_end=472.781,
        include_title_intro=False,
        title_start=0.0,
        title_end=0.0,
        group_overlap=0.62,
        duplicate_similarity=0.92,
        subset_duplicate_window=8.0,
        short_partial_window=2.5,
        partial_superset_window=16.0,
        visual_split_threshold=0.035,
        visual_dedupe_threshold=0.025,
        noisy_visual_dedupe_threshold=0.16,
        text_evolution_similarity=0.84,
    )


def candidate(
    text: str,
    timestamp: float,
    frame_path: str = "dummy.jpg",
    confidence: float = 0.96,
    edge_crop_score: float = 0.0,
    score: float = 100.0,
) -> FrameCandidate:
    normalized = text.lower().replace("?", "").replace(",", "")
    return FrameCandidate(
        frame_path=frame_path,
        timestamp=timestamp,
        raw_text=text,
        normalized_text=normalized,
        word_count=len(normalized.split()),
        avg_confidence=confidence,
        contrast_score=0.9,
        ad_overlay_score=0.0,
        edge_crop_score=edge_crop_score,
        score=score,
    )


class StoryframeAlgorithmTests(unittest.TestCase):
    def test_trim_non_story_edges_drops_intro_and_late_title_cards(self) -> None:
        args = make_args()
        selected = [
            candidate("to lif", 4.5, confidence=0.77),
            candidate("DOE Hi A", 28.25, confidence=0.79),
            candidate("Mommy, Mommy, where's my pet?", 42.25),
            candidate("Where Did Kitty Go?", 395.25),
            candidate("THANK YOU FOR", 444.5),
        ]

        trimmed = trim_non_story_edges(selected, args)

        self.assertEqual(
            [item.normalized_text for item in trimmed],
            ["mommy mommy where's my pet"],
        )

    def test_edge_cropped_partial_is_pruned_across_pan_visual_change(self) -> None:
        args = make_args()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            full_path = temp / "full.jpg"
            crop_path = temp / "crop.jpg"
            Image.new("RGB", (64, 36), (20, 80, 160)).save(full_path)
            Image.new("RGB", (64, 36), (180, 40, 90)).save(crop_path)
            full = candidate(
                "What is rain what falls raindrops",
                81.0,
                str(full_path),
                edge_crop_score=0.0,
                score=110.0,
            )
            cropped = candidate(
                "hat is rain hat falls",
                83.5,
                str(crop_path),
                edge_crop_score=0.9,
                score=88.0,
            )

            self.assertTrue(should_prune_against(cropped, full, args, {}))

    def test_short_ocr_fragment_is_pruned_against_nearby_superset(self) -> None:
        args = make_args()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            fragment_path = temp / "fragment.jpg"
            full_path = temp / "full.jpg"
            Image.new("RGB", (64, 36), (210, 210, 210)).save(fragment_path)
            Image.new("RGB", (64, 36), (212, 212, 212)).save(full_path)
            fragment = candidate("ant to try", 90.25, str(fragment_path), score=75.0)
            fragment.group_frames_seen = 1
            full = candidate(
                "Let's look for kitty want to try",
                91.25,
                str(full_path),
                score=115.0,
            )
            full.group_frames_seen = 4

            self.assertTrue(should_prune_against(fragment, full, args, {}))

    def test_noisy_reconstructed_partial_is_pruned_against_nearby_superset(self) -> None:
        args = make_args()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            partial_path = temp / "partial.jpg"
            full_path = temp / "full.jpg"
            Image.new("RGB", (64, 36), (210, 210, 210)).save(partial_path)
            Image.new("RGB", (64, 36), (212, 212, 212)).save(full_path)
            partial = candidate(
                "do ducks like rain the oil makes water slide off it keeps rs that",
                149.0,
                str(partial_path),
                score=92.0,
            )
            partial.output_source = "reconstructed"
            full = candidate(
                "do ducks like rain the oil makes water slide off it keeps the feathers that are closest to the duck's body dry and warm",
                153.5,
                str(full_path),
                score=118.0,
            )

            self.assertTrue(should_prune_against(partial, full, args, {}))

    def test_transcript_polish_accepts_clear_short_word_correction(self) -> None:
        self.assertTrue(
            should_use_polished_transcript(
                "he says look no rain on mel",
                "He says, Look! No rain on me!",
            )
        )

    def test_transcript_polish_accepts_trailing_noise_removal(self) -> None:
        self.assertTrue(
            should_use_polished_transcript(
                "do birds like rain after a rainstorm passes you can go outside and listen for all the birdie chirps ip",
                "Do birds like rain? After a rainstorm passes, you can go outside and listen for all the birdie chirps!",
            )
        )

    def test_transcript_polish_accepts_trailing_fragment_removal(self) -> None:
        self.assertTrue(
            should_use_polished_transcript(
                "raindrops are falling on mama duck a",
                "Raindrops are falling on Mama Duck.",
            )
        )

    def test_transcript_polish_rejects_missing_meaningful_word(self) -> None:
        self.assertFalse(
            should_use_polished_transcript(
                "what is rain when water heats up it turns into warm wet air called vapor",
                "What is rain? When water heats up, it turns into warm wet air called.",
            )
        )

    def test_column_ghost_detector_requires_multiline_text_panel(self) -> None:
        transcript_items = [
            TextItem("Do", 0.96, 990, 92, 49, 34, 1280, 720),
            TextItem("butterflies", 0.96, 990, 145, 199, 39, 1280, 720),
            TextItem("like", 0.96, 990, 203, 58, 36, 1280, 720),
            TextItem("When", 0.96, 990, 286, 76, 19, 1280, 720),
            TextItem("sun", 0.93, 1040, 331, 48, 14, 1280, 720),
        ]
        ghost_item = TextItem("fl", 0.72, 1123, 409, 14, 15, 1280, 720)
        self.assertTrue(
            is_near_multiline_story_text_column(ghost_item, transcript_items)
        )

        one_line_items = [
            TextItem("Raindrops", 0.97, 637, 38, 157, 38, 1280, 720),
            TextItem("Mama", 0.97, 1025, 38, 95, 29, 1280, 720),
            TextItem("Duck", 0.97, 1130, 37, 86, 30, 1280, 720),
        ]
        rain_art_item = TextItem("af", 0.67, 650, 364, 42, 28, 1280, 720)
        self.assertFalse(
            is_near_multiline_story_text_column(rain_art_item, one_line_items)
        )

    def test_lower_left_story_text_is_not_watermark(self) -> None:
        story_item = TextItem("umbrella!", 0.97, 77, 522, 122, 19, 1280, 720)
        watermark_item = TextItem("VOOKS", 0.97, 43, 635, 149, 45, 1280, 720)

        self.assertFalse(is_lower_left_watermark(story_item))
        self.assertTrue(is_lower_left_watermark(watermark_item))


if __name__ == "__main__":
    unittest.main()
