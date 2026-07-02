from __future__ import annotations

from pathlib import Path

from .bootstrap import add_local_dependency_paths
from .media import extract_audio_wav
from .models import TranscriptUnit
from .text import clean_text, has_reject_phrase

add_local_dependency_paths()


def transcribe_units(
    video_path: Path,
    work_dir: Path,
    start: float,
    end: float,
    backend: str,
    model_size: str,
) -> list[TranscriptUnit]:
    if backend == "none":
        return []
    if backend != "faster-whisper":
        raise RuntimeError(f"Unsupported ASR backend for MVP: {backend}")

    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # pragma: no cover - runtime dependency message.
        raise RuntimeError(
            "faster-whisper is missing. Install local deps into work/.deps/storyframe-local."
        ) from exc

    audio_path = work_dir / "audio" / "story.wav"
    extract_audio_wav(video_path, audio_path, start, end)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=False,
        word_timestamps=True,
    )

    units: list[TranscriptUnit] = []
    for index, segment in enumerate(segments, start=1):
        text = " ".join(str(segment.text).split())
        normalized = clean_text(text)
        if not normalized or has_reject_phrase(normalized):
            continue
        units.append(
            TranscriptUnit(
                unit_id=f"asr-{index:04d}",
                text=text,
                normalized_text=normalized,
                start=start + float(segment.start),
                end=start + float(segment.end),
                source="asr",
            )
        )
    return units

