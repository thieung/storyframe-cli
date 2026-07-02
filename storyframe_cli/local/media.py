from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nSTDERR:\n"
            + proc.stderr[-4000:]
        )
    return proc


def video_duration(video_path: Path) -> float:
    proc = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
    )
    return max(0.1, float(proc.stdout.strip()))


def video_fps(video_path: Path) -> float:
    proc = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "json",
            str(video_path),
        ]
    )
    data = json.loads(proc.stdout)
    rate = data["streams"][0].get("avg_frame_rate", "0/0")
    top, bottom = rate.split("/", 1)
    value = float(top) / max(1.0, float(bottom))
    return value if value > 0 else 24.0


def extract_audio_wav(video_path: Path, output_path: Path, start: float, end: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{max(0.1, end - start):.3f}",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_path),
        ]
    )


def extract_frame(video_path: Path, timestamp: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: RuntimeError | None = None
    for candidate in [timestamp, timestamp - 0.25, timestamp - 0.50, timestamp - 1.00]:
        if candidate < 0:
            continue
        if output_path.exists():
            output_path.unlink()
        try:
            run_command(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{candidate:.3f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    "-pix_fmt",
                    "yuvj420p",
                    str(output_path),
                ]
            )
            if output_path.exists() and output_path.stat().st_size > 0:
                return
        except RuntimeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Could not extract frame at {timestamp:.3f}s")


def scan_fps(mode: str, fps: float, dense_fps: float, video_path: Path) -> float:
    if mode in {"native", "native-windowed"}:
        return video_fps(video_path)
    if mode in {"dense", "dense-windowed"}:
        return max(fps, dense_fps)
    return fps


def timestamps_for_window(start: float, end: float, fps: float) -> list[float]:
    if end <= start:
        return []
    count = max(1, int(math.ceil((end - start) * fps)))
    return [round(start + index / fps, 3) for index in range(count)]
