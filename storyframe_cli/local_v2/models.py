from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OcrBox:
    text: str
    confidence: float
    x: float
    y: float
    width: float
    height: float
    page_width: float
    page_height: float
    ink_score: float = 0.0

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass
class FrameObservation:
    frame_path: str
    timestamp: float
    text: str
    normalized_text: str
    boxes: list[OcrBox]
    avg_confidence: float
    avg_ink_score: float
    word_count: int
    visual_hash: str = ""
    page_id: str = ""
    ad_boxes: list[OcrBox] = field(default_factory=list)


@dataclass
class TranscriptUnit:
    unit_id: str
    text: str
    normalized_text: str
    start: float
    end: float
    source: str


@dataclass
class SelectedFrame:
    unit_id: str
    timestamp: float
    frame_path: str
    transcript: str
    normalized_text: str
    score: float
    status: str
    warnings: list[str] = field(default_factory=list)
    output_source: str = "original"
    page_id: str = ""


@dataclass
class PageInterval:
    page_id: str
    start: float
    end: float
    source: str
