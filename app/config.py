"""Configuration shared by the UI and typing worker."""

from __future__ import annotations

from dataclasses import dataclass

from model.controls import TypingControls


@dataclass(frozen=True)
class TypingConfig:
    text: str
    start_delay_ms: int
    controls: TypingControls
    model_path: str = "models/pytyper.pt"
