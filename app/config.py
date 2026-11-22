"""Configuration shared by the UI and typing worker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from model.controls import TypingControls


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "gelu-16d-hold-mixture.pt"


@dataclass(frozen=True)
class TypingConfig:
    text: str
    start_delay_ms: int
    controls: TypingControls
    model_path: str = str(DEFAULT_MODEL_PATH)
