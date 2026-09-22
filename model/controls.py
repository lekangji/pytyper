"""Lightweight public controls that do not import the PyTorch runtime."""

from __future__ import annotations

from dataclasses import dataclass

SECONDS_PER_CHARACTER_PER_WPM = 12.0  # 60 seconds per five-character word.


@dataclass(frozen=True)
class TypingControls:
    """Friendly controls mapped into the model's learned session style."""

    wpm: int = 75
    error_rate: float = 0.035
    rhythm_variation: float = 1.0
    pause_tendency: float = 1.0
    physical_keys: bool = True

    def validated(self) -> "TypingControls":
        return TypingControls(
            wpm=min(200, max(20, int(self.wpm))),
            error_rate=min(0.20, max(0.0, float(self.error_rate))),
            rhythm_variation=min(1.5, max(0.5, float(self.rhythm_variation))),
            pause_tendency=min(2.0, max(0.25, float(self.pause_tendency))),
            physical_keys=bool(self.physical_keys),
        )
