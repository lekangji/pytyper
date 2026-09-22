"""Live metrics for an emitted typing plan."""

from __future__ import annotations

from dataclasses import dataclass
import time

from model.controls import SECONDS_PER_CHARACTER_PER_WPM


@dataclass(frozen=True)
class ProgressSnapshot:
    percent: int
    wpm: float
    accuracy: float
    seconds_left: float


class ProgressTracker:
    """Measure delivered events, not requested speed or generated event count."""

    def __init__(self, target: str, planned_seconds: float, started_at: float | None = None) -> None:
        self.target = target
        self.planned_seconds = max(0.0, planned_seconds)
        self.started_at = time.monotonic() if started_at is None else started_at
        self.planned_elapsed = 0.0
        self.paused_seconds = 0.0
        self.buffer_length = 0
        self.correct_prefix = 0
        self.typed_keys = 0
        self.wrong_keys = 0

    def snapshot(self, now: float | None = None) -> ProgressSnapshot:
        now = time.monotonic() if now is None else now
        elapsed = max(0.0, now - self.started_at - self.paused_seconds)
        percent = round(100 * self.correct_prefix / len(self.target)) if self.target else 100
        wpm = self.correct_prefix * SECONDS_PER_CHARACTER_PER_WPM / elapsed if elapsed >= 1.0 else 0.0
        accuracy = 100.0 * (self.typed_keys - self.wrong_keys) / self.typed_keys if self.typed_keys else 100.0
        remaining_plan = max(0.0, self.planned_seconds - self.planned_elapsed)
        pace = max(1.0, elapsed / self.planned_elapsed) if self.planned_elapsed >= 1.0 else 1.0
        return ProgressSnapshot(percent, wpm, accuracy, remaining_plan * pace)

    def add_pause(self, seconds: float) -> None:
        self.paused_seconds += max(0.0, seconds)

    def advance(self, key: str, delay_before: float, now: float | None = None) -> ProgressSnapshot:
        self.planned_elapsed += max(0.0, delay_before)
        if key == "BACKSPACE":
            self.buffer_length = max(0, self.buffer_length - 1)
            self.correct_prefix = min(self.correct_prefix, self.buffer_length)
        else:
            correct = (
                self.buffer_length == self.correct_prefix
                and self.correct_prefix < len(self.target)
                and key == self.target[self.correct_prefix]
            )
            self.buffer_length += 1
            self.typed_keys += 1
            self.wrong_keys += not correct
            self.correct_prefix += int(correct)
        return self.snapshot(now)
