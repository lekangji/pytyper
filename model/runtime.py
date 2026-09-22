"""Constrained inference for the trained human typing model."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from model.controls import SECONDS_PER_CHARACTER_PER_WPM, TypingControls
from model.keyboard import transition_features
from model.network import (
    BUFFER_CONTEXT,
    PAD_ID,
    PRINTABLE_KEYS,
    Action,
    TypingEventModel,
    char_id,
    longest_common_prefix,
)


@dataclass(frozen=True)
class NeuralKeyEvent:
    key: str
    delay_before: float
    hold_seconds: float = 0.0


def _encode_state(
    target: str,
    buffer: str,
    previous_key: str,
    previous_action: int,
    style: Tensor,
    target_left: int,
    target_right: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    cursor = longest_common_prefix(target, buffer)
    target_ids = [
        char_id(target[index]) if 0 <= index < len(target) else PAD_ID
        for index in range(cursor - target_left, cursor + target_right)
    ]
    suffix = buffer[-BUFFER_CONTEXT:]
    buffer_ids = [PAD_ID] * (BUFFER_CONTEXT - len(suffix)) + [char_id(value) for value in suffix]
    return (
        torch.tensor([[target_ids]], dtype=torch.long),
        torch.tensor([[buffer_ids]], dtype=torch.long),
        torch.tensor([[[char_id(previous_key)]]], dtype=torch.long),
        torch.tensor([[[previous_action]]], dtype=torch.long),
        style.reshape(1, 1, -1),
        torch.tensor([[transition_features(previous_key, target[cursor] if cursor < len(target) else "")]], dtype=torch.float32),
    )


class NeuralTypingRuntime:
    """Load a trained bundle and generate exact, controllable key streams."""

    def __init__(self, model_path: str | Path, seed: int | None = None) -> None:
        self.path = Path(model_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Trained typing model not found at {self.path}. Restore models/pytyper.pt."
            )
        bundle = torch.load(self.path, map_location="cpu", weights_only=False)
        self.model = TypingEventModel(**bundle["architecture"])
        self.model.load_state_dict(bundle["model_state"])
        self.model.eval()
        self.styles = bundle["style_bank"].float()
        self.style_mean = bundle["style_mean"].float()
        self.style_std = bundle["style_std"].float()
        self.temperature = float(bundle.get("sampling_temperature", 0.9))
        self.reference_error_rate = float(bundle.get("training_mistake_rate", self.style_mean[3].item()))
        self._rng = random.Random(seed)
        self._torch_generator = torch.Generator().manual_seed(self._rng.randrange(2**31))

    def generate(self, target: str, controls: TypingControls | None = None) -> list[NeuralKeyEvent]:
        if not target:
            return []
        controls = (controls or TypingControls()).validated()
        style = self._controlled_style(controls)
        error_offset = self._error_logit_offset(controls.error_rate)
        buffer = ""
        previous_key = ""
        previous_action = len(Action)
        hidden = None
        events: list[NeuralKeyEvent] = []
        max_events = len(target) * 3 + 32

        with torch.inference_mode():
            for step in range(max_events):
                if buffer == target:
                    break
                encoded = _encode_state(
                    target, buffer, previous_key, previous_action, style,
                    self.model.target_left, self.model.target_right,
                )
                outputs = self.model(*encoded[:5], hidden, keyboard_features=encoded[5])
                action_logits, wrong_logits, mix_logits, means, scales, hidden = outputs[:6]
                action_logits = action_logits[0, 0] / self.temperature
                prefix_length = longest_common_prefix(target, buffer)
                divergence = len(buffer) - prefix_length

                if divergence >= 3:
                    action = Action.BACKSPACE
                else:
                    if divergence > 0:
                        action_logits[Action.CORRECT] = -torch.inf
                    elif len(buffer) < len(target):
                        # A correction is only valid after a wrong key. This also
                        # makes a requested zero-error session genuinely clean.
                        action_logits[Action.BACKSPACE] = -torch.inf
                        self._guide_error_probability(action_logits, controls.error_rate, error_offset)
                    action = Action(
                        torch.multinomial(
                            torch.softmax(action_logits, dim=-1), 1, generator=self._torch_generator
                        ).item()
                    )

                if action == Action.CORRECT:
                    key = target[len(buffer)]
                    buffer += key
                elif action == Action.BACKSPACE:
                    key = "BACKSPACE"
                    buffer = buffer[:-1]
                else:
                    distribution = torch.softmax(wrong_logits[0, 0] / self.temperature, dim=-1)
                    if divergence == 0 and len(buffer) < len(target) and target[len(buffer)] in PRINTABLE_KEYS:
                        # A wrong action must not emit the correct character.
                        distribution[PRINTABLE_KEYS.index(target[len(buffer)])] = 0.0
                        distribution /= distribution.sum()
                    key_index = torch.multinomial(distribution, 1, generator=self._torch_generator).item()
                    key = PRINTABLE_KEYS[key_index]
                    buffer += key

                delay, hold = self._sample_event_timing(outputs, action, controls, first=step == 0)
                events.append(NeuralKeyEvent(key, delay, hold))
                previous_key = "" if key == "BACKSPACE" else key
                previous_action = int(action)

            while buffer != target:
                encoded = _encode_state(
                    target, buffer, previous_key, previous_action, style,
                    self.model.target_left, self.model.target_right,
                )
                outputs = self.model(*encoded[:5], hidden, keyboard_features=encoded[5])
                _, _, mix_logits, means, scales, hidden = outputs[:6]
                prefix_length = longest_common_prefix(target, buffer)
                if len(buffer) > prefix_length:
                    action, key, buffer = Action.BACKSPACE, "BACKSPACE", buffer[:-1]
                else:
                    key = target[len(buffer)]
                    action, buffer = Action.CORRECT, buffer + key
                delay, hold = self._sample_event_timing(outputs, action, controls, first=False)
                events.append(NeuralKeyEvent(key, delay, hold))
                previous_key = "" if key == "BACKSPACE" else key
                previous_action = int(action)

        if self.model.event_model_v5:
            return self._calibrate_motor_timing(events, len(target), controls)
        return self._calibrate_timing(events, len(target), controls)

    def _controlled_style(self, controls: TypingControls) -> Tensor:
        if self.model.event_model_v5:
            desired = math.log(SECONDS_PER_CHARACTER_PER_WPM / controls.wpm)
            raw_bank = self.styles * self.style_std + self.style_mean
            distances = (raw_bank[:, 0] - desired).square() + (
                (raw_bank[:, 3] - controls.error_rate) * 4.0
            ).square()
            candidates = torch.topk(distances, min(64, len(distances)), largest=False).indices
            normalized = self.styles[candidates[self._rng.randrange(len(candidates))]]
        else:
            normalized = self.styles[self._rng.randrange(len(self.styles))]
        raw = normalized * self.style_std + self.style_mean
        median_delay = SECONDS_PER_CHARACTER_PER_WPM / controls.wpm
        learned_tail_ratio = max(1.0, math.exp(float(raw[1] - raw[0])))
        correction_ratio = float(raw[2] / raw[3]) if raw[3] > 1e-5 else 1.0
        raw[0] = math.log(median_delay)
        raw[1] = math.log(median_delay * learned_tail_ratio * controls.rhythm_variation)
        raw[3] = controls.error_rate
        raw[2] = min(0.30, controls.error_rate * max(0.5, min(2.0, correction_ratio)))
        return ((raw - self.style_mean) / self.style_std).clamp(-4.0, 4.0)

    def _sample_event_timing(
        self, outputs: tuple[Tensor, ...], action: Action, controls: TypingControls, first: bool
    ) -> tuple[float, float]:
        _, _, mix, means, scales, _, *extra = outputs
        if not self.model.event_model_v5:
            delay = 0.0 if first else self._sample_delay(mix[0, 0], means[0, 0], scales[0, 0])
            return delay, 0.0
        pause_logits, hold_means, hold_scales = extra
        pause_probability = torch.sigmoid(
            pause_logits[0, 0, action] + math.log(controls.pause_tendency)
        ).item()
        pause = not first and self._rng.random() < pause_probability
        mode = int(pause)
        delay = 0.0 if first else self._sample_mode_delay(
            mix[0, 0, action, mode], means[0, 0, action, mode],
            scales[0, 0, action, mode], pause,
        )
        noise = torch.randn((), generator=self._torch_generator).item()
        hold = math.exp(
            hold_means[0, 0, action].item()
            + math.exp(hold_scales[0, 0, action].item()) * noise
        )
        # Keep emitted holds below common OS key-repeat thresholds. Repeated
        # characters or backspaces would invalidate the planned visible buffer.
        return delay, min(0.18, max(0.02, hold))

    def _sample_mode_delay(
        self, logits: Tensor, means: Tensor, scales: Tensor, pause: bool
    ) -> float:
        for _ in range(8):
            delay = self._sample_delay(logits, means, scales)
            if (delay >= 0.5) == pause:
                return delay
        return max(0.5, delay) if pause else min(0.499, delay)

    @staticmethod
    def _calibrate_motor_timing(
        events: list[NeuralKeyEvent], target_characters: int, controls: TypingControls
    ) -> list[NeuralKeyEvent]:
        motor = sorted(event.delay_before for event in events[1:] if 0 < event.delay_before < 0.5)
        if not motor:
            return events
        median = motor[len(motor) // 2]
        scale = min(2.0, max(0.5, (SECONDS_PER_CHARACTER_PER_WPM / controls.wpm) / median))
        result = []
        for event in events:
            delay = event.delay_before
            if 0 < delay < 0.5:
                delay = min(0.499, max(0.008, median * (delay / median) ** controls.rhythm_variation * scale))
            result.append(NeuralKeyEvent(event.key, delay, event.hold_seconds))
        # Match overall target speed after pauses and corrections while keeping
        # their learned relative timing intact.
        planned_seconds = sum(event.delay_before for event in result)
        if planned_seconds <= 0:
            return result
        adjustment = (target_characters * SECONDS_PER_CHARACTER_PER_WPM / controls.wpm) / planned_seconds
        return [NeuralKeyEvent(event.key, event.delay_before * adjustment, event.hold_seconds) for event in result]

    def _error_logit_offset(self, target_rate: float) -> float:
        if target_rate <= 0.0:
            return -math.inf
        reference = min(0.95, max(0.001, self.reference_error_rate))
        return math.log(target_rate / (1.0 - target_rate)) - math.log(reference / (1.0 - reference))

    @staticmethod
    def _guide_error_probability(action_logits: Tensor, target_rate: float, offset: float) -> None:
        if target_rate <= 0.0:
            action_logits[Action.WRONG] = -torch.inf
            return
        # A shared log-odds shift preserves the network's relative preference
        # for making mistakes at difficult versus easy positions.
        action_logits[Action.WRONG] += offset

    @staticmethod
    def _calibrate_timing(
        events: list[NeuralKeyEvent], target_characters: int, controls: TypingControls
    ) -> list[NeuralKeyEvent]:
        if len(events) < 2:
            return events
        positive = [event.delay_before for event in events[1:] if event.delay_before > 0]
        if not positive:
            return events
        median = sorted(positive)[len(positive) // 2]
        shaped = [0.0]
        for event in events[1:]:
            ratio = max(1e-4, event.delay_before / median)
            shaped.append(median * ratio ** controls.rhythm_variation)
        desired_seconds = target_characters * SECONDS_PER_CHARACTER_PER_WPM / controls.wpm
        scale = desired_seconds / sum(shaped)
        return [
            NeuralKeyEvent(event.key, delay * scale)
            for event, delay in zip(events, shaped)
        ]

    def _sample_delay(self, logits: Tensor, means: Tensor, log_scales: Tensor) -> float:
        component = torch.multinomial(
            torch.softmax(logits, dim=-1), 1, generator=self._torch_generator
        ).item()
        noise = torch.randn((), generator=self._torch_generator).item()
        value = math.exp(means[component].item() + math.exp(log_scales[component].item()) * noise)
        return min(5.0, max(0.008, value))
