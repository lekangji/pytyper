"""Constrained inference for the trained human typing model."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path

import torch
from torch import Tensor

from model.checkpoint import read_checkpoint
from model.controls import SECONDS_PER_CHARACTER_PER_WPM, TypingControls
from model.keyboard import transition_features
from model.network import (
    BUFFER_CONTEXT,
    PAD_ID,
    PAUSE_THRESHOLD,
    PRINTABLE_KEYS,
    Action,
    TypingEventModel,
    char_id,
)
from model.timing import (
    MAX_SAMPLED_GAP,
    MIN_SAMPLED_GAP,
    component_log_masses,
    gap_features,
    sample_conditioned_log_gap,
)


# WPM, gap power adjustment, and long-gap slope adjustment, fitted to matched-speed recordings.
SPLIT_SPEED_CALIBRATION = (
    (20, 0.0, 0.0),
    (40, -0.075, 0.25),
    (60, -0.025, 0.70),
    (80, 0.025, 0.85),
    (100, 0.025, -0.05),
    (120, 0.025, 0.0),
    (160, 0.0, 0.0),
)


@dataclass(frozen=True)
class NeuralKeyEvent:
    key: str
    delay_before: float
    hold_seconds: float = 0.0


def _encode_state(
    target: str,
    target_ids: Tensor,
    cursor: int,
    wrong_suffix: str,
    previous_key: str,
    previous_action: int,
    style: Tensor,
    target_width: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    suffix = (target[max(0, cursor - BUFFER_CONTEXT):cursor] + wrong_suffix)[-BUFFER_CONTEXT:]
    buffer_ids = [PAD_ID] * (BUFFER_CONTEXT - len(suffix)) + [char_id(value) for value in suffix]
    return (
        target_ids[..., cursor:cursor + target_width],
        torch.tensor([[buffer_ids]], dtype=torch.long),
        torch.tensor([[[char_id(previous_key)]]], dtype=torch.long),
        torch.tensor([[[previous_action]]], dtype=torch.long),
        style.reshape(1, 1, -1),
        torch.tensor([[transition_features(previous_key, target[cursor] if cursor < len(target) else "")]], dtype=torch.float32),
    )


class NeuralTypingRuntime:
    """Load a trained bundle and generate exact, controllable key streams."""

    def __init__(
        self, model_path: str | Path, seed: int | None = None, *, spec_path: str | Path | None = None
    ) -> None:
        self.path = Path(model_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Restore the trained typing checkpoint at {self.path}.")
        bundle, architecture, self.decoder_version = read_checkpoint(self.path, spec_path)
        self.model = TypingEventModel(**architecture)
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
        return self.plan(target, controls)[1]

    def plan(
        self, target: str, controls: TypingControls | None = None
    ) -> tuple[list[NeuralKeyEvent], list[NeuralKeyEvent]]:
        """Return sampled events and the same events after speed calibration."""
        if not target:
            return [], []
        controls = (controls or TypingControls()).validated()
        if self.decoder_version in {"global-mixture-v3", "global-mixture-v4"} and controls.wpm > 160:
            controls = replace(controls, wpm=160)
        style = self._controlled_style(controls)
        error_offset = self._error_logit_offset(controls.error_rate)
        cursor = 0
        wrong_suffix = ""
        previous_key = ""
        previous_action = len(Action)
        previous_delay = 0.0
        hidden = None
        events: list[NeuralKeyEvent] = []
        max_events = len(target) * 3 + 32

        with torch.inference_mode():
            target_width = self.model.target_left + self.model.target_right
            padded_ids = [PAD_ID] * self.model.target_left
            padded_ids.extend(char_id(key) for key in target)
            padded_ids.extend([PAD_ID] * self.model.target_right)
            target_ids = torch.tensor([[padded_ids]], dtype=torch.long)
            for step in range(max_events):
                if cursor == len(target):
                    break
                encoded = _encode_state(
                    target, target_ids, cursor, wrong_suffix, previous_key, previous_action, style, target_width,
                )
                outputs = self._predict_event(encoded, hidden, previous_delay)
                action_logits, wrong_logits, _, _, _, hidden = outputs[:6]
                action_logits = action_logits[0, 0] / self.temperature
                divergence = len(wrong_suffix)

                if divergence >= 3:
                    action = Action.BACKSPACE
                else:
                    if divergence > 0:
                        action_logits[Action.CORRECT] = -torch.inf
                    else:
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
                    key = target[cursor]
                    cursor += 1
                elif action == Action.BACKSPACE:
                    key = "BACKSPACE"
                    wrong_suffix = wrong_suffix[:-1]
                else:
                    distribution = torch.softmax(wrong_logits[0, 0] / self.temperature, dim=-1)
                    if self.decoder_version != "global-mixture-v3" and divergence == 0 and target[cursor] in PRINTABLE_KEYS:
                        # A wrong action must not emit the correct character.
                        distribution[PRINTABLE_KEYS.index(target[cursor])] = 0.0
                        distribution /= distribution.sum()
                    key_index = torch.multinomial(distribution, 1, generator=self._torch_generator).item()
                    key = PRINTABLE_KEYS[key_index]
                    # The v3 decoder can sample the correct character as a wrong key.
                    if divergence == 0 and key == target[cursor]:
                        cursor += 1
                    else:
                        wrong_suffix += key

                delay, hold = self._sample_event_timing(outputs, action, controls, first=step == 0)
                events.append(NeuralKeyEvent(key, delay, hold))
                previous_key = "" if key == "BACKSPACE" else key
                previous_action = int(action)
                previous_delay = delay

            while cursor < len(target):
                encoded = _encode_state(
                    target, target_ids, cursor, wrong_suffix, previous_key, previous_action, style, target_width,
                )
                outputs = self._predict_event(encoded, hidden, previous_delay)
                hidden = outputs[5]
                if wrong_suffix:
                    action, key = Action.BACKSPACE, "BACKSPACE"
                    wrong_suffix = wrong_suffix[:-1]
                else:
                    action, key = Action.CORRECT, target[cursor]
                    cursor += 1
                delay, hold = self._sample_event_timing(outputs, action, controls, first=False)
                events.append(NeuralKeyEvent(key, delay, hold))
                previous_key = "" if key == "BACKSPACE" else key
                previous_action = int(action)
                previous_delay = delay

        if self.model.event_model_v5:
            return events, self._calibrate_event_timing(
                events, len(target), controls, tune_pauses=not self.model.unified_timing,
            )
        return events, self._calibrate_timing(events, len(target), controls)

    def _predict_event(
        self, encoded: tuple[Tensor, ...], hidden: Tensor | None, previous_delay: float,
    ) -> tuple[Tensor | None, ...]:
        history = None
        if self.model.use_previous_gap:
            previous = encoded[4].new_tensor([[previous_delay]])
            history = gap_features(previous, previous > 0)
        return self.model(*encoded[:5], hidden, keyboard_features=encoded[5], previous_gap=history)

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
        self, outputs: tuple[Tensor | None, ...], action: Action, controls: TypingControls, first: bool
    ) -> tuple[float, float]:
        _, _, mix, means, scales, _, *extra = outputs
        if not self.model.event_model_v5:
            delay = 0.0 if first else self._sample_delay(mix[0, 0], means[0, 0], scales[0, 0])
            return delay, 0.0
        pause_logits, hold_means, hold_scales, *hold_logits = extra
        delay = 0.0
        if not first:
            if self.model.unified_timing:
                delay = self._sample_mixture_delay(
                    mix[0, 0, action], means[0, 0, action], scales[0, 0, action], controls.pause_tendency,
                )
            else:
                pause_probability = torch.sigmoid(
                    pause_logits[0, 0, action] + math.log(controls.pause_tendency)
                ).item()
                pause = self._rng.random() < pause_probability
                mode = int(pause)
                delay = self._sample_mode_delay(
                    mix[0, 0, action, mode], means[0, 0, action, mode],
                    scales[0, 0, action, mode], pause,
                )
        hold_key = (0, 0, action)
        if hold_logits:
            hold = self._sample_lognormal_mixture(hold_logits[0][hold_key], hold_means[hold_key], hold_scales[hold_key])
        else:
            hold = self._sample_lognormal(hold_means[hold_key], hold_scales[hold_key])
        # Keep emitted holds below common OS key-repeat thresholds. Repeated
        # characters or backspaces would invalidate the planned visible buffer.
        return delay, min(0.18, max(0.02, hold))

    def _sample_mixture_delay(
        self, logits: Tensor, means: Tensor, scales: Tensor, pause_tendency: float
    ) -> float:
        if pause_tendency == 1.0:
            return self._sample_delay(logits, means, scales)
        means, scales = means.double(), scales.double()
        weighted = logits.double().log_softmax(0)[:, None] + component_log_masses(means, scales)
        short_mass, long_mass = weighted.logsumexp(0)
        long_gap = self._rng.random() < (long_mass - short_mass + math.log(pause_tendency)).sigmoid().item()
        log_delay = sample_conditioned_log_gap(
            weighted[:, int(long_gap)], means, scales, long_gap, self._rng, self._torch_generator,
        )
        delay = min(MAX_SAMPLED_GAP, max(MIN_SAMPLED_GAP, math.exp(log_delay)))
        return max(PAUSE_THRESHOLD, delay) if long_gap else min(math.nextafter(PAUSE_THRESHOLD, 0.0), delay)

    def _sample_mode_delay(
        self, logits: Tensor, means: Tensor, scales: Tensor, pause: bool
    ) -> float:
        for _ in range(8):
            delay = self._sample_delay(logits, means, scales)
            if (delay >= 0.5) == pause:
                return delay
        return max(0.5, delay) if pause else min(0.499, delay)

    @staticmethod
    def _calibrate_event_timing(
        events: list[NeuralKeyEvent], target_characters: int, controls: TypingControls,
        *, tune_pauses: bool = False,
    ) -> list[NeuralKeyEvent]:
        # Fit to recorded validation gap quantiles. Faster plans need less spread in log-gap space.
        fitted_power = min(1.15, max(0.65, 0.83 - 0.28 * math.log(controls.wpm / 75.0)))
        tail_adjustment = 0.0
        if tune_pauses:
            for left, right in pairwise(SPLIT_SPEED_CALIBRATION):
                if left[0] <= controls.wpm <= right[0]:
                    t = (controls.wpm - left[0]) / (right[0] - left[0])
                    fitted_power += left[1] + t * (right[1] - left[1])
                    tail_adjustment = left[2] + t * (right[2] - left[2])
                    break
        power = fitted_power * controls.rhythm_variation
        shaped = [event.delay_before ** power for event in events]
        planned_seconds = math.fsum(shaped)
        if not planned_seconds:
            return events

        # Match overall target speed after including pauses and corrections.
        target_seconds = target_characters * SECONDS_PER_CHARACTER_PER_WPM / controls.wpm
        scale = target_seconds / planned_seconds
        if tail_adjustment:
            # Bend the long-gap tail continuously, then restore the requested total duration.
            scaled = [delay * scale for delay in shaped]
            shaped = [delay + tail_adjustment * max(0.0, delay - 0.35) for delay in scaled]
            scale = target_seconds / math.fsum(shaped)
        return [
            NeuralKeyEvent(event.key, delay * scale, event.hold_seconds)
            for event, delay in zip(events, shaped)
        ]

    def _error_logit_offset(self, target_rate: float) -> float:
        if target_rate <= 0.0:
            return -math.inf
        reference = min(0.95, max(0.001, self.reference_error_rate))
        return math.log(target_rate / (1.0 - target_rate)) - math.log(reference / (1.0 - reference))

    def _guide_error_probability(self, action_logits: Tensor, target_rate: float, offset: float) -> None:
        if target_rate <= 0.0:
            action_logits[Action.WRONG] = -torch.inf
            return
        if self.decoder_version == "global-mixture-v3":
            other = torch.stack((action_logits[Action.CORRECT], action_logits[Action.BACKSPACE]))
            action_logits[Action.WRONG] = torch.logsumexp(other, dim=0) + math.log(target_rate / (1.0 - target_rate))
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
        value = self._sample_lognormal_mixture(logits, means, log_scales)
        return min(MAX_SAMPLED_GAP, max(MIN_SAMPLED_GAP, value))

    def _sample_lognormal_mixture(self, logits: Tensor, means: Tensor, log_scales: Tensor) -> float:
        component = torch.multinomial(
            torch.softmax(logits, dim=-1), 1, generator=self._torch_generator
        ).item()
        return self._sample_lognormal(means[component], log_scales[component])

    def _sample_lognormal(self, mean: Tensor, log_scale: Tensor) -> float:
        noise = torch.randn((), generator=self._torch_generator).item()
        return math.exp(mean.item() + math.exp(log_scale.item()) * noise)
