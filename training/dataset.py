"""Aalto 136M Keystrokes parsing and neural training examples.

Dataset source: https://userinterfaces.aalto.fi/136Mkeystrokes/
License: non-commercial use with attribution (see MODEL_CARD.md).
"""

from __future__ import annotations

import csv
import io
import math
import statistics
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor

from model.keyboard import transition_features
from model.network import (
    BUFFER_CONTEXT,
    KEY_TO_ID,
    PAD_ID,
    PAUSE_THRESHOLD,
    TARGET_LEFT,
    TARGET_RIGHT,
    Action,
    char_id,
    longest_common_prefix,
)


@dataclass(frozen=True)
class LoggedEvent:
    key: str
    press_ms: int
    release_ms: int | None = None


@dataclass(frozen=True)
class Trial:
    participant_id: str
    target: str
    events: tuple[LoggedEvent, ...]


@dataclass
class TrainingSequence:
    target_context: Tensor
    buffer_context: Tensor
    previous_key: Tensor
    previous_action: Tensor
    keyboard_features: Tensor
    actions: Tensor
    wrong_keys: Tensor
    delays: Tensor
    timing_mask: Tensor
    pauses: Tensor
    holds: Tensor
    hold_mask: Tensor
    style_raw: Tensor
    participant_id: str


def decode_single_zip_entry(path: str | Path) -> tuple[str, bytes]:
    payload = Path(path).read_bytes()
    header = struct.unpack_from("<4s5H3L2H", payload, 0)
    if header[0] != b"PK\x03\x04":
        raise ValueError(f"{path} does not begin with a ZIP local-file record")
    method, compressed_size, name_length, extra_length = header[3], header[7], header[9], header[10]
    name_start = 30
    data_start = name_start + name_length + extra_length
    name = payload[name_start : name_start + name_length].decode("utf-8")
    compressed = payload[data_start : data_start + compressed_size]
    data = zlib.decompress(compressed, -15) if method == 8 else compressed
    return name, data


def iter_zip_segment(path: str | Path) -> Iterator[tuple[str, bytes]]:
    """Yield complete local-file records from a contiguous HTTP ZIP range."""

    payload = Path(path).read_bytes()
    offset = 0
    while offset + 30 <= len(payload) and payload[offset : offset + 4] == b"PK\x03\x04":
        header = struct.unpack_from("<4s5H3L2H", payload, offset)
        method, compressed_size, name_length, extra_length = header[3], header[7], header[9], header[10]
        name_start = offset + 30
        data_start = name_start + name_length + extra_length
        data_end = data_start + compressed_size
        if data_end > len(payload):
            break
        name = payload[name_start : name_start + name_length].decode("utf-8")
        compressed = payload[data_start:data_end]
        data = zlib.decompress(compressed, -15) if method == 8 else compressed
        yield name, data
        offset = data_end


def eligible_participants(metadata_entry: str | Path) -> set[str]:
    _, raw = decode_single_zip_entry(metadata_entry)
    rows = csv.DictReader(io.StringIO(raw.decode("utf-8", "replace"), newline=""), delimiter="\t")
    return {
        row["PARTICIPANT_ID"]
        for row in rows
        if row.get("LAYOUT") == "qwerty" and row.get("KEYBOARD_TYPE") in {"full", "laptop"}
    }


def trials_from_tsv(raw: bytes, eligible: set[str]) -> list[Trial]:
    rows = csv.DictReader(io.StringIO(raw.decode("utf-8", "replace"), newline=""), delimiter="\t")
    grouped: dict[str, tuple[str, str, list[LoggedEvent]]] = {}
    participant_id = ""
    for row in rows:
        participant_id = row["PARTICIPANT_ID"]
        if participant_id not in eligible:
            continue
        section = row["TEST_SECTION_ID"]
        target = row["SENTENCE"]
        final_input = row["USER_INPUT"]
        group = grouped.setdefault(section, (target, final_input, []))
        key = "BACKSPACE" if row["LETTER"] in {"BKSP", "BACKSPACE"} else row["LETTER"]
        if key == "BACKSPACE" or key in KEY_TO_ID:
            # A small number of source files serialize millisecond timestamps in
            # scientific notation, despite the column being integral.
            press = int(float(row["PRESS_TIME"]))
            release = int(float(row["RELEASE_TIME"])) if row.get("RELEASE_TIME") else None
            group[2].append(LoggedEvent(key, press, release))

    result = []
    for target, final_input, events in grouped.values():
        if target == final_input and events:
            result.append(Trial(participant_id, target, tuple(events)))
    return result


def _context_ids(target: str, buffer: str, target_left: int, target_right: int) -> tuple[list[int], list[int]]:
    cursor = longest_common_prefix(target, buffer)
    target_ids = [
        char_id(target[index]) if 0 <= index < len(target) else PAD_ID
        for index in range(cursor - target_left, cursor + target_right)
    ]
    suffix = buffer[-BUFFER_CONTEXT:]
    buffer_ids = [PAD_ID] * (BUFFER_CONTEXT - len(suffix)) + [char_id(key) for key in suffix]
    return target_ids, buffer_ids


def build_sequence(
    trial: Trial, target_left: int = TARGET_LEFT, target_right: int = TARGET_RIGHT
) -> TrainingSequence | None:
    buffer = ""
    previous_key = ""
    previous_action = len(Action)
    previous_time: int | None = None
    contexts: list[list[int]] = []
    buffers: list[list[int]] = []
    previous_keys: list[int] = []
    previous_actions: list[int] = []
    keyboard_features: list[tuple[float, ...]] = []
    actions: list[int] = []
    wrong_keys: list[int] = []
    delays: list[float] = []
    timing_mask: list[bool] = []
    pauses: list[bool] = []
    holds: list[float] = []
    hold_mask: list[bool] = []
    rollovers: list[bool] = []
    previous_release: int | None = None

    for event in trial.events:
        target_context, buffer_context = _context_ids(trial.target, buffer, target_left, target_right)
        cursor = longest_common_prefix(trial.target, buffer)
        upcoming = trial.target[cursor] if cursor < len(trial.target) else ""
        contexts.append(target_context)
        buffers.append(buffer_context)
        previous_keys.append(char_id(previous_key))
        previous_actions.append(previous_action)
        keyboard_features.append(transition_features(previous_key, upcoming))

        if event.key == "BACKSPACE":
            action = Action.BACKSPACE
            wrong_key = 0
            buffer = buffer[:-1]
        elif buffer == trial.target[: len(buffer)] and len(buffer) < len(trial.target) and event.key == trial.target[len(buffer)]:
            action = Action.CORRECT
            wrong_key = 0
            buffer += event.key
        else:
            action = Action.WRONG
            wrong_key = KEY_TO_ID[event.key] - 2
            buffer += event.key

        actions.append(int(action))
        wrong_keys.append(wrong_key)
        if previous_time is None:
            delays.append(0.0)
            timing_mask.append(False)
            pauses.append(False)
        else:
            delay = min(8.0, max(0.008, (event.press_ms - previous_time) / 1000.0))
            delays.append(delay)
            timing_mask.append(True)
            pauses.append(delay >= PAUSE_THRESHOLD)
            if previous_release is not None:
                rollovers.append(event.press_ms < previous_release)
        valid_hold = event.release_ms is not None and 8 <= event.release_ms - event.press_ms <= 2000
        holds.append(min(2.0, max(0.008, (event.release_ms - event.press_ms) / 1000.0)) if valid_hold else 0.0)
        hold_mask.append(valid_hold)
        previous_release = event.release_ms
        previous_time = event.press_ms
        previous_key = "" if event.key == "BACKSPACE" else event.key
        previous_action = int(action)

    if buffer != trial.target or len(actions) < 3:
        return None
    valid_delays = [delay for delay, valid in zip(delays, timing_mask) if valid]
    correction_rate = actions.count(Action.BACKSPACE) / len(actions)
    mistake_rate = actions.count(Action.WRONG) / len(actions)
    valid_holds = [value for value, valid in zip(holds, hold_mask) if valid]
    if not valid_holds:
        return None
    style = torch.tensor(
        [
            math.log(statistics.median(valid_delays)),
            math.log(statistics.quantiles(valid_delays, n=10)[8]) if len(valid_delays) >= 10 else math.log(max(valid_delays)),
            correction_rate,
            mistake_rate,
            math.log(statistics.median(valid_holds)),
            sum(rollovers) / len(rollovers) if rollovers else 0.0,
        ],
        dtype=torch.float32,
    )
    return TrainingSequence(
        torch.tensor(contexts, dtype=torch.uint8),
        torch.tensor(buffers, dtype=torch.uint8),
        torch.tensor(previous_keys, dtype=torch.uint8).unsqueeze(-1),
        torch.tensor(previous_actions, dtype=torch.uint8).unsqueeze(-1),
        torch.tensor(keyboard_features, dtype=torch.float16),
        torch.tensor(actions, dtype=torch.uint8),
        torch.tensor(wrong_keys, dtype=torch.uint8),
        torch.tensor(delays, dtype=torch.float32),
        torch.tensor(timing_mask, dtype=torch.bool),
        torch.tensor(pauses, dtype=torch.bool),
        torch.tensor(holds, dtype=torch.float32),
        torch.tensor(hold_mask, dtype=torch.bool),
        style,
        trial.participant_id,
    )


def load_sequences(
    segment_path: str | Path,
    metadata_entry: str | Path,
    max_participants: int | None = None,
    target_left: int = TARGET_LEFT,
    target_right: int = TARGET_RIGHT,
) -> list[TrainingSequence]:
    eligible = eligible_participants(metadata_entry)
    sequences: list[TrainingSequence] = []
    participants = 0
    for name, raw in iter_zip_segment(segment_path):
        if not name.endswith("_keystrokes.txt"):
            continue
        participant_id = Path(name).name.split("_", 1)[0]
        if participant_id not in eligible:
            continue
        try:
            trials = trials_from_tsv(raw, eligible)
        except (csv.Error, KeyError, TypeError, ValueError):
            # The published archive notes that a small number of participant
            # logs are corrupt; skipping is preferable to repairing behavior.
            continue
        for trial in trials:
            sequence = build_sequence(trial, target_left, target_right)
            if sequence is not None:
                sequences.append(sequence)
        participants += 1
        if max_participants and participants >= max_participants:
            break
    return sequences


def collate_sequences(batch: list[TrainingSequence], style_mean: Tensor, style_std: Tensor) -> dict[str, Tensor]:
    batch_size = len(batch)
    steps = max(len(item.actions) for item in batch)

    def padded(shape: tuple[int, ...], dtype: torch.dtype, value: int | float = 0) -> Tensor:
        return torch.full((batch_size, steps, *shape), value, dtype=dtype)

    output = {
        "target_context": padded((batch[0].target_context.shape[-1],), torch.long),
        "buffer_context": padded((BUFFER_CONTEXT,), torch.long),
        "previous_key": padded((1,), torch.long),
        "previous_action": padded((1,), torch.long, len(Action)),
        "keyboard_features": padded((batch[0].keyboard_features.shape[-1],), torch.float32),
        "actions": padded((), torch.long),
        "wrong_keys": padded((), torch.long),
        "delays": padded((), torch.float32),
        "timing_mask": padded((), torch.bool),
        "pauses": padded((), torch.bool),
        "holds": padded((), torch.float32),
        "hold_mask": padded((), torch.bool),
        "mask": padded((), torch.bool),
        "style": padded((batch[0].style_raw.shape[-1],), torch.float32),
    }
    for index, item in enumerate(batch):
        length = len(item.actions)
        for key in ("target_context", "buffer_context", "previous_key", "previous_action", "keyboard_features", "actions", "wrong_keys", "delays", "timing_mask", "pauses", "holds", "hold_mask"):
            output[key][index, :length] = getattr(item, key)
        output["mask"][index, :length] = True
        normalized_style = (item.style_raw - style_mean) / style_std
        output["style"][index, :length] = normalized_style
    return output
