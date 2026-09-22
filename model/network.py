"""Compact neural network used for training and inference."""

from __future__ import annotations

import math
from enum import IntEnum

import torch
from torch import Tensor, nn

from model.keyboard import KEYBOARD_FEATURES

PRINTABLE_KEYS = tuple(chr(code) for code in range(32, 127))
KEY_TO_ID = {key: index + 2 for index, key in enumerate(PRINTABLE_KEYS)}
PAD_ID = 0
UNKNOWN_ID = 1
VOCAB_SIZE = len(KEY_TO_ID) + 2
TARGET_LEFT = 3
TARGET_RIGHT = 9
TARGET_CONTEXT = TARGET_LEFT + TARGET_RIGHT
BUFFER_CONTEXT = 6
STYLE_FEATURES = 4
STYLE_FEATURES_WITH_HOLDS = 6
PAUSE_THRESHOLD = 0.5


class Action(IntEnum):
    CORRECT = 0
    WRONG = 1
    BACKSPACE = 2


def char_id(character: str) -> int:
    return KEY_TO_ID.get(character, UNKNOWN_ID)


def longest_common_prefix(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


class TypingEventModel(nn.Module):
    """Autoregressive model of keys, corrections, and timing."""

    def __init__(
        self,
        embedding_dim: int = 16,
        hidden_size: int = 128,
        layers: int = 2,
        timing_components: int = 5,
        target_left: int = TARGET_LEFT,
        target_right: int = TARGET_RIGHT,
        use_keyboard_features: bool = False,
        event_model_v5: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.timing_components = timing_components
        self.target_left = target_left
        self.target_right = target_right
        self.use_keyboard_features = use_keyboard_features
        self.event_model_v5 = event_model_v5
        self.style_features = STYLE_FEATURES_WITH_HOLDS if event_model_v5 else STYLE_FEATURES
        self.key_embedding = nn.Embedding(VOCAB_SIZE, embedding_dim, padding_idx=PAD_ID)
        self.action_embedding = nn.Embedding(len(Action) + 1, 8)
        input_size = embedding_dim * (target_left + target_right + BUFFER_CONTEXT + 1) + 8 + self.style_features
        if use_keyboard_features:
            input_size += KEYBOARD_FEATURES
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.gru = nn.GRU(hidden_size, hidden_size, layers, batch_first=True, dropout=0.12)
        self.action_head = nn.Linear(hidden_size, len(Action))
        self.wrong_key_head = nn.Linear(hidden_size, len(PRINTABLE_KEYS))
        if event_model_v5:
            self.pause_head = nn.Linear(hidden_size, len(Action))
            self.timing_head = nn.Linear(hidden_size, len(Action) * 2 * timing_components * 3)
            self.hold_head = nn.Linear(hidden_size, len(Action) * 2)
        else:
            self.timing_head = nn.Linear(hidden_size, timing_components * 3)

    def forward(
        self,
        target_context: Tensor,
        buffer_context: Tensor,
        previous_key: Tensor,
        previous_action: Tensor,
        style: Tensor,
        hidden: Tensor | None = None,
        keyboard_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        batch, steps, _ = target_context.shape
        target_emb = self.key_embedding(target_context).reshape(batch, steps, -1)
        buffer_emb = self.key_embedding(buffer_context).reshape(batch, steps, -1)
        previous_emb = self.key_embedding(previous_key).reshape(batch, steps, -1)
        action_emb = self.action_embedding(previous_action).reshape(batch, steps, -1)
        parts = [target_emb, buffer_emb, previous_emb, action_emb, style]
        if self.use_keyboard_features:
            if keyboard_features is None:
                raise ValueError("This checkpoint requires keyboard transition features")
            parts.append(keyboard_features)
        features = torch.cat(parts, dim=-1)
        recurrent_input = self.input_projection(features)
        output, hidden = self.gru(recurrent_input, hidden)
        if self.event_model_v5:
            timing = self.timing_head(output).reshape(batch, steps, len(Action), 2, self.timing_components, 3)
            hold = self.hold_head(output).reshape(batch, steps, len(Action), 2)
            return (
                self.action_head(output),
                self.wrong_key_head(output),
                timing[..., 0],
                timing[..., 1].clamp(-5.0, 1.7),
                timing[..., 2].clamp(-3.5, 1.2),
                hidden,
                self.pause_head(output),
                hold[..., 0].clamp(-5.0, 0.0),
                hold[..., 1].clamp(-3.5, 1.2),
            )
        timing = self.timing_head(output).reshape(batch, steps, self.timing_components, 3)
        return (
            self.action_head(output),
            self.wrong_key_head(output),
            timing[..., 0],
            timing[..., 1].clamp(-5.0, 1.7),
            timing[..., 2].clamp(-3.5, 1.2),
            hidden,
        )


def lognormal_mixture_nll(
    delays: Tensor,
    mixture_logits: Tensor,
    log_means: Tensor,
    log_scales: Tensor,
) -> Tensor:
    values = delays.clamp(0.008, 8.0).log().unsqueeze(-1)
    scales = log_scales.exp()
    component_log_prob = -0.5 * ((values - log_means) / scales) ** 2
    component_log_prob -= log_scales + 0.5 * math.log(2.0 * math.pi)
    return -torch.logsumexp(torch.log_softmax(mixture_logits, dim=-1) + component_log_prob, dim=-1)
