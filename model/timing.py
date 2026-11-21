"""Numerically stable side masses and truncated log-gap sampling."""

import math
import random

import torch
from torch import Tensor

from model.network import PAUSE_THRESHOLD

LOG_THRESHOLD = math.log(PAUSE_THRESHOLD)
MIN_SAMPLED_GAP = 0.008
MAX_SAMPLED_GAP = 5.0


def gap_features(delays: Tensor, valid: Tensor) -> Tensor:
    logged = delays.clamp(MIN_SAMPLED_GAP, MAX_SAMPLED_GAP).log()
    return torch.stack((logged.masked_fill(~valid, 0.0), (~valid).to(logged.dtype)), dim=-1)


def component_log_masses(means: Tensor, log_scales: Tensor) -> Tensor:
    """Return log probability below and at or above 500 ms for each component."""
    z = (means - LOG_THRESHOLD) * (-log_scales).exp()
    return torch.special.log_ndtr(torch.stack((-z, z), dim=-1))


def sample_conditioned_log_gap(
    log_weights: Tensor, means: Tensor, log_scales: Tensor, long_gap: bool,
    rng: random.Random, generator: torch.Generator,
) -> float:
    component = torch.multinomial(log_weights.softmax(0), 1, generator=generator).item()
    direction = 1 if long_gap else -1
    sigma = math.exp(log_scales[component].item())
    lower = direction * (LOG_THRESHOLD - means[component].item()) / sigma
    if lower <= 0:
        while True:
            sample = rng.gauss(0.0, 1.0)
            if sample >= lower:
                break
    else:
        # An exponential proposal avoids subtracting nearly equal tail probabilities.
        rate = (lower + math.hypot(lower, 2)) / 2
        while True:
            sample = lower + rng.expovariate(rate)
            if rng.random() <= math.exp(-0.5 * (sample - rate) ** 2):
                break
    value = LOG_THRESHOLD + direction * sigma * (sample - lower)
    return value if long_gap else min(value, math.nextafter(LOG_THRESHOLD, -math.inf))
