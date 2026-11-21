"""Build the exact architecture of a recorded experiment without loading weights."""

import json
from pathlib import Path

from model.network import TypingEventModel

_SPECS = Path(__file__).resolve().parents[1] / "models" / "specs"
_CHECKPOINT_VARIANTS = {
    "archive-v3": "archive-v3.json",
    "archive-v4": "archive-v4.json",
    "shipped": "release.json",
}
_CONTEXT_3_10 = {"target_left": 3, "target_right": 11}
_CONTEXT_6_10 = {"target_left": 6, "target_right": 11}
_CONTEXT_3_15 = {"target_left": 3, "target_right": 16}
_UNIFIED = {"unified_timing": True, "timing_components": 3}
_UNIFIED_3_10 = _UNIFIED | _CONTEXT_3_10 | {"embedding_dim": 16, "use_previous_gap": False}

DEFAULT_RUN = "unified-context-3-10"
FULL_RELEASE_VARIANT = "unified-residual-action-mlp-full"
RUN_VARIANTS = {
    "baseline": {},
    "mixture": _UNIFIED,
    "feedback": {"use_previous_gap": True},
    "context-3-15": _CONTEXT_3_15,
    "context-6-10": _CONTEXT_6_10,
    "context-3-10": _CONTEXT_3_10,
    "embedding-8": {"embedding_dim": 8},
    "combined": {"embedding_dim": 8},
    DEFAULT_RUN: _UNIFIED_3_10,
    "unified-context-3-15": _UNIFIED_3_10 | _CONTEXT_3_15,
    "unified-context-6-10": _UNIFIED_3_10 | _CONTEXT_6_10,
    "unified-embedding-8": _UNIFIED_3_10 | {"embedding_dim": 8},
    "unified-feedback": _UNIFIED_3_10 | {"use_previous_gap": True},
    "unified-gru-residual-fixed": _UNIFIED_3_10 | {"gru_residual_alpha": 0.25},
    "unified-gru-residual-learnable": _UNIFIED_3_10 | {
        "gru_residual_alpha": 0.0, "learnable_gru_residual": True,
    },
    "unified-timing-mlp": _UNIFIED_3_10 | {"mlp_head": "timing"},
    "unified-wrong-key-mlp": _UNIFIED_3_10 | {"mlp_head": "wrong_key"},
    "unified-action-mlp": _UNIFIED_3_10 | {"mlp_head": "action"},
    "unified-all-heads-mlp": _UNIFIED_3_10 | {"gru_residual_alpha": 0.25, "mlp_head": "all"},
}
_VARIANT_OVERRIDES = RUN_VARIANTS | {
    FULL_RELEASE_VARIANT: _UNIFIED | _CONTEXT_3_10 | {"gru_residual_alpha": 0.25, "mlp_head": "action"},
}
VARIANT_NAMES = (*_CHECKPOINT_VARIANTS, *_VARIANT_OVERRIDES)


def architecture_for(name: str, base: dict | None = None) -> dict:
    """Return constructor options for a named model or an experiment's supplied base."""
    if name in _CHECKPOINT_VARIANTS:
        spec = _SPECS / _CHECKPOINT_VARIANTS[name]
        return json.loads(spec.read_text(encoding="utf-8"))["architecture"]
    if base is None:
        base = architecture_for("shipped")
        # The recorded combined run used the shorter-context checkpoint as its base.
        if name == "combined":
            base.update(_CONTEXT_6_10)
    return base | _VARIANT_OVERRIDES[name]


def build_model(name: str) -> TypingEventModel:
    """Instantiate a named architecture with fresh weights."""
    return TypingEventModel(**architecture_for(name))
