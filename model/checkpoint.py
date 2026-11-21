"""Versioned checkpoint descriptions for reproducible decoding."""

import hashlib
import io
import json
from pathlib import Path

import torch

SPEC_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
LEGACY_DECODERS = {1: "global-mixture-v3", 2: "global-mixture-v4", 3: "split-mixture-v5"}
UNIFIED_DECODER = "unified-mixture-v1"
FEEDBACK_SUFFIX = "-previous-gap-v1"
DECODERS = (*LEGACY_DECODERS.values(), UNIFIED_DECODER,
            LEGACY_DECODERS[3] + FEEDBACK_SUFFIX, UNIFIED_DECODER + FEEDBACK_SUFFIX)
ARCHITECTURE_DEFAULTS = {
    "target_left": 3, "target_right": 9, "use_keyboard_features": False, "event_model_v5": False,
}


def read_checkpoint(path: Path, spec_path: str | Path | None):
    if spec_path is None:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        return bundle, bundle["architecture"], bundle.get("decoder_version")

    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if spec["spec_version"] != SPEC_VERSION:
        raise ValueError(f"Unsupported checkpoint spec version: {spec['spec_version']}")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != spec["sha256"]:
        raise ValueError(f"Checkpoint SHA-256 does not match spec: {path}")
    bundle = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    architecture = spec["architecture"]
    if (spec["checkpoint_format"] != bundle["format_version"]
            or architecture != ARCHITECTURE_DEFAULTS | bundle["architecture"]):
        raise ValueError("Checkpoint architecture or format does not match spec")
    expected_decoder = bundle.get("decoder_version", LEGACY_DECODERS.get(bundle["format_version"]))
    if spec["decoder_version"] != expected_decoder or expected_decoder not in DECODERS:
        raise ValueError("Checkpoint decoder does not match spec")
    return bundle, architecture, expected_decoder


def write_spec(path: Path, spec_path: Path, paths: list[str], source: str) -> None:
    payload = path.read_bytes()
    bundle = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    architecture = ARCHITECTURE_DEFAULTS | bundle["architecture"]
    state = bundle["model_state"]
    style_names = ["log_median_gap", "log_p90_gap", "correction_rate", "mistake_rate"]
    if bundle["style_mean"].numel() == 6:
        style_names += ["log_median_hold", "rollover_rate"]
    spec = {
        "spec_version": SPEC_VERSION,
        "checkpoint_format": bundle["format_version"],
        "paths": paths,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "architecture": architecture,
        "recurrent_module": "GRU",
        "input_features": state["input_projection.0.weight"].shape[1],
        "buffer_context": 6,
        "target_right_includes_current": True,
        "linear_head_outputs": {
            name.removesuffix(".weight"): value.shape[0]
            for name, value in state.items() if name.endswith("_head.weight")
        },
        "parameter_count": sum(value.numel() for value in state.values()),
        "style": {
            "features": style_names, "bank_rows": bundle["style_bank"].shape[0],
            "mean": bundle["style_mean"].tolist(), "std": bundle["style_std"].tolist(),
        },
        "sampling_temperature": bundle["sampling_temperature"],
        "decoder_version": bundle.get("decoder_version", LEGACY_DECODERS.get(bundle["format_version"])),
        "source": source,
        "inferred_from_source": [
            *[f"architecture.{key}" for key in ARCHITECTURE_DEFAULTS if key not in bundle["architecture"]],
            "style.features", "recurrent_module", "buffer_context", "target_right_includes_current",
            *([] if "decoder_version" in bundle else ["decoder_version"]),
        ],
    }
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    with spec_path.open("x", encoding="utf-8") as output:
        json.dump(spec, output, indent=2)
        output.write("\n")
