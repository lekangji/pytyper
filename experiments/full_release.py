"""Train the unified residual/action model on the original release split."""

import argparse
import hashlib
import json
import random
import time
from functools import partial

import torch
from torch.utils.data import DataLoader, Subset

from experiments.data import CACHE, RELEASE, SPLIT_SEED, PackedTrials, collate_sequences
from experiments.data import participant_split, style_statistics, training_counts
from experiments.evaluate import generated, predict
from experiments.run import BATCH_SIZE, CONTROLS_SEED, INPUT_VARIANTS, evaluation_collate, train_epoch, training_device
from model.checkpoint import ROOT, UNIFIED_DECODER, write_spec
from model.network import Action, TypingEventModel
from model.runtime import NeuralTypingRuntime

NAME = "unified-residual-action-mlp-full"
EPOCHS = 4
LEARNING_RATE = 0.0025
WEIGHT_DECAY = 0.0001


def train(packed, release):
    seed = release["seed"]
    base = CACHE / f"{NAME}-{seed}"
    weight_file = base.with_suffix(".pt")
    spec_file = base.with_suffix(".spec.json")
    if weight_file.exists() or spec_file.exists():
        raise FileExistsError(f"Run artifacts already exist: {base}")
    random.seed(seed)
    torch.manual_seed(seed)
    participants = packed["participants"]
    validation_ids, _ = participant_split(participants, seed)
    train_indices = [i for i, value in enumerate(participants) if value not in validation_ids]
    validation_indices = [i for i, value in enumerate(participants) if value in validation_ids]
    style_mean, style_std, style_bank = style_statistics(packed, train_indices)
    counts, weights = training_counts(packed, train_indices)
    actual = (len(train_indices), len(validation_indices), int(counts.sum()))
    expected = tuple(release[key] for key in ("training_trials", "validation_trials", "training_events"))
    if actual != expected or not all(
        torch.equal(value, release[key])
        for key, value in (("style_mean", style_mean), ("style_std", style_std), ("style_bank", style_bank))
    ):
        raise ValueError("Cached trials do not reproduce the shipped training split")

    architecture = release["architecture"] | INPUT_VARIANTS["context-3-10"] | {
        "unified_timing": True, "timing_components": 3,
        "gru_residual_alpha": 0.25, "mlp_head": "action",
    }
    trials = PackedTrials(packed, architecture["target_left"], architecture["target_right"])
    collate = partial(collate_sequences, style_mean=style_mean, style_std=style_std)
    train_loader = DataLoader(Subset(trials, train_indices), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    validation_loader = DataLoader(Subset(trials, validation_indices), batch_size=BATCH_SIZE, collate_fn=collate)
    device = training_device()
    model = TypingEventModel(**architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    bundle = {
        "format_version": 4, "architecture": architecture, "decoder_version": UNIFIED_DECODER,
        "seed": seed, "split_seed": seed, "style_mean": style_mean, "style_std": style_std, "style_bank": style_bank,
        "sampling_temperature": release["sampling_temperature"],
        "training_trials": actual[0], "validation_trials": actual[1], "training_events": actual[2],
        "training_mistake_rate": float(counts[Action.WRONG] / counts.sum()),
        "trial_index_sha256": [hashlib.sha256(json.dumps(group).encode()).hexdigest()
                               for group in (train_indices, validation_indices)],
        "source_sha256": packed["source_sha256"],
        "schedule": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer": "AdamW",
                     "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "gradient_clip": 1.0},
    }
    print(f"seed={seed} device={device} trials={actual[0]:,}/{actual[1]:,} training_events={actual[2]:,}", flush=True)
    started = time.perf_counter()
    history = []
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, weights)
        with torch.no_grad():
            validation_loss = train_epoch(model, validation_loader, None, weights)
        elapsed = time.perf_counter() - started
        history.append({"epoch": epoch, "train_objective": train_loss, "validation_objective": validation_loss,
                        "elapsed_seconds": elapsed})
        print(f"epoch={epoch}/{EPOCHS} train={train_loss:.6f} validation={validation_loss:.6f} "
              f"elapsed={elapsed:.1f}s", flush=True)
        if validation_loss < bundle.get("best_validation_loss", float("inf")):
            bundle.update(epoch=epoch, best_validation_loss=validation_loss,
                          model_state={key: value.to("cpu", copy=True) for key, value in model.state_dict().items()})
    bundle["history"] = history
    with weight_file.open("xb") as output:
        torch.save(bundle, output)
    write_spec(weight_file, spec_file, [weight_file.relative_to(ROOT).as_posix()], "experiments/full_release.py")
    print(f"Saved {weight_file} (epoch {bundle['epoch']})", flush=True)


def shared_holdout(packed, seed):
    release_validation_ids, _ = participant_split(packed["participants"], seed)
    _, experiment_test_ids = participant_split(packed["participants"], SPLIT_SEED, test=True)
    shared_ids = release_validation_ids & experiment_test_ids
    return len(shared_ids), [i for i, value in enumerate(packed["participants"]) if value in shared_ids]


def evaluate(packed, indices, weight_file, spec_file):
    runtime = NeuralTypingRuntime(weight_file, CONTROLS_SEED, spec_path=spec_file)
    model = runtime.model
    trials = PackedTrials(packed, model.target_left, model.target_right)
    collate = partial(evaluation_collate, style_mean=runtime.style_mean, style_std=runtime.style_std)
    loader = DataLoader(Subset(trials, indices), batch_size=BATCH_SIZE, collate_fn=collate)
    return {
        "checkpoint": weight_file.relative_to(ROOT).as_posix(),
        "spec": spec_file.relative_to(ROOT).as_posix(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "test": predict(model, loader, include_accuracy=True),
        "generated": generated(runtime, PackedTrials(packed), indices),
    }


def compare(packed, release):
    seed = release["seed"]
    base = CACHE / f"{NAME}-{seed}"
    weight_file = base.with_suffix(".pt")
    candidate = torch.load(weight_file, weights_only=True)
    if candidate["source_sha256"] != packed["source_sha256"]:
        raise ValueError("Cached trials do not match the candidate's training corpus")
    participants, indices = shared_holdout(packed, seed)
    comparison = {
        "name": NAME,
        "seed": seed,
        "architecture": candidate["architecture"],
        "schedule": candidate["schedule"],
        "epochs": candidate["history"],
        "best_epoch": candidate["epoch"],
        "training_trials": candidate["training_trials"],
        "validation_trials": candidate["validation_trials"],
        "training_events": candidate["training_events"],
        "source_sha256": candidate["source_sha256"],
        "shared_holdout": {
            "participants": participants, "trials": len(indices), "split_seed": SPLIT_SEED,
            "used_for_validation_selection": True, "generation_seed": CONTROLS_SEED,
        },
        "shipped": evaluate(packed, indices, RELEASE, ROOT / "models/specs/release.json"),
        "candidate": evaluate(packed, indices, weight_file, base.with_suffix(".spec.json")),
    }
    result_file = base.with_suffix(".json")
    result_file.write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved shared-holdout comparison to {result_file}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--compare", action="store_true", help="Score the saved candidate and shipped checkpoint")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    packed = torch.load(CACHE / "trials.pt", weights_only=True)
    release = torch.load(RELEASE, weights_only=True)
    if args.compare:
        compare(packed, release)
    else:
        train(packed, release)


if __name__ == "__main__":
    main()
