"""Train and score matched-seed timing experiments on a frozen participant split."""

import argparse
import hashlib
import json
import random
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from experiments.evaluate import forward, generated, head_losses, predict
from model.checkpoint import FEEDBACK_SUFFIX, ROOT, UNIFIED_DECODER, write_spec
from model.network import TypingEventModel
from model.runtime import NeuralTypingRuntime

SEEDS = (17, 29, 43)
CAPS = (3072, 768, 1536)
EPOCHS = 3
BATCH_SIZE = 48
CONTROLS_SEED = 251
SPLIT_DECODER = "split-mixture-v5"
INPUT_VARIANTS = {
    "context-3-15": {"target_left": 3, "target_right": 16},
    "context-6-10": {"target_left": 6, "target_right": 11},
    "context-3-10": {"target_left": 3, "target_right": 11},
    "embedding-8": {"embedding_dim": 8},
    "combined": {"embedding_dim": 8},
}
DEFAULT_RUN = "unified-context-3-10"
UNIFIED_VARIANTS = {
    DEFAULT_RUN: {},
    "unified-context-3-15": INPUT_VARIANTS["context-3-15"],
    "unified-context-6-10": INPUT_VARIANTS["context-6-10"],
    "unified-embedding-8": {"embedding_dim": 8},
    "unified-feedback": {"use_previous_gap": True},
    "unified-gru-residual-fixed": {"gru_residual_alpha": 0.25},
    "unified-gru-residual-learnable": {"gru_residual_alpha": 0.0, "learnable_gru_residual": True},
    "unified-timing-mlp": {"mlp_head": "timing"},
    "unified-wrong-key-mlp": {"mlp_head": "wrong_key"},
    "unified-action-mlp": {"mlp_head": "action"},
    "unified-all-heads-mlp": {"gru_residual_alpha": 0.25, "mlp_head": "all"},
}


def experiment_indices(packed):
    from experiments.data import SPLIT_SEED, participant_split

    ids = packed["participants"]
    val, test = participant_split(ids, SPLIT_SEED, test=True)
    groups = (
        [i for i, value in enumerate(ids) if value not in val and value not in test],
        [i for i, value in enumerate(ids) if value in val],
        [i for i, value in enumerate(ids) if value in test],
    )
    rng = random.Random(SPLIT_SEED + 1)
    return [sorted(rng.sample(group, min(cap, len(group)))) for group, cap in zip(groups, CAPS)]


def evaluation_collate(items, style_mean, style_std):
    from experiments.data import collate_sequences

    batch = collate_sequences(items, style_mean, style_std)
    batch["style"].zero_()
    return batch


def training_device():
    # PyTorch exposes ROCm GPUs through the same device API as CUDA.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, loader, optimizer, weights):
    model.train(optimizer is not None)
    device = next(model.parameters()).device
    weights = weights.to(device)
    total = 0.0
    coefficients = {"action": 1.0, "wrong_key": 0.35, "timing": 0.55, "pause": 0.25, "hold": 0.15}
    for batch in loader:
        batch = {name: value.to(device) for name, value in batch.items()}
        losses, _ = head_losses(forward(model, batch), batch, weights)
        action_indices = batch["actions"][batch["mask"]]
        loss = losses.pop("action").sum() / weights[action_indices].sum()
        loss = loss + sum(coefficients[name] * values.sum() / max(1, values.numel())
                          for name, values in losses.items())
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if isinstance(model.gru_residual_alpha, torch.nn.Parameter):
                with torch.no_grad():
                    model.gru_residual_alpha.clamp_(0.0, 1.0)
        total += float(loss.detach())
    return total / len(loader)


def run(name, seed, packed, indices, architecture):
    from experiments.data import CACHE, SPLIT_SEED, PackedTrials, collate_sequences, style_statistics, training_counts

    rng = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    train, val, test = indices
    trials = PackedTrials(packed, architecture["target_left"], architecture["target_right"])
    style_mean, style_std, style_bank = style_statistics(packed, train)
    counts, weights = training_counts(packed, train)
    eval_collate = partial(evaluation_collate, style_mean=style_mean, style_std=style_std)
    train_loader = DataLoader(Subset(trials, train), batch_size=BATCH_SIZE, shuffle=True, generator=rng,
                              collate_fn=partial(collate_sequences, style_mean=style_mean, style_std=style_std))
    validation_loader = DataLoader(Subset(trials, val), batch_size=BATCH_SIZE, collate_fn=eval_collate)
    test_loader = DataLoader(Subset(trials, test), batch_size=BATCH_SIZE, collate_fn=eval_collate)
    base = CACHE / f"{name}-{seed}"
    weight_file = base.with_suffix(".pt")
    spec_file = base.with_suffix(".spec.json")
    result_file = base.with_suffix(".json")
    if any(path.exists() for path in (weight_file, spec_file, result_file)):
        raise FileExistsError(f"Run artifacts already exist: {base}")
    started = time.perf_counter()
    device = training_device()
    print(f"{name} seed={seed} device={device}", flush=True)
    model = TypingEventModel(**architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0025, weight_decay=0.0001)
    best_loss = float("inf")
    epochs = []
    for epoch in range(1, EPOCHS + 1):
        objective = train_epoch(model, train_loader, optimizer, weights)
        losses = predict(model, validation_loader)
        score = (losses["action"] + 0.35 * (losses["wrong_key"] or 0)
                 + 0.55 * losses["gap_nll"] + 0.15 * losses["hold"])
        if score < best_loss:
            best_loss = score
            best_epoch = epoch
            best_state = {key: value.to("cpu", copy=True) for key, value in model.state_dict().items()}
        epochs.append({"epoch": epoch, "train_objective": objective, "validation_score": score})
        print(f"{name} seed={seed} epoch={epoch} train={objective:.4f} validation={score:.4f} "
              f"elapsed={time.perf_counter() - started:.1f}s", flush=True)

    decoder = UNIFIED_DECODER if model.unified_timing else SPLIT_DECODER
    if model.use_previous_gap:
        decoder += FEEDBACK_SUFFIX
    bundle = {"format_version": 4, "architecture": architecture, "model_state": best_state,
              "decoder_version": decoder, "seed": seed, "epoch": best_epoch,
              "style_mean": style_mean, "style_std": style_std, "style_bank": style_bank,
              "sampling_temperature": 0.86, "training_mistake_rate": float(counts[1] / counts.sum()),
              "split_seed": SPLIT_SEED, "trial_index_sha256": [
                  hashlib.sha256(json.dumps(group).encode()).hexdigest() for group in indices],
              "source_sha256": packed["source_sha256"]}
    with weight_file.open("xb") as output:
        torch.save(bundle, output)
    write_spec(weight_file, spec_file, [weight_file.relative_to(ROOT).as_posix()], "experiments/run.py and model/")
    runtime = NeuralTypingRuntime(weight_file, CONTROLS_SEED, spec_path=spec_file)
    result = {"name": name, "seed": seed, "architecture": architecture,
              "parameters": sum(param.numel() for param in model.parameters()),
              "wall_seconds": time.perf_counter() - started, "epochs": epochs, "best_epoch": best_epoch,
              "test": predict(runtime.model, test_loader, include_accuracy=True),
              "generated": generated(runtime, PackedTrials(packed), test)}
    if architecture.get("learnable_gru_residual"):
        result["learned_gru_residual_alpha"] = best_state["gru_residual_alpha"].item()
    result["wall_seconds"] = time.perf_counter() - started
    result_file.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{name} seed={seed} gap_nll={result['test']['gap_nll']:.4f} "
          f"rhythm={result['generated']['rhythm_error']:.3f} wall={result['wall_seconds']:.1f}s", flush=True)


def main():
    from experiments.data import CACHE, RELEASE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, action="append", choices=SEEDS)
    parser.add_argument(
        "--run", choices=("baseline", "mixture", "feedback", *INPUT_VARIANTS, *UNIFIED_VARIANTS),
        default=DEFAULT_RUN, help="Experiment configuration (default: %(default)s)",
    )
    parser.add_argument("--base", type=Path, default=RELEASE, help="Checkpoint providing the base architecture")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    packed = torch.load(CACHE / "trials.pt", weights_only=True)
    indices = experiment_indices(packed)
    participant_groups = [{packed["participants"][i] for i in group} for group in indices]
    assert all(left.isdisjoint(right) for i, left in enumerate(participant_groups) for right in participant_groups[i+1:])
    print(f"Trials per split: {list(map(len, indices))}, participants: {list(map(len, participant_groups))}", flush=True)
    architecture = torch.load(args.base, weights_only=True)["architecture"].copy()
    if (architecture["layers"], architecture["hidden_size"], architecture["event_model_v5"]) != (3, 128, True):
        raise ValueError("Timing experiments require the 3×128 GRU event model")
    if args.run == "mixture":
        architecture.update(unified_timing=True, timing_components=3)
    if args.run == "feedback":
        architecture.update(use_previous_gap=True)
    if args.run in INPUT_VARIANTS:
        architecture.update(INPUT_VARIANTS[args.run])
    if args.run in UNIFIED_VARIANTS:
        architecture.update(
            INPUT_VARIANTS["context-3-10"], unified_timing=True, timing_components=3,
            embedding_dim=16, use_previous_gap=False,
        )
        architecture.update(UNIFIED_VARIANTS[args.run])
    for seed in args.seed or SEEDS:
        run(args.run, seed, packed, indices, architecture)


if __name__ == "__main__":
    main()
