"""Train and score the three unified 3/10 models on the full Aalto corpus."""

import argparse
import hashlib
import io
import json
import math
import random
import time

import torch

from experiments.data import EVENT_FIELDS, PackedTrials
from experiments.evaluate import generated, head_losses, predict
from experiments.full_corpus_data import SPLITS, WORK, save_atomic
from experiments.run import CONTROLS_SEED
from experiments.variants import architecture_for
from model.checkpoint import ROOT, UNIFIED_DECODER, write_spec
from model.network import Action, TypingEventModel
from model.runtime import NeuralTypingRuntime

VARIANTS = {
    "gelu-16d": {},
    "gelu-8d": {"embedding_dim": 8},
    "linear-16d": {"mlp_head": None},
}
HOLD_MIXTURE_VARIANT = "gelu-16d-hold-mixture"
HOLD_MIXTURE_PARENT = "gelu-16d"
SEED = 20260920
BATCH_SIZE = 256
EVENT_BUDGET = 16384
PATIENCE = 3
LEARNING_RATE = 0.0025
WEIGHT_DECAY = 0.0001
LOSS_WEIGHTS = {"action": 1.0, "wrong_key": 0.35, "timing": 0.55, "hold": 0.15}
INDEX_FIELDS = {"target_context", "buffer_context", "previous_key", "previous_action", "actions", "wrong_keys"}


def load_shard(meta):
    return torch.load(WORK / meta["file"], weights_only=True, mmap=True, map_location="cpu")


def batch_layout(lengths, rng):
    indices = list(range(len(lengths)))
    if rng is not None:
        rng.shuffle(indices)
    groups = []
    for start in range(0, len(indices), 8192):
        bucket = sorted(indices[start:start + 8192], key=lengths.__getitem__)
        end = len(bucket)
        while end:
            limit = min(BATCH_SIZE, max(1, EVENT_BUDGET // lengths[bucket[end - 1]]))
            start_index = max(0, end - limit)
            groups.append((bucket[start_index:end], lengths[bucket[end - 1]]))
            end = start_index
    if rng is not None:
        rng.shuffle(groups)
    return groups


def trial_indices(meta, excluded_participants, packed=None):
    if excluded_participants.isdisjoint(meta["participants"]):
        return range(meta["trials"])
    if packed is None:
        packed = load_shard(meta)
    return [index for index, participant in enumerate(packed["participants"])
            if participant not in excluded_participants]


def batches(manifest, split, device, epoch=None, *, mean_style=False, target_left=3, target_right=11,
            excluded_participants=frozenset()):
    rng = random.Random(SEED + epoch) if epoch is not None else None
    shards = manifest["shards"][split].copy()
    if rng is not None:
        rng.shuffle(shards)
    mean, std = (manifest[key].to(device) for key in ("style_mean", "style_std"))
    for meta in shards:
        packed = load_shard(meta)
        offsets = packed["offsets"]
        eligible = trial_indices(meta, excluded_participants, packed)
        lengths = offsets.diff().tolist()
        layout = batch_layout([lengths[index] for index in eligible], rng)
        values = {name: packed[name].to(device) for name in EVENT_FIELDS}
        values["target_context"] = values["target_context"][:, 6 - target_left:6 + target_right]
        offsets = offsets.to(device)
        styles = (packed["style"].to(device) - mean) / std
        if mean_style:
            styles.zero_()
        for indices, steps in layout:
            trial = torch.tensor([eligible[index] for index in indices], dtype=torch.long, device=device)
            positions = offsets[trial, None] + torch.arange(steps, device=device)
            mask = positions < offsets[trial + 1, None]
            positions = torch.where(mask, positions, 0)
            batch = {name: value[positions] for name, value in values.items()}
            for name in INDEX_FIELDS:
                batch[name] = batch[name].long()
            batch["keyboard_features"] = batch["keyboard_features"].float()
            batch["style"] = styles[trial, None, :].expand(-1, steps, -1)
            batch["mask"] = mask
            yield batch


def raw_forward(model, batch):
    # Right padding cannot affect valid outputs of a unidirectional GRU.
    return model(batch["target_context"], batch["buffer_context"], batch["previous_key"],
                 batch["previous_action"], batch["style"], keyboard_features=batch["keyboard_features"])


def objective(model, loader, action_weights, optimizer=None, scaler=None):
    device = next(model.parameters()).device
    model.train(optimizer is not None)
    totals = torch.zeros(len(LOSS_WEIGHTS), 2, device=device, dtype=torch.float64)
    with torch.set_grad_enabled(optimizer is not None):
        for step, batch in enumerate(loader, 1):
            with torch.autocast(device.type, dtype=torch.float16, enabled=scaler is not None):
                outputs = raw_forward(model, batch)
            outputs = tuple(value.float() if value is not None else None for value in outputs)
            losses, _ = head_losses(outputs, batch, action_weights)
            valid = batch["mask"]
            denominators = {name: value.numel() for name, value in losses.items()}
            denominators["action"] = action_weights[batch["actions"][valid]].sum()
            sums = {name: value.sum() for name, value in losses.items()}
            loss = sum(weight * sums[name] / torch.as_tensor(
                denominators[name], device=device).clamp_min(1) for name, weight in LOSS_WEIGHTS.items())
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if scaler is None:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                if step % 1000 == 0:
                    print(f"  training batch {step:,}, batch loss {loss.item():.5f}", flush=True)
            for index, name in enumerate(LOSS_WEIGHTS):
                totals[index, 0] += sums[name].detach()
                totals[index, 1] += denominators[name]
    heads = dict(zip(LOSS_WEIGHTS, (totals[:, 0] / totals[:, 1].clamp_min(1)).tolist()))
    combined = sum(LOSS_WEIGHTS[name] * value for name, value in heads.items())
    if not math.isfinite(combined):
        raise ValueError(f"Non-finite epoch objective: {heads}")
    return combined, heads


def split_summary(manifest):
    result = {}
    for split in SPLITS:
        shards = manifest["shards"][split]
        participants = {p for shard in shards for p in shard["participants"]}
        counts = torch.stack([shard["action_counts"] for shard in shards]).sum(0)
        result[split] = {key: sum(shard[key] for shard in shards)
                         for key in ("trials", "events", "pause_count", "gap_count", "hold_count")}
        result[split].update(participants=len(participants), action_counts=counts.tolist(),
                             participant_sha256=hashlib.sha256("\n".join(sorted(participants)).encode()).hexdigest(),
                             mistake_rate=int(counts[1]) / result[split]["events"])
    return result


def generation_eval(runtime, manifest, excluded_participants=frozenset()):
    shards = [(shard, trial_indices(shard, excluded_participants)) for shard in manifest["shards"]["test"]]
    population = sum(len(indices) for _, indices in shards)
    selected = random.Random(101).sample(range(population), min(population, 128))
    sample = {}
    start = 0
    for shard, indices in shards:
        end = start + len(indices)
        local = [index for index in selected if start <= index < end]
        if local:
            trials = PackedTrials(load_shard(shard))
            for index in local:
                sample[index] = trials[indices[index - start]]
        start = end

    # generated() samples with seed 101 internally. Arrange the small population
    # so it replays the same prompt order as sampling directly from the full test.
    order = random.Random(101).sample(range(len(selected)), len(selected))
    trials = [None] * len(selected)
    for index, position in zip(selected, order):
        trials[position] = sample[index]
    return generated(runtime, trials, list(range(len(trials))))


def write_json(value, path):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def warm_start_hold_mixture(model, parent_state):
    weight_key, bias_key = "hold_head.2.weight", "hold_head.2.bias"
    parent_weight, parent_bias = parent_state[weight_key], parent_state[bias_key]
    old_width = len(Action) * 2
    state = dict(parent_state)
    output = model.hold_head[-1]
    weight, bias = torch.zeros_like(output.weight), torch.zeros_like(output.bias)
    weight[:old_width].copy_(parent_weight)
    bias[:old_width].copy_(parent_bias)
    # Start near the original distribution and give both new components gradients.
    extra_weight = weight[old_width:].reshape(len(Action), 7, -1)
    extra_bias = bias[old_width:].reshape(len(Action), 7)
    extra_bias[:, 1:3] = -3.5
    for start, offset in ((3, -0.25), (5, 0.25)):
        extra_weight[:, start:start + 2].copy_(parent_weight.reshape(len(Action), 2, -1))
        extra_bias[:, start:start + 2].copy_(parent_bias.reshape(len(Action), 2))
        extra_bias[:, start] += offset
    state[weight_key], state[bias_key] = weight, bias
    model.load_state_dict(state)


def train(name, manifest, splits, device, amp):
    result_path = WORK / f"{name}.json"
    best_path = WORK / f"{name}.pt"
    spec_path = WORK / f"{name}.spec.json"
    resume_path = WORK / f"{name}.resume.pt"
    history_path = WORK / f"{name}.epochs.json"
    warm_start = name == HOLD_MIXTURE_VARIANT
    overrides = {"hold_components": 3} if warm_start else VARIANTS[name]
    architecture = architecture_for("unified-all-heads-mlp") | overrides
    learning_rate = 0.00025 if warm_start else LEARNING_RATE
    schedule = {"patience": PATIENCE, "min_delta": 0.0, "batch_size_max": BATCH_SIZE,
                "padded_events_max": EVENT_BUDGET, "optimizer": "AdamW", "learning_rate": learning_rate,
                "weight_decay": WEIGHT_DECAY, "gradient_clip": 1.0, "loss_weights": LOSS_WEIGHTS,
                "device": str(device), "amp": amp}
    provenance = {"architecture": architecture, "schedule": schedule, "seed": SEED,
                  "split_seed": manifest["split_seed"], "source_sha256": manifest["source_sha256"],
                  "splits": splits}
    if warm_start:
        parent_path = WORK / f"{HOLD_MIXTURE_PARENT}.pt"
        parent_bytes = parent_path.read_bytes()
        parent = torch.load(io.BytesIO(parent_bytes), weights_only=True, map_location="cpu")
        if (parent["architecture"] | overrides != architecture
                or any(parent[key] != provenance[key] for key in ("split_seed", "source_sha256", "splits"))):
            raise ValueError(f"Warm-start checkpoint architecture or corpus split differs: {parent_path}")
        provenance["warm_start"] = {"checkpoint": parent_path.relative_to(ROOT).as_posix(),
                                    "sha256": hashlib.sha256(parent_bytes).hexdigest(), "epoch": parent["epoch"],
                                    "parent_validation_objective": parent["best_validation_loss"]}
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if any(result.get(key) != value for key, value in provenance.items()):
            raise ValueError(f"Run inputs changed for {name}")
        return result

    torch.manual_seed(SEED)
    model = TypingEventModel(**architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY,
                                  fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if amp else None
    counts = torch.tensor(splits["train"]["action_counts"], dtype=torch.float32, device=device)
    weights = (counts.sum() / counts.clamp_min(1)).sqrt()
    weights = (weights / weights[0]).clamp(max=8.0)
    history, best, initial_validation = [], None, None
    if resume_path.exists():
        state = torch.load(resume_path, weights_only=True, map_location="cpu")
        if state["provenance"] != provenance:
            raise ValueError(f"Run inputs changed for {name}")
        history, best = state["history"], state["best"]
        initial_validation = state.get("initial_validation")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"])
        if scaler is not None:
            scaler.load_state_dict(state["scaler"])
    elif warm_start:
        warm_start_hold_mixture(model, parent["model_state"])

    def checkpoint(epoch, val_loss):
        return {**provenance, "format_version": 4, "decoder_version": UNIFIED_DECODER,
                "epoch": epoch, "best_validation_loss": val_loss, "sampling_temperature": 0.86,
                "training_mistake_rate": splits["train"]["mistake_rate"],
                "model_state": {key: value.to("cpu", copy=True) for key, value in model.state_dict().items()},
                **{key: manifest[key] for key in ("style_mean", "style_std", "style_bank")}}

    print(f"{name}: {device}, {sum(p.numel() for p in model.parameters()):,} parameters, "
          f"resuming after epoch {len(history)}", flush=True)
    started = time.perf_counter()
    previous_time = history[-1]["elapsed_seconds"] if history else 0.0
    if warm_start and best is None:
        val_loss, val_heads = objective(model, batches(manifest, "validation", device), weights)
        initial_validation = {"objective": val_loss, "heads": val_heads}
        best = checkpoint(0, val_loss)
        save_atomic(best, best_path)
        print(f"{name}: initialized from {HOLD_MIXTURE_PARENT} epoch {parent['epoch']}, "
              f"validation={val_loss:.6f}", flush=True)
    while best is None or len(history) - best["epoch"] < PATIENCE:
        epoch = len(history) + 1
        train_loss, train_heads = objective(model, batches(manifest, "train", device, epoch),
                                           weights, optimizer, scaler)
        val_loss, val_heads = objective(model, batches(manifest, "validation", device), weights)
        row = {"epoch": epoch, "train_objective": train_loss, "validation_objective": val_loss,
               "train_heads": train_heads, "validation_heads": val_heads,
               "elapsed_seconds": previous_time + time.perf_counter() - started}
        history.append(row)
        improved = best is None or val_loss < best["best_validation_loss"]
        if improved:
            best = checkpoint(epoch, val_loss)
        state = {"provenance": provenance, "history": history, "best": best, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None,
                 "scaler": scaler.state_dict() if scaler is not None else None}
        if warm_start:
            state["initial_validation"] = initial_validation
        save_atomic(state, resume_path)
        if improved:
            save_atomic(best, best_path)
        write_json(history, history_path)
        print(f"{name} epoch {epoch}: train={train_loss:.6f} validation={val_loss:.6f} "
              f"best={best['epoch']} elapsed={row['elapsed_seconds']:.1f}s", flush=True)
    save_atomic({**best, "history": history}, best_path)
    temp_spec = spec_path.with_suffix(".tmp")
    temp_spec.unlink(missing_ok=True)
    write_spec(best_path, temp_spec, [best_path.relative_to(ROOT).as_posix()], "experiments/full_corpus.py")
    temp_spec.replace(spec_path)
    model.load_state_dict(best["model_state"])
    runtime = NeuralTypingRuntime(best_path, CONTROLS_SEED, spec_path=spec_path)
    print(f"{name}: validation stopped improving. Scoring checkpoint from epoch {best['epoch']}.", flush=True)
    result = {"name": name, **provenance, "checkpoint": best_path.relative_to(ROOT).as_posix(),
              "spec": spec_path.relative_to(ROOT).as_posix(), "parameters": sum(p.numel() for p in model.parameters()),
              "best_epoch": best["epoch"], "best_validation_objective": best["best_validation_loss"],
              "epochs": history, "test": predict(model, batches(manifest, "test", device, mean_style=True),
                                                  include_accuracy=True),
              "generated": generation_eval(runtime, manifest)}
    if warm_start:
        result["initial_validation"] = initial_validation
    write_json(result, result_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-amp", action="store_true", help="Use full precision on CUDA")
    parser.add_argument("--variant", choices=(*VARIANTS, HOLD_MIXTURE_VARIANT), help="Train one model")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = torch.load(WORK / "manifest.pt", weights_only=True)
    splits = split_summary(manifest)
    print(json.dumps(splits, indent=2), flush=True)
    amp = device.type == "cuda" and not args.no_amp
    if args.variant:
        train(args.variant, manifest, splits, device, amp)
        return
    results = [train(name, manifest, splits, device, amp) for name in VARIANTS]
    write_json(results, WORK / "results.json")
    if (WORK / f"{HOLD_MIXTURE_VARIANT}.json").exists():
        results.append(train(HOLD_MIXTURE_VARIANT, manifest, splits, device, amp))
    from experiments.full_corpus_report import update_report
    update_report(manifest, results)


if __name__ == "__main__":
    main()
