"""Reconstruct the release split and prepare local timing experiment data."""

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import fields
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.checkpoint import ROOT
from model.network import Action, TypingEventModel

# Use the trainer and parser that produced the release.
BACKUP = ROOT / ".backup/v5"
sys.path.append(str(BACKUP))
from training.dataset import TrainingSequence, collate_sequences, load_sequences
from training.train import run_epoch

RELEASE = ROOT / "models/pytyper.pt"
CACHE = ROOT / ".data/timing"
SPLIT_SEED = 20261119
EVENT_FIELDS = [field.name for field in fields(TrainingSequence)
                if field.name not in {"participant_id", "style_raw"}]


class PackedTrials:
    def __init__(self, packed, left=6, right=16):
        self.packed = packed
        self.context = slice(6 - left, 6 + right)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(index)
        start, end = self.packed["offsets"][index:index + 2]
        values = {key: self.packed[key][start:end] for key in EVENT_FIELDS}
        values["target_context"] = values["target_context"][:, self.context]
        return TrainingSequence(**values, style_raw=self.packed["style"][index],
                                participant_id=self.packed["participants"][index])

    def __len__(self):
        return len(self.packed["participants"])


def participant_split(participants, seed, test=False):
    ids = sorted(set(participants))
    random.Random(seed).shuffle(ids)
    size = max(1, int(len(ids) * 0.1))
    return set(ids[:size]), set(ids[size:2 * size]) if test else set()


def style_statistics(packed, indices):
    styles = packed["style"][indices]
    mean = styles.mean(0)
    std = styles.std(0).clamp_min(1e-4)
    bank = ((styles - mean) / std)[::max(1, len(styles) // 2048)].clone()
    return mean, std, bank


def training_counts(packed, indices):
    trials = PackedTrials(packed)
    counts = torch.bincount(torch.cat([trials[i].actions for i in indices]).long(), minlength=len(Action))
    weights = (counts.sum() / counts.float().clamp_min(1)).sqrt()
    return counts, (weights / weights[0]).clamp(max=8.0)


def source_hash(path):
    with path.open("rb") as source:
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def release_validation(packed, bundle):
    started = time.perf_counter()
    ids = packed["participants"]
    validation_ids, _ = participant_split(ids, bundle["seed"])
    train = [i for i, value in enumerate(ids) if value not in validation_ids]
    val = [i for i, value in enumerate(ids) if value in validation_ids]
    mean, std, bank = style_statistics(packed, train)
    counts, weights = training_counts(packed, train)
    offsets = packed["offsets"]
    facts = {
        "training_trials": len(train), "validation_trials": len(val), "training_events": int(counts.sum()),
        "training_mistake_rate": float(counts[Action.WRONG] / counts.sum()),
        "training_pause_rate": sum(int(packed["pauses"][offsets[i]:offsets[i + 1]].sum())
                                   for i in train) / int(counts.sum()),
        "training_rollover_rate": float(mean[5]),
    }
    compare = {name: {"reconstructed": value, "checkpoint": bundle[name]} for name, value in facts.items()}
    style_match = {name: torch.equal(value, bundle[name]) for name, value in
                   (("style_mean", mean), ("style_std", std), ("style_bank", bank))}
    model = TypingEventModel(**bundle["architecture"])
    model.load_state_dict(bundle["model_state"])
    dataset = torch.utils.data.Subset(PackedTrials(packed), val)
    valid_loader = DataLoader(dataset, batch_size=48,
                              collate_fn=partial(collate_sequences, style_mean=mean, style_std=std))
    print(json.dumps({"counts": compare, "style_exact": style_match}), flush=True)
    loss = run_epoch(model, valid_loader, None, weights)
    return {"counts": compare, "style_exact": style_match, "validation_participants": len(validation_ids),
            "training_participants": len(set(ids) - validation_ids),
            "validation_loss": loss, "reported_loss": bundle["best_validation_loss"],
            "wall_seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / "trials.pt"
    bundle = torch.load(RELEASE, weights_only=True)
    if cache_path.exists():
        packed = torch.load(cache_path, weights_only=True)
    else:
        started = time.perf_counter()
        segment = BACKUP / bundle["segments"][0].replace("\\", "/")
        metadata = BACKUP / ".data/metadata-entry.bin"
        trials = load_sequences(segment, metadata, target_left=6, target_right=16)
        if not trials:
            raise RuntimeError("No valid trials found in release archive range")
        offsets = [0]
        for trial in trials:
            offsets.append(offsets[-1] + len(trial.actions))
        packed = {key: torch.cat([getattr(item, key) for item in trials]) for key in EVENT_FIELDS}
        packed.update(offsets=offsets, participants=[trial.participant_id for trial in trials],
                      style=torch.stack([trial.style_raw for trial in trials]),
                      source_sha256={path.name: source_hash(path) for path in (segment, metadata)})
        del trials
        temporary = cache_path.with_suffix(".tmp")
        torch.save(packed, temporary)
        temporary.replace(cache_path)
        print(f"Prepared {len(packed['participants']):,} trials in {time.perf_counter() - started:.1f}s", flush=True)
    result = release_validation(packed, bundle)
    val, test = participant_split(packed["participants"], SPLIT_SEED, test=True)
    release_val, _ = participant_split(packed["participants"], bundle["seed"])
    result.update(source_sha256=packed["source_sha256"], experiment_validation_participants=len(val),
                  experiment_test_participants=len(test), experiment_test_in_release_train=len(test - release_val))
    (CACHE / "release.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
