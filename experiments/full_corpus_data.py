"""Prepare bounded-size shards from the official Aalto ZIP."""

import argparse
import csv
import io
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import torch

from experiments.data import EVENT_FIELDS, SPLIT_SEED, source_hash
from model.checkpoint import ROOT
from training.dataset import build_sequence, trials_from_tsv

ARCHIVE = ROOT / ".data/Keystrokes.zip"
WORK = ROOT / ".data/full-corpus"
SPLITS = ("train", "validation", "test")
GROUP_SIZE = 1536


def save_atomic(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def prepare_shard(job):
    archive, output, number, members = job
    torch.set_num_threads(1)
    groups = {name: [] for name in SPLITS}
    malformed = 0
    with ZipFile(archive) as source:
        for filename, participant, split in members:
            try:
                trials = trials_from_tsv(source.read(filename), {participant})
            except (csv.Error, KeyError, TypeError, ValueError, OverflowError):
                malformed += 1
                continue
            for trial in trials:
                sequence = build_sequence(trial, target_left=6, target_right=16)
                if sequence is not None:
                    groups[split].append(sequence)
    result = {}
    for split, trials in groups.items():
        if not trials:
            continue
        offsets = [0]
        for trial in trials:
            offsets.append(offsets[-1] + len(trial.actions))
        fields = {key: torch.cat([getattr(trial, key) for trial in trials]) for key in EVENT_FIELDS}
        participants = [trial.participant_id for trial in trials]
        styles = torch.stack([trial.style_raw for trial in trials])
        name = f"{split}-{number:03}.pt"
        save_atomic(dict(**fields, participants=participants, offsets=torch.tensor(offsets), style=styles),
                    output / name)
        result[split] = {
            "file": name, "participants": sorted(set(participants)), "trials": len(trials),
            "events": offsets[-1], "action_counts": torch.bincount(fields["actions"].long(), minlength=3),
            "pause_count": int(fields["pauses"].sum()), "gap_count": int(fields["timing_mask"].sum()),
            "hold_count": int(fields["hold_mask"].sum()),
        }
        if split == "train":
            result[split]["styles"] = styles
    return result, malformed


def prepare(archive, output, workers):
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.pt"
    source_sha256 = {archive.name: source_hash(archive)}
    if manifest_path.exists():
        manifest = torch.load(manifest_path, weights_only=True, map_location="cpu")
        if manifest["source_sha256"] != source_sha256 or manifest["split_seed"] != SPLIT_SEED:
            raise ValueError(f"Prepared cache archive or split differs: {manifest_path}")
        print("Full corpus cache already prepared.", flush=True)
        return
    torch.set_num_threads(2)
    with ZipFile(archive) as source:
        names = source.namelist()
        metadata, = [name for name in names if Path(name).name == "metadata_participants.txt"]
        rows = csv.DictReader(io.StringIO(source.read(metadata).decode("utf-8", "replace"), newline=""),
                              delimiter="\t")
        eligible = {row["PARTICIPANT_ID"] for row in rows
                    if row.get("LAYOUT") == "qwerty" and row.get("KEYBOARD_TYPE") in {"full", "laptop"}}
        entries = [(name, Path(name).name.removesuffix("_keystrokes.txt")) for name in names
                   if name.endswith("_keystrokes.txt")]
    entries = [(name, participant) for name, participant in entries if participant in eligible]
    participants = sorted({participant for _, participant in entries})
    random.Random(SPLIT_SEED).shuffle(participants)
    size = len(participants) // 10
    if not size:
        raise ValueError("Full archive needs at least ten eligible participants")
    assignment = {participant: SPLITS[2 if i < size else 1 if i < 2 * size else 0]
                  for i, participant in enumerate(participants)}
    members = [(name, participant, assignment[participant]) for name, participant in entries]
    jobs = [(archive, output, i // GROUP_SIZE, members[i:i + GROUP_SIZE])
            for i in range(0, len(members), GROUP_SIZE)]
    print(f"Preparing {len(members):,} entries from {len(participants):,} eligible participants "
          f"in {len(jobs)} groups.", flush=True)
    shards = {split: [] for split in SPLITS}
    styles = []
    malformed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (result, skipped) in enumerate(pool.map(prepare_shard, jobs), 1):
            malformed += skipped
            for split, shard in result.items():
                if split == "train":
                    styles.append(shard.pop("styles"))
                shards[split].append(shard)
            print(f"Prepared group {i}/{len(jobs)}", flush=True)
    actual = {split: {p for shard in shards[split] for p in shard["participants"]} for split in SPLITS}
    if any(not actual[split] for split in SPLITS):
        raise ValueError("A prepared participant split is empty")
    for split, ids in actual.items():
        if any(assignment[p] != split for p in ids):
            raise ValueError(f"Participant assigned to the wrong split: {split}")
    styles = torch.cat(styles)
    mean, std = styles.mean(0), styles.std(0).clamp_min(1e-4)
    manifest = {
        "shards": shards, "eligible_participants": len(eligible), "eligible_entries": len(entries),
        "malformed_entries": malformed, "split_seed": SPLIT_SEED,
        "source_sha256": source_sha256,
        "style_mean": mean, "style_std": std,
        "style_bank": ((styles[::max(1, len(styles) // 2048)] - mean) / std).clone(),
    }
    save_atomic(manifest, manifest_path)
    for split in SPLITS:
        print(f"{split}: {len(actual[split]):,} participants, "
              f"{sum(shard['trials'] for shard in shards[split]):,} trials", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    prepare(args.archive, WORK, args.workers)
