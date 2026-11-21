"""Train the human typing event model on local Aalto archive segments."""

from __future__ import annotations

import argparse
import random
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from model.network import Action, TypingEventModel, lognormal_mixture_nll
from training.dataset import collate_sequences, load_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", type=Path, action="append", required=True, help="Contiguous ZIP local-record range; repeat for more ranges")
    parser.add_argument("--metadata-entry", type=Path, required=True, help="ZIP local record for metadata_participants.txt")
    parser.add_argument("--output", type=Path, default=Path("models/human_typist_v5.pt"))
    parser.add_argument("--participants", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--target-left", type=int, default=6)
    parser.add_argument("--target-right", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--no-keyboard-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing checkpoint: {args.output}")
    if args.target_left < 1 or args.target_right < 1 or args.layers < 1:
        raise ValueError("Context widths and GRU layer count must be positive")
    sequences = []
    for segment in args.segment:
        sequences.extend(
            load_sequences(segment, args.metadata_entry, args.participants, args.target_left, args.target_right)
        )
    if len(sequences) < 100:
        raise RuntimeError(f"Only {len(sequences)} valid trials were found; at least 100 are required")
    participant_ids = sorted({sequence.participant_id for sequence in sequences})
    random.shuffle(participant_ids)
    validation_count = max(1, int(len(participant_ids) * 0.1))
    validation_ids = set(participant_ids[:validation_count])
    train_sequences = [item for item in sequences if item.participant_id not in validation_ids]
    validation_sequences = [item for item in sequences if item.participant_id in validation_ids]
    styles = torch.stack([sequence.style_raw for sequence in train_sequences])
    style_mean = styles.mean(0)
    style_std = styles.std(0).clamp_min(1e-4)
    normalized_styles = (styles - style_mean) / style_std

    collate = partial(collate_sequences, style_mean=style_mean, style_std=style_std)
    train_loader = DataLoader(train_sequences, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    validation_loader = DataLoader(validation_sequences, batch_size=args.batch_size, collate_fn=collate)

    architecture = {
        "embedding_dim": 16,
        "hidden_size": 128,
        "layers": args.layers,
        "timing_components": 5,
        "target_left": args.target_left,
        "target_right": args.target_right,
        "use_keyboard_features": not args.no_keyboard_features,
        "event_model_v5": True,
    }
    model = TypingEventModel(**architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    counts = torch.bincount(torch.cat([item.actions for item in train_sequences]).long(), minlength=len(Action)).float()
    mistake_rate = float(counts[Action.WRONG] / counts.sum())
    print(
        f"Loaded {len(sequences):,} valid trials from {len(participant_ids):,} participants; "
        f"training={len(train_sequences):,} trials/{int(counts.sum()):,} events, "
        f"validation={len(validation_sequences):,} trials",
        flush=True,
    )
    action_weights = (counts.sum() / counts.clamp_min(1)).sqrt()
    action_weights = (action_weights / action_weights[0]).clamp(max=8.0)

    best_validation = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, action_weights)
        validation_loss = run_epoch(model, validation_loader, None, action_weights)
        print(f"epoch={epoch} train={train_loss:.4f} validation={validation_loss:.4f}", flush=True)
        if validation_loss < best_validation:
            best_validation = validation_loss
            torch.save(
                {
                    "format_version": 3,
                    "architecture": architecture,
                    "model_state": model.state_dict(),
                    "style_mean": style_mean,
                    "style_std": style_std,
                    "style_bank": normalized_styles[:: max(1, len(normalized_styles) // 2048)].clone(),
                    "sampling_temperature": 0.86,
                    "dataset": "Aalto 136M Keystrokes; physical QWERTY subset",
                    "dataset_url": "https://userinterfaces.aalto.fi/136Mkeystrokes/",
                    "dataset_license": "Non-commercial use with attribution",
                    "training_trials": len(train_sequences),
                    "validation_trials": len(validation_sequences),
                    "training_events": int(counts.sum()),
                    "training_mistake_rate": mistake_rate,
                    "training_pause_rate": sum(int(item.pauses.sum()) for item in train_sequences) / int(counts.sum()),
                    "training_rollover_rate": float(styles[:, 5].mean()),
                    "training_loss": train_loss,
                    "epoch": epoch,
                    "segments": [str(path) for path in args.segment],
                    "seed": args.seed,
                    "best_validation_loss": best_validation,
                },
                args.output,
            )


def run_epoch(
    model: TypingEventModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    action_weights: torch.Tensor,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batches = 0
    for batch in loader:
        with torch.set_grad_enabled(training):
            action_logits, wrong_logits, mix, means, scales, _, pause_logits, hold_means, hold_scales = model(
                batch["target_context"],
                batch["buffer_context"],
                batch["previous_key"],
                batch["previous_action"],
                batch["style"],
                keyboard_features=batch["keyboard_features"],
            )
            mask = batch["mask"]
            action_loss = nn.functional.cross_entropy(
                action_logits[mask], batch["actions"][mask], weight=action_weights
            )
            wrong_mask = mask & (batch["actions"] == Action.WRONG)
            wrong_loss = (
                nn.functional.cross_entropy(wrong_logits[wrong_mask], batch["wrong_keys"][wrong_mask])
                if wrong_mask.any()
                else action_loss.new_zeros(())
            )
            timing_mask = mask & batch["timing_mask"]
            action_index = batch["actions"]
            pause_index = batch["pauses"].long()
            row = torch.arange(action_index.shape[0])[:, None]
            step = torch.arange(action_index.shape[1])[None, :]
            selected_mix = mix[row, step, action_index, pause_index]
            selected_means = means[row, step, action_index, pause_index]
            selected_scales = scales[row, step, action_index, pause_index]
            timing_loss = lognormal_mixture_nll(
                batch["delays"][timing_mask],
                selected_mix[timing_mask], selected_means[timing_mask], selected_scales[timing_mask],
            ).mean()
            selected_pause_logits = pause_logits.gather(2, action_index.unsqueeze(-1)).squeeze(-1)
            pause_loss = nn.functional.binary_cross_entropy_with_logits(
                selected_pause_logits[timing_mask], batch["pauses"][timing_mask].float()
            )
            hold_mask = mask & batch["hold_mask"]
            selected_hold_means = hold_means.gather(2, action_index.unsqueeze(-1)).squeeze(-1)
            selected_hold_scales = hold_scales.gather(2, action_index.unsqueeze(-1)).squeeze(-1)
            hold_loss = lognormal_mixture_nll(
                batch["holds"][hold_mask],
                torch.zeros_like(selected_hold_means[hold_mask]).unsqueeze(-1),
                selected_hold_means[hold_mask].unsqueeze(-1),
                selected_hold_scales[hold_mask].unsqueeze(-1),
            ).mean()
            loss = action_loss + 0.35 * wrong_loss + 0.55 * timing_loss + 0.25 * pause_loss + 0.15 * hold_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += loss.item()
        batches += 1
    return total_loss / max(1, batches)


if __name__ == "__main__":
    main()
