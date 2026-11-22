"""Write the full-corpus comparison into the repository's experiment report."""

from model.checkpoint import ROOT

HEADING = "## Full Aalto corpus · Unified 3/10 context"


def row(*values):
    return "| " + " | ".join(str(value) for value in values) + " |"


def update_report(manifest, results):
    first = results[0]
    lines = [
        HEADING, "",
        "The full Aalto ZIP was scanned with the existing physical QWERTY "
        "and valid-trial filters. Participants were assigned to train, validation, "
        f"or test in proportions of 80/10/10 with split seed `{first['split_seed']}`. "
        "All eligible trials from each participant stayed in the same split. "
        "Style normalization, the sampled style bank, and loss class weights were fit on train only.", "",
        f"Archive SHA-256: `{next(iter(manifest['source_sha256'].values()))}`. "
        f"Eligible keystroke entries: {manifest['eligible_entries']:,}. "
        f"Malformed entries skipped: {manifest['malformed_entries']:,}.", "",
        "| Split | Participants | Trials | Events | Mistake rate | Gaps | Pauses | Holds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, split in first["splits"].items():
        lines.append(row(name, f"{split['participants']:,}", f"{split['trials']:,}",
                         f"{split['events']:,}", f"{split['mistake_rate']:.2%}",
                         *(f"{split[key]:,}" for key in ("gap_count", "pause_count", "hold_count"))))
    schedule = first["schedule"]
    precision = "FP16 autocast with FP32 loss accumulation" if schedule["amp"] else "FP32"
    lines.extend([
        "",
        "All models were trained from fresh weights with unified three-component timing, "
        "three characters behind and ten ahead of the current character, a three-layer 128-unit GRU, "
        "and a fixed 0.25 input skip. The GELU models add a 64-unit hidden layer to each output head. "
        "Only the embedding size and output heads vary across the three runs.", "",
        f"Training seed `{first['seed']}`. AdamW used learning rate {schedule['learning_rate']}, "
        f"weight decay {schedule['weight_decay']}, and gradient clipping at {schedule['gradient_clip']}. "
        f"Length-bucketed batches used up to {schedule['batch_size_max']} trials and "
        f"{schedule['padded_events_max']:,} padded positions, except for any single longer trial. "
        f"Training used {schedule['device']} with {precision}. "
        "The objective weights were 1.0 for class-weighted action, 0.35 for wrong key, "
        "0.55 for timing and 0.15 for hold. The validation objective uses recorded styles "
        "and accumulates each head over its valid events before applying its coefficient. "
        f"Training stopped after {schedule['patience']} consecutive epochs without a lower validation "
        "objective and saved the lowest-loss checkpoint. Predictive test metrics use the mean "
        "training style and recorded history. Generation uses the same 128 seeded test prompts per "
        "model and reports both raw and rescaled plans.", "",
        "| Model | Parameters | Epochs | Best epoch | Best validation objective | Training min | Gap NLL | Hold NLL | Rhythm error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for result in results:
        test = result["test"]
        lines.append(row(result["name"], f"{result['parameters']:,}", len(result["epochs"]),
                         result["best_epoch"], f"{result['best_validation_objective']:.6f}",
                         f"{result['epochs'][-1]['elapsed_seconds'] / 60:.2f}", f"{test['gap_nll']:.6f}",
                         f"{test['hold']:.6f}", f"{result['generated']['rhythm_error']:.3f}"))
    lines.extend(["", "### Predictive test metrics", "",
                  "| Model | Action NLL | Wrong-key NLL | Long-gap Brier | Action accuracy | Wrong-action recall | Wrong-key accuracy | Wrong-action AP | Wrong-action AUROC | Hold MAE (ms) | Hold within 20 ms |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for result in results:
        test = result["test"]
        lines.append(row(result["name"], *(f"{test[key]:.5f}" for key in ("action", "wrong_key", "long_gap_brier")),
                         *(f"{test[key]:.2%}" for key in ("action_accuracy", "wrong_action_recall", "wrong_key_accuracy")),
                         *(f"{test[key]:.5f}" for key in ("wrong_action_average_precision", "wrong_action_auroc")),
                         f"{test['hold_mae_ms']:.2f}", f"{test['hold_within_20ms']:.2%}"))
    lines.extend(["", "### Recorded and generated rhythm", "",
                  "| Model | Mode | Lag 1 | Lag 2 | Mean burst | Over 500 ms | Over 1 s | Mistake rate | Correction p90 (ms) | Mean Backspace run | Hold p50 (ms) | Word entry / within word p50 (ms) |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for index, result in enumerate(results):
        modes = ("observed", "raw", "rescaled") if index == 0 else ("raw", "rescaled")
        for mode in modes:
            stats = result["generated"][mode]
            lines.append(row(result["name"] if mode != "observed" else "Reference", mode,
                             *(f"{stats[f'log_gap_acf_{lag}']:.4f}" for lag in (1, 2)),
                             f"{stats['bursts']['mean']:.2f}",
                             *(f"{stats[key]:.2%}" for key in ("over_500ms", "over_1s", "mistake_rate")),
                             milliseconds(stats["correction_seconds"]["p90"]),
                             f"{stats['backspace_runs']['mean']:.2f}" if stats["backspace_runs"]["mean"] else "—",
                             milliseconds(stats["holds"]["p50"]),
                             " / ".join(milliseconds(stats[key]["p50"]) for key in ("word_entry", "within_word"))))
    for result in results:
        name = result["name"]
        lines.extend(["", f"### {name} epoch history", "",
                      f"Local artifacts include `{result['checkpoint']}`, `{result['spec']}`, "
                      f"and `.data/full-corpus/{name}.json`.", "",
                      "| Epoch | Train objective | Validation objective | Elapsed (s) |",
                      "| ---: | ---: | ---: | ---: |"])
        for epoch in result["epochs"]:
            lines.append(row(epoch["epoch"], f"{epoch['train_objective']:.6f}",
                             f"{epoch['validation_objective']:.6f}", f"{epoch['elapsed_seconds']:.1f}"))
    lines.extend(["", "The local `.data/full-corpus/results.json` includes all per-head epoch "
                  "losses, test denominators, calibration bins, mean mixture weights, and full recorded and "
                  "generated distributions with counts, means and quantiles.", "",
                  "Reproduce with `python -m experiments.full_corpus_data` followed by "
                  "`python -m experiments.full_corpus`. The archive belongs at `.data/Keystrokes.zip`.", ""])
    path = ROOT / "TIMING_EXPERIMENTS.md"
    previous = path.read_text(encoding="utf-8")
    before, separator, existing = previous.partition("\n" + HEADING + "\n")
    if separator and "\n## " in existing:
        _, _, later = existing.partition("\n## ")
        after = "\n## " + later
    else:
        after = ""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(before.rstrip() + "\n\n" + "\n".join(lines) + after, encoding="utf-8")
    temporary.replace(path)
    print(f"Updated {path}", flush=True)


def milliseconds(value):
    return "—" if value is None else f"{value * 1000:.0f}"
