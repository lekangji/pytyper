"""Predictive and generated timing metrics shared by all experiment runs."""

import math
import random
from collections import defaultdict

import torch
from torch.nn import functional as F

from model.controls import TypingControls
from model.network import PAUSE_THRESHOLD, PRINTABLE_KEYS, Action, lognormal_mixture_nll
from model.runtime import NeuralKeyEvent
from model.timing import component_log_masses, gap_features



def head_losses(outputs, batch, action_weights=None):
    action, wrong, mix, means, scales, _, pause, hold_means, hold_scales = outputs
    actions = batch["actions"]
    mask = batch["mask"]
    timing = mask & batch["timing_mask"]
    holds = mask & batch["hold_mask"]
    wrong_keys = mask & (actions == Action.WRONG)
    side = batch["pauses"][timing]
    action_indices = actions[timing]
    row, step = timing.nonzero(as_tuple=True)
    key = (row, step, action_indices)
    result = {
        "action": F.cross_entropy(action[mask], actions[mask], weight=action_weights, reduction="none"),
        "wrong_key": F.cross_entropy(wrong[wrong_keys], batch["wrong_keys"][wrong_keys], reduction="none"),
        "hold": lognormal_mixture_nll(
            batch["holds"][holds], torch.zeros_like(hold_means[holds][:, :1]),
            hold_means[holds].gather(1, actions[holds].unsqueeze(1)),
            hold_scales[holds].gather(1, actions[holds].unsqueeze(1)),
        ),
    }
    selected = (*key, side.long()) if pause is not None else key
    params = mix[selected], means[selected], scales[selected]
    result["timing"] = lognormal_mixture_nll(batch["delays"][timing], *params)
    if pause is not None:
        pause_logits = pause[key]
        result["pause"] = F.binary_cross_entropy_with_logits(pause_logits, side.float(), reduction="none")
    return result, (timing, side, params, None if pause is None else pause_logits)


def head_accuracy(outputs, batch):
    actions = batch["actions"]
    mask = batch["mask"]
    wrong_mask = mask & (actions == Action.WRONG)
    hold_mask = mask & batch["hold_mask"]
    action_correct = outputs[0].argmax(-1) == actions
    hold_medians = outputs[7][hold_mask].gather(1, actions[hold_mask].unsqueeze(1)).squeeze(1).exp()
    hold_error_ms = 1000 * (hold_medians - batch["holds"][hold_mask]).abs()
    return {
        "action_accuracy": action_correct[mask],
        "wrong_action_recall": action_correct[wrong_mask],
        "wrong_key_accuracy": outputs[1][wrong_mask].argmax(-1) == batch["wrong_keys"][wrong_mask],
        "hold_mae_ms": hold_error_ms,
        "hold_within_20ms": hold_error_ms <= 20,
    }


def wrong_action_ranking(probabilities, is_wrong):
    order = probabilities.argsort(descending=True)
    _, sizes = probabilities[order].unique_consecutive(return_counts=True)
    thresholds = sizes.cumsum(0)
    true_positives = is_wrong[order].cumsum(0)[thresholds - 1].double()
    false_positives = thresholds - true_positives
    positives = int(is_wrong.sum())
    negatives = is_wrong.numel() - positives
    average_precision = auroc = None
    if positives:
        recall = true_positives / positives
        precision = true_positives / thresholds
        average_precision = float((precision * recall.diff(prepend=recall.new_zeros(1))).sum())
        if negatives:
            auroc = float(torch.trapezoid(F.pad(recall, (1, 0)), F.pad(false_positives / negatives, (1, 0))))
    return {"wrong_action_average_precision": average_precision, "wrong_action_auroc": auroc}


def mixture_side_log_masses(params):
    logits, means, scales = params
    return (logits.log_softmax(-1).unsqueeze(-1) + component_log_masses(means, scales)).logsumexp(-2)


def previous_gap_inputs(batch):
    previous = F.pad(batch["delays"][:, :-1], (1, 0))
    valid = F.pad((batch["timing_mask"] & batch["mask"])[:, :-1], (1, 0), value=False) & batch["mask"]
    return gap_features(previous, valid)


def forward(model, batch):
    history = previous_gap_inputs(batch) if model.use_previous_gap else None
    mask = batch["mask"]
    lengths = None if mask[:, -1].all() else mask.sum(1)
    return model(batch["target_context"], batch["buffer_context"], batch["previous_key"],
                 batch["previous_action"], batch["style"], keyboard_features=batch["keyboard_features"],
                 previous_gap=history, lengths=lengths)


@torch.inference_mode()
def predict(model, loader, *, include_accuracy=False):
    totals, counts = defaultdict(float), defaultdict(int)
    device = next(model.parameters()).device
    calibration = torch.zeros(10, 3, dtype=torch.float64, device=device)
    usage = torch.zeros(model.timing_components, dtype=torch.float64, device=device)
    wrong_probabilities, wrong_labels = [], []
    model.eval()
    for batch in loader:
        batch = {name: value.to(device) for name, value in batch.items()}
        outputs = forward(model, batch)
        scores, (_, side, params, pause) = head_losses(outputs, batch)
        sides = mixture_side_log_masses(params)
        if pause is None:
            scores["gap_nll"] = scores["timing"]
            probabilities = sides[:, 1].exp()
        else:
            scores["gap_nll"] = scores["timing"] + scores["pause"] + sides.gather(1, side.long().unsqueeze(1)).squeeze(1)
            probabilities = pause.sigmoid()
        scores["long_gap_brier"] = (probabilities - side.float()).square()
        if include_accuracy:
            scores.update(head_accuracy(outputs, batch))
            mask = batch["mask"]
            wrong_probabilities.append(outputs[0][mask].softmax(-1)[:, Action.WRONG])
            wrong_labels.append(batch["actions"][mask] == Action.WRONG)
        for name, values in scores.items():
            totals[name] += values.double().sum().item()
            counts[name] += values.numel()
        bins = (probabilities * 10).long().clamp(max=9)
        values = torch.stack((torch.ones_like(probabilities), probabilities, side.float()), 1)
        calibration.index_add_(0, bins, values.double())
        usage += params[0].softmax(-1).double().sum(0)
    summary = {name: value / counts[name] if counts[name] else None for name, value in totals.items()}
    if include_accuracy:
        if wrong_labels:
            summary.update(wrong_action_ranking(torch.cat(wrong_probabilities), torch.cat(wrong_labels)))
        summary["accuracy_counts"] = {
            "actions": counts["action_accuracy"],
            "wrong_actions": counts["wrong_key_accuracy"],
            "holds": counts["hold_mae_ms"],
        }
    return {**summary,
            "calibration": [{"count": int(n), "predicted": p / n, "observed": y / n}
                            if n else {"count": 0, "predicted": None, "observed": None}
                            for n, p, y in calibration.tolist()],
            "component_usage": (usage / usage.sum()).tolist()}


def recorded_plan(trial, left=6):
    events = []
    buffer = []
    for i, action in enumerate(trial.actions.tolist()):
        if action == Action.BACKSPACE:
            key = "BACKSPACE"
            if buffer:
                buffer.pop()
        else:
            index = int(trial.target_context[i, left]) - 2 if action == Action.CORRECT else int(trial.wrong_keys[i])
            key = PRINTABLE_KEYS[index]
            buffer.append(key)
        events.append(NeuralKeyEvent(key, float(trial.delays[i]), float(trial.holds[i])))
    return "".join(buffer), events


def timing_metrics(plans, *, recorded=False):
    gaps, bursts, holds, runs, corrections, within, word_entry = ([] for _ in range(7))
    lag_pairs = {1: ([], []), 2: ([], [])}
    mistakes = events_count = 0
    for target, events in plans:
        delays = [event.delay_before for event in events[1:]]
        gaps.extend(delays)
        log_gaps = [math.log(gap) for gap in delays]
        for lag, (previous, current) in lag_pairs.items():
            previous.extend(log_gaps[:-lag])
            current.extend(log_gaps[lag:])
        buffer = []
        burst = 1
        run = 0
        elapsed = 0.0
        last_correct = ""
        first_error = None
        for index, event in enumerate(events):
            key, gap, hold = event.key, event.delay_before, event.hold_seconds
            elapsed += gap
            events_count += 1
            if hold > 0:
                holds.append(hold)
            if index:
                if gap >= PAUSE_THRESHOLD:
                    bursts.append(burst)
                    burst = 0
                burst += 1
            if key == "BACKSPACE":
                run += 1
                if not buffer and not recorded:
                    raise ValueError("Invalid Backspace on empty buffer")
                if buffer:
                    buffer.pop()
                if all(buffer):
                    if first_error is not None:
                        corrections.append(elapsed - first_error)
                    first_error = None
                last_correct = ""
                continue
            if run:
                runs.append(run)
                run = 0
            correct = first_error is None and len(buffer) < len(target) and key == target[len(buffer)]
            if correct and key.isalpha():
                if last_correct == " ":
                    word_entry.append(gap)
                elif last_correct.isalpha():
                    within.append(gap)
            last_correct = key if correct else ""
            if not correct:
                mistakes += 1
                if first_error is None:
                    first_error = elapsed
            buffer.append(correct)
        if first_error is not None or len(buffer) != len(target):
            raise ValueError("Plan does not replay to the requested text")
        if run:
            runs.append(run)
        bursts.append(burst)

    def summary(values):
        if not values:
            return {"count": 0, "mean": None, "p50": None, "p90": None}
        values = torch.tensor(values, dtype=torch.float64)
        p50, p90 = values.quantile(torch.tensor([0.5, 0.9], dtype=values.dtype)).tolist()
        return {"count": len(values), "mean": values.mean().item(), "p50": p50, "p90": p90}

    result = dict(bursts=summary(bursts), backspace_runs=summary(runs), correction_seconds=summary(corrections),
                  holds=summary(holds), word_entry=summary(word_entry), within_word=summary(within),
                  events=events_count, mistakes=mistakes,
                  mistake_rate=mistakes / events_count,
                  over_500ms=sum(gap > 0.5 for gap in gaps) / len(gaps),
                  over_1s=sum(gap > 1.0 for gap in gaps) / len(gaps))
    for lag, pairs in lag_pairs.items():
        result[f"log_gap_acf_{lag}"] = float(torch.corrcoef(torch.tensor(pairs, dtype=torch.float64))[0, 1])
    return result


def generated(runtime, trials, test_indices):
    sample = random.Random(101).sample(test_indices, min(128, len(test_indices)))
    references, raw, calibrated = [], [], []
    for index in sample:
        prompt, observed = recorded_plan(trials[index])
        before, after = runtime.plan(prompt, TypingControls(error_rate=0.045))
        references.append((prompt, observed))
        raw.append((prompt, before))
        calibrated.append((prompt, after))
    observed = timing_metrics(references, recorded=True)
    result = {"observed": observed, "raw": timing_metrics(raw), "rescaled": timing_metrics(calibrated)}
    result["rhythm_error"] = sum(
        abs(result[mode][name] - observed[name]) / scale
        for mode in ("raw", "rescaled")
        for name, scale in (("log_gap_acf_1", 0.1), ("log_gap_acf_2", 0.1),
                            ("over_500ms", 0.03), ("over_1s", 0.02))
    ) + sum(abs(result[mode]["bursts"]["mean"] - observed["bursts"]["mean"]) / 3.0
            for mode in ("raw", "rescaled"))
    return result
