"""Apply the thresholds in TIMING_EXPERIMENTS.md to matched runs."""

import argparse
import json
from statistics import mean

from model.checkpoint import ROOT


def load_runs(name, seeds=(17, 29, 43)):
    return [json.loads((ROOT / f".data/timing/{name}-{seed}.json").read_text(encoding="utf-8"))
            for seed in seeds]


def rhythm(run):
    return run["generated"]["rhythm_error"]


def gap(run):
    return run["test"]["gap_nll"]


def improvement(baseline, candidate, metric, margin, *, relative=False):
    differences = [metric(old) - metric(new) for old, new in zip(baseline, candidate)]
    thresholds = [margin * metric(old) if relative else margin for old in baseline]
    return (mean(differences) >= mean(thresholds) and
            sum(delta >= threshold for delta, threshold in zip(differences, thresholds)) >= 2)


def generated_error(run, mode, name, stat=None):
    recorded, generated = (run["generated"][key] for key in ("observed", mode))
    if name == "word_boundary":
        return abs((generated["word_entry"]["p50"] - generated["within_word"]["p50"])
                   - (recorded["word_entry"]["p50"] - recorded["within_word"]["p50"]))
    if stat:
        generated, recorded = generated[name][stat], recorded[name][stat]
    else:
        generated, recorded = generated[name], recorded[name]
    return abs(generated - recorded) if generated is not None and recorded is not None else float("inf")


def decision(baseline, candidate, *, smaller=False):
    if [run["seed"] for run in baseline] != [run["seed"] for run in candidate] or len(baseline) < 3:
        raise ValueError("Need at least three matched seeds")
    reasons = []
    gap_delta = mean(map(gap, candidate)) - mean(map(gap, baseline))
    rhythm_delta = mean(map(rhythm, candidate)) / mean(map(rhythm, baseline)) - 1
    if gap_delta > 0.02:
        reasons.append("gap NLL regression")
    if rhythm_delta > 0.05:
        reasons.append("rhythm regression")
    for name, margin in (("action", 0.02), ("wrong_key", 0.05), ("hold", 0.02)):
        delta = mean(new["test"][name] - old["test"][name] for old, new in zip(baseline, candidate))
        if delta > margin:
            reasons.append(f"{name} loss regression")
    for mode in ("raw", "rescaled"):
        for name, stat, margin in (("mistake_rate", None, 0.015),
                                   ("correction_seconds", "mean", 0.15),
                                   ("backspace_runs", "mean", 0.3), ("holds", "p50", 0.02),
                                   ("word_boundary", None, 0.05)):
            new_error = mean(generated_error(run, mode, name, stat) for run in candidate)
            old_error = mean(generated_error(run, mode, name, stat) for run in baseline)
            if new_error == float("inf") or new_error - old_error > margin:
                reasons.append(f"{mode} {name} regression")
    if smaller:
        if candidate[0]["parameters"] >= baseline[0]["parameters"]:
            reasons.append("no parameter savings")
    elif not (improvement(baseline, candidate, gap, 0.02) or
              improvement(baseline, candidate, rhythm, 0.10, relative=True)):
        reasons.append("improvement below threshold")
    return {"keep": not reasons, "gap_delta": gap_delta, "rhythm_delta": rhythm_delta, "reasons": reasons}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--smaller", action="store_true")
    args = parser.parse_args()
    print(json.dumps(decision(load_runs(args.baseline), load_runs(args.candidate), smaller=args.smaller), indent=2))
