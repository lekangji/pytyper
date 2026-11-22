# Timing experiments

## Phase 0 · Checkpoint compatibility

Five weight files, three distinct SHA-256 hashes. All three formats load.
The v3 and v4 backups retain their original two-layer GRUs. Experiments use
the shipped model's three-layer, 128-unit GRU and linear heads.

| Spec | Checkpoint format | Parameters | Decoder |
| --- | ---: | ---: | --- |
| [Archive v3](models/specs/archive-v3.json) | 1 | 255,137 | Global mixture, v3 error control |
| [Archive v4](models/specs/archive-v4.json) | 2 | 276,257 | Global mixture, v4 error control |
| [Release](models/specs/release.json) | 3 | 386,421 | Per-action split mixture, pause and hold heads |

Twelve golden plans captured from the unmodified release and the matching v3,
v4, and v5 backup runtimes. Spec-based loading matched every key, delay, and
hold on PyTorch 2.15.0+cpu with one thread. Independent replay reconstructed all
twelve target strings and exercised Backspaces in every runtime. The release's
normal weight-file entry point matched explicit spec loading. The SHA-256
mismatch test passed.

## Phase 1 · Baseline protocol

The backup trainer and saved seed 20260920 reconstructed 11,367 training and
1,263 validation participants, with 99,684 and 11,261 trials respectively.
All 4,349,953 training events, the mistake, pause, and rollover rates, and all
three style tensors matched the bundle. Scoring the shipped weights with the
original objective, batch size 48, and trial styles reproduced 1.1811789330
against the saved 1.1811789355. The absolute difference was 2.54 × 10⁻⁹.
The experiment test pool contains 1,148 participants from release training,
so the shipped loss serves as a historical reference.

Freeze the experiment split by shuffling sorted participant IDs with seed
20261119 and assigning the first 10% to validation and next 10% to held-out
scoring. Fit style normalization and build the style bank on the training
partition. Use the mean training style for predictive validation and scoring.
For each generated prompt, sample one of the 64 training-bank rows closest
to the requested log-median gap and mistake rate. The runtime applies WPM,
rhythm, and error controls to that row and keeps its hold and rollover
features.

Use matched training seeds 17, 29, and 43. Lower held-out gap NLL by at least
0.02 nats/gap or lower rhythm error by at least 10% to retain a timing change.
Require the improvement on the seed mean and in at least two matched seeds.
Smaller inputs require fewer parameters. Allow at most +0.02 nats/gap, +5% rhythm
error, +0.02 nats for action or hold, and +0.05 nats for wrong keys across
variants. Use 3,072 training, 768 validation, and 1,536 held-out trials for the
matched CPU experiments. Use three epochs and select by validation loss.

Use 128 fixed held-out prompts with generation seed 251 and controls of 75 WPM,
4.5% errors, neutral rhythm and neutral pause tendency. The prompts contribute
text only. Compare generated plans with recorded timing on those prompts.
Compute rhythm error as the sum of absolute deviations from recorded lag-1 and
lag-2 log-gap correlation, gap rates above 500 ms and 1 s, and mean burst length.
Divide each deviation by 0.10, 0.10, 0.03, 0.02, and 3 keys respectively, and
include both raw and speed-rescaled metrics. A gap of at least 500 ms starts a
new burst. Exclude each trial's first gap.

Guardrails use increases in absolute error relative to the observed traces.
Allow at most 1.5 percentage points for mistake rate, 150 ms for mean time to
correct, 0.3 keys for mean Backspace run length, 20 ms for median hold, and
50 ms for the difference between median word-entry and within-word gaps. Apply
these to the seed mean in both raw and speed-rescaled plans. Require exact
visible-buffer replay for every generated plan.

The capped partitions contained 2,603 training, 587 validation, and 867
held-out participants, with no overlap. Baseline gap NLL across seeds was
0.9264–0.9568 nats per valid gap, with a sample standard deviation of 0.0165.
Rhythm error ranged from 14.274 to 18.023, with a sample standard deviation
of 1.875. The three runs provide the matched-seed reference for later phases.

## Phase 2 · Unified mixture

Predict three components per action with one linear timing head. Both models
score valid gaps in log-seconds with log means bounded to [−5, 1.7], log
standard deviations to [−3.5, 1.2], and training gaps to [8 ms, 8 s].
For the split model, combine the observed side's Bernoulli log probability
with the branch density divided by its mass on that side. The unified
model scores the full mixture. These scores exclude sampling clamps and
fallback point masses. Per-head action, wrong-key, and hold losses are
event means over their applicable masks. The split timing and pause losses
remain available as separate training-head diagnostics.

The unified decoder samples directly at neutral pause tendency. Other settings
multiply the odds of a gap of at least 500 ms, then sample a component weighted
by its mass on the chosen side. Truncated normal sampling in log-seconds uses
an exponential proposal for distant tails and has no rejection-count fallback.
Both decoders clamp sampled gaps to 8 ms–5 s and holds to 20–180 ms. The legacy
split decoder tries eight draws before using 499 or 500 ms on the requested
side. The first event has zero delay. Speed calibration follows sampling and
can move gaps across thresholds.

The unified mixture lowered mean gap NLL by 0.0109 nats/gap. Mean rhythm
error rose 13.1%. It failed both the improvement rule and the rhythm
guardrail, so phase 3 starts from split timing. Long-gap Brier score moved
from 0.07322 to 0.07256. The lowest probability bin averaged 2.42% predicted
versus 5.70% observed for the unified mixture and 3.17% versus 5.90% for the
split model. Mean unified component weights were 48.5%, 21.6%, and 30.0%
across the three runs. Component labels are exchangeable across seeds.

## Phase 3 · Previous-gap feedback

Feed the preceding event's log-gap and a missing flag into the input
projection. Both training and generation use the sampling bounds of 8 ms–5 s
for this input. The first two events lack a previous timed gap. Scoring and
training keep the original timing targets and masks. Generation feeds back the
sampled gap before speed rescaling, including during forced completion.

Compared with the split baseline, mean gap NLL rose 0.0146 nats/gap, rhythm
error fell 0.8%, and hold loss exceeded the +0.02 nats guardrail. The feedback
candidate failed acceptance. The noise trial was conditional on a held-out
recorded-history improvement with a free-running regression. That condition
did not occur. Phase 4 uses split timing without feedback.

## Phase 4 · Smaller inputs

Compare each context reduction and the standalone 8-dimensional embedding
against the same selected baseline. The shipped width is 6 characters behind,
the current key, and 15 ahead. A runtime right width of 11 keeps the current key and 10 ahead.
Context slices preserve the current-key position in both training and
generation. Word-entry gaps measure consecutive correct events from a space
to a letter. Within-word gaps measure consecutive correct letters.

Pair the 8-dimensional embedding with the smallest context that meets the
guardrails. If none does, use the context with the lowest mean validation
score. Confirm the pairing with all three matched seeds. Among configurations
that pass, favor the largest parameter reduction.

| Standalone input | Parameters saved | Gap NLL change | Rhythm error change | Decision |
| --- | ---: | ---: | ---: | --- |
| 3 behind, 15 ahead | 6,144 | −0.0033 | +2.4% | Pass |
| 6 behind, 10 ahead | 10,240 | +0.0058 | +4.4% | Pass |
| 3 behind, 10 ahead | 16,384 | −0.0097 | +6.0% | Rhythm regression |
| 8-dimensional embedding | 30,472 | +0.0063 | +8.6% | Rhythm and mistake-rate regressions |

The confirmation run uses 6 characters behind, the current key, 10 ahead,
and an 8-dimensional embedding.
It saves 35,592 parameters. Mean gap NLL fell 0.0018 nats/gap, but rhythm
error rose 8.4% and action loss exceeded its guardrail. The selected smaller
configuration keeps the 16-dimensional embedding and uses 376,181 parameters,
down 2.65% from the split baseline. Its three checkpoints and matching specs
are in `.data/timing/context-6-10-*`.

## Shipped checkpoint comparison

For the production decision, intersect the original validation participants
with the frozen experiment test participants. All 1,031 trials from these
115 participants were excluded from training for each scored model. Seed 17
had the lowest validation score for the selected smaller configuration.
The same predictive and generated metrics apply on this common subset.

| Model | Seed | Parameters | Eval. time (s) | Action NLL | Wrong-key NLL | Hold NLL | Pause NLL | Timing NLL | Gap NLL | Long-gap Brier | Rhythm error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Shipped | 20260920 | 386,421 | 13.0 | 0.3838 | 1.9996 | 0.3396 | 0.2255 | 0.6021 | 0.8144 | 0.0610 | 19.526 |
| Retrained split baseline | 17 | 386,421 | 13.7 | 0.1867 | 3.3039 | 0.3893 | 0.2351 | 0.6668 | 0.8899 | 0.0638 | 16.231 |
| Shorter context | 17 | 376,181 | 12.0 | 0.1893 | 3.2716 | 0.4156 | 0.2266 | 0.6546 | 0.8584 | 0.0616 | 14.661 |

The shorter context improved rhythm on this subset, with gap NLL 0.0440
nats higher than the shipped weights. This exceeds the +0.02 predictive
regression tolerance. The app uses the shipped weights. The smaller model
passed the matched-seed experiment against the retrained baseline and is
available as an experiment checkpoint. Full-corpus retraining and scoring
of the smaller architecture remain unverified.

## Reproduction

Every tested architecture has a name in [experiments/variants.py](experiments/variants.py).
Build any variant with fresh weights using `build_model(name)`.

```python
from experiments.variants import build_model

model = build_model("unified-all-heads-mlp")
```

The repository includes the [training code](training/) and the shipped checkpoint.
Training and evaluation also need local Aalto archive segments under `.backup/v5/.data/`
or the prepared `.data/timing/trials.pt` cache. These runs used Windows,
Python 3.12.14, and PyTorch 2.15.0+cpu. Existing experiment artifacts are protected
from overwrite.

```powershell
python -m unittest discover -s tests
python -m experiments.data --threads 4
python -m experiments.run --run baseline --threads 4
python -m experiments.run --run mixture --threads 4
python -m experiments.compare baseline mixture
python -m experiments.run --run feedback --base .data/timing/baseline-17.pt --threads 4
python -m experiments.compare baseline feedback
foreach ($variant in 'context-3-15', 'context-6-10', 'context-3-10', 'embedding-8') {
    python -m experiments.run --run $variant --base .data/timing/baseline-17.pt --threads 3
    python -m experiments.compare baseline $variant --smaller
}
python -m experiments.run --run combined --base .data/timing/context-6-10-17.pt --threads 3
python -m experiments.compare baseline combined --smaller
```

The input runs shared the CPU. Reported wall time includes training,
checkpoint writing, and held-out evaluation, with concurrency affecting
timing. Every run saves one versioned spec for its selected checkpoint.
Pass the weights path and the JSON path via `spec_path=` to
`NeuralTypingRuntime` to validate the architecture and SHA-256 when loading.

The [experiment results](experiments/results.json) include calibration bin
counts, component weights, per-head losses, and recorded, raw, and rescaled
timing metrics for every run.

## Generated timing on the frozen test prompts

Each model row averages seeds 17, 29, and 43. The recorded row uses the same 128 prompts.

| Model | Plan | Log-gap lag 1 | Lag 2 | Mean burst | >500 ms | >1 s | Mistake rate | Mean correction (ms) | Mean Backspace run | Median hold (ms) | Median entry / within (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recorded | Observed | 0.261 | 0.255 | 9.90 | 8.07% | 1.58% | 4.20% | 824 | 1.72 | 100 | 211 / 148 |
| baseline | Raw | 0.070 | 0.056 | 11.46 | 6.97% | 1.20% | 7.16% | 596 | 1.28 | 103 | 199 / 146 |
| baseline | Rescaled | 0.101 | 0.088 | 25.09 | 1.99% | 0.21% | 7.16% | 338 | 1.28 | 103 | 133 / 97 |
| mixture | Raw | 0.071 | 0.051 | 12.44 | 6.18% | 0.95% | 7.30% | 574 | 1.29 | 105 | 210 / 141 |
| mixture | Rescaled | 0.081 | 0.063 | 27.79 | 1.60% | 0.15% | 7.30% | 349 | 1.29 | 105 | 144 / 96 |
| feedback | Raw | 0.088 | 0.073 | 12.15 | 6.25% | 0.99% | 5.71% | 593 | 1.30 | 99 | 199 / 144 |
| feedback | Rescaled | 0.100 | 0.085 | 24.27 | 2.06% | 0.13% | 5.71% | 375 | 1.30 | 99 | 144 / 102 |
| context-3-15 | Raw | 0.078 | 0.034 | 11.54 | 6.81% | 0.99% | 6.14% | 544 | 1.24 | 106 | 209 / 147 |
| context-3-15 | Rescaled | 0.091 | 0.049 | 24.15 | 2.10% | 0.16% | 6.14% | 331 | 1.24 | 106 | 140 / 99 |
| context-6-10 | Raw | 0.078 | 0.034 | 10.88 | 7.40% | 1.26% | 8.57% | 687 | 1.43 | 105 | 204 / 143 |
| context-6-10 | Rescaled | 0.105 | 0.063 | 26.96 | 1.82% | 0.17% | 8.57% | 384 | 1.43 | 105 | 132 / 91 |
| context-3-10 | Raw | 0.070 | 0.033 | 12.00 | 6.37% | 0.94% | 5.39% | 668 | 1.28 | 110 | 206 / 148 |
| context-3-10 | Rescaled | 0.075 | 0.041 | 24.09 | 2.07% | 0.20% | 5.39% | 421 | 1.28 | 110 | 145 / 103 |
| embedding-8 | Raw | 0.074 | 0.048 | 10.73 | 7.90% | 1.32% | 9.81% | 780 | 1.39 | 112 | 215 / 153 |
| embedding-8 | Rescaled | 0.121 | 0.098 | 27.97 | 1.77% | 0.14% | 9.81% | 392 | 1.39 | 112 | 127 / 89 |
| combined | Raw | 0.061 | 0.036 | 11.47 | 6.86% | 1.12% | 7.70% | 576 | 1.27 | 109 | 212 / 153 |
| combined | Rescaled | 0.076 | 0.052 | 26.52 | 1.82% | 0.24% | 7.70% | 335 | 1.27 | 109 | 133 / 94 |
| unified-context-3-15 | Raw | 0.062 | 0.044 | 10.04 | 8.01% | 1.39% | 5.36% | 744 | 1.26 | 114 | 223 / 154 |
| unified-context-3-15 | Rescaled | 0.065 | 0.049 | 21.91 | 2.53% | 0.28% | 5.36% | 435 | 1.26 | 114 | 144 / 97 |
| unified-context-6-10 | Raw | 0.059 | 0.045 | 10.19 | 8.05% | 1.42% | 7.28% | 651 | 1.30 | 104 | 216 / 155 |
| unified-context-6-10 | Rescaled | 0.070 | 0.057 | 23.71 | 2.23% | 0.26% | 7.28% | 376 | 1.30 | 104 | 135 / 94 |
| unified-context-3-10 | Raw | 0.082 | 0.030 | 10.69 | 7.79% | 1.19% | 6.34% | 702 | 1.28 | 108 | 223 / 148 |
| unified-context-3-10 | Rescaled | 0.080 | 0.029 | 25.29 | 1.96% | 0.12% | 6.34% | 421 | 1.28 | 108 | 145 / 95 |
| unified-embedding-8 | Raw | 0.061 | 0.024 | 10.83 | 7.30% | 1.03% | 6.11% | 703 | 1.28 | 105 | 213 / 149 |
| unified-embedding-8 | Rescaled | 0.060 | 0.024 | 24.10 | 2.10% | 0.19% | 6.11% | 426 | 1.28 | 105 | 140 / 96 |
| unified-feedback | Raw | 0.047 | 0.045 | 10.06 | 9.10% | 1.63% | 14.31% | 704 | 1.33 | 104 | 224 / 153 |
| unified-feedback | Rescaled | 0.119 | 0.116 | 40.02 | 1.34% | 0.11% | 14.31% | 309 | 1.33 | 104 | 118 / 77 |
| unified-gru-residual-fixed | Raw | 0.057 | 0.025 | 10.62 | 7.51% | 1.26% | 6.34% | 647 | 1.27 | 110 | 216 / 144 |
| unified-gru-residual-fixed | Rescaled | 0.056 | 0.024 | 24.54 | 2.05% | 0.17% | 6.34% | 393 | 1.27 | 110 | 144 / 95 |
| unified-gru-residual-learnable | Raw | 0.067 | 0.026 | 11.53 | 6.75% | 1.22% | 6.31% | 712 | 1.39 | 108 | 213 / 143 |
| unified-gru-residual-learnable | Rescaled | 0.072 | 0.031 | 23.60 | 2.20% | 0.21% | 6.31% | 434 | 1.39 | 108 | 147 / 97 |
| unified-timing-mlp | Raw | 0.076 | 0.044 | 10.54 | 7.78% | 1.45% | 10.56% | 704 | 1.38 | 106 | 226 / 146 |
| unified-timing-mlp | Rescaled | 0.157 | 0.126 | 27.31 | 1.99% | 0.19% | 10.56% | 364 | 1.38 | 106 | 132 / 86 |
| unified-wrong-key-mlp | Raw | 0.067 | 0.050 | 11.18 | 7.21% | 1.20% | 10.32% | 689 | 1.31 | 107 | 206 / 149 |
| unified-wrong-key-mlp | Rescaled | 0.136 | 0.120 | 29.75 | 1.73% | 0.17% | 10.32% | 362 | 1.31 | 107 | 126 / 89 |
| unified-action-mlp | Raw | 0.086 | 0.070 | 10.73 | 7.60% | 1.54% | 7.35% | 722 | 1.33 | 104 | 206 / 141 |
| unified-action-mlp | Rescaled | 0.129 | 0.115 | 23.88 | 2.23% | 0.27% | 7.35% | 391 | 1.33 | 104 | 136 / 93 |

## Run table

All 54 runs use the 1,536-trial frozen experiment holdout. Losses are nats
per applicable event. Gap NLL uses the shared log-seconds score. Pause and
timing are the model-specific training-head losses. Unified models have no
separate pause head, so their pause score is N/A. A dash in the decision
column means the round recorded no formal pass or reject decision.

| Model | Seed | Best epoch | Parameters | Wall (s) | Action NLL | Wrong-key NLL | Hold NLL | Pause NLL | Timing NLL | Gap NLL | Long-gap Brier | Rhythm error | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 17 | 2 | 386,421 | 90.4 | 0.1914 | 3.3241 | 0.4398 | 0.2699 | 0.7003 | 0.9568 | 0.0737 | 16.231 | Reference |
| baseline | 29 | 2 | 386,421 | 87.4 | 0.1886 | 3.3475 | 0.3927 | 0.2641 | 0.6825 | 0.9264 | 0.0725 | 14.274 | Reference |
| baseline | 43 | 3 | 386,421 | 87.6 | 0.2696 | 3.2797 | 0.3841 | 0.2753 | 0.6798 | 0.9304 | 0.0734 | 18.023 | Reference |
| mixture | 17 | 2 | 377,907 | 91.2 | 0.2095 | 3.3102 | 0.4032 | N/A | 0.9389 | 0.9389 | 0.0719 | 16.210 | Reject |
| mixture | 29 | 3 | 377,907 | 85.2 | 0.2100 | 3.2533 | 0.4113 | N/A | 0.9320 | 0.9320 | 0.0727 | 19.196 | Reject |
| mixture | 43 | 2 | 377,907 | 85.6 | 0.2323 | 3.3021 | 0.3750 | N/A | 0.9100 | 0.9100 | 0.0731 | 19.501 | Reject |
| feedback | 17 | 3 | 386,677 | 91.1 | 0.1913 | 3.3029 | 0.5601 | 0.2650 | 0.6938 | 0.9463 | 0.0718 | 15.545 | Reject |
| feedback | 29 | 3 | 386,677 | 87.0 | 0.1832 | 3.2783 | 0.3841 | 0.2721 | 0.7144 | 0.9650 | 0.0724 | 15.106 | Reject |
| feedback | 43 | 2 | 386,677 | 88.5 | 0.2124 | 3.3117 | 0.3586 | 0.2636 | 0.7014 | 0.9462 | 0.0721 | 17.480 | Reject |
| context-3-15 | 17 | 3 | 380,277 | 106.0 | 0.1909 | 3.3005 | 0.3836 | 0.2618 | 0.7040 | 0.9523 | 0.0716 | 14.983 | Pass |
| context-3-15 | 29 | 3 | 380,277 | 101.3 | 0.2145 | 3.2787 | 0.3673 | 0.2678 | 0.6844 | 0.9232 | 0.0726 | 17.484 | Pass |
| context-3-15 | 43 | 3 | 380,277 | 99.1 | 0.2261 | 3.2367 | 0.3654 | 0.2663 | 0.6854 | 0.9281 | 0.0729 | 17.232 | Pass |
| context-6-10 | 17 | 3 | 376,181 | 105.8 | 0.1951 | 3.3028 | 0.4516 | 0.2628 | 0.7017 | 0.9389 | 0.0716 | 16.532 | Selected for experiments |
| context-6-10 | 29 | 2 | 376,181 | 103.2 | 0.2050 | 3.3205 | 0.3540 | 0.2735 | 0.7107 | 0.9651 | 0.0729 | 15.276 | Selected for experiments |
| context-6-10 | 43 | 3 | 376,181 | 98.2 | 0.2220 | 3.2861 | 0.3846 | 0.2609 | 0.6922 | 0.9270 | 0.0717 | 18.874 | Selected for experiments |
| context-3-10 | 17 | 3 | 370,037 | 104.5 | 0.2415 | 3.2993 | 0.3906 | 0.2641 | 0.6996 | 0.9459 | 0.0716 | 18.137 | Reject |
| context-3-10 | 29 | 3 | 370,037 | 99.9 | 0.2165 | 3.2532 | 0.4616 | 0.2716 | 0.6671 | 0.9073 | 0.0725 | 17.292 | Reject |
| context-3-10 | 43 | 3 | 370,037 | 98.0 | 0.2143 | 3.2503 | 0.3802 | 0.2654 | 0.6886 | 0.9312 | 0.0719 | 16.000 | Reject |
| embedding-8 | 17 | 1 | 355,949 | 107.4 | 0.2238 | 3.3469 | 0.4697 | 0.2704 | 0.7032 | 0.9555 | 0.0741 | 16.270 | Reject |
| embedding-8 | 29 | 3 | 355,949 | 100.0 | 0.2606 | 3.2537 | 0.4256 | 0.2700 | 0.6999 | 0.9394 | 0.0725 | 18.959 | Reject |
| embedding-8 | 43 | 3 | 355,949 | 97.3 | 0.2007 | 3.2837 | 0.3632 | 0.2731 | 0.6862 | 0.9375 | 0.0738 | 17.490 | Reject |
| combined | 17 | 3 | 350,829 | 106.0 | 0.2727 | 3.3039 | 0.4456 | 0.2681 | 0.6892 | 0.9351 | 0.0723 | 18.099 | Reject |
| combined | 29 | 1 | 350,829 | 106.9 | 0.2194 | 3.3629 | 0.4124 | 0.2723 | 0.6871 | 0.9372 | 0.0747 | 17.576 | Reject |
| combined | 43 | 3 | 350,829 | 106.0 | 0.2376 | 3.2878 | 0.3874 | 0.2626 | 0.6922 | 0.9360 | 0.0726 | 16.935 | Reject |
| unified-context-3-15 | 17 | 3 | 371,763 | 111.5 | 0.2012 | 3.2651 | 0.3869 | N/A | 0.9354 | 0.9354 | 0.0722 | 14.316 | — |
| unified-context-3-15 | 29 | 2 | 371,763 | 94.5 | 0.2266 | 3.3074 | 0.3806 | N/A | 0.9121 | 0.9121 | 0.0718 | 15.826 | — |
| unified-context-3-15 | 43 | 3 | 371,763 | 101.9 | 0.2016 | 3.2607 | 0.3983 | N/A | 0.9323 | 0.9323 | 0.0713 | 14.703 | — |
| unified-context-6-10 | 17 | 2 | 367,667 | 111.4 | 0.2023 | 3.3189 | 0.4885 | N/A | 0.9144 | 0.9144 | 0.0721 | 14.823 | — |
| unified-context-6-10 | 29 | 2 | 367,667 | 94.3 | 0.2255 | 3.3326 | 0.3513 | N/A | 0.9052 | 0.9052 | 0.0716 | 15.836 | — |
| unified-context-6-10 | 43 | 3 | 367,667 | 101.7 | 0.2163 | 3.2384 | 0.4007 | N/A | 0.9722 | 0.9722 | 0.0724 | 17.432 | — |
| unified-context-3-10 | 17 | 3 | 361,523 | 110.4 | 0.2747 | 3.2151 | 0.4091 | N/A | 0.9092 | 0.9092 | 0.0717 | 17.770 | — |
| unified-context-3-10 | 29 | 3 | 361,523 | 92.5 | 0.2341 | 3.1840 | 0.3684 | N/A | 0.9251 | 0.9251 | 0.0724 | 19.613 | — |
| unified-context-3-10 | 43 | 3 | 361,523 | 100.9 | 0.2528 | 3.2071 | 0.4074 | N/A | 0.9135 | 0.9135 | 0.0718 | 14.685 | — |
| unified-embedding-8 | 17 | 1 | 339,243 | 108.7 | 0.1994 | 3.3476 | 0.4029 | N/A | 0.9418 | 0.9418 | 0.0743 | 16.908 | — |
| unified-embedding-8 | 29 | 3 | 339,243 | 91.2 | 0.2742 | 3.2226 | 0.3756 | N/A | 0.9045 | 0.9045 | 0.0720 | 17.162 | — |
| unified-embedding-8 | 43 | 3 | 339,243 | 99.5 | 0.2289 | 3.2121 | 0.3755 | N/A | 0.9368 | 0.9368 | 0.0718 | 16.657 | — |
| unified-feedback | 17 | 1 | 361,779 | 105.0 | 0.2444 | 3.3375 | 0.4003 | N/A | 0.9175 | 0.9175 | 0.0717 | 29.100 | — |
| unified-feedback | 29 | 3 | 361,779 | 98.7 | 0.2456 | 3.2405 | 0.3795 | N/A | 0.9247 | 0.9247 | 0.0723 | 21.942 | — |
| unified-feedback | 43 | 3 | 361,779 | 98.1 | 0.2642 | 3.2125 | 0.3451 | N/A | 0.8890 | 0.8890 | 0.0701 | 15.583 | — |
| unified-gru-residual-fixed | 17 | 2 | 361,523 | 90.1 | 0.2633 | 3.0557 | 0.3582 | N/A | 0.9304 | 0.9304 | 0.0715 | 15.181 | Reject |
| unified-gru-residual-fixed | 29 | 3 | 361,523 | 74.1 | 0.2660 | 2.9084 | 0.3440 | N/A | 0.8936 | 0.8936 | 0.0712 | 17.866 | Reject |
| unified-gru-residual-fixed | 43 | 3 | 361,523 | 75.3 | 0.2472 | 2.9611 | 0.3986 | N/A | 0.9230 | 0.9230 | 0.0711 | 17.839 | Reject |
| unified-gru-residual-learnable | 17 | 2 | 361,524 | 90.5 | 0.2160 | 3.2409 | 0.4998 | N/A | 0.9228 | 0.9228 | 0.0713 | 16.940 | Reject |
| unified-gru-residual-learnable | 29 | 3 | 361,524 | 74.0 | 0.2423 | 3.1244 | 0.3840 | N/A | 0.9216 | 0.9216 | 0.0724 | 16.093 | Reject |
| unified-gru-residual-learnable | 43 | 3 | 361,524 | 75.2 | 0.2463 | 3.0519 | 0.4083 | N/A | 0.9140 | 0.9140 | 0.0714 | 17.240 | Reject |
| unified-timing-mlp | 17 | 2 | 368,051 | 85.2 | 0.2180 | 3.3221 | 0.3530 | N/A | 0.9158 | 0.9158 | 0.0715 | 17.056 | Reject |
| unified-timing-mlp | 29 | 3 | 368,051 | 75.2 | 0.2213 | 3.2227 | 0.3803 | N/A | 0.9191 | 0.9191 | 0.0712 | 16.189 | Reject |
| unified-timing-mlp | 43 | 3 | 368,051 | 76.1 | 0.2241 | 3.2311 | 0.3901 | N/A | 0.9368 | 0.9368 | 0.0709 | 15.878 | Reject |
| unified-wrong-key-mlp | 17 | 1 | 363,699 | 85.3 | 0.2298 | 3.3609 | 0.3434 | N/A | 0.9205 | 0.9205 | 0.0739 | 18.104 | Reject |
| unified-wrong-key-mlp | 29 | 3 | 363,699 | 75.5 | 0.2487 | 3.2861 | 0.3622 | N/A | 0.9118 | 0.9118 | 0.0713 | 16.350 | Reject |
| unified-wrong-key-mlp | 43 | 3 | 363,699 | 75.4 | 0.2454 | 3.2568 | 0.3569 | N/A | 0.9278 | 0.9278 | 0.0715 | 17.937 | Reject |
| unified-action-mlp | 17 | 2 | 369,587 | 73.0 | 0.1931 | 3.3216 | 0.4301 | N/A | 0.9284 | 0.9284 | 0.0713 | 12.862 | Reject |
| unified-action-mlp | 29 | 3 | 369,587 | 69.0 | 0.2230 | 3.2574 | 0.4162 | N/A | 0.9394 | 0.9394 | 0.0725 | 15.826 | Reject |
| unified-action-mlp | 43 | 3 | 369,587 | 67.3 | 0.2046 | 3.1870 | 0.4150 | N/A | 0.9119 | 0.9119 | 0.0726 | 15.620 | Reject |

## Unified-head follow-up

New experiments default to 3 characters behind, the current character,
and 10 ahead. The runtime uses `target_left=3` and `target_right=11`.
Each run uses the 3×128 GRU and 27-output timing head. All fifteen runs
reuse the frozen 3,072/768/1,536 trial split and seeds 17, 29, and 43.
The existing unified 6-behind/15-ahead runs are the context reference.
The 8D embedding and feedback each change one feature of the new default.

| Configuration | Parameters | Gap NLL | Action | Wrong key | Hold | Long-gap Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 behind, 15 ahead (reference) | 377,907 | 0.9270 | 0.2173 | 3.2885 | 0.3965 | 0.0726 |
| 3 behind, 15 ahead | 371,763 | 0.9266 | 0.2098 | 3.2777 | 0.3886 | 0.0718 |
| 6 behind, 10 ahead | 367,667 | 0.9306 | 0.2147 | 3.2967 | 0.4135 | 0.0720 |
| 3 behind, 10 ahead (default) | 361,523 | 0.9159 | 0.2539 | 3.2020 | 0.3950 | 0.0720 |
| 3 behind, 10 ahead + 8D embedding | 339,243 | 0.9277 | 0.2342 | 3.2608 | 0.3846 | 0.0727 |
| 3 behind, 10 ahead + previous gap | 361,779 | 0.9104 | 0.2514 | 3.2635 | 0.3750 | 0.0714 |

Positive differences indicate higher held-out gap loss. Each column uses the same training seed on both sides.

| Comparison | Seed 17 | Seed 29 | Seed 43 | Mean |
| --- | ---: | ---: | ---: | ---: |
| 3 behind, 15 ahead vs 6-behind/15-ahead reference | -0.0036 | -0.0199 | +0.0224 | -0.0004 |
| 6 behind, 10 ahead vs 6-behind/15-ahead reference | -0.0245 | -0.0267 | +0.0623 | +0.0037 |
| 3 behind, 10 ahead (default) vs 6-behind/15-ahead reference | -0.0298 | -0.0069 | +0.0035 | -0.0111 |
| 3 behind, 10 ahead + 8D embedding vs default | +0.0326 | -0.0205 | +0.0233 | +0.0118 |
| 3 behind, 10 ahead + previous gap vs default | +0.0083 | -0.0004 | -0.0244 | -0.0055 |

Run `.\.venv\Scripts\python.exe -m experiments.run` for the default. Use `--run` with
`unified-context-3-15`, `unified-context-6-10`, `unified-embedding-8`, or
`unified-feedback` for the other configurations. The existing run
artifacts are protected from overwrite. Every checkpoint has a
matching spec and per-seed metrics under `.data/timing/`.

| Run | Seed | Best epoch | Wall (s) | Gap NLL | Action | Wrong key | Hold | Long-gap Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unified-context-3-15 | 17 | 3 | 111.5 | 0.9354 | 0.2012 | 3.2651 | 0.3869 | 0.0722 |
| unified-context-3-15 | 29 | 2 | 94.5 | 0.9121 | 0.2266 | 3.3074 | 0.3806 | 0.0718 |
| unified-context-3-15 | 43 | 3 | 101.9 | 0.9323 | 0.2016 | 3.2607 | 0.3983 | 0.0713 |
| unified-context-6-10 | 17 | 2 | 111.4 | 0.9144 | 0.2023 | 3.3189 | 0.4885 | 0.0721 |
| unified-context-6-10 | 29 | 2 | 94.3 | 0.9052 | 0.2255 | 3.3326 | 0.3513 | 0.0716 |
| unified-context-6-10 | 43 | 3 | 101.7 | 0.9722 | 0.2163 | 3.2384 | 0.4007 | 0.0724 |
| unified-context-3-10 | 17 | 3 | 110.4 | 0.9092 | 0.2747 | 3.2151 | 0.4091 | 0.0717 |
| unified-context-3-10 | 29 | 3 | 92.5 | 0.9251 | 0.2341 | 3.1840 | 0.3684 | 0.0724 |
| unified-context-3-10 | 43 | 3 | 100.9 | 0.9135 | 0.2528 | 3.2071 | 0.4074 | 0.0718 |
| unified-embedding-8 | 17 | 1 | 108.7 | 0.9418 | 0.1994 | 3.3476 | 0.4029 | 0.0743 |
| unified-embedding-8 | 29 | 3 | 91.2 | 0.9045 | 0.2742 | 3.2226 | 0.3756 | 0.0720 |
| unified-embedding-8 | 43 | 3 | 99.5 | 0.9368 | 0.2289 | 3.2121 | 0.3755 | 0.0718 |
| unified-feedback | 17 | 1 | 105.0 | 0.9175 | 0.2444 | 3.3375 | 0.4003 | 0.0717 |
| unified-feedback | 29 | 3 | 98.7 | 0.9247 | 0.2456 | 3.2405 | 0.3795 | 0.0723 |
| unified-feedback | 43 | 3 | 98.1 | 0.8890 | 0.2642 | 3.2125 | 0.3451 | 0.0701 |

## Action, wrong-key, mistake, and hold metrics by seed

Action accuracy counts the top prediction among correct key, wrong key, and
Backspace. Wrong-action recall measures how often the action head identified
a recorded wrong action. Wrong-key accuracy measures the wrong-key head's
top prediction on those actual wrong events, regardless of the action prediction.
These are held-out scores given the recorded typing history.

Hold MAE is the mean absolute error in milliseconds between the predicted
median hold for the recorded action and its recorded duration. The last
column gives the share of valid holds within 20 ms of their recorded duration.
Both use the model's median before runtime sampling and clipping.

Generated mistakes count wrong keypresses in the existing 128-prompt plans.
The denominator includes all emitted events, including Backspaces. Timing
rescaling preserves the emitted keys and therefore preserves both counts.

### Frozen experiment holdout · 1,536 trials

Each row scores 67,339 recorded actions, including 2,896 wrong actions, and 67,304 valid recorded holds.

| Model | Seed | Gap NLL ↓ | Action accuracy ↑ | Wrong-action recall ↑ | Wrong-key accuracy ↑ | Generated mistakes / events | Mistake rate ↓ | Hold MAE, ms ↓ | Holds within 20 ms ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 17 | 0.9568 | 94.44% | 0.10% | 10.53% | 422 / 6,207 | 6.80% | 29.91 | 45.47% |
| baseline | 29 | 0.9264 | 94.73% | 0.00% | 11.50% | 466 / 6,295 | 7.40% | 27.92 | 45.80% |
| baseline | 43 | 0.9304 | 94.47% | 3.07% | 13.02% | 457 / 6,277 | 7.28% | 28.05 | 43.68% |
| mixture | 17 | 0.9389 | 94.79% | 0.31% | 10.98% | 607 / 6,577 | 9.23% | 28.41 | 44.19% |
| mixture | 29 | 0.9320 | 94.97% | 1.31% | 13.43% | 317 / 5,997 | 5.29% | 27.87 | 46.42% |
| mixture | 43 | 0.9100 | 94.87% | 0.24% | 12.02% | 465 / 6,293 | 7.39% | 27.99 | 45.80% |
| feedback | 17 | 0.9463 | 95.01% | 0.79% | 10.98% | 394 / 6,151 | 6.41% | 31.67 | 44.00% |
| feedback | 29 | 0.9650 | 95.14% | 0.35% | 12.60% | 316 / 5,995 | 5.27% | 28.02 | 46.37% |
| feedback | 43 | 0.9462 | 94.87% | 0.10% | 11.67% | 329 / 6,021 | 5.46% | 28.38 | 45.07% |
| context-3-15 | 17 | 0.9523 | 94.90% | 0.24% | 11.57% | 413 / 6,189 | 6.67% | 27.98 | 45.33% |
| context-3-15 | 29 | 0.9232 | 95.06% | 1.04% | 12.88% | 389 / 6,141 | 6.33% | 28.00 | 46.75% |
| context-3-15 | 43 | 0.9281 | 94.84% | 2.38% | 14.61% | 325 / 6,013 | 5.40% | 27.87 | 44.61% |
| context-6-10 | 17 | 0.9389 | 95.06% | 0.48% | 11.60% | 400 / 6,163 | 6.49% | 29.28 | 41.63% |
| context-6-10 | 29 | 0.9651 | 94.97% | 4.87% | 12.02% | 1,008 / 7,379 | 13.66% | 27.86 | 46.72% |
| context-6-10 | 43 | 0.9270 | 95.19% | 4.28% | 12.19% | 336 / 6,035 | 5.57% | 27.88 | 44.53% |
| context-3-10 | 17 | 0.9459 | 94.57% | 1.38% | 11.05% | 297 / 5,957 | 4.99% | 27.96 | 44.26% |
| context-3-10 | 29 | 0.9073 | 94.76% | 2.21% | 13.40% | 393 / 6,149 | 6.39% | 27.70 | 45.02% |
| context-3-10 | 43 | 0.9312 | 95.16% | 0.73% | 13.67% | 285 / 5,933 | 4.80% | 28.07 | 44.51% |
| embedding-8 | 17 | 0.9555 | 94.53% | 0.24% | 11.12% | 1,405 / 8,173 | 17.19% | 29.88 | 39.57% |
| embedding-8 | 29 | 0.9394 | 94.47% | 10.84% | 13.50% | 516 / 6,395 | 8.07% | 28.64 | 42.51% |
| embedding-8 | 43 | 0.9375 | 95.12% | 0.69% | 12.53% | 244 / 5,851 | 4.17% | 28.09 | 44.97% |
| combined | 17 | 0.9351 | 93.79% | 3.11% | 12.09% | 363 / 6,089 | 5.96% | 31.28 | 40.02% |
| combined | 29 | 0.9372 | 94.12% | 0.00% | 10.88% | 718 / 6,799 | 10.56% | 28.62 | 44.15% |
| combined | 43 | 0.9360 | 94.63% | 1.76% | 12.67% | 406 / 6,175 | 6.57% | 27.94 | 46.47% |
| unified-context-3-15 | 17 | 0.9354 | 94.82% | 0.48% | 11.64% | 386 / 6,135 | 6.29% | 27.89 | 44.73% |
| unified-context-3-15 | 29 | 0.9121 | 94.62% | 0.38% | 11.46% | 363 / 6,089 | 5.96% | 27.64 | 45.22% |
| unified-context-3-15 | 43 | 0.9323 | 95.22% | 0.52% | 14.23% | 222 / 5,807 | 3.82% | 28.96 | 41.56% |
| unified-context-6-10 | 17 | 0.9144 | 94.69% | 0.21% | 12.15% | 576 / 6,515 | 8.84% | 29.63 | 45.35% |
| unified-context-6-10 | 29 | 0.9052 | 94.42% | 0.59% | 12.05% | 494 / 6,351 | 7.78% | 28.11 | 45.69% |
| unified-context-6-10 | 43 | 0.9722 | 95.12% | 2.21% | 15.02% | 313 / 5,989 | 5.23% | 28.01 | 44.24% |
| unified-context-3-10 | 17 | 0.9092 | 93.09% | 4.11% | 13.67% | 549 / 6,461 | 8.50% | 28.77 | 46.36% |
| unified-context-3-10 | 29 | 0.9251 | 94.91% | 1.48% | 15.81% | 279 / 5,921 | 4.71% | 27.93 | 44.58% |
| unified-context-3-10 | 43 | 0.9135 | 94.50% | 3.80% | 14.85% | 352 / 6,067 | 5.80% | 27.75 | 45.63% |
| unified-embedding-8 | 17 | 0.9418 | 94.72% | 0.03% | 11.60% | 456 / 6,275 | 7.27% | 29.64 | 45.99% |
| unified-embedding-8 | 29 | 0.9045 | 94.06% | 3.52% | 15.81% | 354 / 6,071 | 5.83% | 28.05 | 45.93% |
| unified-embedding-8 | 43 | 0.9368 | 94.71% | 1.45% | 14.09% | 314 / 5,991 | 5.24% | 28.19 | 45.54% |
| unified-feedback | 17 | 0.9175 | 93.47% | 1.38% | 11.29% | 3,424 / 12,211 | 28.04% | 29.70 | 45.73% |
| unified-feedback | 29 | 0.9247 | 94.67% | 4.59% | 14.23% | 557 / 6,477 | 8.60% | 27.81 | 44.65% |
| unified-feedback | 43 | 0.8890 | 94.07% | 4.52% | 15.71% | 385 / 6,133 | 6.28% | 27.89 | 45.03% |
| unified-gru-residual-fixed | 17 | 0.9304 | 93.34% | 5.70% | 21.62% | 443 / 6,249 | 7.09% | 27.71 | 45.40% |
| unified-gru-residual-fixed | 29 | 0.8936 | 93.87% | 3.04% | 24.93% | 431 / 6,225 | 6.92% | 27.52 | 45.88% |
| unified-gru-residual-fixed | 43 | 0.9230 | 94.41% | 2.14% | 23.10% | 298 / 5,959 | 5.00% | 28.89 | 43.56% |
| unified-gru-residual-learnable | 17 | 0.9228 | 94.72% | 6.80% | 14.47% | 532 / 6,427 | 8.28% | 28.90 | 46.48% |
| unified-gru-residual-learnable | 29 | 0.9216 | 94.74% | 1.93% | 16.64% | 283 / 5,929 | 4.77% | 27.96 | 45.10% |
| unified-gru-residual-learnable | 43 | 0.9140 | 94.55% | 2.69% | 20.03% | 358 / 6,079 | 5.89% | 28.16 | 44.08% |
| unified-timing-mlp | 17 | 0.9158 | 94.18% | 0.79% | 11.43% | 1,798 / 8,959 | 20.07% | 28.63 | 46.23% |
| unified-timing-mlp | 29 | 0.9191 | 94.78% | 2.00% | 15.37% | 267 / 5,897 | 4.53% | 28.08 | 44.41% |
| unified-timing-mlp | 43 | 0.9368 | 94.84% | 9.84% | 15.30% | 442 / 6,247 | 7.08% | 28.54 | 43.61% |
| unified-wrong-key-mlp | 17 | 0.9205 | 93.87% | 1.04% | 10.77% | 1,862 / 9,087 | 20.49% | 28.05 | 44.87% |
| unified-wrong-key-mlp | 29 | 0.9118 | 94.63% | 1.76% | 11.98% | 314 / 5,991 | 5.24% | 27.47 | 45.18% |
| unified-wrong-key-mlp | 43 | 0.9278 | 94.73% | 2.42% | 12.91% | 314 / 5,991 | 5.24% | 27.55 | 46.14% |
| unified-action-mlp | 17 | 0.9284 | 94.35% | 0.45% | 11.71% | 731 / 6,825 | 10.71% | 29.47 | 46.02% |
| unified-action-mlp | 29 | 0.9394 | 94.71% | 1.73% | 13.47% | 437 / 6,237 | 7.01% | 28.52 | 46.27% |
| unified-action-mlp | 43 | 0.9119 | 95.19% | 1.17% | 16.40% | 255 / 5,873 | 4.34% | 28.17 | 43.73% |

### Shared holdout with the shipped model · 1,031 trials

Each row scores 45,214 recorded actions, including 1,871 wrong actions, and 45,211 valid recorded holds.

| Model | Seed | Gap NLL ↓ | Action accuracy ↑ | Wrong-action recall ↑ | Wrong-key accuracy ↑ | Generated mistakes / events | Mistake rate ↓ | Hold MAE, ms ↓ | Holds within 20 ms ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shipped | 20260920 | 0.8144 | 89.20% | 7.48% | 45.86% | 302 / 5,822 | 5.19% | 26.21 | 47.90% |
| baseline | 17 | 0.8899 | 94.60% | 0.00% | 11.81% | 353 / 5,924 | 5.96% | 28.86 | 46.08% |
| context-6-10 | 17 | 0.8584 | 95.23% | 0.21% | 12.35% | 394 / 6,006 | 6.56% | 28.56 | 42.05% |

### Generated timing on the shared holdout

These rows use the 128 prompts sampled from the 1,031-trial shared holdout.

| Model | Plan | Log-gap lag 1 | Lag 2 | Mean burst | >500 ms | >1 s | Mistake rate | Mean correction (ms) | Mean Backspace run | Median hold (ms) | Median entry / within (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recorded | Observed | 0.210 | 0.242 | 9.31 | 8.66% | 1.70% | 3.85% | 790 | 1.55 | 104 | 225 / 160 |
| shipped | Raw | 0.047 | 0.014 | 11.64 | 6.53% | 0.39% | 5.19% | 569 | 1.23 | 100 | 205 / 148 |
| shipped | Rescaled | 0.039 | 0.003 | 27.72 | 1.44% | 0.05% | 5.19% | 391 | 1.23 | 100 | 150 / 106 |
| baseline | Raw | 0.071 | 0.054 | 12.19 | 6.18% | 0.60% | 5.96% | 514 | 1.25 | 96 | 192 / 141 |
| baseline | Rescaled | 0.119 | 0.104 | 25.10 | 1.86% | 0.09% | 5.96% | 298 | 1.25 | 96 | 151 / 105 |
| context-6-10 | Raw | 0.086 | 0.053 | 11.31 | 6.86% | 1.07% | 6.56% | 762 | 1.27 | 109 | 218 / 144 |
| context-6-10 | Rescaled | 0.162 | 0.137 | 25.34 | 1.85% | 0.10% | 6.56% | 401 | 1.27 | 109 | 153 / 100 |

## GRU residual and GELU-head follow-up

The reference is the saved unified 3-behind/10-ahead model with 16D embeddings.
Each variant changes one feature. All 15 new runs match baseline seeds 17, 29,
and 43, the frozen 3,072/768/1,536 trial split, and the three-epoch training
protocol. Generated metrics use the same 128 prompts and generation seed 251.

The GRU skip adds α times the projected input to the output of the full
three-layer GRU before the heads. The learned variant uses one scalar
initialized to 0 and projected into [0, 1] after each optimizer step.
Each MLP replaces one linear head with a 64-unit GELU hidden layer. The output
widths are 27 for timing, 95 for wrong key, and 3 for action. Tables report seed means
unless a seed is specified.

| Configuration | Parameters | Gap NLL ↓ | Δ gap NLL | Action loss ↓ | Wrong-key loss ↓ | Hold loss ↓ | Long-gap Brier ↓ | Rhythm error ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unified 3 behind, 10 ahead, 16D (reference) | 361,523 | 0.9159 | — | 0.2539 | 3.2020 | 0.3950 | 0.0720 | 17.356 |
| GRU residual, fixed α = 0.25 | 361,523 | 0.9156 | -0.0002 | 0.2588 | 2.9751 | 0.3670 | 0.0713 | 16.962 |
| GRU residual, learnable α | 361,524 | 0.9195 | +0.0036 | 0.2348 | 3.1390 | 0.4307 | 0.0717 | 16.758 |
| Timing GELU head | 368,051 | 0.9239 | +0.0080 | 0.2211 | 3.2586 | 0.3745 | 0.0712 | 16.375 |
| Wrong-key GELU head | 363,699 | 0.9200 | +0.0042 | 0.2413 | 3.3013 | 0.3542 | 0.0722 | 17.463 |
| Action GELU head | 369,587 | 0.9266 | +0.0107 | 0.2069 | 3.2554 | 0.4204 | 0.0721 | 14.769 |

| Configuration | Action accuracy ↑ | Wrong-action recall ↑ | Wrong-key accuracy ↑ | Generated mistake rate | Hold MAE (ms) ↓ | Holds within 20 ms ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unified 3 behind, 10 ahead, 16D (reference) | 94.16% | 3.13% | 14.78% | 6.34% | 28.15 | 45.52% |
| GRU residual, fixed α = 0.25 | 93.87% | 3.63% | 23.22% | 6.34% | 28.04 | 44.95% |
| GRU residual, learnable α | 94.67% | 3.81% | 17.05% | 6.31% | 28.34 | 45.22% |
| Timing GELU head | 94.60% | 4.21% | 14.03% | 10.56% | 28.42 | 44.75% |
| Wrong-key GELU head | 94.41% | 1.74% | 11.89% | 10.32% | 27.69 | 45.40% |
| Action GELU head | 94.75% | 1.12% | 13.86% | 7.35% | 28.72 | 45.34% |

Gap NLL differences are paired by training seed. Negative values indicate
lower loss. Learned α values come from the validation-selected checkpoints.

| Configuration | Δ gap NLL, seed 17 | Seed 29 | Seed 43 | Learned α, seeds 17 / 29 / 43 |
| --- | ---: | ---: | ---: | --- |
| GRU residual, fixed α = 0.25 | +0.0212 | -0.0314 | +0.0095 | — |
| GRU residual, learnable α | +0.0136 | -0.0034 | +0.0006 | 0.1742 / 0.2082 / 0.3310 |
| Timing GELU head | +0.0067 | -0.0060 | +0.0233 | — |
| Wrong-key GELU head | +0.0114 | -0.0133 | +0.0144 | — |
| Action GELU head | +0.0192 | +0.0144 | -0.0015 | — |

No variant passed the existing timing gate against this reference.
See the [raw results](experiments/results.json) for per-seed metrics and gate decisions.

## Full release training · Fixed GRU skip and GELU action head

Trained from scratch on the shipped corpus and participant split with seed
20260920. The split contains 99,684 training trials (4,349,953 events) and
11,261 validation trials. Training used four epochs, batches of 48, AdamW at
0.0025 with 0.0001 weight decay, and gradient clipping at 1.0. The loss weights
were 1.0 for class-weighted action, 0.35 for wrong key, 0.55 for unified timing,
and 0.15 for hold. Checkpoint selection used the lowest validation objective
with recorded trial styles. Epoch 4 was selected after 37.9 minutes of training
and validation on CPU.

The candidate has 369,587 parameters. It uses 16D embeddings, 3 characters
behind, the current character, and 10 ahead. The projected input contributes
a fixed α = 0.25 skip to the full three-layer GRU output. The action head uses
128 → 64 → 3 with GELU between the two linear layers.

The shared comparison covers 1,031 trials from 115 participants. Both
models used these participants for validation and checkpoint selection.
Predictive metrics use recorded history and the mean training style.
Wrong-action ranking uses the predicted wrong-action probability across all
45,214 valid actions. Recall counts argmax wrong predictions on the 1,871
recorded wrong actions. Generated metrics use 128 matched prompts and the
frozen controls. The recorded mistake rate on those prompts was 3.85%.

| Metric | Shipped | Candidate | Candidate − shipped |
| --- | ---: | ---: | ---: |
| Gap NLL, nats per gap ↓ | 0.8144 | 0.8205 | +0.0061 |
| Hold NLL, nats per hold ↓ | 0.3396 | 0.3787 | +0.0390 |
| Wrong-key top-1 accuracy ↑ | 45.86% | 49.33% | +3.47 pp |
| Wrong-action average precision ↑ | 0.0404 | 0.0401 | -0.0003 |
| Wrong-action AUROC ↑ | 0.4632 | 0.4552 | -0.0080 |
| Wrong-action recall ↑ | 7.48% | 6.41% | -1.07 pp |
| Generated mistake rate | 5.19% | 5.28% | +0.09 pp |
| Rhythm error, raw + rescaled ↓ | 19.526 | 20.184 | +0.658 |

Reproduce the run with `python -m experiments.full_release` and score the saved
checkpoint with `python -m experiments.full_release --compare`.
See the [checkpoint](.data/timing/unified-residual-action-mlp-full-20260920.pt), [spec](.data/timing/unified-residual-action-mlp-full-20260920.spec.json),
and [full metrics](experiments/results.json).

## Full Aalto corpus · Unified 3/10 context

The full Aalto ZIP was scanned with the existing physical QWERTY and valid-trial filters. Participants were assigned to train, validation, or test in proportions of 80/10/10 with split seed `20261119`. All eligible trials from each participant stayed in the same split. Style normalization, the sampled style bank, and loss class weights were fit on train only.

Archive SHA-256: `5fb217e0e1273017a6789c5c7ebcc00eed624f2f6426c284da2077a88e270420`. Eligible keystroke entries: 162,056. Malformed entries skipped: 83.

| Split | Participants | Trials | Events | Mistake rate | Gaps | Pauses | Holds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 112,457 | 990,283 | 43,290,250 | 4.35% | 42,299,967 | 3,653,243 | 43,275,579 |
| validation | 14,028 | 123,808 | 5,403,627 | 4.30% | 5,279,819 | 455,818 | 5,401,664 |
| test | 14,066 | 123,328 | 5,394,821 | 4.38% | 5,271,493 | 461,770 | 5,393,422 |

The three baseline models were trained from fresh weights with unified three-component timing, three characters behind and ten ahead of the current character, a three-layer 128-unit GRU, and a fixed 0.25 input skip. The GELU models add a 64-unit hidden layer to each output head. Only the embedding size and output heads vary across the three runs.

Baseline training seed `20260920`. AdamW used learning rate 0.0025, weight decay 0.0001, and gradient clipping at 1.0. Length-bucketed batches used up to 256 trials and 16,384 padded positions, except for any single longer trial. Training used cuda with FP16 autocast with FP32 loss accumulation. The objective weights were 1.0 for class-weighted action, 0.35 for wrong key, 0.55 for timing and 0.15 for hold. The validation objective uses recorded styles and accumulates each head over its valid events before applying its coefficient. Training stopped after 3 consecutive epochs without a lower validation objective and saved the lowest-loss checkpoint. Predictive test metrics use the mean training style and recorded history. Generation uses the same 128 seeded test prompts per model and reports both raw and rescaled plans.

The [hold-mixture run](#three-component-hold-mixture) fine-tunes `gelu-16d` on the same split.

| Model | Parameters | Epochs | Best epoch | Best validation objective | Training min | Gap NLL | Hold NLL | Rhythm error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gelu-16d | 386,163 | 17 | 14 | 1.041216 | 21.72 | 0.881396 | 0.375375 | 27.477 |
| gelu-8d | 363,883 | 21 | 18 | 1.073169 | 21.89 | 0.863279 | 0.373603 | 26.255 |
| linear-16d | 361,523 | 11 | 8 | 1.078125 | 11.20 | 0.886819 | 0.348692 | 25.665 |
| gelu-16d-hold-mixture | 387,528 | 58 | 55 | 0.976423 | 67.33 | 0.873668 | 0.363428 | 26.370 |

### Predictive test metrics

| Model | Action NLL | Wrong-key NLL | Long-gap Brier | Action accuracy | Wrong-action recall | Wrong-key accuracy | Wrong-action AP | Wrong-action AUROC | Hold MAE (ms) | Hold within 20 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gelu-16d | 0.49182 | 1.60098 | 0.07294 | 89.98% | 9.03% | 53.90% | 0.04606 | 0.44982 | 26.98 | 46.73% |
| gelu-8d | 0.43170 | 1.66681 | 0.07225 | 89.81% | 9.20% | 52.79% | 0.04654 | 0.46197 | 26.94 | 46.70% |
| linear-16d | 0.38511 | 1.69408 | 0.07241 | 90.89% | 7.16% | 51.76% | 0.04554 | 0.46952 | 27.16 | 46.40% |
| gelu-16d-hold-mixture | 0.61789 | 1.51780 | 0.07260 | 90.57% | 8.04% | 56.13% | 0.04632 | 0.44043 | 26.84 | 46.68% |

### Recorded and generated rhythm

| Model | Mode | Lag 1 | Lag 2 | Mean burst | Over 500 ms | Over 1 s | Mistake rate | Correction p90 (ms) | Mean Backspace run | Hold p50 (ms) | Word entry / within word p50 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference | observed | 0.3464 | 0.3500 | 9.08 | 8.88% | 1.62% | 5.23% | 1620 | 2.06 | 104 | 207 / 153 |
| gelu-16d | raw | 0.0746 | 0.0184 | 12.02 | 6.11% | 0.51% | 4.96% | 775 | 1.16 | 105 | 203 / 149 |
| gelu-16d | rescaled | 0.0583 | 0.0006 | 36.07 | 0.43% | 0.02% | 4.96% | 513 | 1.16 | 105 | 150 / 113 |
| gelu-8d | raw | 0.0488 | 0.0303 | 11.54 | 6.49% | 0.80% | 5.43% | 866 | 1.15 | 104 | 200 / 150 |
| gelu-8d | rescaled | 0.0253 | 0.0057 | 32.76 | 0.74% | 0.02% | 5.43% | 481 | 1.15 | 104 | 148 / 111 |
| linear-16d | raw | 0.0762 | 0.0078 | 11.44 | 6.56% | 0.77% | 5.12% | 764 | 1.13 | 104 | 203 / 152 |
| linear-16d | rescaled | 0.0587 | -0.0121 | 31.78 | 0.82% | 0.00% | 5.12% | 469 | 1.13 | 104 | 144 / 112 |
| gelu-16d-hold-mixture | raw | 0.0479 | 0.0242 | 11.61 | 6.41% | 0.66% | 4.84% | 868 | 1.16 | 106 | 199 / 147 |
| gelu-16d-hold-mixture | rescaled | 0.0319 | 0.0071 | 32.72 | 0.72% | 0.00% | 4.84% | 528 | 1.16 | 106 | 154 / 113 |

### gelu-16d epoch history

Local artifacts include `.data/full-corpus/gelu-16d.pt`, `.data/full-corpus/gelu-16d.spec.json`, and `.data/full-corpus/gelu-16d.json`.

| Epoch | Train objective | Validation objective | Elapsed (s) |
| ---: | ---: | ---: | ---: |
| 1 | 1.297510 | 1.140583 | 118.5 |
| 2 | 1.119048 | 1.096652 | 233.8 |
| 3 | 1.091197 | 1.076366 | 326.8 |
| 4 | 1.077607 | 1.074464 | 433.8 |
| 5 | 1.069061 | 1.062037 | 517.0 |
| 6 | 1.062388 | 1.062536 | 581.5 |
| 7 | 1.058145 | 1.053498 | 646.0 |
| 8 | 1.054271 | 1.044598 | 714.5 |
| 9 | 1.052106 | 1.053069 | 782.0 |
| 10 | 1.050018 | 1.051023 | 847.3 |
| 11 | 1.049100 | 1.044477 | 911.4 |
| 12 | 1.047760 | 1.045062 | 977.6 |
| 13 | 1.044473 | 1.043173 | 1041.7 |
| 14 | 1.046024 | 1.041216 | 1111.7 |
| 15 | 1.044599 | 1.046482 | 1175.8 |
| 16 | 1.044469 | 1.043598 | 1241.3 |
| 17 | 1.045799 | 1.043088 | 1303.1 |

### gelu-8d epoch history

Local artifacts include `.data/full-corpus/gelu-8d.pt`, `.data/full-corpus/gelu-8d.spec.json`, and `.data/full-corpus/gelu-8d.json`.

| Epoch | Train objective | Validation objective | Elapsed (s) |
| ---: | ---: | ---: | ---: |
| 1 | 1.324697 | 1.163784 | 60.7 |
| 2 | 1.150988 | 1.117396 | 122.6 |
| 3 | 1.121768 | 1.099118 | 186.4 |
| 4 | 1.107431 | 1.097227 | 249.3 |
| 5 | 1.098855 | 1.087995 | 310.4 |
| 6 | 1.092844 | 1.088065 | 370.7 |
| 7 | 1.090094 | 1.078268 | 432.7 |
| 8 | 1.087281 | 1.076942 | 494.6 |
| 9 | 1.085346 | 1.083982 | 556.2 |
| 10 | 1.085757 | 1.077233 | 616.6 |
| 11 | 1.087004 | 1.076323 | 676.7 |
| 12 | 1.086576 | 1.075554 | 737.6 |
| 13 | 1.085854 | 1.081596 | 797.6 |
| 14 | 1.084547 | 1.076080 | 866.0 |
| 15 | 1.085580 | 1.075033 | 932.5 |
| 16 | 1.086216 | 1.076043 | 994.3 |
| 17 | 1.084866 | 1.076914 | 1055.5 |
| 18 | 1.086396 | 1.073169 | 1116.7 |
| 19 | 1.086066 | 1.080854 | 1177.2 |
| 20 | 1.088467 | 1.088758 | 1248.9 |
| 21 | 1.091422 | 1.080320 | 1313.6 |

### linear-16d epoch history

Local artifacts include `.data/full-corpus/linear-16d.pt`, `.data/full-corpus/linear-16d.spec.json`, and `.data/full-corpus/linear-16d.json`.

| Epoch | Train objective | Validation objective | Elapsed (s) |
| ---: | ---: | ---: | ---: |
| 1 | 1.269821 | 1.137751 | 59.5 |
| 2 | 1.135759 | 1.102748 | 118.8 |
| 3 | 1.114745 | 1.092644 | 177.8 |
| 4 | 1.105487 | 1.100100 | 242.0 |
| 5 | 1.099999 | 1.088587 | 307.5 |
| 6 | 1.097383 | 1.089342 | 366.9 |
| 7 | 1.095115 | 1.081312 | 427.7 |
| 8 | 1.093925 | 1.078125 | 487.5 |
| 9 | 1.093659 | 1.085755 | 548.7 |
| 10 | 1.094110 | 1.086198 | 610.6 |
| 11 | 1.096171 | 1.091326 | 672.1 |

### gelu-16d-hold-mixture epoch history

Local artifacts include `.data/full-corpus/gelu-16d-hold-mixture.pt`, `.data/full-corpus/gelu-16d-hold-mixture.spec.json`, and `.data/full-corpus/gelu-16d-hold-mixture.json`.

| Epoch | Train objective | Validation objective | Elapsed (s) |
| ---: | ---: | ---: | ---: |
| 0 | — | 1.040716 | — |
| 1 | 1.008756 | 1.005680 | 72.4 |
| 2 | 1.000485 | 1.001766 | 143.9 |
| 3 | 0.996891 | 0.999627 | 210.6 |
| 4 | 0.994605 | 0.996969 | 275.3 |
| 5 | 0.992674 | 0.995399 | 340.6 |
| 6 | 0.991084 | 0.995844 | 406.7 |
| 7 | 0.989828 | 0.993311 | 482.3 |
| 8 | 0.988532 | 0.992058 | 556.2 |
| 9 | 0.987492 | 0.991803 | 626.3 |
| 10 | 0.986341 | 0.990921 | 694.5 |
| 11 | 0.985419 | 0.989679 | 762.6 |
| 12 | 0.984727 | 0.989563 | 836.0 |
| 13 | 0.983857 | 0.988224 | 905.8 |
| 14 | 0.982939 | 0.988044 | 978.0 |
| 15 | 0.982605 | 0.987466 | 1046.9 |
| 16 | 0.981953 | 0.987500 | 1113.8 |
| 17 | 0.981366 | 0.986799 | 1185.5 |
| 18 | 0.980774 | 0.985936 | 1254.0 |
| 19 | 0.980414 | 0.986070 | 1324.0 |
| 20 | 0.979702 | 0.985063 | 1389.9 |
| 21 | 0.979197 | 0.984872 | 1459.5 |
| 22 | 0.978663 | 0.984608 | 1530.0 |
| 23 | 0.978156 | 0.983732 | 1603.6 |
| 24 | 0.977781 | 0.983666 | 1672.9 |
| 25 | 0.977357 | 0.982903 | 1739.8 |
| 26 | 0.976885 | 0.982817 | 1812.1 |
| 27 | 0.976484 | 0.983241 | 1883.5 |
| 28 | 0.976191 | 0.982670 | 1958.8 |
| 29 | 0.975642 | 0.981797 | 2027.8 |
| 30 | 0.975241 | 0.981997 | 2093.7 |
| 31 | 0.974877 | 0.981739 | 2163.5 |
| 32 | 0.974523 | 0.981158 | 2232.5 |
| 33 | 0.974090 | 0.981168 | 2309.4 |
| 34 | 0.973899 | 0.980429 | 2378.6 |
| 35 | 0.973471 | 0.980450 | 2447.2 |
| 36 | 0.973306 | 0.979769 | 2517.5 |
| 37 | 0.972747 | 0.980024 | 2595.4 |
| 38 | 0.972469 | 0.981296 | 2669.0 |
| 39 | 0.972247 | 0.979502 | 2730.8 |
| 40 | 0.971917 | 0.979214 | 2793.2 |
| 41 | 0.971637 | 0.980379 | 2862.9 |
| 42 | 0.971517 | 0.978654 | 2929.7 |
| 43 | 0.971047 | 0.978608 | 3006.5 |
| 44 | 0.970827 | 0.977980 | 3079.0 |
| 45 | 0.970554 | 0.978488 | 3147.5 |
| 46 | 0.970306 | 0.978268 | 3215.8 |
| 47 | 0.970084 | 0.977934 | 3281.8 |
| 48 | 0.969814 | 0.977742 | 3354.6 |
| 49 | 0.969560 | 0.977835 | 3423.4 |
| 50 | 0.969457 | 0.977391 | 3493.8 |
| 51 | 0.968979 | 0.978231 | 3560.3 |
| 52 | 0.968871 | 0.977201 | 3628.0 |
| 53 | 0.968532 | 0.976844 | 3698.3 |
| 54 | 0.968361 | 0.976744 | 3769.9 |
| 55 | 0.968108 | 0.976423 | 3838.7 |
| 56 | 0.967909 | 0.976809 | 3905.0 |
| 57 | 0.967636 | 0.976745 | 3972.0 |
| 58 | 0.967498 | 0.977600 | 4039.7 |

The [published metrics](experiments/full_corpus_results.json) include per-head epoch losses, test denominators, calibration bins, mean mixture weights, and observed and generated distributions with counts, means, and quantiles for each run.

Reproduce with `python -m experiments.full_corpus_data` followed by `python -m experiments.full_corpus`. The archive belongs at `.data/Keystrokes.zip`.

## Three-component hold mixture

`gelu-16d-hold-mixture` starts from epoch 14 of `gelu-16d`. The hold head grows
from 6 to 27 outputs while preserving the learned rows for the original
component. All parameters are fine-tuned on the same split with AdamW at
`0.00025`. The initial validation objective was 1.040716. Epoch 55 reached
0.976423, and training stopped at epoch 58 after three epochs without improvement.

To reproduce the fine-tuning after training `gelu-16d`, run
`python -m experiments.full_corpus --variant gelu-16d-hold-mixture`.
The selected checkpoint is [bundled as the default model](models/gelu-16d-hold-mixture.pt).

## Shipped checkpoint · Common participant holdout

We reconstructed the shipped data range as a contiguous span of 12,630
eligible participants in Aalto archive order. The resulting split matched the
checkpoint's complete style bank, style mean and standard deviation, 99,684
training trials, 11,261 validation trials, and 4,349,953 training events exactly.

The full-corpus test split included 1,091 shipped training participants and 110
shipped validation participants. We excluded both groups from the comparison
below. The common holdout contains 113,078 trials from 12,865 participants and
4,948,395 scored actions and 4,835,317 gaps. Each checkpoint keeps its native context and decoder.
Predictive evaluation uses the mean training style and recorded event history.
Gap MAE compares the predicted median with the recorded gap before speed
calibration. The original model's predictive distribution weights the short and
long mixtures by the pause classifier and normalizes each to its side of 500 ms.
Generation uses the same 128 prompts from this subset for all five models,
with seed 101 for sampling, seed 251 for generation, and a 75 WPM target.

| Model | Parameters | Gap MAE (ms) ↓ | Gap NLL ↓ | Hold MAE (ms) ↓ | Hold NLL ↓ | Action accuracy ↑ | Wrong-key accuracy ↑ | Wrong-action recall ↑ | Long-gap Brier ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original shipped | 386,421 | 108.95 | 0.914349 | 27.63 | 0.428353 | 89.22% | 45.54% | 7.69% | 0.07263 |
| gelu-16d | 386,163 | 107.66 | 0.881201 | 26.99 | 0.372098 | 89.98% | 53.90% | 9.02% | 0.07307 |
| gelu-8d | 363,883 | 107.72 | 0.863017 | 26.95 | 0.370441 | 89.81% | 52.80% | 9.19% | 0.07238 |
| linear-16d | 361,523 | 108.46 | 0.886457 | 27.18 | 0.346107 | 90.88% | 51.77% | 7.16% | 0.07253 |
| gelu-16d-hold-mixture | 387,528 | 107.34 | 0.873492 | 26.85 | 0.362558 | 90.57% | 56.12% | 8.03% | 0.07272 |

| Model | Raw rhythm error ↓ | Rescaled rhythm error ↓ | Combined rhythm error ↓ | Mean burst, raw / rescaled | Gaps over 500 ms, raw / rescaled |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original shipped | 7.533 | 14.063 | 21.596 | 13.43 / 28.72 | 5.37% / 1.32% |
| gelu-16d | 6.775 | 17.747 | 24.522 | 12.67 / 38.50 | 5.82% / 0.40% |
| gelu-8d | 6.360 | 17.082 | 23.442 | 11.55 / 35.72 | 6.61% / 0.63% |
| linear-16d | 5.629 | 15.996 | 21.625 | 11.47 / 34.21 | 6.66% / 0.74% |
| gelu-16d-hold-mixture | 5.755 | 14.864 | 20.619 | 11.79 / 31.63 | 6.41% / 0.97% |

The recordings averaged 9.91 keystrokes per burst, and 8.04% of gaps exceeded
500 ms. All four full-corpus models improved predictive gap and hold likelihood,
wrong-key accuracy, and raw rhythm error over the original checkpoint. The
hold-mixture model had the lowest combined rhythm error, gap MAE, and hold MAE.
The original decoder had the lowest rhythm error after rescaling.

The shipped validation objective of 1.181179 comes from its original split and
split-timing loss, which also scores a pause classifier. The common holdout
above uses comparable predictive metrics for both timing formulations.

The initial four-model common-holdout run wrote `.data/full-corpus/common-holdout.json`.
