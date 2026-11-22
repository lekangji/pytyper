# pytyper v1 model card

## Intended use and source

pytyper generates visible keyboard events for local UI automation and
demonstrations. It models desktop sentence-transcription typing, including
pauses, mistakes, corrections, key holds, and some overlapping presses. It is not
intended for biometric impersonation.

Both checkpoints use physical QWERTY data from
Dhakal et al., [*Observations on Typing from 136 Million Keystrokes*](https://userinterfaces.aalto.fi/136Mkeystrokes/)
(CHI 2018). The dataset source permits research and non-commercial use with
attribution. Commercial use requires permission from the dataset authors.

## Bundled checkpoints

**Default model.** `models/gelu-16d-hold-mixture.pt` uses three 128-unit GRU
layers, 16-dimensional key embeddings, and 64-unit GELU output heads. It sees
three characters behind and ten ahead of the current character. Separate
three-component log-normal mixtures predict the gap before a keypress and the
duration of its hold, conditioned on the action. Its full-corpus training split
contains 990,283 trials and 43,290,250 events. Fine-tuning started from the
`gelu-16d` checkpoint and selected epoch 55 at a validation loss of 0.976423.
See the [checkpoint spec](models/specs/gelu-16d-hold-mixture.json).

**Original model.** `models/pytyper.pt` uses a three-layer GRU with a separate
pause classifier and a single log-normal hold distribution per action. Its
metadata reports 99,684 training trials, 11,261 validation trials, 4,349,953
training events, and a best validation loss of 1.1812 under its original
objective. See the [checkpoint spec](models/specs/release.json).

Both checkpoints use intended-text and visible-buffer context, previous key and
action, keyboard transition features, and a six-value session style. The
[common holdout](TIMING_EXPERIMENTS.md#shipped-checkpoint--common-participant-holdout)
compares predictive accuracy and generated rhythm under the same protocol.

## Controls and limits

Speed, error rate, rhythm, and pause controls guide a sampled typing style.
Requested WPM and error rate are approximate in the target application.
Speed calibration uses a smooth curve fitted to recorded typing to adjust
gap variation with WPM. It then fits the plan's duration to the requested
speed. The decoder constrains planned output to the requested final text,
but delivery depends on target focus and keyboard handling.

Training data contain English sentence transcription rather than spontaneous
composition. Wrong-key predictions cover printable ASCII. Other target
characters use an unknown input embedding and the keyboard library's
correct-character path. Physical rollover depends on the application and
keyboard layout. Key holds are capped at 180 ms to reduce key-repeat risk.
