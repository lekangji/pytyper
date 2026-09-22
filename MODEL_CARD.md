# pytyper v1 model card

## Intended use and source

pytyper generates visible keyboard events for local UI automation and
demonstrations. It models desktop sentence-transcription typing, including
pauses, mistakes, corrections, key holds, and some overlapping presses. It is not
intended for biometric impersonation.

The checkpoint metadata identifies the source as a physical QWERTY subset of
Dhakal et al., [*Observations on Typing from 136 Million Keystrokes*](https://userinterfaces.aalto.fi/136Mkeystrokes/)
(CHI 2018). The dataset source permits research and non-commercial use with
attribution. Commercial use requires permission from the dataset authors.

## Bundled checkpoint

`models/pytyper.pt` contains a three-layer GRU with intended-text and visible
buffer context, previous key and action, keyboard transition features, and a
six-value session style. Output heads predict the next action, a wrong key when
applicable, pause choice, event timing, and key-hold duration.

Checkpoint metadata reports 99,684 training trials, 11,261 validation trials,
4,349,953 training events, and a best validation loss of 1.1812. These are
training records, not a measure of perceived realism. No blinded human
evaluation is supplied.

The checkpoint stores internal architecture and format identifiers needed for
loading. These identifiers do not describe the public release version.

## Controls and limits

Speed, error rate, rhythm, and pause controls guide a sampled typing style.
Requested WPM and error rate are approximate. The decoder constrains planned
output to the requested final text, but delivery depends on target focus and
keyboard handling.

Training data contain English sentence transcription rather than spontaneous
composition. Wrong-key predictions cover printable ASCII. Other target
characters use an unknown input embedding and the keyboard library's
correct-character path. Physical rollover depends on the application and
keyboard layout. Key holds are capped at 180 ms to reduce key-repeat risk.
