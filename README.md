# pytyper

pytyper generates keyboard events from a trained neural model. It types requested
text with variable rhythm, pauses, temporary mistakes, and corrections. The
decoder plans the exact final text; focus changes or input failures in the target
application can still interrupt delivery.

## Run

Install Python 3.10 or newer, then run from the project directory:

```powershell
python -m pip install -r requirements.txt
python main.py
```

Paste text or load a UTF-8 file, adjust typing style, then select **Start typing**.
Focus the destination window before the countdown ends. Press **Esc** or select
**Stop typing** to cancel. The progress window reports delivered text, observed
speed, key-entry accuracy, and estimated time left.

The bundled checkpoint is `models/pytyper.pt`. The app does not train or download
weights on launch. Loaded text files have a 2 MB limit. On simple keys, the model
can use separate key presses and releases with overlapping holds; turn off
**Key holds / rollover** if a target app mishandles held keys.

## Model and use limits

The checkpoint was trained on a physical QWERTY subset of the
[Aalto 136M Keystrokes dataset](https://userinterfaces.aalto.fi/136Mkeystrokes/).
That source permits research and non-commercial use with attribution. See
[MODEL_CARD.md](MODEL_CARD.md) for provenance, checkpoint metadata, and limits.

## Project layout

```text
app/       desktop UI and keyboard worker
model/     neural network, controls, and inference runtime
models/    bundled pytyper checkpoint
main.py    application entrypoint
```
