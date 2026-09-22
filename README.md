# pytyper

**Have your computer type like a human with natural rhythm, pauses, typos, and corrections.**

`pytyper` generates keyboard events using a small neural net. It types requested text with variable rhythm, pauses, temporary mistakes, and realistic corrections.

## Getting Started

### Requirements

- Python **3.10+**
- Dependencies listed in `requirements.txt`

### Run

From the project directory:

```powershell
python -m pip install -r requirements.txt
python main.py
```

Then:

1. Paste text or load a UTF-8 file.
2. Adjust the typing style.
3. Select **Start typing**.
4. Focus the destination window before the countdown ends.

Press **Esc** or select **Stop typing** at any time to cancel.

## Features

The progress window reports:

- Delivered text
- Observed typing speed
- Key-entry accuracy
- Estimated time remaining

The bundled model checkpoint is located at:

```text
models/pytyper.pt
```

The application does **not** train or download model weights on launch.

### Input Limits

Loaded text files have a maximum size of **2 MB**.

### Key Holds and Rollover

For simple keys, the model can generate separate key presses and releases with overlapping holds to produce more natural keyboard behavior.

If a target application mishandles held keys, disable **Key holds / rollover**.

## Model and Usage Limits

The bundled checkpoint was trained on a physical QWERTY subset of the [Aalto 136M Keystrokes dataset](https://userinterfaces.aalto.fi/136Mkeystrokes/).

The source dataset permits **research and non-commercial use with attribution**.

See [`MODEL_CARD.md`](MODEL_CARD.md) for:

- Dataset provenance
- Checkpoint metadata
- Model limitations
- Usage restrictions

## Project Structure

```text
pytyper/
├── app/       Desktop UI and keyboard worker
├── model/     Neural network, controls, and inference runtime
├── models/    Bundled pytyper checkpoint
└── main.py    Application entry point
```
