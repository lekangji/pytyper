"""Golden plans captured before changing the release or backup decoders."""

import json
import math
import unittest
from pathlib import Path

import torch

from app.config import TypingConfig
from model.controls import SECONDS_PER_CHARACTER_PER_WPM, TypingControls
from model.runtime import NeuralTypingRuntime

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).with_name("golden_plans.json")


class CheckpointTests(unittest.TestCase):
    def test_new_bundled_default_loads_and_generates(self):
        config = TypingConfig("The quick brown fox.", 0, TypingControls())
        checkpoint = ROOT / config.model_path
        self.assertEqual(checkpoint.name, "gelu-16d-hold-mixture.pt")
        spec = ROOT / "models" / "specs" / f"{checkpoint.stem}.json"
        runtime = NeuralTypingRuntime(checkpoint, seed=251, spec_path=spec)
        self.assertTrue(runtime.model.unified_timing)
        self.assertEqual(runtime.model.hold_components, 3)
        events = runtime.generate(config.text, TypingControls(error_rate=0.0))
        self.assertEqual("".join(event.key for event in events), config.text)
        self.assertTrue(all(0.02 <= event.hold_seconds <= 0.18 for event in events))

    def test_original_plans_and_replay(self):
        torch.set_num_threads(1)
        fixture = json.loads(GOLDEN.read_text(encoding="utf-8"))
        for group in fixture["checkpoints"]:
            checkpoint = ROOT / group["checkpoint"]
            if not checkpoint.exists() and group["checkpoint"].startswith(".backup/"):
                continue
            backspaces = 0
            for case in group["cases"]:
                with self.subTest(checkpoint=checkpoint.name, seed=case["seed"]):
                    runtime = NeuralTypingRuntime(
                        checkpoint, case["seed"], spec_path=ROOT / group["spec"],
                    )
                    controls = TypingControls(**case["controls"]).validated()
                    sampled, events = runtime.plan(case["text"], controls)
                    sampled_timing = group.get("timing") == "sampled"
                    golden_events = sampled if sampled_timing else events
                    actual = [[event.key, event.delay_before, event.hold_seconds] for event in golden_events]
                    if fixture["torch"] == torch.__version__:
                        self.assertEqual(actual, case["events"])
                    if sampled_timing:
                        self.assertAlmostEqual(
                            math.fsum(event.delay_before for event in events),
                            len(case["text"]) * SECONDS_PER_CHARACTER_PER_WPM / controls.wpm,
                        )
                    buffer = ""
                    for raw, event in zip(sampled, events, strict=True):
                        key, delay, hold = event.key, event.delay_before, event.hold_seconds
                        if sampled_timing:
                            self.assertEqual((key, hold), (raw.key, raw.hold_seconds))
                        self.assertTrue(math.isfinite(delay) and delay >= 0)
                        self.assertTrue(math.isfinite(hold) and 0 <= hold <= 0.18)
                        if key == "BACKSPACE":
                            self.assertTrue(buffer)
                            buffer = buffer[:-1]
                            backspaces += 1
                        else:
                            buffer += key
                    self.assertEqual(buffer, case["text"])
                    if group["checkpoint"] == "models/pytyper.pt":
                        raw = NeuralTypingRuntime(checkpoint, case["seed"])
                        self.assertEqual(events, raw.generate(case["text"], TypingControls(**case["controls"])))
            self.assertGreater(backspaces, 0)

    def test_spec_rejects_wrong_weight_file(self):
        fixture = json.loads(GOLDEN.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            NeuralTypingRuntime(
                ROOT / "models/pytyper.pt", spec_path=ROOT / fixture["checkpoints"][0]["spec"],
            )
