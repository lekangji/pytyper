"""Check teacher-forced and free-running previous-gap features."""

import math
import unittest

import torch

from experiments.evaluate import previous_gap_inputs
from model.checkpoint import ROOT
from model.controls import TypingControls
from model.network import Action, TypingEventModel
from model.runtime import NeuralTypingRuntime


class FeedbackTests(unittest.TestCase):
    def test_recorded_history_is_shifted_masked_and_clamped(self):
        batch = {
            "delays": torch.tensor([[0.0, 0.001, 8.0, 0.2, 0.6, 0.0]]),
            "timing_mask": torch.tensor([[False, True, True, False, True, False]]),
            "mask": torch.tensor([[True, True, True, True, True, False]]),
        }
        original = batch["delays"].clone()
        expected = torch.tensor([[[0, 1], [0, 1], [math.log(0.008), 0], [math.log(5), 0], [0, 1], [0, 1]]])
        torch.testing.assert_close(previous_gap_inputs(batch), expected)
        torch.testing.assert_close(batch["delays"], original, rtol=0, atol=0)

    def test_runtime_uses_pre_rescale_feedback_during_forced_completion(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        runtime = NeuralTypingRuntime(ROOT / "models/pytyper.pt", seed=26)
        runtime.model = TypingEventModel(layers=3, event_model_v5=True, use_previous_gap=True).eval()
        with torch.no_grad():
            runtime.model.action_head.bias[Action.WRONG] = 1000
        observed = []

        def record(_model, _args, kwargs):
            observed.append(kwargs["previous_gap"])

        runtime.model.register_forward_pre_hook(record, with_kwargs=True)
        target = "A short line."
        before, after = runtime.plan(target, TypingControls(wpm=190, error_rate=0.2))
        self.assertGreater(len(before), len(target) * 3 + 32)
        self.assertTrue(any(a.delay_before != b.delay_before for a, b in zip(before, after)))
        inputs = {"delays": torch.tensor([[event.delay_before for event in before]]),
                  "timing_mask": torch.tensor([[False] + [True] * (len(before) - 1)]),
                  "mask": torch.ones(1, len(before), dtype=torch.bool)}
        torch.testing.assert_close(torch.cat(observed, dim=1), previous_gap_inputs(inputs), rtol=0, atol=0)
        buffer = ""
        for event in after:
            buffer = buffer[:-1] if event.key == "BACKSPACE" else buffer + event.key
        self.assertEqual(buffer, target)
