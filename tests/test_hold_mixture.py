import math
import unittest

import torch

from experiments.full_corpus import warm_start_hold_mixture
from model.network import BUFFER_CONTEXT, Action, TypingEventModel


class HoldMixtureTests(unittest.TestCase):
    def test_warm_start_preserves_original_rows_and_other_predictions(self):
        architecture = {
            "hidden_size": 8, "event_model_v5": True, "timing_components": 3,
            "unified_timing": True, "mlp_head": "all",
        }
        parent = TypingEventModel(**architecture).eval()
        model = TypingEventModel(**architecture, hold_components=3).eval()
        warm_start_hold_mixture(model, parent.state_dict())
        for key, expected in parent.state_dict().items():
            actual = model.state_dict()[key]
            if key.startswith("hold_head.2."):
                actual = actual[:len(expected)]
            self.assertTrue(torch.equal(actual, expected), key)
        inputs = (
            torch.zeros(1, 2, model.target_left + model.target_right, dtype=torch.long),
            torch.zeros(1, 2, BUFFER_CONTEXT, dtype=torch.long),
            torch.zeros(1, 2, 1, dtype=torch.long),
            torch.zeros(1, 2, 1, dtype=torch.long),
            torch.zeros(1, 2, model.style_features),
        )
        with torch.inference_mode():
            previous = parent(*inputs)
            current = model(*inputs)
        self.assertEqual(model.hold_head[-1].out_features, 27)
        for index in (0, 1, 2, 3, 4, 5):
            torch.testing.assert_close(current[index], previous[index])
        for index in (7, 8, 9):
            self.assertEqual(current[index].shape, (1, 2, len(Action), 3))
        for index in (7, 8):
            torch.testing.assert_close(current[index][..., 0], previous[index])
        self.assertGreater(current[9].softmax(-1)[..., 0].min().item(), 0.9)

    def test_hold_scoring_uses_mixture_and_action(self):
        from experiments.evaluate import head_accuracy, head_losses

        mask = torch.tensor([[True]])
        actions = torch.tensor([[Action.BACKSPACE]])
        batch = {"actions": actions, "mask": mask, "timing_mask": mask,
                 "hold_mask": mask, "pauses": ~mask, "wrong_keys": torch.zeros_like(actions),
                 "delays": torch.full((1, 1), 0.1), "holds": torch.full((1, 1), 0.12)}
        gap = torch.zeros(1, 1, 3, 3)
        means = torch.full_like(gap, math.log(0.04))
        means[0, 0, Action.BACKSPACE, 1] = math.log(0.12)
        logits = torch.full_like(gap, -100.0)
        logits[..., 1] = 0
        scales = torch.full_like(gap, math.log(0.2))
        outputs = (torch.zeros(1, 1, 3), torch.zeros(1, 1, 95), gap, gap, gap,
                   None, None, means, scales, logits)
        losses, _ = head_losses(outputs, batch)
        expected_nll = math.log(0.2) + 0.5 * math.log(2 * math.pi)
        self.assertAlmostEqual(losses["hold"].item(), expected_nll, places=5)
        self.assertLess(head_accuracy(outputs, batch)["hold_mae_ms"].item(), 0.001)
