"""Check what counts toward action, wrong-key, and hold accuracy."""

import math
import unittest

import torch

from experiments.evaluate import head_accuracy
from model.network import BUFFER_CONTEXT, VOCAB_SIZE, Action, TypingEventModel


class EvaluationAccuracyTests(unittest.TestCase):
    def test_training_and_scoring_on_selected_device(self):
        from experiments.evaluate import predict
        from experiments.run import train_epoch, training_device

        device = training_device()
        self.assertEqual(device.type, "cuda" if torch.cuda.is_available() else "cpu")
        model = TypingEventModel(
            hidden_size=8, event_model_v5=True, timing_components=3, unified_timing=True,
        ).to(device)
        mask = torch.tensor([[True, True, True], [True, True, False]])
        batch = {
            "target_context": torch.full((2, 3, model.target_left + model.target_right), 2),
            "buffer_context": torch.zeros(2, 3, BUFFER_CONTEXT, dtype=torch.long),
            "previous_key": torch.zeros(2, 3, 1, dtype=torch.long),
            "previous_action": torch.zeros(2, 3, 1, dtype=torch.long),
            "style": torch.zeros(2, 3, model.style_features),
            "keyboard_features": torch.empty(2, 3, 0),
            "actions": torch.tensor([[Action.CORRECT, Action.WRONG, Action.BACKSPACE],
                                     [Action.CORRECT, Action.CORRECT, Action.CORRECT]]),
            "wrong_keys": torch.zeros(2, 3, dtype=torch.long),
            "holds": torch.full((2, 3), 0.1),
            "delays": torch.full((2, 3), 0.2),
            "mask": mask,
            "timing_mask": torch.tensor([[False, True, True], [False, True, False]]),
            "pauses": torch.zeros_like(mask),
            "hold_mask": mask,
        }
        optimizer = torch.optim.AdamW(model.parameters())
        self.assertTrue(math.isfinite(train_epoch(model, [batch], optimizer, torch.ones(len(Action)))))
        scores = predict(model, [batch], include_accuracy=True)
        self.assertTrue(math.isfinite(scores["gap_nll"]))
        self.assertEqual(scores["accuracy_counts"], {"actions": 5, "wrong_actions": 1, "holds": 5})

    def test_packed_predictions_match_dense_on_valid_steps(self):
        model = TypingEventModel(
            hidden_size=16, layers=2, event_model_v5=True, timing_components=3,
            unified_timing=True, gru_residual_alpha=0.25, mlp_head="action",
        ).eval()
        lengths = torch.tensor([2, 5, 3])
        mask = torch.arange(5) < lengths[:, None]
        inputs = (
            torch.randint(VOCAB_SIZE, (3, 5, model.target_left + model.target_right)),
            torch.randint(VOCAB_SIZE, (3, 5, BUFFER_CONTEXT)),
            torch.randint(VOCAB_SIZE, (3, 5, 1)),
            torch.randint(len(Action) + 1, (3, 5, 1)),
            torch.randn(3, 5, model.style_features),
        )
        with torch.inference_mode():
            dense = model(*inputs)
            packed = model(*inputs, lengths=lengths)
        for index in (0, 1, 2, 3, 4, 7, 8):
            torch.testing.assert_close(packed[index][mask], dense[index][mask], atol=1e-6, rtol=1e-5)

    def test_wrong_action_ranking_groups_tied_scores(self):
        from experiments.evaluate import wrong_action_ranking

        scores = torch.tensor([0.4, 0.8, 0.9, 0.8])
        wrong = torch.tensor([False, False, True, True])
        ranking = wrong_action_ranking(scores, wrong)
        self.assertAlmostEqual(ranking["wrong_action_average_precision"], 5 / 6)
        self.assertAlmostEqual(ranking["wrong_action_auroc"], 0.875)

    def test_scores_apply_their_own_masks(self):
        batch = {
            "actions": torch.tensor([[Action.CORRECT, Action.WRONG, Action.WRONG, Action.BACKSPACE, Action.WRONG]]),
            "wrong_keys": torch.tensor([[0, 1, 1, 0, 1]]),
            "mask": torch.tensor([[True, True, True, True, False]]),
            "holds": torch.tensor([[0.100, 0.100, 0.100, 0.100, 0.100]]),
            "hold_mask": torch.tensor([[True, False, True, True, True]]),
        }
        action_logits = torch.nn.functional.one_hot(torch.tensor([[0, 1, 0, 2, 1]]), num_classes=3)
        wrong_logits = torch.nn.functional.one_hot(torch.tensor([[0, 0, 1, 0, 1]]), num_classes=2)
        hold_medians = torch.tensor([[[0.090, 0.200, 0.200], [0.200, 0.200, 0.200],
                                     [0.200, 0.150, 0.200], [0.200, 0.200, 0.105], [0.100, 0.100, 0.100]]])
        scores = head_accuracy((action_logits, wrong_logits, None, None, None, None, None,
                                hold_medians.log(), None), batch)
        self.assertEqual(scores["action_accuracy"].tolist(), [True, True, False, True])
        self.assertEqual(scores["wrong_action_recall"].tolist(), [True, False])
        self.assertEqual(scores["wrong_key_accuracy"].tolist(), [False, True])
        torch.testing.assert_close(scores["hold_mae_ms"], torch.tensor([10.0, 50.0, 5.0]))
        self.assertEqual(scores["hold_within_20ms"].tolist(), [True, False, True])
