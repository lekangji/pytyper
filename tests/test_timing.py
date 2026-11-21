"""Distribution and runtime checks for unified timing."""

import math
import random
import unittest

import torch

from experiments.evaluate import mixture_side_log_masses
from model.checkpoint import ROOT
from model.controls import TypingControls
from model.network import PAUSE_THRESHOLD, Action, TypingEventModel, lognormal_mixture_nll
from model.runtime import NeuralKeyEvent, NeuralTypingRuntime
from model.timing import LOG_THRESHOLD, component_log_masses, sample_conditioned_log_gap


class CalibrationTests(unittest.TestCase):
    def test_faster_plans_compress_relative_gap_spread(self):
        events = [NeuralKeyEvent(key, gap, 0.1) for key, gap in zip("abcd", (0, 0.10, 0.20, 0.85))]
        slow = NeuralTypingRuntime._calibrate_event_timing(events, 4, TypingControls(wpm=20))
        fast = NeuralTypingRuntime._calibrate_event_timing(events, 4, TypingControls(wpm=200))
        raw_ratio = events[-1].delay_before / events[1].delay_before
        self.assertGreater(slow[-1].delay_before / slow[1].delay_before, raw_ratio)
        self.assertLess(fast[-1].delay_before / fast[1].delay_before, raw_ratio)
        for plan, duration in ((slow, 2.4), (fast, 0.24)):
            self.assertEqual(plan[0], events[0])
            self.assertEqual([(e.key, e.hold_seconds) for e in plan], [(e.key, e.hold_seconds) for e in events])
            self.assertAlmostEqual(sum(e.delay_before for e in plan), duration)

    def test_speed_fit_is_continuous_at_the_sampling_cutoff(self):
        events = [NeuralKeyEvent(key, gap) for key, gap in zip("abc", (0, 0.49, 0.51))]
        result = NeuralTypingRuntime._calibrate_event_timing(events, 3, TypingControls(wpm=200))
        self.assertAlmostEqual(sum(e.delay_before for e in result), 0.18)
        self.assertTrue(0 < result[1].delay_before < result[-1].delay_before < PAUSE_THRESHOLD)
        self.assertLess(result[-1].delay_before / result[1].delay_before, 0.51 / 0.49)

    def test_rhythm_control_still_changes_gap_spread(self):
        events = [NeuralKeyEvent(key, gap) for key, gap in zip("abc", (0, 0.1, 0.3))]
        even = NeuralTypingRuntime._calibrate_event_timing(events, 3, TypingControls(rhythm_variation=0.5))
        varied = NeuralTypingRuntime._calibrate_event_timing(events, 3, TypingControls(rhythm_variation=1.5))
        self.assertGreater(varied[2].delay_before / varied[1].delay_before,
                           even[2].delay_before / even[1].delay_before)

    def test_split_calibration_restores_a_pause_at_80_wpm(self):
        events = [NeuralKeyEvent(key, gap, 0.1) for key, gap in zip("abcdefgh", (0, *([0.1] * 6), 0.53))]
        controls = TypingControls(wpm=80)
        baseline = NeuralTypingRuntime._calibrate_event_timing(events, 8, controls)
        tuned = NeuralTypingRuntime._calibrate_event_timing(events, 8, controls, tune_pauses=True)
        self.assertLess(baseline[-1].delay_before, PAUSE_THRESHOLD)
        self.assertGreaterEqual(tuned[-1].delay_before, PAUSE_THRESHOLD)
        self.assertAlmostEqual(sum(event.delay_before for event in tuned), 1.2)
        self.assertEqual([(e.key, e.hold_seconds) for e in tuned], [(e.key, e.hold_seconds) for e in events])

    def test_pause_only_plan_can_change_speed(self):
        events = [NeuralKeyEvent("a", 0.0), NeuralKeyEvent("b", 0.75)]
        result = NeuralTypingRuntime._calibrate_event_timing(events, 2, TypingControls(wpm=24))
        self.assertEqual(result[0], events[0])
        self.assertAlmostEqual(result[1].delay_before, 1.0)

    def test_untimed_single_event_stays_untimed(self):
        events = [NeuralKeyEvent("a", 0.0)]
        self.assertEqual(NeuralTypingRuntime._calibrate_event_timing(events, 1, TypingControls()), events)


class MixtureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.rng = random.Random(71)
        self.logits = torch.tensor([1.1, 0.2, -0.8], dtype=torch.float64)
        self.means = torch.tensor([-2.0, -0.3, 0.0], dtype=torch.float64)
        self.scales = torch.tensor([-1.0, -1.2, -1.3], dtype=torch.float64)
        self.generator = torch.Generator().manual_seed(71)

    def test_mixture_side_mass_partitions_probability(self):
        masses = component_log_masses(self.means, self.scales)
        torch.testing.assert_close(masses.exp().sum(-1), torch.ones_like(self.means))
        weights = self.logits.softmax(-1)
        p = (weights[:, None] * masses.exp()).sum(0)
        expected = torch.special.ndtr((self.means - LOG_THRESHOLD) / self.scales.exp())
        self.assertAlmostEqual(p[1].item(), (weights * expected).sum().item(), places=13)
        self.assertAlmostEqual(p.sum().item(), 1.0, places=13)

    def test_conditional_sampling_reweights_components(self):
        weighted = self.logits.log_softmax(-1)[:, None] + component_log_masses(self.means, self.scales)
        for side in (0, 1):
            draws = [sample_conditioned_log_gap(
                weighted[:, side], self.means, self.scales, bool(side), self.rng, self.generator,
            ) for _ in range(3000)]
            self.assertTrue(all((value >= LOG_THRESHOLD) == side for value in draws))
            cut = LOG_THRESHOLD + (0.25 if side else -0.25)
            cdf = torch.special.ndtr((cut - self.means) / self.scales.exp())
            probability = weighted[:, side].logsumexp(0).exp()
            if side:
                expected = 1 - ((1 - cdf) * self.logits.softmax(-1)).sum() / probability
            else:
                expected = (cdf * self.logits.softmax(-1)).sum() / probability
            self.assertAlmostEqual(sum(value < cut for value in draws) / len(draws), expected.item(), delta=0.035)

    def test_full_and_side_normalized_log_gap_densities_integrate(self):
        log_gaps = torch.linspace(math.log(0.008), math.log(8.0), 4001, dtype=torch.float64)
        params = self.logits, self.means, self.scales
        density = (-lognormal_mixture_nll(log_gaps.exp(), *params)).exp()
        self.assertAlmostEqual(torch.trapezoid(density, log_gaps).item(), 1.0, places=6)
        for side, mask in enumerate((log_gaps < LOG_THRESHOLD, log_gaps >= LOG_THRESHOLD)):
            conditional = density[mask] / mixture_side_log_masses(params)[side].exp()
            self.assertAlmostEqual(torch.trapezoid(conditional, log_gaps[mask]).item(), 1.0, delta=0.004)

    def test_extreme_tail_returns_finite_sample(self):
        means = torch.tensor([-5.0, -4.999, -4.998], dtype=torch.float64)
        scales = torch.full_like(means, -3.5)
        weighted = self.logits + component_log_masses(means, scales)[:, 1]
        self.assertTrue(torch.isfinite(weighted).all())
        for _ in range(20):
            value = sample_conditioned_log_gap(weighted, means, scales, True, self.rng, self.generator)
            self.assertTrue(math.isfinite(value) and LOG_THRESHOLD <= value < LOG_THRESHOLD + 0.01)

    def test_neutral_slider_samples_directly(self):
        path = ROOT / "models/pytyper.pt"
        runtime = NeuralTypingRuntime(path, seed=13)
        direct = NeuralTypingRuntime(path, seed=13)
        runtime.model = TypingEventModel(event_model_v5=True, unified_timing=True, timing_components=3)
        timing = [value.expand(1, 1, 3, 3) for value in (self.logits, self.means, self.scales)]
        holds = torch.full((1, 1, 3), -2.1)
        outputs = (None, None, *timing, None, None, holds, holds)
        self.assertEqual(
            runtime._sample_event_timing(outputs, Action.WRONG, TypingControls(), first=False)[0],
            direct._sample_delay(self.logits, self.means, self.scales),
        )
        self.assertEqual(runtime._sample_event_timing(outputs, Action.WRONG, TypingControls(), first=True)[0], 0)

    def test_pause_slider_multiplies_odds_at_half_second(self):
        runtime = NeuralTypingRuntime(ROOT / "models/pytyper.pt", seed=19)
        weighted = self.logits.log_softmax(-1)[:, None] + component_log_masses(self.means, self.scales)
        short, long = weighted.logsumexp(0).exp().tolist()
        for multiplier in (0.25, 2.0):
            draws = [runtime._sample_mixture_delay(self.logits, self.means, self.scales, multiplier)
                     for _ in range(3500)]
            self.assertTrue(all(0.008 <= value <= 5.0 for value in draws))
            self.assertAlmostEqual(sum(value >= 0.5 for value in draws) / len(draws),
                                   multiplier * long / (short + multiplier * long), delta=0.025)
