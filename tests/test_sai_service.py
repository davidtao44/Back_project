"""Unit tests for sai_service (SAI metric)."""

import os
import sys
import unittest

# Allow running directly with `python tests/test_sai_service.py`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.sai_service import compute_sai, compute_sai_from_runs, interpret_sai


class ComputeSAITests(unittest.TestCase):
    def test_both_zero_returns_none(self):
        self.assertIsNone(compute_sai(0.0, 0.0))

    def test_symmetric_returns_zero(self):
        self.assertEqual(compute_sai(0.5, 0.5), 0.0)
        self.assertEqual(compute_sai(0.123, 0.123), 0.0)

    def test_only_s1_returns_plus_one(self):
        self.assertEqual(compute_sai(0.0, 1.0), 1.0)
        self.assertEqual(compute_sai(0.0, 0.42), 1.0)

    def test_only_s0_returns_minus_one(self):
        self.assertEqual(compute_sai(1.0, 0.0), -1.0)
        self.assertEqual(compute_sai(0.42, 0.0), -1.0)

    def test_partial_asymmetry(self):
        self.assertAlmostEqual(compute_sai(0.2, 0.8), 0.6)
        self.assertAlmostEqual(compute_sai(0.8, 0.2), -0.6)


class ComputeSAIFromRunsTests(unittest.TestCase):
    def test_basic_summary(self):
        s = compute_sai_from_runs(
            {"n_inj": 100, "n_prop": 10},
            {"n_inj": 100, "n_prop": 30},
        )
        self.assertEqual(s["n_inj_s0"], 100)
        self.assertEqual(s["n_inj_s1"], 100)
        self.assertEqual(s["f_prop_s0"], 0.1)
        self.assertEqual(s["f_prop_s1"], 0.3)
        self.assertEqual(s["sai"], 0.5)
        self.assertIn("s@1", s["interpretation"])

    def test_no_propagation_returns_none_sai(self):
        s = compute_sai_from_runs(
            {"n_inj": 50, "n_prop": 0},
            {"n_inj": 50, "n_prop": 0},
        )
        self.assertIsNone(s["sai"])
        self.assertIn("undefined", s["interpretation"])

    def test_zero_inj_treated_as_zero_f_prop(self):
        s = compute_sai_from_runs(
            {"n_inj": 0, "n_prop": 0},
            {"n_inj": 10, "n_prop": 5},
        )
        self.assertEqual(s["f_prop_s0"], 0.0)
        self.assertEqual(s["f_prop_s1"], 0.5)
        self.assertEqual(s["sai"], 1.0)


class InterpretSAITests(unittest.TestCase):
    def test_none_undefined(self):
        self.assertIn("undefined", interpret_sai(None))

    def test_symmetric(self):
        self.assertIn("symmetric", interpret_sai(0.05))
        self.assertIn("symmetric", interpret_sai(-0.09))

    def test_positive_means_s1(self):
        self.assertIn("s@1", interpret_sai(0.5))

    def test_negative_means_s0(self):
        self.assertIn("s@0", interpret_sai(-0.5))


if __name__ == "__main__":
    unittest.main()
