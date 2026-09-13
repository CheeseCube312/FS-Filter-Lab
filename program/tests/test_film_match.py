"""Film Match deviation metrics (UX_SPEC.md §7.2)."""
import unittest

import numpy as np

from models.constants import INTERP_GRID
from services.presenter import _channel_deviation


def _gauss(center: float, width: float, height: float = 1.0) -> np.ndarray:
    return height * np.exp(-((INTERP_GRID - center) ** 2) / (2 * width ** 2))


class TestChannelDeviation(unittest.TestCase):
    def test_identical_curves_zero_rms_unit_ratio(self):
        curve = _gauss(550, 40)
        peak, ratio, rms = _channel_deviation(curve, curve.copy())
        self.assertEqual(peak, 550)
        self.assertAlmostEqual(ratio, 1.0)
        self.assertAlmostEqual(rms, 0.0)

    def test_scaled_target_peak_normalizes(self):
        """Only channel ratio changes when the target is uniformly scaled."""
        curve = _gauss(550, 40)
        peak, ratio, rms = _channel_deviation(curve, curve * 0.5)
        self.assertEqual(peak, 550)
        self.assertAlmostEqual(ratio, 2.0)
        self.assertAlmostEqual(rms, 0.0)

    def test_shifted_curve_has_nonzero_rms_and_peak(self):
        achieved = _gauss(580, 40)
        target = _gauss(550, 40)
        peak, _, rms = _channel_deviation(achieved, target)
        self.assertEqual(peak, 580)
        self.assertGreater(rms, 0.01)

    def test_flat_zero_curves_guarded(self):
        zero = np.zeros_like(INTERP_GRID, dtype=float)
        peak, ratio, rms = _channel_deviation(zero, zero)
        self.assertIsNone(peak)
        self.assertEqual(ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
