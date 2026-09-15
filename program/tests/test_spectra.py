"""services.spectra — the physics core.

Hand-computed oracles on 4-point toy grids: the multiplication chain, the
NaN policy ("one bad point spoils the point" for products, excluded points
for integrals), and green-referenced white balance.
"""
import unittest

import numpy as np

from models.constants import DEFAULT_WB_GAINS
from models.core import ChannelMixerSettings
from services import spectra
from services.channel_mixer import apply_channel_mixing_to_responses
from tests.helpers import TestCase, flat, toy, with_nan


class CombineTransmission(TestCase):
    def test_empty_stack_returns_identity_spectrum(self):
        out = spectra.combine_transmission([])
        self.assertEqual(len(out), 801)  # ones_like(INTERP_GRID)
        self.assertTrue(np.all(out == 1.0))

    def test_single_filter_passes_through_unchanged(self):
        t = toy()
        self.assertNanEqual(spectra.combine_transmission([t]), t)

    def test_multi_filter_multiplies_per_wavelength(self):
        a, b = toy() * 0.5, toy() * 2.0
        self.assertNanEqual(spectra.combine_transmission([a, b]), toy() ** 2)

    def test_nan_in_any_operand_spoils_only_that_wavelength(self):
        a = with_nan(toy() * 0.5, 1)
        out = spectra.combine_transmission([a, toy() * 2.0])
        self.assertTrue(np.isnan(out[1]))
        self.assertNanEqual(np.delete(out, 1), np.delete(toy() ** 2, 1))

    def test_no_clipping_values_above_one_survive(self):
        out = spectra.combine_transmission([toy(), toy()])
        self.assertNanEqual(out, toy() ** 2)  # 16.0 at the last point, not 1.0


class IntegratedResponse(TestCase):
    def chain(self, trans, reflector=None):
        return spectra.integrated_response(
            trans, np.ones(4), np.full(4, 2.0), reflector)

    def test_plain_sum_of_the_chain(self):
        # 0.5 * 1 * 2 summed over 4 points = 4.0
        self.assertClose(self.chain(np.full(4, 0.5)), 4.0)

    def test_reflector_multiplies_into_the_chain(self):
        self.assertClose(self.chain(np.full(4, 0.5), np.full(4, 0.5)), 2.0)

    def test_nan_wavelength_is_excluded_not_zero_filled(self):
        self.assertClose(self.chain(with_nan(np.full(4, 0.5), 1)), 3.0)

    def test_all_wavelengths_invalid_yields_nan(self):
        self.assertTrue(np.isnan(self.chain(np.full(4, np.nan))))


class SensorChannelCurves(TestCase):
    def test_per_channel_product(self):
        curves = spectra.sensor_channel_curves(
            np.full(4, 0.5), {"R": np.full(4, 2.0), "G": np.ones(4)})
        self.assertNanEqual(curves["R"], np.full(4, 1.0))
        self.assertNanEqual(curves["G"], np.full(4, 0.5))

    def test_illuminant_none_means_no_illuminant_factor(self):
        curves = spectra.sensor_channel_curves(
            np.full(4, 0.5), {"R": np.full(4, 2.0)})
        self.assertNanEqual(curves["R"], np.full(4, 1.0))

    def test_display_scale_scales_output_percent_scale(self):
        curves = spectra.sensor_channel_curves(
            np.full(4, 0.5), {"R": np.full(4, 2.0)}, np.full(4, 2.0),
            display_scale=100.0)
        self.assertNanEqual(curves["R"], np.full(4, 200.0))

    def test_nan_in_transmission_poisons_that_wavelength_only(self):
        curves = spectra.sensor_channel_curves(
            with_nan(np.full(4, 0.5), 2), {"R": np.ones(4)})
        self.assertTrue(np.isnan(curves["R"][2]))
        self.assertNanEqual(np.delete(curves["R"], 2), np.full(3, 0.5))


class WbGains(TestCase):
    """Green-referenced normalization, hand-computed.

    trans=0.5 flat, illuminant=2 flat over 4 points:
      R = 0.5*2*2*4 = 8   G = 0.5*1*2*4 = 4   B = 4  ->  gains R:2, G:1, B:1
    """

    def setUp(self):
        self.trans = np.full(4, 0.5)
        self.illum = np.full(4, 2.0)
        self.qe = {"R": np.full(4, 2.0), "G": np.ones(4), "B": np.ones(4)}

    def gains(self, trans=None, qe=None, reflector=None):
        return spectra.wb_gains(
            self.trans if trans is None else trans,
            self.qe if qe is None else qe,
            self.illum, reflector=reflector)

    def test_green_referenced_gains_hand_computed(self):
        self.assertEqual(self.gains(), {"R": 2.0, "G": 1.0, "B": 1.0})

    def test_neutral_reflector_leaves_gains_unchanged(self):
        # A neutral grey scales every channel equally: ratios are untouched.
        self.assertEqual(self.gains(reflector=np.full(4, 0.5)),
                         {"R": 2.0, "G": 1.0, "B": 1.0})

    def test_invalid_transmission_falls_back_to_unity(self):
        self.assertEqual(self.gains(trans=np.full(4, np.nan)), DEFAULT_WB_GAINS)

    def test_missing_channel_gets_unity_gain(self):
        self.assertEqual(self.gains(qe={"G": np.ones(4)}), DEFAULT_WB_GAINS.copy())

    def test_dead_green_channel_falls_back_to_defaults(self):
        qe = {"R": np.full(4, 2.0), "G": np.zeros(4), "B": np.ones(4)}
        self.assertEqual(self.gains(qe=qe), DEFAULT_WB_GAINS)

    def test_nan_green_response_falls_back_to_defaults(self):
        qe = {"R": np.full(4, 2.0), "G": np.full(4, np.nan), "B": np.ones(4)}
        self.assertEqual(self.gains(qe=qe), DEFAULT_WB_GAINS)


class IsValidSpectrum(TestCase):
    def test_rejects_none_empty_and_all_nan(self):
        self.assertFalse(spectra.is_valid_spectrum(None))
        self.assertFalse(spectra.is_valid_spectrum(np.array([])))
        self.assertFalse(spectra.is_valid_spectrum(np.array([np.nan, np.nan])))

    def test_accepts_any_array_with_one_finite_entry(self):
        self.assertTrue(spectra.is_valid_spectrum(np.array([np.nan, 0.0])))
        self.assertTrue(spectra.is_valid_spectrum(np.zeros(3)))

    def test_rejects_non_array_like_input(self):
        self.assertFalse(spectra.is_valid_spectrum(5))
        self.assertFalse(spectra.is_valid_spectrum("nope"))


class ChannelMixerResponses(TestCase):
    """The response-curve mixer path: curves stack into (n, 3), multiply by
    the mixer matrix, and come back per channel — unclamped, zero-filled for
    missing keys."""

    def test_disabled_mixer_returns_curves_untouched(self):
        responses = {"R": flat(2.0), "G": flat(1.0), "B": flat(0.5)}
        out = apply_channel_mixing_to_responses(responses, ChannelMixerSettings())
        self.assertNanEqual(out["R"], flat(2.0))
        self.assertNanEqual(out["G"], flat(1.0))
        self.assertNanEqual(out["B"], flat(0.5))

    def test_rb_swap_moves_energy_between_channels(self):
        swap = ChannelMixerSettings(enabled=True, red_r=0.0, red_b=1.0,
                                    blue_r=1.0, blue_b=0.0)
        out = apply_channel_mixing_to_responses(
            {"R": flat(2.0), "G": flat(1.0), "B": flat(0.5)}, swap)
        self.assertNanEqual(out["R"], flat(0.5))
        self.assertNanEqual(out["G"], flat(1.0))
        self.assertNanEqual(out["B"], flat(2.0))

    def test_missing_channel_is_zero_filled_not_keyerror(self):
        swap = ChannelMixerSettings(enabled=True, red_r=0.0, red_b=1.0,
                                    blue_r=1.0, blue_b=0.0)
        out = apply_channel_mixing_to_responses({"R": flat(2.0)}, swap)
        # Missing G/B read as zeros; the R energy shows up in B's output.
        self.assertNanEqual(out["R"], flat(0.0))
        self.assertNanEqual(out["G"], flat(0.0))
        self.assertNanEqual(out["B"], flat(2.0))


if __name__ == "__main__":
    unittest.main()
