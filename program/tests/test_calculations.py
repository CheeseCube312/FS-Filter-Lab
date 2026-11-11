"""services.calculations — stack math, metrics, and color identity.

Oracles: EPSILON-clipped vs raw combination (the two combinators differ),
photon-flux-weighted effective stops, the exposure-referenced surface chain,
and the sRGB transfer function.  Collections are built on the real grid.
"""
import math
import unittest

import numpy as np

from models.constants import DEFAULT_HEX_COLOR, EPSILON, INTERP_GRID
from models.core import ChannelMixerSettings
from services import calculations as calc
from tests.helpers import GRID, TestCase, band, display_name, flat, make_collection


def collection():
    """Half=0.5, Quarter=0.25, Zero=0.0 — products stay hand-checkable."""
    return make_collection([
        {"name": "Half", "number": "H1", "curve": flat(0.5), "hex": "#204060"},
        {"name": "Quarter", "number": "Q1", "curve": flat(0.25), "hex": "#ff0000"},
        {"name": "Zero", "number": "Z9", "curve": flat(0.0), "hex": ""},
    ])


class ComputeFilterTransmission(TestCase):
    def test_empty_stack_is_identity_labeled_no_filter(self):
        trans, label = calc.compute_filter_transmission([], collection().filter_matrix)
        self.assertEqual(label, "No Filter")
        self.assertNanEqual(trans, np.ones_like(GRID))

    def test_single_filter_returns_its_row_labeled_single(self):
        c = collection()
        trans, label = calc.compute_filter_transmission([1], c.filter_matrix)
        self.assertEqual(label, "Single")
        self.assertNanEqual(trans, c.filter_matrix[1])

    def test_combined_is_epsilon_clipped_product(self):
        # Half x Zero = 0.0 everywhere -> clipped UP to EPSILON.
        trans, label = calc.compute_filter_transmission(
            [0, 2], collection().filter_matrix)
        self.assertEqual(label, "Combined")
        self.assertTrue(np.all(trans == EPSILON))

    def test_combined_product_of_two_real_filters(self):
        trans, label = calc.compute_filter_transmission(
            [0, 1], collection().filter_matrix)
        self.assertEqual(label, "Combined")
        self.assertTrue(np.all(trans == 0.125))


class ComputeSelectedFilterIndices(TestCase):
    def test_multipliers_repeat_the_row(self):
        indices = calc.compute_selected_filter_indices(
            [display_name("Half", "H1", "Test"),
             display_name("Quarter", "Q1", "Test")],
            {display_name("Half", "H1", "Test"): 2}, collection())
        self.assertEqual(indices, [0, 0, 1])

    def test_missing_multiplier_defaults_to_one(self):
        indices = calc.compute_selected_filter_indices(
            [display_name("Half", "H1", "Test")], {}, collection())
        self.assertEqual(indices, [0])

    def test_unknown_names_are_skipped(self):
        indices = calc.compute_selected_filter_indices(
            [display_name("Half", "H1", "Test"),
             display_name("Ghost", "G9", "Nowhere")], {}, collection())
        self.assertEqual(indices, [0])

    def test_empty_selection(self):
        self.assertEqual(
            calc.compute_selected_filter_indices([], {}, collection()), [])


class EffectiveTransmission(TestCase):
    def test_names_and_multipliers_become_the_combined_curve(self):
        c = collection()
        trans = calc.effective_transmission(
            [display_name("Half", "H1", "Test")],
            {display_name("Half", "H1", "Test"): 3}, c)
        self.assertTrue(np.all(trans == 0.125))  # 0.5 ** 3

    def test_no_filters_is_identity(self):
        trans = calc.effective_transmission([], {}, collection())
        self.assertNanEqual(trans, np.ones_like(GRID))

    def test_unknown_names_only_is_identity(self):
        trans = calc.effective_transmission(
            [display_name("Ghost", "G9", "Nowhere")], {}, collection())
        self.assertNanEqual(trans, np.ones_like(GRID))

    def test_zero_product_stays_zero_unclipped(self):
        # The raw combinator: contrast with compute_filter_transmission's clip.
        trans = calc.effective_transmission(
            [display_name("Half", "H1", "Test"),
             display_name("Zero", "Z9", "Test")], {}, collection())
        self.assertTrue(np.all(trans == 0.0))

    def test_repeated_indices_multiply_repeatedly(self):
        trans = calc.effective_transmission(
            [display_name("Half", "H1", "Test")],
            {display_name("Half", "H1", "Test"): 2}, collection())
        self.assertTrue(np.all(trans == 0.25))


class ComputeEffectiveStops(TestCase):
    def test_uniform_half_is_exactly_one_stop(self):
        avg, stops = calc.compute_effective_stops(
            np.array([0.5, 0.5]), np.array([1.0, 1.0]))
        self.assertClose(avg, 0.5)
        self.assertClose(stops, 1.0)

    def test_photon_flux_weighting_hand_computed(self):
        # weights = qe*illum = [1, 3]; avg = (0.5*1 + 0.25*3)/4 = 0.3125
        avg, stops = calc.compute_effective_stops(
            np.array([0.5, 0.25]), np.array([1.0, 3.0]), np.array([1.0, 1.0]))
        self.assertClose(avg, 0.3125)
        self.assertClose(stops, -math.log2(0.3125))

    def test_illuminant_contributes_to_the_weights(self):
        # weights = illum*qe = [3, 1]; avg = (0.5*3 + 0.25*1)/4 = 0.4375
        avg, _ = calc.compute_effective_stops(
            np.array([0.5, 0.25]), np.array([1.0, 1.0]), np.array([3.0, 1.0]))
        self.assertClose(avg, 0.4375)

    def test_zero_transmission_clips_to_epsilon_before_log(self):
        avg, stops = calc.compute_effective_stops(
            np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        self.assertClose(avg, EPSILON)
        self.assertClose(stops, -math.log2(EPSILON))

    def test_nan_wavelengths_are_excluded(self):
        avg, _ = calc.compute_effective_stops(
            np.array([0.5, np.nan]), np.array([1.0, 1.0]), np.array([1.0, 1.0]))
        self.assertClose(avg, 0.5)

    def test_no_valid_data_returns_nans(self):
        avg, stops = calc.compute_effective_stops(
            np.array([np.nan, np.nan]), np.array([1.0, 1.0]))
        self.assertTrue(np.isnan(avg) and np.isnan(stops))

    def test_all_zero_weights_return_nans(self):
        avg, stops = calc.compute_effective_stops(
            np.array([0.5, 0.5]), np.array([0.0, 0.0]))
        self.assertTrue(np.isnan(avg) and np.isnan(stops))


class ReflectorColors(TestCase):
    """GRID-wide flat scene through a 0.5 reflector:
    R = 0.5*2*2*0.5*801 = 801.0, G = B = 400.5 (exact)."""

    def setUp(self):
        self.refl = flat(0.5)
        self.trans = flat(0.5)
        self.qe = {"R": flat(2.0), "G": flat(1.0), "B": flat(1.0)}
        self.illum = flat(2.0)

    def color(self, gains=None, refl=None, mixer=None):
        return calc.compute_reflector_color(
            self.refl if refl is None else refl, self.trans, self.qe,
            self.illum, channel_mixer=mixer, white_balance_gains=gains)

    def test_raw_integrated_response_per_channel(self):
        self.assertNanEqual(self.color(), [801.0, 400.5, 400.5])

    def test_white_balance_gains_divide_the_response(self):
        self.assertNanEqual(self.color(gains={"R": 2.0, "G": 1.0, "B": 1.0}),
                            [400.5, 400.5, 400.5])

    def test_invalid_transmission_returns_black(self):
        rgb = calc.compute_reflector_color(
            self.refl, np.full(len(GRID), np.nan), self.qe, self.illum)
        self.assertNanEqual(rgb, [0.0, 0.0, 0.0])

    def test_identity_mixer_enabled_leaves_color_unchanged(self):
        mixer = ChannelMixerSettings(enabled=True)  # identity matrix
        self.assertNanEqual(self.color(mixer=mixer), [801.0, 400.5, 400.5])

    def test_mixer_swaps_red_and_blue_without_clipping(self):
        # Photoshop-style R<->B swap: the 3x3 matrix mixes linearly, so the
        # unclamped R value (801) must travel to the blue channel intact.
        mixer = ChannelMixerSettings(enabled=True, red_r=0.0, red_b=1.0,
                                     blue_r=1.0, blue_b=0.0)
        self.assertNanEqual(self.color(mixer=mixer), [400.5, 400.5, 801.0])

    def test_white_reference_is_a_perfect_diffuser(self):
        anchor = calc.compute_white_reference_color(
            self.trans, self.qe, self.illum)
        self.assertNanEqual(anchor, [1602.0, 801.0, 801.0])  # refl = 1.0

    def test_reference_to_display_whites_the_anchor(self):
        display = calc.reference_color_to_display(
            np.array([801.0, 801.0, 801.0]), np.array([1602.0, 801.0, 801.0]))
        self.assertNanEqual(display, [0.5, 1.0, 1.0])

    def test_reference_to_display_guards_a_dead_white_anchor(self):
        display = calc.reference_color_to_display(
            np.array([0.5, 0.25, 1.0]), np.array([0.0, 0.0, 0.0]))
        self.assertNanEqual(display, [0.5, 0.25, 1.0])  # divisor falls back to 1

    def test_reference_to_display_clips_into_unit_range(self):
        display = calc.reference_color_to_display(
            np.array([-0.5, 2.0, 1.0]), np.array([1.0, 1.0, 1.0]))
        self.assertNanEqual(display, [0.0, 1.0, 1.0])


class CameraLook(TestCase):
    def test_neutral_grey_is_a_fixed_point(self):
        out = calc.apply_camera_look(np.array([0.5, 0.5, 0.5]))
        self.assertNanEqual(out, [0.5, 0.5, 0.5])

    def test_smoothstep_tone_curve_hand_computed(self):
        # Neutral triplet: the chroma stage is the identity, so the values
        # isolate the per-channel smoothstep.
        out = calc.apply_camera_look(np.array([0.25, 0.25, 0.25]))
        self.assertClose(out[0], 0.15625)   # 0.25^2 * (3 - 0.5)
        self.assertClose(out[1], 0.15625)
        self.assertClose(out[2], 0.15625)

    def test_negative_inversion_precedes_the_tone_curve(self):
        out = calc.apply_camera_look(np.array([0.25, 0.25, 0.25]), invert=True)
        self.assertClose(out[0], 0.84375)   # 1 - 0.25 = 0.75 goes in

    def test_chroma_boost_widens_the_channel_spread(self):
        rgb = np.array([0.6, 0.4, 0.4])
        out = calc.apply_camera_look(rgb)
        self.assertGreater(out[0] - out[2], rgb[0] - rgb[2])
        self.assertClose(out[0], 0.682, places=3)
        self.assertClose(out[2], 0.343, places=3)

    def test_output_is_clamped_to_unit_range(self):
        out = calc.apply_camera_look(np.array([2.0, -1.0, 0.5]))
        self.assertTrue(((out >= 0.0) & (out <= 1.0)).all())


class SrgbHelpers(TestCase):
    def test_encode_endpoints_exact(self):
        self.assertNanEqual(calc.srgb_encode(np.array([0.0, 1.0])),
                            [0.0, 1.0])  # 1.0 stays exactly 1.0

    def test_encode_low_segment_is_linear(self):
        self.assertClose(calc.srgb_encode(np.array([0.002]))[0], 0.002 * 12.92)

    def test_encode_midtone_uses_the_power_law(self):
        expected = 1.055 * 0.5 ** (1.0 / 2.4) - 0.055
        self.assertClose(calc.srgb_encode(np.array([0.5]))[0], expected)

    def test_encode_clips_out_of_range_input(self):
        out = calc.srgb_encode(np.array([-1.0, 2.0]))
        self.assertNanEqual(out, [0.0, 1.0])

    def test_to_hex_is_lowercase_rrggbb(self):
        self.assertEqual(calc.srgb_to_hex(np.array([1.0, 0.0, 0.0])), "#ff0000")

    def test_to_hex_midtone_is_bcbcbc(self):
        grey = calc.srgb_to_hex(np.full(3, 0.5))
        self.assertEqual(grey, "#bcbcbc")  # round(0.735356983...*255) = 188 = 0xbc


class FilterDisplayColor(TestCase):
    red_band = staticmethod(lambda: band(600, 700))

    def test_valid_metadata_hex_wins_over_the_spectrum(self):
        self.assertEqual(calc.filter_display_color(self.red_band(), "#00ff00"),
                         "#00ff00")

    def test_default_sentinel_hex_is_treated_as_no_data(self):
        derived = calc.filter_display_color(self.red_band(), DEFAULT_HEX_COLOR)
        self.assertNotEqual(derived, DEFAULT_HEX_COLOR)

    def test_derived_color_of_a_red_band_is_red_dominant(self):
        hex_color = calc.filter_display_color(self.red_band(), "")
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        self.assertTrue(r > g and r > b, f"expected red-dominant, got {hex_color}")

    def test_none_curve_falls_back_to_default(self):
        self.assertEqual(calc.filter_display_color(None, ""), DEFAULT_HEX_COLOR)

    def test_wrong_shape_curve_falls_back_to_default(self):
        self.assertEqual(calc.filter_display_color(np.ones(5), ""),
                         DEFAULT_HEX_COLOR)

    def test_curve_with_no_visible_light_falls_back_to_default(self):
        uv_only = band(320, 340)  # all-zero across the visible band
        self.assertEqual(calc.filter_display_color(uv_only, ""),
                         DEFAULT_HEX_COLOR)


class SpectralLocusColors(TestCase):
    """CIE-1931 single-wavelength colors behind the human-vision strip."""

    def test_locus_shape_is_n_by_3_and_never_negative(self):
        rgb = calc.spectral_locus_linear_srgb(GRID)
        self.assertEqual(rgb.shape, (len(GRID), 3))
        self.assertTrue(np.all(rgb >= 0.0))  # out-of-gamut lobes clipped to 0

    def test_locus_hues_follow_the_spectrum(self):
        def at(nm):
            return calc.spectral_locus_linear_srgb(np.array([float(nm)]))[0]
        blue, green, red = at(460), at(532), at(610)
        self.assertGreaterEqual(blue[2], blue[0])    # 460 nm: blue over red
        self.assertGreaterEqual(blue[2], blue[1])
        self.assertGreaterEqual(green[1], green[0])  # 532 nm: green dominant
        self.assertGreaterEqual(green[1], green[2])
        self.assertGreaterEqual(red[0], red[1])      # 610 nm: red dominant
        self.assertGreaterEqual(red[0], red[2])

    def test_locus_magnitude_falls_toward_the_spectrum_ends(self):
        locus = calc.spectral_locus_linear_srgb(np.array([550.0, 650.0, 700.0]))
        norms = np.linalg.norm(locus, axis=1)
        self.assertGreater(norms[0], norms[1])  # 550 nm outshines 650 nm
        self.assertGreater(norms[1], norms[2])  # 650 nm outshines 700 nm


class PhotopicBrightness(TestCase):
    """V(lambda) profile normalized over the passed (visible) range."""

    def test_profile_peaks_at_one_near_555(self):
        visible = GRID[(GRID >= 380) & (GRID <= 700)]
        v = calc.photopic_relative_brightness(visible)
        self.assertEqual(v.shape, visible.shape)
        self.assertTrue(np.all(v >= 0.0))
        self.assertClose(float(v.max()), 1.0)
        peak_nm = visible[int(np.argmax(v))]
        self.assertTrue(550 <= peak_nm <= 575,
                        f"fit peak lands at {peak_nm} nm, expected near 555")

    def test_mid_spectrum_dwarfs_the_deep_red_end(self):
        v = calc.photopic_relative_brightness(np.array([550.0, 650.0, 700.0]))
        self.assertGreater(v[0], 100 * v[2])  # 550 nm >> 700 nm
        self.assertGreater(v[0], v[1])
        self.assertGreater(v[1], v[2])


if __name__ == "__main__":
    unittest.main()
