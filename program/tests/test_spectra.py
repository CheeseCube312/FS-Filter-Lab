"""
Regression suite for the spectra domain layer (roadmap S2).

Locks the S2 delegations -- services/calculations.py (S2a) and
services/presenter.py (S2b) route their spectral physics through
services/spectra.py -- by comparing the public API against embedded
reference implementations of the chain semantics, plus property tests
for the domain functions themselves.

The references were validated against the true pre-refactor inline code
during the S2a parity run (32/32 pass, recorded in TECHNICAL.md):
plain white balance was bitwise identical, surface white balance within
1 ulp.  Going forward this suite guards against drift in the shared
physics -- change spectra.py and these tests tell you what moved.

Run from the program/ directory:

    python -m unittest discover -s tests -v

The suite is stdlib-only (unittest + numpy); no pytest dependency.
"""
import inspect
import math
import os
import sys
import unittest
from types import SimpleNamespace

# Make the program/ package root importable regardless of invocation cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from models.constants import DEFAULT_WB_GAINS, EPSILON
from services import calculations as calc
from services import spectra
from services import visualization as vis
from services.presenter import _compute_film_emulation_target


# ============================================================================
# Shared fixtures: noisy spectra with NaNs, mimicking real interpolated data
# ============================================================================

N = 801


def _noisy_spectrum(seed, n=N, nan_rate=0.04):
    rng = np.random.RandomState(seed)
    values = rng.rand(n)
    values[rng.rand(n) < nan_rate] = np.nan
    return values


TRANSMISSION = _noisy_spectrum(7)
ILLUMINANT = 0.4 + 0.6 * np.abs(np.sin(np.linspace(0.0, 3.1, N)))
QE = {ch: _noisy_spectrum(seed) for ch, seed in (('R', 11), ('G', 12), ('B', 13))}
REFLECTOR = _noisy_spectrum(17, nan_rate=0.02)

# Deterministic small arrays for the presenter film-target tests
FILL_ILLUMINANT = np.linspace(1.0, 0.6, 64)
FILM_CHANNELS = {
    'R': np.linspace(0.9, 0.1, 64),
    'G': np.linspace(0.5, 0.8, 64),
    'B': np.linspace(0.2, 0.4, 64),
}
FILM_CHANNELS['B'][7] = np.nan  # NaN must propagate through the chain


class _FakeFilmProfile:
    """Stand-in for SensorProfile: channels + display metadata."""

    def __init__(self, channels, display_name='Portra 400', sensor_type='color'):
        self._channels = channels
        self.display_name = display_name
        self.sensor_type = sensor_type

    def get_channels(self):
        return self._channels


def _film_state(wb, profile, illuminant=FILL_ILLUMINANT, filters=()):
    return SimpleNamespace(
        film_emulation_profile=profile,
        illuminant=illuminant,
        film_emulation_filters=list(filters),
        film_emulation_wb=wb,
        film_emulation_wb_gains=None,
    )


_FILM_COLLECTION = SimpleNamespace(filter_matrix={})


# ============================================================================
# Reference implementations (the chain semantics S2 locked in)
# ============================================================================

def reference_combine(transmission_values, combine=True):
    """Stack multiplication: NaN-aware product over the stack."""
    if not transmission_values:
        return np.ones_like(TRANSMISSION)
    if combine and len(transmission_values) > 1:
        stack = np.array(transmission_values)
        combined = np.nanprod(stack, axis=0)
        combined[np.any(np.isnan(stack), axis=0)] = np.nan
        return combined
    return transmission_values[0]


def reference_wb_gains(transmission, quantum_efficiency, illuminant, reflector=None):
    """Mask -> per-channel product chain -> green-referenced normalization."""
    channel_responses = {}
    for channel in ('R', 'G', 'B'):
        sensitivity = quantum_efficiency.get(channel)
        if sensitivity is None:
            channel_responses[channel] = np.nan
            continue
        terms = [
            np.asarray(transmission, dtype=float),
            np.asarray(sensitivity, dtype=float),
            np.asarray(illuminant, dtype=float),
        ]
        if reflector is not None:
            terms.append(np.asarray(reflector, dtype=float))
        valid = ~np.isnan(terms[0])
        for term in terms[1:]:
            valid &= ~np.isnan(term)
        if not valid.any():
            channel_responses[channel] = np.nan
            continue
        product = terms[0][valid]
        for term in terms[1:]:
            product = product * term[valid]
        channel_responses[channel] = float(np.sum(product))

    g_response = channel_responses.get('G', np.nan)
    if not np.isnan(g_response) and g_response > EPSILON:
        return {
            channel: (
                channel_responses.get(channel, 0.0) / g_response
                if not np.isnan(channel_responses.get(channel, 0.0))
                else 1.0
            )
            for channel in ('R', 'G', 'B')
        }
    return DEFAULT_WB_GAINS.copy()


def assert_nan_equal(testcase, actual, expected):
    """Exact (bitwise) array equality, treating NaN == NaN as equal."""
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


class TestFilterTransmission(unittest.TestCase):
    """compute_filter_transmission: combine + clip, per-path behavior."""

    def test_multi_filter_combined_and_clipped(self):
        t1 = np.clip(_noisy_spectrum(21), 0.0, 1.0)
        t2 = np.clip(_noisy_spectrum(22), 0.0, 1.0)
        t3 = np.clip(_noisy_spectrum(23), 0.0, 1.0)
        matrix = np.vstack([t1, t2, t3])
        result, label, combined = calc.compute_filter_transmission([0, 1, 2], matrix)
        expected = np.clip(reference_combine([t1, t2, t3]), EPSILON, 1.0)
        assert_nan_equal(self, result, expected)
        self.assertIs(combined, result)
        self.assertEqual(label, "Combined")

    def test_multi_filter_nan_propagates(self):
        t1 = np.clip(_noisy_spectrum(21), 0.0, 1.0)
        t2 = np.clip(_noisy_spectrum(22), 0.0, 1.0)
        matrix = np.vstack([t1, t2])
        result, _, _ = calc.compute_filter_transmission([0, 1], matrix)
        # A NaN in any operand makes that wavelength NaN
        assert_nan_equal(
            self, np.isnan(result), np.isnan(t1) | np.isnan(t2)
        )

    def test_single_filter_returned_unclipped(self):
        t1 = _noisy_spectrum(21)  # raw values, no clip expected
        matrix = np.vstack([t1, np.ones(N)])
        result, label, combined = calc.compute_filter_transmission([0], matrix)
        assert_nan_equal(self, result, t1)
        self.assertEqual(label, "Single")
        self.assertIsNone(combined)

    def test_no_filter_returns_identity(self):
        result, label, combined = calc.compute_filter_transmission([], np.zeros((1, N)))
        assert_nan_equal(self, result, np.ones(N))
        self.assertEqual(label, "No Filter")
        self.assertIsNone(combined)

    def test_combine_transmission_empty_and_single(self):
        assert_nan_equal(self, spectra.combine_transmission([]), np.ones(N))
        single = _noisy_spectrum(5)
        self.assertIs(spectra.combine_transmission([single]), single)


class TestWhiteBalanceGains(unittest.TestCase):
    """The white-balance path: plain, surface, and degenerate inputs."""

    def test_plain_matches_reference_bitwise(self):
        gains = calc.compute_white_balance_gains(TRANSMISSION, QE, ILLUMINANT)
        self.assertEqual(gains, reference_wb_gains(TRANSMISSION, QE, ILLUMINANT))

    def test_surface_matches_reference_bitwise(self):
        gains = calc.compute_white_balance_gains_from_surface(
            REFLECTOR, TRANSMISSION, QE, ILLUMINANT
        )
        self.assertEqual(
            gains,
            reference_wb_gains(TRANSMISSION, QE, ILLUMINANT, reflector=REFLECTOR),
        )

    def test_invalid_transmission_returns_defaults(self):
        all_nan = np.full(N, np.nan)
        self.assertEqual(
            calc.compute_white_balance_gains(all_nan, QE, ILLUMINANT),
            dict(DEFAULT_WB_GAINS),
        )
        self.assertEqual(
            calc.compute_white_balance_gains(None, QE, ILLUMINANT),
            dict(DEFAULT_WB_GAINS),
        )

    def test_missing_channel_matches_reference(self):
        partial_qe = {'R': QE['R'], 'G': QE['G']}
        gains = calc.compute_white_balance_gains(TRANSMISSION, partial_qe, ILLUMINANT)
        self.assertEqual(
            gains, reference_wb_gains(TRANSMISSION, partial_qe, ILLUMINANT)
        )

    def test_dead_green_returns_defaults(self):
        zero_green = {'R': QE['R'], 'G': np.zeros(N), 'B': QE['B']}
        gains = calc.compute_white_balance_gains(TRANSMISSION, zero_green, ILLUMINANT)
        self.assertEqual(gains, dict(DEFAULT_WB_GAINS))

    def test_gain_relative_to_green(self):
        clean = {  # NaN-free for a hand-checkable ratio
            'R': np.full(N, 0.5),
            'G': np.full(N, 0.25),
            'B': np.full(N, 0.5),
        }
        gains = calc.compute_white_balance_gains(np.ones(N), clean, np.ones(N))
        self.assertEqual(gains, {'R': 2.0, 'G': 1.0, 'B': 2.0})


class TestRgbResponse(unittest.TestCase):
    """compute_rgb_response: percent curves, gain flooring, visibility."""

    def test_visible_channels_match_formula(self):
        gains = {'R': 1.0, 'G': 2.0, 'B': 1.5}
        visible = {'R': True, 'G': True, 'B': True}
        responses, rgb_matrix, max_response = calc.compute_rgb_response(
            TRANSMISSION, QE, gains, visible
        )
        expected_max = 0.0
        for channel in ('R', 'G', 'B'):
            gain = max(gains.get(channel, 1.0), EPSILON)
            expected = np.nan_to_num(TRANSMISSION * QE[channel]) / gain * 100
            assert_nan_equal(self, responses[channel], expected)
            expected_max = max(expected_max, np.nanmax(expected))
        self.assertEqual(max_response, expected_max)

    def test_hidden_channel_zeroed(self):
        gains = {'R': 1.0, 'G': 1.0, 'B': 1.0}
        visible = {'R': True, 'G': False, 'B': True}
        responses, _, _ = calc.compute_rgb_response(
            TRANSMISSION, QE, gains, visible
        )
        assert_nan_equal(self, responses['G'], np.zeros(N))
        self.assertFalse(np.all(responses['R'] == 0.0))

    def test_invalid_transmission_returns_zeros(self):
        all_nan = np.full(N, np.nan)
        responses, rgb_matrix, max_response = calc.compute_rgb_response(
            all_nan, QE, {'R': 1.0, 'G': 1.0, 'B': 1.0},
            {'R': True, 'G': True, 'B': True}
        )
        self.assertEqual(max_response, 0.0)
        for channel in ('R', 'G', 'B'):
            assert_nan_equal(self, responses[channel], np.zeros(N))
        self.assertEqual(rgb_matrix.shape, (N, 3))


class TestSpectraCurves(unittest.TestCase):
    """The per-wavelength chain and the integral over it."""

    def test_sensor_channel_curves_matches_film_formula(self):
        curves = spectra.sensor_channel_curves(
            TRANSMISSION, QE, ILLUMINANT, display_scale=100
        )
        for channel, sensitivity in QE.items():
            # The exact expression the film block used before S2b
            assert_nan_equal(
                self, curves[channel], TRANSMISSION * sensitivity * ILLUMINANT * 100
            )

    def test_sensor_channel_curves_without_illuminant(self):
        curves = spectra.sensor_channel_curves(TRANSMISSION, QE)
        for channel, sensitivity in QE.items():
            assert_nan_equal(self, curves[channel], TRANSMISSION * sensitivity)

    def test_nan_propagates_elementwise(self):
        curves = spectra.sensor_channel_curves(
            TRANSMISSION, QE, ILLUMINANT, display_scale=100
        )
        any_nan = np.isnan(TRANSMISSION) | np.isnan(ILLUMINANT)
        for channel, sensitivity in QE.items():
            expected_nan = any_nan | np.isnan(sensitivity)
            assert_nan_equal(self, np.isnan(curves[channel]), expected_nan)

    def test_integrated_response_sums_valid_wavelengths(self):
        value = spectra.integrated_response(TRANSMISSION, QE['R'], ILLUMINANT)
        valid = ~np.isnan(TRANSMISSION) & ~np.isnan(QE['R']) & ~np.isnan(ILLUMINANT)
        product = TRANSMISSION[valid]
        product = product * QE['R'][valid]
        product = product * ILLUMINANT[valid]
        self.assertEqual(value, float(np.sum(product)))

    def test_integrated_response_toy_example(self):
        ones = np.ones(3)
        qe = np.array([1.0, np.nan, 1.0])
        self.assertEqual(spectra.integrated_response(ones, qe, ones), 2.0)

    def test_integrated_response_all_nan_returns_nan(self):
        all_nan = np.full(N, np.nan)
        self.assertTrue(
            math.isnan(spectra.integrated_response(all_nan, QE['G'], ILLUMINANT))
        )

    def test_integrated_response_with_reflector(self):
        value = spectra.integrated_response(
            TRANSMISSION, QE['R'], ILLUMINANT, reflector=REFLECTOR
        )
        valid = (
            ~np.isnan(TRANSMISSION) & ~np.isnan(QE['R'])
            & ~np.isnan(ILLUMINANT) & ~np.isnan(REFLECTOR)
        )
        product = TRANSMISSION[valid]
        product = product * QE['R'][valid]
        product = product * ILLUMINANT[valid]
        product = product * REFLECTOR[valid]
        self.assertEqual(value, float(np.sum(product)))


class TestSpectraHelpers(unittest.TestCase):
    """Normalization, film inversion, and validity checks."""

    def test_normalize_wb_gains_relative_to_green(self):
        normalized = spectra.normalize_wb_gains({'R': 4.0, 'G': 2.0, 'B': 1.0})
        self.assertEqual(normalized, {'R': 2.0, 'G': 1.0, 'B': 0.5})

    def test_normalize_wb_gains_nan_green_returns_defaults(self):
        self.assertEqual(
            spectra.normalize_wb_gains({'R': 4.0, 'G': np.nan, 'B': 1.0}),
            dict(DEFAULT_WB_GAINS),
        )

    def test_visualization_delegates_inversion(self):
        self.assertIs(vis.invert_for_display, spectra.invert_for_display)

    def test_invert_for_display_peak_relative(self):
        responses = {'R': np.array([1.0, 2.0]), 'G': np.array([3.0, 1.0])}
        inverted = spectra.invert_for_display(responses)
        assert_nan_equal(self, inverted['R'], np.array([2.0, 1.0]))
        assert_nan_equal(self, inverted['G'], np.array([0.0, 2.0]))

    def test_is_valid_spectrum(self):
        self.assertTrue(spectra.is_valid_spectrum(np.ones(4)))
        self.assertFalse(spectra.is_valid_spectrum(np.full(4, np.nan)))
        self.assertFalse(spectra.is_valid_spectrum(None))
        self.assertFalse(spectra.is_valid_spectrum(np.array([])))


class TestFilmEmulationTarget(unittest.TestCase):
    """S2b: the presenter film-overlay path, now a unit-testable helper.

    Each expected value re-derives the original inline formula
    (film_filter * sens * illuminant * 100, then guarded WB division).
    """

    def test_no_profile_returns_defaults(self):
        state = _film_state(wb=True, profile=None)
        self.assertEqual(
            _compute_film_emulation_target(state, _FILM_COLLECTION),
            (None, "Film Target", False),
        )

    def test_no_illuminant_returns_defaults(self):
        state = _film_state(wb=True, profile=_FakeFilmProfile(FILM_CHANNELS),
                            illuminant=None)
        self.assertEqual(
            _compute_film_emulation_target(state, _FILM_COLLECTION),
            (None, "Film Target", False),
        )

    def test_without_wb_matches_original_formula(self):
        state = _film_state(wb=False, profile=_FakeFilmProfile(FILM_CHANNELS))
        curves, label, is_bw = _compute_film_emulation_target(state, _FILM_COLLECTION)
        ones = np.ones_like(FILL_ILLUMINANT)
        for channel, sensitivity in FILM_CHANNELS.items():
            assert_nan_equal(self, curves[channel], ones * sensitivity * FILL_ILLUMINANT * 100)
        self.assertEqual(label, "Portra 400")
        self.assertFalse(is_bw)
        self.assertIsNone(state.film_emulation_wb_gains)

    def test_with_wb_matches_original_block(self):
        state = _film_state(wb=True, profile=_FakeFilmProfile(FILM_CHANNELS))
        curves, _, _ = _compute_film_emulation_target(state, _FILM_COLLECTION)

        # Original inline computation, reproduced here as the oracle
        ones = np.ones_like(FILL_ILLUMINANT)
        expected = {
            ch: ones * sens * FILL_ILLUMINANT * 100
            for ch, sens in FILM_CHANNELS.items()
        }
        gains = calc.compute_white_balance_gains(ones, FILM_CHANNELS, FILL_ILLUMINANT)
        for channel in expected:
            gain = gains.get(channel, 1.0)
            expected[channel] = (
                expected[channel] / gain if gain > 0 else expected[channel]
            )

        for channel in expected:
            assert_nan_equal(self, curves[channel], expected[channel])
        self.assertEqual(state.film_emulation_wb_gains, gains)

    def test_bw_flag_from_profile(self):
        profile = _FakeFilmProfile(FILM_CHANNELS, display_name='HP5', sensor_type='bw')
        state = _film_state(wb=False, profile=profile)
        _, label, is_bw = _compute_film_emulation_target(state, _FILM_COLLECTION)
        self.assertEqual(label, 'HP5')
        self.assertTrue(is_bw)

    def test_nonpositive_gain_skips_division(self):
        # Negative sensitivities produce negative gains; the guarded division
        # must leave those curves untouched.
        negative_channels = {
            'R': np.linspace(-0.9, -0.1, 64),
            'G': np.linspace(0.5, 0.8, 64),
            'B': np.linspace(-0.2, -0.4, 64),
        }
        state = _film_state(wb=True, profile=_FakeFilmProfile(negative_channels))
        curves, _, _ = _compute_film_emulation_target(state, _FILM_COLLECTION)

        ones = np.ones_like(FILL_ILLUMINANT)
        gains = calc.compute_white_balance_gains(ones, negative_channels, FILL_ILLUMINANT)
        self.assertLessEqual(gains['R'], 0.0)
        self.assertLessEqual(gains['B'], 0.0)
        assert_nan_equal(self, curves['R'], ones * negative_channels['R'] * FILL_ILLUMINANT * 100)
        assert_nan_equal(self, curves['B'], ones * negative_channels['B'] * FILL_ILLUMINANT * 100)
        self.assertGreater(gains['G'], 0.0)
        assert_nan_equal(
            self, curves['G'],
            ones * negative_channels['G'] * FILL_ILLUMINANT * 100 / gains['G'],
        )


class TestPublicSignatures(unittest.TestCase):
    """The public API S2 promised to preserve, locked by parameter list."""

    EXPECTED = {
        calc.compute_filter_transmission: ['filter_indices', 'filter_matrix'],
        calc.compute_active_transmission: [
            'selected_filters', 'selected_indices', 'filter_matrix'],
        calc.compute_selected_filter_indices: [
            'selected_filters', 'filter_multipliers', 'filter_collection'],
        calc.compute_effective_stops: ['transmission', 'sensor_qe', 'illuminant'],
        calc.compute_rgb_response: [
            'transmission', 'quantum_efficiency', 'white_balance_gains',
            'visible_channels', 'channel_mixer'],
        calc.compute_white_balance_gains: [
            'transmission', 'quantum_efficiency', 'illuminant'],
        calc.compute_white_balance_gains_from_surface: [
            'reflector', 'transmission', 'quantum_efficiency', 'illuminant'],
        spectra.combine_transmission: ['transmission_values', 'combine'],
        spectra.sensor_channel_curves: [
            'transmission', 'sensitivities', 'illuminant', 'display_scale'],
        spectra.integrated_response: [
            'transmission', 'sensitivity', 'illuminant', 'reflector'],
        spectra.normalize_wb_gains: ['channel_responses'],
        spectra.wb_gains: ['transmission', 'quantum_efficiency', 'illuminant', 'reflector'],
        spectra.invert_for_display: ['responses'],
        spectra.is_valid_spectrum: ['spectrum'],
    }

    def test_signatures_preserved(self):
        for function, expected in self.EXPECTED.items():
            with self.subTest(function=function.__name__):
                self.assertEqual(list(inspect.signature(function).parameters), expected)


class TestConsumerChain(unittest.TestCase):
    """The refactored layers still import as a whole."""

    def test_services_layers_import(self):
        import services.app_operations  # noqa: F401
        import services.presenter  # noqa: F401
        import services.visualization  # noqa: F401
        self.assertTrue(callable(services.presenter.prepare_sensor_analysis))
        self.assertTrue(callable(services.presenter._compute_film_emulation_target))


if __name__ == '__main__':
    unittest.main()