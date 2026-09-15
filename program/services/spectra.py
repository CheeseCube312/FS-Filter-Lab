"""
spectra.py — the domain core of FS FilterLab.

Every physical entity in the program is a *spectrum* sampled on one shared
wavelength grid (``INTERP_GRID``, 300-1100 nm at 1 nm).  The entire domain
model is a multiplication chain:

    filter stack       transmission spectra               combine_transmission()
    capture response   transmission x sensitivity         sensor_channel_curves()
                       [x illuminant] (per wavelength)
    scalar response    sum of the chain over wavelength   integrated_response()
    white balance      green-referenced normalisation     wb_gains()

Capture media (digital sensor, slide film, colour-negative film, B&W film)
are all the same thing here: a dict of per-channel sensitivity curves.
Film's display inversion stays in the presentation layer.

This module is pure: numpy arithmetic only.  No I/O, no NiceGUI, no app
state.  Both the calculation service and the chart layer consume it, so
there is exactly one implementation of the physics.

NaN policy ("one bad point spoils the point"):
  * Products (``combine_transmission``, ``sensor_channel_curves``): a NaN
    in any operand makes that wavelength NaN.  We never silently zero-fill.
  * Integrals (``integrated_response``): wavelengths where *any* operand is
    NaN are excluded from the sum; remaining points are summed plainly.
  * Validity (``is_valid_spectrum``): an array is usable only if it has at
    least one finite entry; empty or all-NaN inputs are rejected up front.
"""

from typing import Dict, Optional, Sequence

import numpy as np

from models.constants import DEFAULT_WB_GAINS, EPSILON, INTERP_GRID

# Canonical channel keys — every capture profile reports these three keys
# (B&W film repeats one sensitivity curve across all of them).
RGB_CHANNELS = ('R', 'G', 'B')


# ============================================================================
# VALIDITY
# ============================================================================

def is_valid_spectrum(spectrum: Optional[np.ndarray]) -> bool:
    """True when a spectrum is a non-empty array with at least one finite entry."""
    try:
        return (spectrum is not None and
                hasattr(spectrum, '__len__') and
                len(spectrum) > 0 and
                np.any(np.isfinite(spectrum)))
    except (TypeError, ValueError):
        # Handle cases where the input is not array-like or has invalid shape
        return False


# ============================================================================
# THE MULTIPLICATION CHAIN
# ============================================================================

def combine_transmission(transmission_values: Sequence[np.ndarray]) -> np.ndarray:
    """
    Multiply a filter stack into one transmission spectrum, preserving the
    fractional (0-1) scale; the identity spectrum (ones) for an empty stack.
    """
    if not transmission_values:
        return np.ones_like(INTERP_GRID)

    if len(transmission_values) > 1:
        stack = np.array(transmission_values)
        combined = np.nanprod(stack, axis=0)
        combined[np.any(np.isnan(stack), axis=0)] = np.nan
        return combined

    return transmission_values[0]


def sensor_channel_curves(
    transmission: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    illuminant: Optional[np.ndarray] = None,
    display_scale: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Per-wavelength product of the chain: transmission x sensitivity
    [x illuminant] [x display_scale], per channel.  A NaN in any operand
    makes that wavelength NaN in the curve.
    """
    curves = {}
    for channel, sensitivity in sensitivities.items():
        curve = transmission * sensitivity
        if illuminant is not None:
            curve = curve * illuminant
        if display_scale != 1.0:
            curve = curve * display_scale
        curves[channel] = curve
    return curves


def integrated_response(
    transmission: np.ndarray,
    sensitivity: np.ndarray,
    illuminant: np.ndarray,
    reflector: Optional[np.ndarray] = None
) -> float:
    """
    Scalar channel response: the chain summed over fully-valid wavelengths.
    NaN wavelengths are excluded (not zero-filled); NaN when none are valid.
    """
    terms = [
        np.asarray(transmission, dtype=float),
        np.asarray(sensitivity, dtype=float),
        np.asarray(illuminant, dtype=float),
    ]
    if reflector is not None:
        terms.append(np.asarray(reflector, dtype=float))

    # Keep only wavelengths where every operand is valid
    valid = ~np.isnan(terms[0])
    for term in terms[1:]:
        valid &= ~np.isnan(term)
    if not valid.any():
        return np.nan

    product = terms[0][valid]
    for term in terms[1:]:
        product = product * term[valid]
    return float(np.sum(product))


# ============================================================================
# WHITE BALANCE
# ============================================================================

def _normalize_wb_gains(channel_responses: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize integrated channel responses into white-balance gains.

    Green is the reference: each gain is the channel's response relative to
    green, so multiplying a captured image by these gains neutralises the
    current lighting/filter combination.  Channels whose response is NaN
    fall back to unity gain; if green itself is unusable (NaN or ~0), the
    DEFAULT_WB_GAINS (all unity) are returned.

    Args:
        channel_responses: Integrated response per channel {'R','G','B'}

    Returns:
        White-balance gains per channel {'R','G','B'}
    """
    g_response = channel_responses.get('G', np.nan)
    if not np.isnan(g_response) and g_response > EPSILON:
        return {
            channel: (
                channel_responses.get(channel, 0.0) / g_response
                if not np.isnan(channel_responses.get(channel, 0.0))
                else 1.0
            )
            for channel in RGB_CHANNELS
        }

    # Fall back to defaults if we can't normalize
    return DEFAULT_WB_GAINS.copy()


def wb_gains(
    transmission: np.ndarray,
    quantum_efficiency: Dict[str, np.ndarray],
    illuminant: np.ndarray,
    reflector: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    White-balance gains for the current stack over a capture profile.

    Integrates the chain per channel (optionally through a reference
    surface's reflectance, i.e. "white balance on this surface") and
    normalizes against green.

    Args:
        transmission: Combined filter transmission (0-1 fractional scale)
        quantum_efficiency: Per-channel sensitivity curves {'R','G','B'}
        illuminant: Illuminant spectrum
        reflector: Optional reference-surface reflectance spectrum

    Returns:
        White-balance gains per channel {'R','G','B'}
    """
    # Early exit for invalid data
    if not is_valid_spectrum(transmission):
        return DEFAULT_WB_GAINS.copy()

    # Calculate response per channel
    channel_responses = {}
    for channel in RGB_CHANNELS:
        sensitivity = quantum_efficiency.get(channel)
        if sensitivity is None:
            channel_responses[channel] = np.nan
            continue
        channel_responses[channel] = integrated_response(
            transmission, sensitivity, illuminant, reflector
        )

    return _normalize_wb_gains(channel_responses)

