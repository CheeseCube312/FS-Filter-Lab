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
Film differs only in display (negative inversion, B&W luminance collapse),
which stays in the presentation layer — see ``invert_for_display()``.

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
    """
    Check whether a spectrum array is usable for computation.

    Args:
        spectrum: Spectrum values to check (any array-like)

    Returns:
        True if the spectrum is a non-empty array with finite entries
    """
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

def combine_transmission(
    transmission_values: Sequence[np.ndarray],
    combine: bool = True
) -> np.ndarray:
    """
    Multiply a filter stack into one transmission spectrum.

    All transmission values are expected to be in fractional scale (0-1).
    Filter combination uses natural multiplication which preserves this scale.

    Args:
        transmission_values: Sequence of transmission arrays (each 0-1 scale)
        combine: Whether to combine the transmissions (multiply them)

    Returns:
        Combined transmission in fractional scale (0-1);
        the identity spectrum (ones) for an empty stack
    """
    if not transmission_values:
        return np.ones_like(INTERP_GRID)

    if combine and len(transmission_values) > 1:
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
    Per-wavelength product of the chain: transmission x sensitivity [x illuminant].

    This is the response *curve* of each capture channel under the current
    filter stack — what the response chart and the film-emulation overlay
    plot.  A NaN in any operand makes that wavelength NaN in the curve.

    Args:
        transmission: Combined filter transmission (0-1 fractional scale)
        sensitivities: Per-channel sensitivity/QE curves {'R': ..., 'G': ..., 'B': ...}
        illuminant: Optional illuminant spectrum; omitted for raw capture response
        display_scale: Constant factor applied last (e.g. 100 for percent display)

    Returns:
        Dict of response curves keyed like ``sensitivities``
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

    Wavelengths where any operand is NaN are excluded from the integral
    (not zero-filled).  Returns NaN when no wavelength is valid.

    Args:
        transmission: Combined filter transmission (0-1 fractional scale)
        sensitivity: Single-channel sensitivity/QE curve
        illuminant: Illuminant spectrum
        reflector: Optional surface reflectance in the optical path

    Returns:
        Integrated response as a float, or NaN if there is no valid data
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

def normalize_wb_gains(channel_responses: Dict[str, float]) -> Dict[str, float]:
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

    return normalize_wb_gains(channel_responses)


# ============================================================================
# FILM DISPLAY TRANSFORMS
# ============================================================================

def invert_for_display(responses: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Colour-negative film: flip the latent image for display.

    The latent image of a negative stores dye densities, so the visualisation
    applies a peak-relative inversion (peak - response) to render a positive.
    Peak-relative rather than absolute so it works at any display scale.

    Args:
        responses: Per-channel response curves {'R','G','B'}

    Returns:
        Inverted response curves keyed like the input
    """
    max_val = max(
        (np.nanmax(curve) for curve in responses.values() if len(curve) > 0),
        default=1.0
    )
    max_val = max_val if max_val > 0 else 1.0
    return {channel: max_val - curve for channel, curve in responses.items()}

