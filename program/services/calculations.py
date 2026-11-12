"""
Mathematical calculations for FS FilterLab.

Stack/metrics/surface math on top of the pure spectral core
(:mod:`services.spectra`) plus color identity helpers.  Presentation-only
string formatting lives in the presenter layer.
"""
# Third-party imports
import numpy as np
from typing import Dict, List, Tuple, Optional

# Local imports
from models.constants import DEFAULT_HEX_COLOR, EPSILON, INTERP_GRID
from models.core import FilterCollection, ChannelMixerSettings
from services.spectra import (
    apply_channel_mixing_to_colors, combine_transmission,
    integrated_response, is_valid_spectrum
)


# ============================================================================
# TRANSMISSION CALCULATIONS
# ============================================================================

def compute_filter_transmission(
    filter_indices: List[int],
    filter_matrix: np.ndarray
) -> Tuple[np.ndarray, str]:
    """
    Active stack curve and its display label ("No Filter"/"Single"/"Combined").

    The array is always the curve downstream code should use: identity for an
    empty stack, the single filter row, or the EPSILON-clipped combination.
    """
    if not filter_indices:
        return np.ones_like(INTERP_GRID), "No Filter"

    if len(filter_indices) > 1:
        transmissions = [filter_matrix[idx] for idx in filter_indices]
        combined = combine_transmission(transmissions)
        combined = np.clip(combined, EPSILON, 1.0)
        return combined, "Combined"

    return filter_matrix[filter_indices[0]], "Single"


def effective_transmission(
    selected_filters: List[str],
    filter_multipliers: Dict[str, int],
    filter_collection: FilterCollection
) -> np.ndarray:
    """
    Current stack transmission on the interpolation grid (ones if empty).

    Single shared derivation of the live stack for the action layer and for
    every presenter that needs transmission (surfaces, reflectors shelf),
    so presenters stay read-only instead of trusting a render-time cache.

    Args:
        selected_filters: List of selected filter display names
        filter_multipliers: Dictionary mapping filter names to multiplier counts
        filter_collection: Collection of available filters

    Returns:
        Combined transmission array over INTERP_GRID
    """
    indices = compute_selected_filter_indices(
        selected_filters, filter_multipliers, filter_collection
    )
    if indices and filter_collection.filter_matrix is not None:
        transmissions = [filter_collection.filter_matrix[idx] for idx in indices]
        return combine_transmission(transmissions)

    return np.ones_like(INTERP_GRID)


def compute_selected_filter_indices(
    selected_filters: List[str],
    filter_multipliers: Dict[str, int],
    filter_collection: FilterCollection
) -> List[int]:
    """Filter matrix indices for the selected display names, repeated by multiplier."""
    if not selected_filters:
        return []
    
    display_to_index = filter_collection.get_display_to_index_map()
    selected_indices = []
    
    for name in selected_filters:
        if name not in display_to_index:
            continue
            
        idx = display_to_index[name]
        count = filter_multipliers.get(name, 1)
        selected_indices.extend([idx] * count)
        
    return selected_indices


# ============================================================================
# METRICS CALCULATIONS
# ============================================================================

def compute_effective_stops(
    transmission: np.ndarray, 
    sensor_qe: np.ndarray,
    illuminant: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Compute effective stops from transmission, sensor QE, and illuminant.
    
    Args:
        transmission: Transmission values (0-1 fractional scale)
        sensor_qe: Sensor quantum efficiency values (0-1 fractional scale)
        illuminant: Illuminant spectrum (arbitrary units, optional)
    
    Returns:
        Tuple of (avg_transmission, effective_stops)
    """
    # Ensure inputs are numpy arrays
    transmission = np.asarray(transmission)
    sensor_qe = np.asarray(sensor_qe)
    
    # Default to uniform illuminant if not provided
    if illuminant is None:
        illuminant = np.ones_like(transmission)
    else:
        illuminant = np.asarray(illuminant)
    
    # Find valid indices where none are NaN
    valid = (~np.isnan(transmission) & ~np.isnan(sensor_qe) & ~np.isnan(illuminant))
    
    # If no valid data, return NaNs immediately
    if not np.any(valid):
        return np.nan, np.nan
    
    clipped_trans = np.clip(transmission[valid], EPSILON, 1.0)
    clipped_qe = sensor_qe[valid]
    clipped_illuminant = illuminant[valid]
    
    # Weight by actual photon flux (illuminant * QE)
    photometric_weights = clipped_illuminant * clipped_qe
    
    # If all weights are zero, cannot compute weighted average
    if np.all(photometric_weights == 0):
        return np.nan, np.nan
    
    # Weighted average transmission by photon flux
    avg_trans = np.average(clipped_trans, weights=photometric_weights)
    
    # Prevent log2 of zero or negative (should be prevented by clipping but be safe)
    if avg_trans <= 0:
        return np.nan, np.nan
    
    effective_stops = -np.log2(avg_trans)
    
    return avg_trans, effective_stops



# ============================================================================
# REFLECTOR CALCULATIONS
# ============================================================================

def compute_reflector_color(
    reflector: np.ndarray,
    transmission: np.ndarray,
    quantum_efficiency: Dict[str, np.ndarray],
    illuminant: np.ndarray,
    channel_mixer: Optional[ChannelMixerSettings] = None,
    white_balance_gains: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Linear sRGB [R, G, B] of a reflector under the current conditions:
    per-channel integration (reflector x transmission x QE x illuminant),
    division by white balance gains, then optional channel mixing.
    """
    # Early exit for invalid data
    if not is_valid_spectrum(transmission) or reflector is None:
        return np.zeros(3)

    # Per channel: integrate reflector x transmission x QE x illuminant
    # (NaN points excluded), then divide by the white balance gain.
    rgb_values = np.zeros(3)
    for i, ch in enumerate(['R', 'G', 'B']):
        qe_curve = quantum_efficiency.get(ch)
        if qe_curve is None:
            continue

        response = integrated_response(transmission, qe_curve, illuminant, reflector)
        channel_response = 0.0 if np.isnan(response) else response

        wb_gain = white_balance_gains.get(ch, 1.0) if white_balance_gains is not None else 1.0
        if wb_gain > EPSILON:
            rgb_values[i] = channel_response / wb_gain
        else:
            rgb_values[i] = channel_response

    # Apply channel mixing if enabled
    if channel_mixer is not None and channel_mixer.enabled:
        rgb_values = apply_channel_mixing_to_colors(rgb_values, channel_mixer)

    return rgb_values



# ============================================================================
# EXPOSURE-REFERENCED SURFACE COLORS
# ============================================================================

def compute_white_reference_color(
    transmission: np.ndarray,
    quantum_efficiency: Dict[str, np.ndarray],
    illuminant: np.ndarray,
    white_balance_gains: Optional[Dict[str, float]] = None,
    channel_mixer: Optional[ChannelMixerSettings] = None
) -> np.ndarray:
    """
    Sensor response to a perfect diffuse reflector (reflectance = 1.0 at every
    wavelength) under the current stack x illuminant x sensor x WB (+ mixer).

    This is the shared white anchor for surface color previews: the exposure
    is defined so a perfect white just saturates.  Physically, a surface's
    display color is its response divided by this anchor.

    Returns:
        RGB color as numpy array [R, G, B] (unnormalized integrated response)
    """
    ones = np.ones_like(INTERP_GRID)
    return compute_reflector_color(
        ones, transmission, quantum_efficiency, illuminant,
        channel_mixer=channel_mixer, white_balance_gains=white_balance_gains,
    )


def reference_color_to_display(
    surface_rgb: np.ndarray, white_ref_rgb: np.ndarray
) -> np.ndarray:
    """
    White-referenced linear display color: ``clip(surface / white_ref, 0, 1)``.

    A surface whose response equals the white anchor renders exactly white; a
    near-black surface stays near-black (fixing the per-channel auto-exposure
    bug that amplified every swatch's brightest channel to 1.0).

    Returns:
        Linear RGB in [0, 1] per channel.
    """
    surface = np.asarray(surface_rgb, dtype=float)
    white = np.asarray(white_ref_rgb, dtype=float)
    safe_white = np.where(white > EPSILON, white, 1.0)
    return np.clip(surface / safe_white, 0.0, 1.0)


# ============================================================================
# CAMERA-LOOK RENDER MODE (filmic tone curve + chroma, pre-sRGB encoding)
# ============================================================================

# Chroma boost applied around Rec.709 luma before the tone curve.
CAMERA_LOOK_CHROMA_BOOST = 1.15


def apply_camera_look(linear_rgb: np.ndarray, invert: bool = False) -> np.ndarray:
    """
    "Camera look" render of a white-referenced linear color: optional
    negative-film inversion (``1 - x`` for ``invert_for_display`` profiles),
    a chroma boost around luma, then a smoothstep filmic S-curve per channel.

    Post-ratio / pre-sRGB: input and output are linear RGB in [0, 1].  The
    S-curve maps 0, 0.5 and 1 onto themselves, so neutrals keep their tone
    while midtone contrast increases — the default-on Scene Preview look.
    """
    rgb = np.clip(np.asarray(linear_rgb, dtype=float), 0.0, 1.0)
    if invert:
        rgb = 1.0 - rgb
    luma = float(np.dot(rgb, (0.2126, 0.7152, 0.0722)))
    rgb = luma + (rgb - luma) * CAMERA_LOOK_CHROMA_BOOST
    return np.clip(rgb * rgb * (3.0 - 2.0 * rgb), 0.0, 1.0)


def srgb_encode(linear_rgb: np.ndarray) -> np.ndarray:
    """
    IEC 61966-2-1 sRGB transfer function (vectorized), linear [0,1] in,
    encoded [0,1] out.  Browsers assume sRGB — without this encoding
    white-referenced midtones render too dark.
    """
    lin = np.clip(np.asarray(linear_rgb, dtype=float), 0.0, 1.0)
    encoded = np.where(
        lin <= 0.0031308,
        lin * 12.92,
        1.055 * np.power(lin, 1.0 / 2.4) - 0.055,
    )
    # Exactly white stays exactly white (1.055 - 0.055 loses a ULP in float).
    return np.where(lin >= 1.0, 1.0, encoded)


def srgb_to_hex(linear_rgb: np.ndarray) -> str:
    """sRGB-encode linear RGB and format it as a lowercase '#rrggbb' string."""
    encoded = srgb_encode(linear_rgb)
    return "#{:02x}{:02x}{:02x}".format(
        *(int(round(float(c) * 255)) for c in encoded)
    )


# ============================================================================
# FILTER IDENTITY COLORS (spectrum-derived swatch fallback)
# ============================================================================

def _is_valid_hex_color(text: str) -> bool:
    """True for exact '#RRGGBB' strings (6 hex digits)."""
    if not isinstance(text, str) or not text.startswith("#") or len(text) != 7:
        return False
    try:
        int(text[1:], 16)
        return True
    except ValueError:
        return False


def _cmf_lobe(
    wavelengths: np.ndarray, mu: float, sigma_low: float, sigma_high: float
) -> np.ndarray:
    """Piecewise Gaussian lobe: sigma_low below mu, sigma_high above."""
    sigma = np.where(wavelengths < mu, sigma_low, sigma_high)
    return np.exp(-((wavelengths - mu) ** 2) / (2.0 * sigma ** 2))


# CIE-1931 XYZ -> linear sRGB conversion matrix (D65 reference white)
_XYZ_TO_LINEAR_SRGB = np.array([
    [3.2406255, -1.5372080, -0.4986286],
    [-0.9689307, 1.8757561, 0.0415175],
    [0.0557101, -0.2040211, 1.0569959],
])


def _cie1931_xyz(wavelengths: np.ndarray) -> np.ndarray:
    """
    Analytic CIE-1931 2-deg color matching functions at single wavelengths
    (multi-lobe Gaussian fit; Wyman, Sloan & Shirley, JCGT 2013) as an
    (..., 3) array of (X, Y, Z).
    """
    wl = np.asarray(wavelengths, dtype=float)
    x_bar = (
        1.056 * _cmf_lobe(wl, 599.8, 37.9, 31.0)
        + 0.362 * _cmf_lobe(wl, 442.0, 16.0, 26.7)
        - 0.065 * _cmf_lobe(wl, 501.1, 20.4, 26.2)
    )
    y_bar = (
        0.821 * _cmf_lobe(wl, 568.8, 46.9, 40.5)
        + 0.286 * _cmf_lobe(wl, 530.9, 16.3, 31.1)
    )
    z_bar = (
        1.217 * _cmf_lobe(wl, 437.0, 11.8, 36.0)
        + 0.681 * _cmf_lobe(wl, 459.0, 26.0, 13.8)
    )
    return np.stack([x_bar, y_bar, z_bar], axis=-1)


def _spectral_to_linear_srgb(
    transmission: np.ndarray, wavelengths: np.ndarray
) -> np.ndarray:
    """
    XYZ of a transmission curve under equal-energy white via _cie1931_xyz,
    converted to linear sRGB.  Unnormalized output.
    """
    t = np.asarray(transmission, dtype=float)
    xyz = np.trapezoid(t[:, np.newaxis] * _cie1931_xyz(wavelengths), wavelengths, axis=0)
    return _XYZ_TO_LINEAR_SRGB @ xyz


def filter_display_color(transmission: Optional[np.ndarray], hex_color: str = "") -> str:
    """
    Identity color for a filter: metadata hex when valid, spectrum otherwise.

    Swatches and curve strokes trust the file's 'Hex Color' metadata when it is
    a valid '#RRGGBB' — except DEFAULT_HEX_COLOR, which is the loader's
    "no color data" sentinel, indistinguishable from real metadata once
    substituted.  Many manufacturers ship no color metadata at all, so the
    fallback derives the color from the visible-band transmission (CIE-1931
    observer, equal-energy white, peak-normalized, sRGB-encoded) — a vividly
    blue filter previews blue instead of the neutral-gray default.  Spectra
    with effectively no visible light (UV/IR-only, all-zero), and missing or
    wrong-sized curves, fall back to DEFAULT_HEX_COLOR.
    """
    text = str(hex_color or "").strip()
    if _is_valid_hex_color(text) and text.upper() != DEFAULT_HEX_COLOR.upper():
        return text
    if transmission is None:
        return DEFAULT_HEX_COLOR
    t = np.asarray(transmission, dtype=float)
    if t.shape != INTERP_GRID.shape or not np.any(np.isfinite(t)):
        return DEFAULT_HEX_COLOR
    t = np.where(np.isfinite(t), t, 0.0)
    visible = (INTERP_GRID >= 380) & (INTERP_GRID <= 730)
    linear = np.clip(_spectral_to_linear_srgb(t[visible], INTERP_GRID[visible]), 0.0, None)
    peak = float(linear.max())
    if peak <= EPSILON:
        return DEFAULT_HEX_COLOR
    return srgb_to_hex(linear / peak)


# ============================================================================
# HUMAN-VISION SPECTRUM COLORS (single wavelengths → perceived color)
# ============================================================================

def spectral_locus_linear_srgb(wavelengths: np.ndarray) -> np.ndarray:
    """
    Linear sRGB color of individual wavelengths (same CIE-1931 2-deg observer
    as _spectral_to_linear_srgb), out-of-gamut negatives clipped to 0.

    Magnitude carries the observer's ȳ(λ).  For an explicit V(λ) brightness
    profile, normalize the locus to its own peak first (hue/chroma), then
    scale by photopic_relative_brightness.
    """
    return np.clip(_cie1931_xyz(wavelengths) @ _XYZ_TO_LINEAR_SRGB.T, 0.0, None)


def photopic_relative_brightness(wavelengths: np.ndarray) -> np.ndarray:
    """
    Photopic brightness ȳ(λ) of the same fit, normalized to its maximum over
    the passed range — pass the full visible band so 1.0 lands on the
    brightest wavelength (~555 nm).
    """
    y_bar = _cie1931_xyz(wavelengths)[..., 1]
    return y_bar / y_bar.max()
