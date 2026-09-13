"""
Mathematical calculations for FS FilterLab.

This module provides all mathematical computation functions for:
- Transmission calculations and filtering
- Color processing and RGB response
- Channel mixing transformations
- Metrics computation and formatting
- White balance calculations
- Deviation analysis
"""
# Third-party imports
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Local imports
from models.constants import EPSILON, INTERP_GRID
from models.core import FilterCollection, ChannelMixerSettings
from services.channel_mixer import apply_channel_mixing_to_responses, apply_channel_mixing_to_colors
from services.spectra import combine_transmission, integrated_response, is_valid_spectrum, wb_gains

# Constants for calculations
MIN_RGB_VALUE = 1/255  # Minimum RGB value to prevent complete black pixels


# ============================================================================
# TRANSMISSION CALCULATIONS
# ============================================================================

def compute_filter_transmission(
    filter_indices: List[int], 
    filter_matrix: np.ndarray
) -> Tuple[np.ndarray, str, Optional[np.ndarray]]:
    """
    Compute filter transmission from filter indices.
    
    Args:
        filter_indices: List of filter indices
        filter_matrix: Matrix of filter transmissions
    
    Returns:
        Tuple of (transmission, label, combined_transmission)
    """
    if not filter_indices:
        return np.ones_like(INTERP_GRID), "No Filter", None
    
    if len(filter_indices) > 1:
        transmissions = [filter_matrix[idx] for idx in filter_indices]
        combined = combine_transmission(transmissions, combine=True)
        combined = np.clip(combined, EPSILON, 1.0)
        return combined, "Combined", combined
    
    transmission = filter_matrix[filter_indices[0]]
    return transmission, "Single", None


def compute_active_transmission(
    selected_filters: List[str],
    selected_indices: List[int],
    filter_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute the active transmission based on selected filters.
    
    Args:
        selected_filters: List of selected filter display names
        selected_indices: List of corresponding filter indices
        filter_matrix: Matrix of filter transmissions
    
    Returns:
        Active transmission array
    """
    if selected_filters and selected_indices and filter_matrix is not None:
        transmissions = [filter_matrix[idx] for idx in selected_indices]
        return combine_transmission(transmissions, combine=True)
    
    return np.ones_like(INTERP_GRID)  # Identity transmission (no filter effect)


def compute_selected_filter_indices(
    selected_filters: List[str],
    filter_multipliers: Dict[str, int],
    filter_collection: FilterCollection
) -> List[int]:
    """
    Compute indices of selected filters with their multipliers.
    
    Args:
        selected_filters: List of selected filter display names
        filter_multipliers: Dictionary mapping filter names to their multiplier counts
        filter_collection: Collection of available filters
    
    Returns:
        List of filter indices
    """
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
    
    # Defensive: Check if arrays are empty before averaging
    if clipped_trans.size == 0 or photometric_weights.size == 0:
        return np.nan, np.nan
    
    # Weighted average transmission by photon flux
    avg_trans = np.average(clipped_trans, weights=photometric_weights)
    
    # Prevent log2 of zero or negative (should be prevented by clipping but be safe)
    if avg_trans <= 0:
        return np.nan, np.nan
    
    effective_stops = -np.log2(avg_trans)
    
    return avg_trans, effective_stops


# ============================================================================
# COLOR PROCESSING AND RGB RESPONSE
# ============================================================================

def compute_rgb_response(
    transmission: np.ndarray,
    quantum_efficiency: Dict[str, np.ndarray],
    white_balance_gains: Dict[str, float],
    visible_channels: Dict[str, bool],
    channel_mixer: Optional[ChannelMixerSettings] = None
) -> Tuple[Dict[str, np.ndarray], np.ndarray, float]:
    """
    Compute RGB response from transmission and quantum efficiency.
    
    Args:
        transmission: Transmission values
        quantum_efficiency: Dictionary of quantum efficiency values by channel
        white_balance_gains: Dictionary of white balance gains by channel
        visible_channels: Dictionary of channel visibility flags
        channel_mixer: Optional channel mixer settings for RGB manipulation
    
    Returns:
        Tuple of (responses_by_channel, rgb_matrix, max_response)
    """
    # Create empty arrays for responses
    responses = {}
    rgb_stack = []
    
    # Check for valid transmission data - early exit if invalid
    if not is_valid_spectrum(transmission) or not quantum_efficiency:
        # Return zero arrays with correct dimensions
        sample_size = len(next(iter(quantum_efficiency.values()))) if quantum_efficiency else len(INTERP_GRID)
        zero_array = np.zeros(sample_size)
        for channel in ['R', 'G', 'B']:
            responses[channel] = zero_array
            rgb_stack.append(zero_array)
        return responses, np.stack(rgb_stack, axis=1) if rgb_stack else np.array([]), 0.0

    # Process each color channel
    max_response = 0.0
    for channel in ['R', 'G', 'B']:
        qe_curve = quantum_efficiency.get(channel)
        
        # Create zero response if no QE data or size mismatch
        if qe_curve is None or len(qe_curve) != len(transmission):
            responses[channel] = np.zeros_like(transmission)
        else:
            # Calculate weighted response with white balance
            gain = max(white_balance_gains.get(channel, 1.0), EPSILON)
            weighted = np.nan_to_num(transmission * qe_curve) / gain * 100
            max_response = max(max_response, np.nanmax(weighted))
            
            # Apply channel visibility
            responses[channel] = weighted if visible_channels.get(channel, True) else np.zeros_like(weighted)
            
        rgb_stack.append(responses[channel])
    
    # Apply channel mixing if enabled
    if channel_mixer is not None and channel_mixer.enabled:
        responses = apply_channel_mixing_to_responses(responses, channel_mixer)
        # Update rgb_stack with mixed responses
        rgb_stack = [responses['R'], responses['G'], responses['B']]

    # Create RGB matrix and normalize
    rgb_matrix = np.stack(rgb_stack, axis=1)
    max_val = np.nanmax(rgb_matrix)
    
    if max_val > 0:
        rgb_matrix = rgb_matrix / max_val
        
    # Clip to valid range
    rgb_matrix = np.clip(rgb_matrix, MIN_RGB_VALUE, 1.0)

    return responses, rgb_matrix, max_response


def compute_white_balance_gains(
    transmission: np.ndarray,
    quantum_efficiency: Dict[str, np.ndarray],
    illuminant: np.ndarray
) -> Dict[str, float]:
    """
    Compute white balance gains from transmission, sensor QE, and illuminant.

    Integrates the spectral chain per channel and normalizes the responses
    against green.  Delegates to :func:`services.spectra.wb_gains` — the
    single implementation of the multiplication model.

    Args:
        transmission: Combined filter transmission values
        quantum_efficiency: Dictionary of quantum efficiency curves by channel
        illuminant: Illuminant curve

    Returns:
        Dictionary of white balance gains by channel
    """
    return wb_gains(transmission, quantum_efficiency, illuminant)


def compute_white_balance_gains_from_surface(
    reflector: np.ndarray,
    transmission: np.ndarray,
    quantum_efficiency: Dict[str, np.ndarray],
    illuminant: np.ndarray
) -> Dict[str, float]:
    """
    Compute white balance gains referenced to a specific surface.

    Simulates in-field white balancing on a reference surface (e.g., foliage
    in IR photography): the surface reflectance participates in the chain
    integration before green normalization.  Delegates to
    :func:`services.spectra.wb_gains` with the optional reflector operand.

    Args:
        reflector: Reflectance spectrum of the reference surface
        transmission: Combined filter transmission values
        quantum_efficiency: Dictionary of quantum efficiency curves by channel
        illuminant: Illuminant curve

    Returns:
        Dictionary of white balance gains by channel
    """
    return wb_gains(transmission, quantum_efficiency, illuminant, reflector=reflector)


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
    Compute reflector color from reflector, transmission, QE, and illuminant.
    
    Args:
        reflector: Reflector spectrum
        transmission: Transmission values
        quantum_efficiency: Dictionary of quantum efficiency values by channel
        illuminant: Illuminant curve
        white_balance_gains: Pre-computed white balance gains (single source of truth)
        channel_mixer: Optional channel mixer settings for color manipulation
    
    Returns:
        RGB color as numpy array [R, G, B]
    """
    # Early exit for invalid data
    if not is_valid_spectrum(transmission) or reflector is None:
        return np.zeros(3)

    # Process each channel
    rgb_resp = {}
    for ch in ['R', 'G', 'B']:
        qe_curve = quantum_efficiency.get(ch)
        if qe_curve is None:
            rgb_resp[ch] = 0.0
            continue

        # Calculate channel response:
        # reflector x transmission x QE x illuminant (NaN points excluded)
        response = integrated_response(transmission, qe_curve, illuminant, reflector)
        rgb_resp[ch] = 0.0 if np.isnan(response) else response

    # Apply white balance with safety against division by zero
    rgb_values = np.zeros(3)
    for i, ch in enumerate(['R', 'G', 'B']):
        # Handle case where white_balance_gains is None
        if white_balance_gains is not None:
            wb_gain = white_balance_gains.get(ch, 1.0)
        else:
            wb_gain = 1.0
        
        if wb_gain > EPSILON:
            rgb_values[i] = rgb_resp.get(ch, 0.0) / wb_gain
        else:
            rgb_values[i] = rgb_resp.get(ch, 0.0)
    
    # Apply channel mixing if enabled
    if channel_mixer is not None and channel_mixer.enabled:
        rgb_values = apply_channel_mixing_to_colors(rgb_values, channel_mixer)
            
    return rgb_values


# ============================================================================
# FORMATTING FUNCTIONS
# ============================================================================

def format_transmission_metrics(
    trans: np.ndarray, 
    label: str, 
    avg_trans: float, 
    effective_stops: float
) -> Dict[str, str]:
    """
    Format transmission metrics for display.
    
    Args:
        trans: Transmission values
        label: Label for the transmission
        avg_trans: Average transmission (0-1)
        effective_stops: Effective light loss in stops
        
    Returns:
        Dictionary containing formatted metrics
    """
    return {
        "label": label,
        "effective_stops": f"{effective_stops:.2f}",
        "avg_transmission_pct": f"{avg_trans * 100:.1f}%"
    }


def format_white_balance_data(
    white_balance_gains: Dict[str, float],
    selected_filters: List[str]
) -> Dict[str, Any]:
    """
    Format white balance data for display.
    
    Args:
        white_balance_gains: Dictionary of white balance gains by channel
        selected_filters: List of selected filter names
        
    Returns:
        Dictionary with formatted white balance data
    """
    # Calculate relative channel intensities (inverted gains)
    intensities = {
        k: (1.0 / v if v != 0 else 0.0)
        for k, v in white_balance_gains.items()
    }
    
    # Add a note if no filters are selected
    has_filters = len(selected_filters) > 0
    
    return {
        "has_filters": has_filters,
        "intensities": {
            "R": f"{intensities['R']:.3f}",
            "G": f"{intensities['G']:.3f}",
            "B": f"{intensities['B']:.3f}"
        }
    }
