"""
Channel Mixer Service for FS FilterLab.

This module provides channel mixing functionality for RGB response manipulation,
commonly used in IR photography for channel swapping and false color enhancement.
"""
# Third-party imports
import numpy as np
from typing import Dict, Optional

# Local imports

from models.core import ChannelMixerSettings


def apply_channel_mixing_to_responses(
    rgb_responses: Dict[str, np.ndarray],
    mixer_settings: ChannelMixerSettings
) -> Dict[str, np.ndarray]:
    """
    Apply channel mixer transformation to RGB spectral response curves.
    
    This function transforms the RGB spectral response curves using the channel
    mixer matrix, enabling real-time visualization of how channel mixing affects
    the sensor's spectral sensitivity.
    
    Args:
        rgb_responses: Dictionary with 'R', 'G', 'B' keys containing response arrays
        mixer_settings: Channel mixer configuration settings
    
    Returns:
        Dictionary with mixed RGB response curves
        
    Example:
        For R-G swap: red output gets green response, green output gets red response
    """
    if not mixer_settings.enabled or not rgb_responses:
        return rgb_responses
    
    reference_shape = next(iter(rgb_responses.values())).shape
    zero_array = np.zeros(reference_shape)
    
    # Stack into (n_wavelengths, 3) matrix, apply transform, unstack
    stacked = np.column_stack([
        rgb_responses.get('R', zero_array),
        rgb_responses.get('G', zero_array),
        rgb_responses.get('B', zero_array),
    ])
    mixed = stacked @ mixer_settings.to_matrix().T
    
    return {'R': mixed[:, 0], 'G': mixed[:, 1], 'B': mixed[:, 2]}


def apply_channel_mixing_to_colors(
    rgb_colors: np.ndarray,
    mixer_settings: ChannelMixerSettings
) -> np.ndarray:
    """
    Apply channel mixer transformation to RGB color arrays.
    
    Used for transforming computed color values (like reflector previews)
    to match the channel-mixed spectral response visualization.
    
    Args:
        rgb_colors: Array of RGB values, shape (..., 3) where last dimension is RGB
        mixer_settings: Channel mixer configuration settings
    
    Returns:
        Array of mixed RGB values with same shape as input
        
    Note:
        Input RGB values should be in the same units/scale as the spectral responses
        for consistent results across the application.
    """
    if not mixer_settings.enabled:
        return rgb_colors
    
    # Handle different input shapes - validate that last dimension is RGB (3)
    if rgb_colors.shape[-1] != 3:
        raise ValueError(f"Invalid RGB array shape: {rgb_colors.shape}. Last dimension must be 3.")
    
    original_shape = rgb_colors.shape
    
    # Flatten to 2D for matrix multiplication, preserving RGB dimension
    colors_flat = rgb_colors.reshape(-1, 3)
    
    # Apply transformation matrix
    transform_matrix = mixer_settings.to_matrix()
    mixed_colors_flat = colors_flat @ transform_matrix.T
    
    # Reshape back to original form
    return mixed_colors_flat.reshape(original_shape)



