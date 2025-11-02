"""
Channel Mixer Service for FS FilterLab.

This module provides channel mixing functionality for RGB response manipulation,
commonly used in IR photography for channel swapping and false color enhancement.
"""
# Third-party imports
import numpy as np
from typing import Dict, List, Optional, Any

# Local imports
from models.constants import DEFAULT_CHANNEL_MIXER
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
    if not mixer_settings.enabled:
        return rgb_responses
        
    # Get original responses
    r_orig = rgb_responses.get('R', np.zeros_like(next(iter(rgb_responses.values()))))
    g_orig = rgb_responses.get('G', np.zeros_like(next(iter(rgb_responses.values()))))  
    b_orig = rgb_responses.get('B', np.zeros_like(next(iter(rgb_responses.values()))))
    
    # Apply mixing transformation
    r_mixed = (r_orig * mixer_settings.red_r + 
               g_orig * mixer_settings.red_g + 
               b_orig * mixer_settings.red_b)
    
    g_mixed = (r_orig * mixer_settings.green_r + 
               g_orig * mixer_settings.green_g + 
               b_orig * mixer_settings.green_b)
    
    b_mixed = (r_orig * mixer_settings.blue_r + 
               g_orig * mixer_settings.blue_g + 
               b_orig * mixer_settings.blue_b)
    
    return {
        'R': r_mixed,
        'G': g_mixed, 
        'B': b_mixed
    }


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
    
    # Handle different input shapes
    original_shape = rgb_colors.shape
    
    # Reshape to (..., 3) for matrix multiplication  
    if rgb_colors.ndim == 1 and len(rgb_colors) == 3:
        # Single RGB triplet
        colors_flat = rgb_colors.reshape(1, 3)
    elif rgb_colors.ndim >= 2 and rgb_colors.shape[-1] == 3:
        # Multiple RGB values
        colors_flat = rgb_colors.reshape(-1, 3)
    else:
        raise ValueError(f"Invalid RGB array shape: {original_shape}. Last dimension must be 3.")
    
    # Get transformation matrix and apply
    transform_matrix = mixer_settings.to_matrix()
    mixed_colors_flat = np.dot(colors_flat, transform_matrix.T)
    
    # Reshape back to original form
    return mixed_colors_flat.reshape(original_shape)


def apply_channel_mixing_to_matrix(
    rgb_matrix: np.ndarray,
    mixer_settings: ChannelMixerSettings
) -> np.ndarray:
    """
    Apply channel mixer to RGB matrix for visualization purposes.
    
    Specifically designed for the RGB response matrix used in spectral visualization,
    where the matrix has shape (wavelengths, 3) representing RGB values across spectrum.
    
    Args:
        rgb_matrix: RGB matrix with shape (n_wavelengths, 3)
        mixer_settings: Channel mixer configuration settings
    
    Returns:
        Mixed RGB matrix with same shape as input
    """
    if not mixer_settings.enabled:
        return rgb_matrix
        
    if rgb_matrix.shape[-1] != 3:
        raise ValueError(f"RGB matrix must have 3 channels, got shape: {rgb_matrix.shape}")
    
    # Apply transformation matrix
    transform_matrix = mixer_settings.to_matrix()
    return np.dot(rgb_matrix, transform_matrix.T)


def create_default_mixer_settings() -> ChannelMixerSettings:
    """
    Create default channel mixer settings (identity transformation).
    
    Returns:
        ChannelMixerSettings with identity matrix (no mixing)
    """
    return ChannelMixerSettings(**DEFAULT_CHANNEL_MIXER)


def validate_mixer_settings(mixer_settings: ChannelMixerSettings) -> bool:
    """
    Validate channel mixer settings for reasonable values.
    
    Args:
        mixer_settings: Settings to validate
        
    Returns:
        True if settings are valid, False otherwise
        
    Note:
        Allows values outside [-2, 2] range but warns about extreme values
        that might produce unusual results.
    """
    # Check for extreme values that might indicate user error
    all_values = [
        mixer_settings.red_r, mixer_settings.red_g, mixer_settings.red_b,
        mixer_settings.green_r, mixer_settings.green_g, mixer_settings.green_b,
        mixer_settings.blue_r, mixer_settings.blue_g, mixer_settings.blue_b
    ]
    
    # Allow wide range but flag extreme values
    for value in all_values:
        if not np.isfinite(value):
            return False
        if abs(value) > 5.0:  # Very extreme mixing
            return False
            
    return True


def is_identity_matrix(mixer_settings: ChannelMixerSettings) -> bool:
    """
    Check if mixer settings represent identity transformation (no mixing).
    
    Args:
        mixer_settings: Settings to check
        
    Returns:
        True if settings represent identity matrix
    """
    matrix = mixer_settings.to_matrix()
    identity = np.eye(3)
    return np.allclose(matrix, identity, atol=1e-6)
