# models/constants.py
"""
Application constants and configuration values for FS FilterLab.

This module centralizes all constant values used throughout the application:
- Spectral data configuration (wavelength ranges, interpolation grids)
- Default UI settings and colors
- File paths and caching configuration
- Mathematical constants and error prevention values

Constants defined here ensure consistency across all modules and provide
a single location for configuration changes.
"""
import numpy as np

# Basic interpolation grid for all spectral data
INTERP_GRID = np.arange(300, 1101, 1)  # 300–1100 nm, step 1 nm (standard optical range)

# Default QE file name pattern for camera sensor data
DEFAULT_QE_FILE = "Default_QE"

# Default RGB channel colors for visualization consistency
QE_COLORS = {
    'R': 'red',      # Red channel display color
    'G': 'green',    # Green channel display color  
    'B': 'blue'      # Blue channel display color
}

# Default white balance multiplier values (unity gain)
DEFAULT_WB_GAINS = {
    'R': 1.0,        # Red channel multiplier
    'G': 1.0,        # Green channel multiplier (reference)
    'B': 1.0         # Blue channel multiplier
}

# Default channel visibility settings for UI controls
DEFAULT_RGB_VISIBILITY = {
    'R': True,       # Show red channel by default
    'G': True,       # Show green channel by default
    'B': True        # Show blue channel by default
}

# Default application state values
DEFAULT_ILLUMINANT = "AM1.5_Global_REL"  # Standard solar spectrum reference
DEFAULT_HEX_COLOR = "#838383"             # Default filter color (neutral gray)

# Cache configuration
from pathlib import Path
CACHE_DIR = Path("cache")  # Directory for storing cached computation results

# Mathematical constants for numerical stability
EPSILON = 1e-6  # Small value to prevent division by zero and log domain errors

# Channel Mixer constants
DEFAULT_CHANNEL_MIXER = {
    # Red output channel (R_out = R*red_r + G*red_g + B*red_b)
    'red_r': 1.0, 'red_g': 0.0, 'red_b': 0.0,
    # Green output channel (G_out = R*green_r + G*green_g + B*green_b)  
    'green_r': 0.0, 'green_g': 1.0, 'green_b': 0.0,
    # Blue output channel (B_out = R*blue_r + G*blue_g + B*blue_b)
    'blue_r': 0.0, 'blue_g': 0.0, 'blue_b': 1.0,
    # Control flags
    'enabled': False
}

# Channel Mixer UI Constants
CHANNEL_MIXER_RANGE = (-2.0, 2.0)  # Slider range for channel mixing values
CHANNEL_MIXER_STEP = 0.01          # Step size for channel mixing sliders

# Vegetation Preview Constants
VEGETATION_PREVIEW_FILES = [
    "Leaf 1",
    "Leaf 2",
    "Leaf 3",
    "Leaf 4"
]  # Exact reflector names required for vegetation preview (2x2 grid)
