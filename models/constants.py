# models/constants.py
"""
Application constants and configuration values for FS FilterLab.

This module centralizes all constant values used throughout the application:
- Spectral data configuration (wavelength ranges, interpolation grids)
- Default application settings and values
- User interface text and styling constants  
- Chart rendering configuration
- File paths and caching configuration
- Mathematical constants for numerical stability

Constants defined here ensure consistency across all modules and provide
a single location for configuration changes.
"""

# Standard library imports
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Callable, Any, Tuple, Optional

# Third-party imports
import numpy as np

# =============================================================================
# SPECTRAL DATA CONFIGURATION
# =============================================================================

# Standard wavelength grid for all spectral data interpolation
INTERP_GRID = np.arange(300, 1101, 1)  # 300–1100 nm, step 1 nm (standard optical range)

# Mathematical constants for numerical stability
EPSILON = 1e-6  # Small value to prevent division by zero and log domain errors

# =============================================================================
# APPLICATION DEFAULTS
# =============================================================================

# File and data defaults
DEFAULT_QE_FILE = "Default_QE"                # Default QE file name pattern
DEFAULT_ILLUMINANT = "AM1.5_Global_REL"      # Standard solar spectrum reference
DEFAULT_HEX_COLOR = "#838383"                # Default filter color (neutral gray)

# Cache configuration
CACHE_DIR = Path("cache")  # Directory for storing cached computation results

# RGB channel configuration
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

# Default channel visibility settings
DEFAULT_RGB_VISIBILITY = {
    'R': True,       # Show red channel by default
    'G': True,       # Show green channel by default
    'B': True        # Show blue channel by default
}

# =============================================================================
# CHANNEL MIXER CONFIGURATION
# =============================================================================

# Default channel mixer settings (identity matrix = no color mixing)
DEFAULT_CHANNEL_MIXER = {
    # Red output channel: R_out = R*red_r + G*red_g + B*red_b
    'red_r': 1.0, 'red_g': 0.0, 'red_b': 0.0,
    
    # Green output channel: G_out = R*green_r + G*green_g + B*green_b  
    'green_r': 0.0, 'green_g': 1.0, 'green_b': 0.0,
    
    # Blue output channel: B_out = R*blue_r + G*blue_g + B*blue_b
    'blue_r': 0.0, 'blue_g': 0.0, 'blue_b': 1.0,
    
    # Control flags
    'enabled': False
}

# Channel mixer UI control settings
CHANNEL_MIXER_RANGE = (-2.0, 2.0)  # Slider range for mixing coefficients
CHANNEL_MIXER_STEP = 0.01           # Step size for sliders

# =============================================================================
# VEGETATION PREVIEW CONFIGURATION  
# =============================================================================

# Required reflector names for vegetation preview (2x2 grid display)
VEGETATION_PREVIEW_FILES = [
    "Leaf 1",    # Top-left quadrant
    "Leaf 2",    # Top-right quadrant
    "Leaf 3",    # Bottom-left quadrant
    "Leaf 4"     # Bottom-right quadrant
]

# =============================================================================
# USER INTERFACE TEXT CONSTANTS
# =============================================================================

# Button text with emoji icons
UI_BUTTONS = {
    'apply': "🔄 Apply",
    'done': "✅ Done", 
    'cancel': "✖ Cancel",
    'close_importers': "✖️ Close Importers",
    'rebuild_cache': "🔄 Rebuild Cache",
    'csv_importers': "📊 WebPlotDigitizer .csv importers",
    'generate_report': "📄 Generate Report (PNG)",
    'download_report': "⬇️ Download Last Report"
}

# Main section and panel titles
UI_SECTIONS = {
    'filter_plotter': "Filter Plotter",
    'extras': "Extras",
    'settings': "Settings",
    'vegetation_preview': "Vegetation Color Preview",
    'surface_preview': "Surface Preview",
    'show_advanced_search': "Show Advanced Search",
    'show_channel_mixer': "Show Channel Mixer",
    'sensor_response_channels': "Sensor-Weighted Response Channels",
    'display_options': "Display Options",
    'reflectance_illuminant_curves': "Show Reflectance and Illuminant Curves"
}

# Form field labels and control text
UI_LABELS = {
    'select_filters': "Select filters to plot",
    'scene_illuminant': "Scene Illuminant", 
    'sensor_qe_profile': "Sensor QE Profile",
    'reference_target': "Reference Target",
    'surface_reflectance': "Surface Reflectance Spectrum",
    'set_filter_counts': "Set Filter Stack Counts",
    'stop_view_toggle': "Show stop-view (logarithmic)",
    'apply_white_balance': "Apply White Balance to Response"
}

# User feedback messages
UI_INFO_MESSAGES = {
    'no_target_overlap': "ℹ️ No valid overlap with target for deviation calculation.",
    'leaf_data_required': "ℹ️ Leaf reflectance data requires files named: Leaf 1, Leaf 2, Leaf 3, Leaf 4",
    'no_illuminant': "ℹ️ No illuminant loaded.",
    'no_reflectors': "No reflectance spectra found.",
    'qe_illuminant_required': "Select a QE & illuminant profile to compute white balance.",
    'color_compute_failed': "Unable to compute color for selected surface"
}

UI_WARNING_MESSAGES = {
    'no_illuminants': "⚠️ No illuminants found.",
    'invalid_hex_colors': "⚠ Found {count} filters with invalid hex color codes:",
    'incomplete_reflector_data': "Some reflector data appears incomplete. Check data files.",
    'vegetation_preview_required': (
        "⚠️ Vegetation Color Preview requires these exact reflector names:\n"
        "• Leaf 1\n"
        "• Leaf 2\n" 
        "• Leaf 3\n"
        "• Leaf 4\n"
        "Make sure the TSV files have a 'Name' column with these exact values."
    )
}

UI_SUCCESS_MESSAGES = {
    'report_generated': "✅ Report generated successfully!",
    'cache_rebuilt': "✅ Cache rebuilt successfully! Reloading application..."
}

# Tooltip and help text
UI_HELP_TEXT = {
    'channel_mixer': "Open channel mixer panel for RGB channel manipulation",
    'stop_view': "Display transmission in camera stops (logarithmic scale) instead of percentage"
}

# Chart and visualization titles  
UI_CHART_TITLES = {
    'combined_filter_response': "Combined Filter Response",
    'sensor_weighted_response': "Sensor-Weighted Response (QE × Transmission)",
    'leaf_reflectance': "Leaf Reflectance Spectra",
    'qe_profile': "Sensor Quantum Efficiency (QE)",
    'illuminant_spectrum': "Illuminant Spectrum"
}

# =============================================================================
# CHART RENDERING AND VISUALIZATION CONSTANTS
# =============================================================================

# Chart dimensions (heights in pixels)
CHART_HEIGHTS = {
    'default': 300,              # Standard chart height
    'standard_plot': 400,        # Larger plots  
    'plot_with_spectrum': 450,   # Plots with spectrum strips
    'sparkline': 150             # Compact inline sparklines
}

# Line rendering styles
CHART_LINE_STYLES = {
    'default': {'width': 2},               # Standard line width
    'thick': {'width': 3},                 # Emphasized lines (combined filters)
    'sparkline': {'width': 1.5},           # Sparkline thickness
    'standard_width': 2,                   # Standard matplotlib line width
    'extrapolated_style': '--',            # Extrapolated data line style (dashed)
    'extrapolated_alpha': 0.7,             # Extrapolated data transparency
    'extrapolated': {                      # Styling for extrapolated data regions
        'alpha': 0.7,
        'dash': 'dot'
    }
}

# Color scheme for different chart elements
CHART_COLORS = {
    # Primary chart elements
    'illuminant': 'orange',        # Illuminant spectrum curves
    'target': 'red',              # Target/reference lines  
    'combined': 'black',          # Combined filter response
    'warning': 'red',             # Warning indicators
    'text': 'black',              # General text and borders
    
    # Specialized colors
    'single_reflector': 'brown',   # Individual reflector curves
    'leaf_colors': ['#228B22', '#32CD32', '#90EE90', '#006400'],  # Vegetation (various greens)
    'rgb_colors': {'R': 'red', 'G': 'green', 'B': 'blue'},        # RGB channels
    
    # UI colors
    'grid': 'rgba(200,200,200,0.4)',      # Chart grid lines
    'transparent': 'rgba(0,0,0,0)'        # Transparent backgrounds
}

# Layout configuration for complex plots
PLOT_LAYOUT = {
    'spectrum_strip_height_pct': 0.05,              # Height percentage for spectrum indicators
    'grid_height_ratios': [1.2, 0.6, 3.2, 0.8, 3.2]  # Relative heights for multi-panel plots
}

# Sensor response plot configuration
SENSOR_RESPONSE_DEFAULTS = {
    'spectrum_strip_height_pct': 0.05,     # Height of spectrum color strip
    'spectrum_strip_position_pct': 1.02,   # Position of spectrum strip (relative to max response)
    'saturation_scaling_factor': 5.0,      # Color saturation enhancement factor
    'min_saturation': 0.15                 # Minimum saturation value for visibility
}

# Report generation configuration
REPORT_CONFIG = {
    'figure_size': (8, 14),                # Figure dimensions (width, height) in inches
    'dpi': 150,                            # Figure DPI for high quality output
    'swatch_line_width': 0.5,             # Filter color swatch border width
    'combined_line_width': 2.5,           # Combined filter line width
    'channel_line_width': 2,              # Individual channel line width
    'font_sizes': {
        'filter_label': 10,                # Filter name labels
        'section_header': 12,              # Section headers
        'title': 16,                       # Main title
        'subtitle': 8,                     # Subtitles and legends
        'axis_title': 14,                  # Chart axis titles
    }
}

# Matplotlib style configuration
MPL_STYLE_CONFIG = {
    "font.family": "DejaVu Sans",
    "axes.facecolor": "white",
    "axes.edgecolor": "#CCCCCC",
    "axes.grid": True,
    "grid.color": "#EEEEEE",
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "#444444",
    "ytick.color": "#444444", 
    "text.color": "#333333",
    "axes.labelcolor": "#333333",
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.frameon": False,
    "legend.fontsize": 8,
}

# =============================================================================
# DATA CLASSES FOR PARAMETER MANAGEMENT
# =============================================================================

@dataclass
class ReportConfig:
    """Configuration for report generation parameters."""
    selected_filters: List[str]
    current_qe: Dict[str, np.ndarray]
    camera_name: str
    illuminant_name: str
    illuminant_curve: np.ndarray

@dataclass
class FilterData:
    """Container for filter-related data structures."""
    filter_matrix: np.ndarray
    df: Any
    display_to_index: Dict[str, int]
    masks: np.ndarray
    interp_grid: np.ndarray

@dataclass  
class ComputationFunctions:
    """Container for computation functions used in report generation."""
    compute_selected_indices_fn: Callable[[List[str]], List[int]]
    compute_filter_transmission_fn: Callable[[List[int]], Tuple[np.ndarray, str, np.ndarray]]
    compute_effective_stops_fn: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], Tuple[float, float]]
    compute_white_balance_gains_fn: Callable[[np.ndarray, Dict[str, np.ndarray], np.ndarray], Dict[str, float]]
    add_curve_fn: Callable
    sanitize_fn: Callable[[str], str]

@dataclass
class SensorData:
    """Container for sensor-related parameters."""
    sensor_qe: np.ndarray

@dataclass
class ChartConfig:
    """Configuration for chart styling and layout parameters."""
    title: str = ""
    x_title: str = "Wavelength (nm)" 
    y_title: str = "Response"
    height: Optional[int] = None
    template: str = "plotly_white"
    hovermode: str = "x unified"
    log_scale: bool = False
    show_legend: bool = True
    show_spectrum_strip: bool = True
