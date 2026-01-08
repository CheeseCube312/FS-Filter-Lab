# models/constants.py
"""
Application constants and configuration values for FS FilterLab.

SCALE CONVENTION:
All spectral data (filters, QE, reflectors) use FRACTIONAL scale (0-1) internally:
- 0.0 = 0% (no transmission/response)  
- 1.0 = 100% (full transmission/response)
- This enables natural multiplication for filter combinations
- UI conversions to percentages (* 100) happen only for display

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
WAVELENGTH_RANGE = (300, 1100)  # Min and max wavelengths for the standard grid

# Mathematical constants for numerical stability
EPSILON = 1e-6  # Small value to prevent division by zero and log domain errors

# =============================================================================
# DATA FOLDER STRUCTURE
# =============================================================================

# Data directory paths - centralized for consistency
DATA_FOLDERS = {
    'filters': "program/data/filters_data",
    'qe': "program/data/QE_data", 
    'illuminants': "program/data/illuminants",
    'reflectors': "program/data/reflectors"
}

# Output directory paths
OUTPUT_FOLDERS = {
    'reports': "program/output",  # Main output directory for generated reports
    'ecosis': "program/data/reflectors/Ecosis",  # ECOSIS import destination
    'filter_import': "program/data/filters_data"  # Filter import destination
}

# File extensions used throughout the application
FILE_EXTENSIONS = {
    'data': '.tsv',      # Tab-separated values for all spectral data
    'image': '.png',     # Portable Network Graphics for chart exports
    'export': '.tsv'     # Export format for processed data
}

# =============================================================================
# TSV FILE STRUCTURE CONSTANTS
# =============================================================================

# Standard column names used in TSV files
TSV_COLUMNS = {
    'wavelength': 'Wavelength',
    'transmittance': 'Transmittance', 
    'reflectance': 'Reflectance',
    'filter_number': 'Filter Number',
    'filter_name': 'Name',
    'manufacturer': 'Manufacturer',
    'hex_color': 'hex_color'
}

# Metadata field names used in comment-based TSV files (# key\tvalue format)
METADATA_FIELDS = {
    'name': 'Name',                    # Display name for the spectrum
    'is_default': 'IsDefault',         # Vegetation preview default marker
    'name_for_search': 'name_for_search',  # User-selected naming field
    'species': 'species',              # Species name (ECOSIS data)
    'sample_type': 'sample_type',      # Sample type classification
    'collector': 'collector',          # Data collector information
    'package_title': 'Package Title'   # ECOSIS package title
}

# =============================================================================
# VEGETATION PREVIEW CONFIGURATION
# =============================================================================

# Settings for the vegetation color preview functionality
VEGETATION_PREVIEW = {
    'required_count': 4,          # Number of default reflectors needed (2x2 grid)
    'grid_size': (2, 2),          # Display grid dimensions
    'default_prefix': 'Default ', # Prefix for IsDefault metadata values
    'default_numbers': [1, 2, 3, 4]  # Required default numbers
}

# =============================================================================
# SPECTRAL DATA PROCESSING CONSTANTS
# =============================================================================

# Configuration for spectral data validation and processing
SPECTRAL_CONFIG = {
    'min_data_points': 2,          # Minimum valid data points required
    'normalization_threshold': 1.5, # Values above this treated as percentages
    'precision_decimals': 3         # Decimal places for processed values
}

# =============================================================================
# APPLICATION DEFAULTS
# =============================================================================

# File and data defaults
DEFAULT_QE_FILE = "Default_QE"                # Default QE file name pattern
DEFAULT_ILLUMINANT = "AM1.5_Global_REL"      # Standard solar spectrum reference
DEFAULT_HEX_COLOR = "#838383"                # Default filter color (neutral gray)

# Cache configuration
CACHE_DIR = Path("program/cache")  # Directory for storing cached computation results

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
# USER INTERFACE TEXT CONSTANTS
# =============================================================================

# Button text
UI_BUTTONS = {
    'apply': "Apply",
    'done': "Done",
    'cancel': "Cancel",
    'close_importers': "Close Import Data",
    'rebuild_cache': "Rebuild Cache",
    'csv_importers': "Import Data (CSV/ECOSIS)",
    'generate_full_report': "Generate Full Report",
}

# Main section and panel titles
UI_SECTIONS = {
    'filter_plotter': "Filter Plotter",
    'analysis_setup': "Analysis Setup",
    'display_visualization': "Display & Visualization",
    'export_reports': "Export & Reports",
    'data_management': "Data Management",
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
    'no_target_overlap': "No valid overlap with target for deviation calculation.",
    'leaf_data_required': "Leaf reflectance data requires files named: Leaf 1, Leaf 2, Leaf 3, Leaf 4",
    'no_illuminant': "No illuminant loaded.",
    'no_reflectors': "No reflectance spectra found.",
    'qe_illuminant_required': "Select a QE & illuminant profile to compute white balance.",
    'color_compute_failed': "Unable to compute color for selected surface"
}

UI_WARNING_MESSAGES = {
    'no_illuminants': "No illuminants found.",
    'invalid_hex_colors': "Found {count} filters with invalid hex color codes:",
    'incomplete_reflector_data': "Some reflector data appears incomplete. Check data files.",
    'vegetation_preview_required': (
        "Vegetation Color Preview requires 4 reflector files with IsDefault metadata:\n"
        "- File with #IsDefault\tDefault 1\n"
        "- File with #IsDefault\tDefault 2\n"
        "- File with #IsDefault\tDefault 3\n"
        "- File with #IsDefault\tDefault 4\n"
        "Make sure the TSV files have IsDefault metadata with these exact values."
    )
}

UI_SUCCESS_MESSAGES = {
    'report_generated': "Report generated successfully!",
    'cache_rebuilt': "Cache rebuilt successfully! Reloading application...",
    # Additional action success messages
    'full_report_generated': "Full report generated. Files saved to output folder.",
    'tsv_generated': "TSV generated and ready for download!"
}

# Operation error messages for try_operation calls
UI_OPERATION_ERRORS = {
    'report_generation': "Report generation failed",
    'full_report_generation': "Full report generation failed", 
    'tsv_generation': "TSV generation failed",
    'cache_rebuild': "Cache rebuild failed"
}

# Action type constants to eliminate magic strings
ACTION_TYPES = {
    'generate_report': 'generate_report',
    'generate_full_report': 'generate_full_report',
    'export_tsv': 'export_tsv',
    'rebuild_cache': 'rebuild_cache'
}

# Reusable error message templates for consistent formatting
ERROR_MESSAGE_TEMPLATES = {
    'compute_failed': "Cannot compute {metric} for {item}: {reason}.",
    'import_failed': "Import failed: {reason}",
    'operation_failed': "{operation} failed",
    'invalid_format': "Invalid {item} format",
    'data_not_found': "{data_type} data not found. Make sure you have .tsv files in {directory}.",
    'file_error': "Failed to {action} file {filename}: {reason}",
    'validation_error': "{validation_type} validation failed: {details}"
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
        'main_title': 18,                  # Report main title
        'legend': 8,                       # Legend font size
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
