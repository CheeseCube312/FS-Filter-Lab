# models/constants.py
"""
Application constants and configuration values for FS FilterLab.

SCALE CONVENTION:
All data FILES store spectral values in PERCENTAGE scale (0-100).
Internally, all spectral data (filters, QE, reflectors) is converted to FRACTIONAL scale (0-1):
- 0.0 = 0% (no transmission/response)  
- 1.0 = 100% (full transmission/response)
- This enables natural multiplication for filter combinations
- UI conversions to percentages (* 100) happen only for display
- Filter and reflector loaders divide by 100 unconditionally on load, so
  importers write files in 0-100 scale
- Exception: the film-profile loader accepts either scale; it divides only
  when the interpolated maximum exceeds 1.5

"""

from pathlib import Path

import numpy as np

# =============================================================================
# SPECTRAL DATA CONFIGURATION
# =============================================================================

# Standard wavelength grid for all spectral data interpolation
INTERP_GRID = np.arange(300, 1101, 1)  # 300–1100 nm, step 1 nm (standard optical range)

# Photographic visible band: bounds of the human-vision spectrum strip and
# the UV / VIS / IR chart zoning
VISIBLE_MIN_NM = 380
VISIBLE_MAX_NM = 700

# Mathematical constants for numerical stability
EPSILON = 1e-6  # Small value to prevent division by zero and log domain errors

# =============================================================================
# FEATURE FLAGS
# =============================================================================
# Reflectance workflow (Reflectors shelf tab, Scene Preview card, surface
# white-balance references, surface-spectrum import) is gated OFF for this
# release: only a handful of reflector spectra exist, so the panels read as
# half-finished.  Services, models, tests and reflector data stay intact;
# flip the flag to True to restore the full UI.
FEATURES = {
    "reflectance": False,
}

# =============================================================================
# DATA FOLDER STRUCTURE
# =============================================================================

# Repo-root anchor: resolved from this file's location so data, output and
# cache paths are identical no matter which working directory the app is
# launched from (a CWD=program/ launch used to write imports into a stray
# program/program/ tree).
_REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = _REPO_ROOT  # Public alias: path anchoring for state_manager/tests

# Data directory paths - centralized for consistency
DATA_FOLDERS = {
    'filters': str(_REPO_ROOT / "program/data/filters_data"),
    'qe': str(_REPO_ROOT / "program/data/QE_data"),
    'film': str(_REPO_ROOT / "program/data/film_data"),
    'illuminants': str(_REPO_ROOT / "program/data/illuminants"),
    'reflectors': str(_REPO_ROOT / "program/data/reflectors"),
}

# Human-readable labels for sensor/film profile types
SENSOR_TYPE_LABELS = {
    'rgb_sensor':       "Digital Sensor",
    'color_slide':      "Slide Film",
    'color_negative':   "Color Negative Film",
    'bw':               "B&W Film",
}

# Valid film-profile types: film CSV import validation + Data shelf select
FILM_TYPES = ('color_negative', 'color_slide', 'bw')

# Default advanced-search wavelength criterion row (0-100 % scale; AppState
# owns the field, views use this when adding or resetting a criterion row)
DEFAULT_CRITERION = {"wavelength": 550, "trans_min": 0, "trans_max": 100}

# Output directory paths
OUTPUT_FOLDERS = {
    'reports': str(_REPO_ROOT / "program/output"),  # Generated reports
    'filter_import': str(_REPO_ROOT / "program/data/filters_data")  # Filter import destination
}

# Persistent JSON files
MY_FILTERS_FILE = str(_REPO_ROOT / "program/data/filters_data/my_filters.json")

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
    'sample_type': 'sample_type',      # Sample type classification
    'collector': 'collector',          # Data collector information
}

# Header field names used in sensor/film TSV files (# Key: Value format)
# Required fields: Manufacturer, Name, Type
# Optional fields: ISO, Process, Notes
FILM_TSV_FIELDS = {
    'manufacturer': 'Manufacturer',    # Required: brand name
    'name':         'Name',            # Required: film/sensor model name
    'type':         'Type',            # Required: one of the SENSOR_TYPE_LABELS keys
    'iso':          'ISO',             # Optional: film speed (e.g. "400")
    'process':      'Process',         # Optional: development process (e.g. "E-6", "D-76 1:1")
    'notes':        'Notes',           # Optional: free-text notes
}

# =============================================================================
# SPECTRAL DATA PROCESSING CONSTANTS
# =============================================================================

MIN_DATA_POINTS = 2      # Minimum valid data points required
PRECISION_DECIMALS = 3   # Decimal places for processed values

# =============================================================================
# APPLICATION DEFAULTS
# =============================================================================

# File and data defaults
DEFAULT_HEX_COLOR = "#838383"                # Default filter color (neutral gray)

# Cache configuration
CACHE_DIR = _REPO_ROOT / "program" / "cache"  # Directory for storing cached computation results

# Default white balance multiplier values (unity gain)
DEFAULT_WB_GAINS = {
    'R': 1.0,        # Red channel multiplier
    'G': 1.0,        # Green channel multiplier (reference)
    'B': 1.0         # Blue channel multiplier
}

# Scene preview coverage guard: a surface is flagged "partial data" when more
# than this fraction of the chain's spectral weight (illuminant x stack
# transmission x sensor QE) falls on wavelengths where its reflectance has
# no data.
SCENE_COVERAGE_LIMIT = 0.10

# =============================================================================
# CHART RENDERING AND VISUALIZATION CONSTANTS
# =============================================================================

# Chart dimensions (heights in pixels)
CHART_HEIGHTS = {
    'default': 300,              # Standard chart height
    'standard_plot': 400,        # Larger plots  
    'plot_with_spectrum': 450,   # Plots with spectrum strips
}

# Line rendering styles
CHART_LINE_STYLES = {
    # Plotly line-width presets (keyed by role)
    'default': {'width': 2},               # Standard line width
    'thick': {'width': 3},                 # Emphasized lines (combined filters)
    # Matplotlib line-width presets
    'matplotlib': {'width': 2},            # Standard matplotlib line width
    # Extrapolated-region styling
    'extrapolated': {
        'width': 2,
        'style': '--',                     # Matplotlib dash style
        'alpha': 0.7,
        'dash': 'dot',                     # Plotly dash style
    },
}

# Color scheme for chart elements (only keys actually consumed by services/)
CHART_COLORS = {
    'illuminant': 'orange',   # Illuminant spectrum curves
    'text': 'black',          # General text and borders
    'rgb_colors': {'R': 'red', 'G': 'green', 'B': 'blue'},  # RGB channels
}

# Relative row heights for the multi-panel report grid (GridSpec)
GRID_HEIGHT_RATIOS = [1.2, 0.6, 3.2, 0.8, 3.2]

# Sensor response plot configuration
SENSOR_RESPONSE_DEFAULTS = {
    'spectrum_strip_position_pct': 1.02,   # Camera-capture strip y (relative to max response)
    'human_strip_position_pct': 1.075,     # Human-vision strip y, stacked above the camera strip
    'saturation_scaling_factor': 5.0,      # Color saturation enhancement factor
    'min_saturation': 0.15                 # Minimum saturation value for visibility
}

# Human-vision spectrum strip: hue legend — per-wavelength perceived color
# (CIE-1931 observer, normalized per wavelength so core hues are readable;
# true photopic brightness is shown in the hover text).  The strip covers
# only the visible band: its edges taper to black over 'edge_taper_nm',
# marking where human vision fades out — UV / IR are not drawn at all.
SPECTRUM_STRIP = {
    'solo_position_pct': 1.05,       # filter chart: strip y above the curves
    'marker_size': 9,                # px — matches the camera strip
    'edge_taper_nm': 60,             # visible-band edge fade to black
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
