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
- Data loading code divides by 100 unconditionally on load

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
# DATA FOLDER STRUCTURE
# =============================================================================

# Data directory paths - centralized for consistency
DATA_FOLDERS = {
    'filters': "program/data/filters_data",
    'qe': "program/data/QE_data",
    'film': "program/data/film_data",
    'illuminants': "program/data/illuminants",
    'reflectors': "program/data/reflectors",
}

# Human-readable labels for sensor/film profile types
SENSOR_TYPE_LABELS = {
    'rgb_sensor':       "Digital Sensor",
    'color_slide':      "Slide Film",
    'color_negative':   "Color Negative Film",
    'bw':               "B&W Film",
}

# Output directory paths
OUTPUT_FOLDERS = {
    'reports': "program/output",  # Main output directory for generated reports
    'ecosis': "program/data/reflectors/Ecosis",  # ECOSIS import destination
    'filter_import': "program/data/filters_data"  # Filter import destination
}

# Persistent JSON files
MY_FILTERS_FILE = "program/data/filters_data/my_filters.json"

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
    'name_for_search': 'name_for_search',  # Column name to use for naming (not the value)
    'relevant_metadata': 'relevant_metadata',  # Column names for Surface Color Preview (pipe-separated)
    'species': 'species',              # Species name (ECOSIS data)
    'sample_type': 'sample_type',      # Sample type classification
    'collector': 'collector',          # Data collector information
    'package_title': 'Package Title'   # ECOSIS package title
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

# Surface Color Preview metadata display configuration
SURFACE_COLOR_METADATA = {
    'api_attribution_fields': ['Organization', 'Package Title', 'Citation', 'License', 'DOI URL'],
    'fallback_fields': ['Target Type', 'Common Name', 'Latin Genus', 'Latin Species']
}

# TSV attribution fields for ECOSIS imports (display_name, internal_key)
TSV_ATTRIBUTION_FIELDS = [
    # Core identification
    ('Organization', 'organization'),
    ('Package Title', 'package_title'),
    ('Author', 'author'),
    ('Year', 'year'),
    ('License', 'license'),
    # Scientific citation
    ('DOI', 'doi'),
    ('DOI URL', 'doi_url'),
    ('Citation', 'citation'),
    ('Related Publications', 'related_publications'),
    # Funding
    ('Funding Source', 'funding_source'),
    ('Grant Numbers', 'funding_grant_numbers'),
    # Instrument
    ('Instrument Manufacturer', 'instrument_manufacturer'),
    ('Instrument Model', 'instrument_model'),
    ('Acquisition Method', 'acquisition_method'),
    ('Foreoptic Type', 'foreoptic_type'),
    ('Light Source', 'light_source'),
    # Sample
    ('Target Type', 'target_type'),
    ('Target Status', 'target_status'),
    ('Ecosystem Type', 'ecosystem_type'),
    ('Measurement Venue', 'measurement_venue'),
    ('Measurement Date', 'measurement_date'),
    # Data characteristics
    ('Measurement Units', 'measurement_units'),
    ('Measurement Quantity', 'measurement_quantity'),
    ('Processing Info', 'processing_info'),
    ('Spectra Count', 'spectra_count'),
    # Keywords
    ('Keywords', 'keywords'),
    ('NASA GCMD Keywords', 'nasa_gcmd_keywords'),
    ('Theme', 'theme'),
    # Dataset metadata
    ('Description', 'description'),
    ('Created', 'created'),
    ('Modified', 'modified'),
    # Import source
    ('Source CSV File', 'source_csv_file'),
    ('API URL', 'api_url'),
]
# =============================================================================
# SPECTRAL DATA PROCESSING CONSTANTS
# =============================================================================

# Configuration for spectral data validation and processing
SPECTRAL_CONFIG = {
    'min_data_points': 2,          # Minimum valid data points required
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

# Default white balance multiplier values (unity gain)
DEFAULT_WB_GAINS = {
    'R': 1.0,        # Red channel multiplier
    'G': 1.0,        # Green channel multiplier (reference)
    'B': 1.0         # Blue channel multiplier
}

# =============================================================================
# CHANNEL MIXER CONFIGURATION
# =============================================================================

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
    'csv_importers': "Import Data (CSV/ECOSIS)",
    'generate_full_report': "Generate Full Report",
    'download_report': "Download PNG Report",
    'import_filter': "Import Filter",
    'import_illuminant': "Import Illuminant",
    'import_camera_qe': "Import Camera QE",
    'import_single_spectrum': "Import Single Spectrum",
    'import_ecosis_file': "Import ECOSIS File",
    'add_all': "Add All",
    'remove_all': "Remove All",
    'clear_all_owned': "Clear All Owned Filters",
}

# Main section and panel titles
UI_SECTIONS = {
    'filter_plotter': "Filter Plotter",
    'analysis_setup': "Analysis Setup",
    'film_emulation': "Film Emulation Target",
    'display_visualization': "Display & Visualization",
    'export_reports': "Export & Reports",
    'data_management': "Data Management",
    'show_advanced_search': "Show Advanced Filter Search",
    'show_reflector_search': "Show Reflector Search",
    'show_channel_mixer': "Show Channel Mixer",
    'show_my_filters_manager': "Manage Owned Filters",
    'sensor_response_channels': "Sensor-Weighted Response Channels",
    'display_options': "Display Options",
    'reflectance_illuminant_curves': "Show Reflectance and Illuminant Curves",
    'default_reflector_list': "Surface Color Preview",
    'advanced_filter_search': "Advanced Filter Search",
    'advanced_reflector_search': "Advanced Reflector Search",
    'my_filters_manager': "Manage Owned Filters",
    'import_data': "Import Data",
    'ecosis_import': "ECOSIS Import",
    'import_reflectance_absorption': "Import Reflectance/Absorption Data",
    'column_selection': "Column Selection:",
}

# Form field labels and control text
UI_LABELS = {
    'select_filters': "Select filters to plot",
    'scene_illuminant': "Scene Illuminant", 
    'sensor_qe_profile': "Sensor / Film Profile",
    'reference_target': "Reference Target",
    'film_emulation_target': "Film Emulation Target",
    'film_emulation_wb': "White Balance Target Independently",
    'film_emulation_filters': "Filters on Film Target",
    'surface_reflectance': "Surface Reflectance Spectrum",
    'set_filter_counts': "Set Filter Stack Counts",
    'stop_view_toggle': "Show stop-view (logarithmic)",
    'apply_white_balance': "Apply White Balance to Response",
    'search_by_manufacturer': "Search by manufacturer, color, or spectral transmittance.",
    'filter_reflectors': "Filter reflectors by metadata, then add to your default list.",
    'upload_csv_wl_trans': "Upload CSV (Wavelength, Transmittance)",
    'upload_csv_wl_power': "Upload CSV (Wavelength, Power)",
    'upload_csv_wl_rgb': "Upload CSV (Wavelength, R, G, B)",
    'upload_csv': "Upload CSV",
    'upload_ecosis_csv': "Upload ECOSIS CSV",
    'ecosis_api_url': "ECOSIS API URL (optional, recommended)",
    'choose_name_column': "Choose column for spectrum names",
    'relevant_metadata': "Relevant metadata for Surface Color Preview",
    'upload_file_first': "Please upload a file first",
}

# User feedback messages
UI_INFO_MESSAGES = {
    'no_target_overlap': "No valid overlap with target for deviation calculation.",
    'no_illuminant': "No illuminant loaded.",
    'no_reflectors': "No reflectance spectra found.",
    'qe_illuminant_required': "Select a QE & illuminant profile to compute white balance.",
    'color_compute_failed': "Unable to compute color for selected surface",
    'select_qe_prompt': "Select a QE profile in Analysis Setup (sidebar) to see sensor response and surface color analysis."
}

UI_WARNING_MESSAGES = {
    'no_illuminants': "No illuminants found.",
    'invalid_hex_colors': "Found {count} filters with invalid hex color codes:",
    'incomplete_reflector_data': "Some reflector data appears incomplete. Check data files.",
    'no_owned_filters': "No filters marked as owned. Use \"Manage Owned Filters\" to add filters first.",
    'unowned_selected': "Selected filter(s) not in owned list:",
}

UI_SUCCESS_MESSAGES = {
    'report_generated': "Report generated successfully!",
    'full_report_generated': "Full report generated. Files saved to output folder."
}

# Operation error messages for try_operation calls
UI_OPERATION_ERRORS = {
    'report_generation': "Report generation failed",
    'full_report_generation': "Full report generation failed"
}

# Action type constants to eliminate magic strings
ACTION_TYPES = {
    'generate_report': 'generate_report',
    'generate_full_report': 'generate_full_report'
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
    'stop_view': "Display transmission in camera stops (logarithmic scale) instead of percentage",
    'apply_white_balance_sensor': "Normalize channel responses so a neutral grey scene produces equal R/G/B output.",
    'apply_white_balance_film': "For film, this simulates white balance correction applied during scanning or printing. Disable to see the raw dye-layer response.",
    'film_emulation_wb': "Apply white balance to the film target independently of the sensor white balance. The target is always scaled so its peak matches the sensor peak — only the channel ratio changes.",
    'film_emulation_filters': "Filters applied to the film emulation target only. Independent from the main filter stack. Leave empty for unfiltered film response.",
}

# Chart and visualization titles  
UI_CHART_TITLES = {
    'combined_filter_response': "Combined Filter Response",
    'sensor_weighted_response': "Sensor-Weighted Response (QE × Transmission)",
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

# Sparkline plot configuration
SPARKLINE_CONFIG = {
    'default_width': 300,        # Default sparkline width in pixels
    'margins': {'l': 40, 'r': 10, 't': 10, 'b': 30},  # Chart margins (left, right, top, bottom)
    'font_sizes': {
        'axis_title': 10,        # Axis label font size
        'tick_label': 8,         # Tick label font size
    },
    'wavelength_tick_intervals': {  # Wavelength axis tick spacing by range
        'large': {'threshold': 500, 'interval': 200},   # >500nm range
        'medium': {'threshold': 200, 'interval': 100},  # 200-500nm range  
        'small': {'threshold': 0, 'interval': 50},      # <200nm range
    }
}

# Line rendering styles
CHART_LINE_STYLES = {
    # Plotly line-width presets (keyed by role)
    'default': {'width': 2},               # Standard line width
    'thick': {'width': 3},                 # Emphasized lines (combined filters)
    'sparkline': {'width': 1.5},           # Sparkline thickness
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
# IMPORT DIALOG CONFIGURATION
# =============================================================================
# Tab labels for the import dialog. Each key maps to a tab panel in forms.py.
# To add a new import type:
# 1. Add entry here with key and display label
# 2. Create _import_<type>_panel() function in views/forms.py
# 3. Add corresponding tab and tab_panel in render_import_dialog()
IMPORT_TABS = {
    'filters': "Filters",
    'illuminants': "Illuminants",
    'camera_qe': "Camera QE",
    'film': "Film",
    'reflectance_ecosis': "Reflectance/ECOSIS",
}

IMPORT_DATA_TYPES = ["Reflectance", "Absorption"]
IMPORT_CATEGORIES = ["Plant", "Other"]
IMPORT_ECOSIS_MODES = ["Single Spectrum CSV", "ECOSIS Multi-Spectrum CSV"]
