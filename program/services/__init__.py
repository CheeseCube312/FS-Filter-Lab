"""
Services package for FS FilterLab.

This package provides all core business logic and processing services:

Core Services:
- app_operations: High-level application operations and workflow management
- calculations: Mathematical computations for transmission and color analysis  
- data: Data loading and caching services with automatic validation
- importing: File import utilities for various spectral data formats
- state_manager: Unified application state management using Streamlit session state
- visualization: Chart generation and report creation services

Note: While this __init__.py provides re-exports for convenience, most code
imports directly from submodules (e.g., `from services.calculations import ...`).
"""

# =============================================================================
# CALCULATION SERVICES
# =============================================================================
from services.calculations import (
    # Transmission calculations
    compute_combined_transmission,
    compute_filter_transmission,
    compute_active_transmission,
    compute_effective_stops,
    calculate_transmission_deviation_metrics,
    compute_selected_filter_indices,
    
    # Color and RGB processing
    compute_rgb_response,
    compute_white_balance_gains,
    compute_white_balance_gains_from_surface,
    compute_reflector_color,
    compute_reflector_preview_colors,
    find_vegetation_preview_reflectors,
    is_reflector_data_valid,
    check_reflector_wavelength_validity,
    
    # Formatting utilities
    format_transmission_metrics,
    format_deviation_metrics,
    format_white_balance_data,
)

# =============================================================================
# DATA LOADING SERVICES
# =============================================================================
from services.data import (
    load_filter_collection,
    load_quantum_efficiencies, 
    load_illuminant_collection,
    load_reflector_collection,
)

# =============================================================================
# VISUALIZATION SERVICES
# =============================================================================
from services.visualization import (
    # Chart creation
    create_filter_response_plot,
    create_sensor_response_plot,
    create_sparkline_plot,
    create_qe_figure,
    create_illuminant_figure,
    
    # Curve utilities
    add_filter_curve_to_plotly,
    add_filter_curve_to_matplotlib,
    
    # Report generation
    generate_report_png,
)

# =============================================================================
# IMPORT SERVICES
# =============================================================================
from services.importing import (
    import_filter_from_csv,
    import_illuminant_from_csv,
    import_qe_from_csv,
    import_reflectance_absorption_from_csv,
)

# =============================================================================
# APPLICATION OPERATIONS
# =============================================================================
from services.app_operations import (
    sanitize_filename_component,
    generate_application_report,
    generate_full_report,
    rebuild_application_cache,
    generate_tsv_for_download,
)
