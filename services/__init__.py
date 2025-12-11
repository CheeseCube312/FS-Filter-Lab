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

Service Categories:

Data Processing:
- Filter transmission calculations and combination
- RGB sensor response computations  
- White balance and color correction
- Deviation metrics and statistical analysis

Visualization:
- Interactive Plotly charts for real-time analysis
- Static Matplotlib reports for documentation
- Sparkline plots and color swatches
- Multi-panel technical reports

File Operations:
- TSV/CSV import with format auto-detection
- Spectral data interpolation and extrapolation
- Cache management for performance optimization
- PNG report generation and download

The services layer maintains separation of concerns by keeping UI logic in views
and data models in the models package, while providing all business logic here.
"""
# Import general utilities
from services.app_operations import (
    sanitize_filename_component
)

# Import transmission services
from services.calculations import (
    # Calculation functions
    compute_combined_transmission,
    compute_filter_transmission,
    compute_active_transmission,
    compute_effective_stops,
    calculate_transmission_deviation_metrics,
    
    # Formatting functions
    format_transmission_metrics,
    format_deviation_metrics,
    format_white_balance_data,
    
    # Selection utilities
    compute_selected_filter_indices
)

# Import color processing services
from services.calculations import (
    normalize_pixels,
    compute_rgb_response,
    compute_white_balance_gains,
    compute_reflector_color,
    compute_reflector_preview_colors,
    find_vegetation_preview_reflectors,
    is_reflector_data_valid,
    check_reflector_wavelength_validity
)

# Import data loading services
from services.data import (
    load_filter_collection,
    load_quantum_efficiencies, 
    load_illuminant_collection,
    load_reflector_collection
)

# Import plotting services
from services.visualization import (
    create_filter_response_plot,
    create_sensor_response_plot,
    create_sparkline_plot,
    create_qe_figure,
    create_illuminant_figure,
    add_filter_curve_to_plotly,
    add_filter_curve_to_matplotlib
)

# Import importing services
from services.importing import (
    import_filter_from_csv,
    import_illuminant_from_csv,
    import_qe_from_csv,
    import_reflectance_absorption_from_csv
)

# Import reporting services
from services.visualization import (
    generate_report_png
)

# Import application operations
from services.app_operations import (
    process_reflector_data,
    generate_application_report,
    generate_full_report,
    rebuild_application_cache,
    generate_tsv_for_download
)
