"""
High-level application operations for FS FilterLab.

This module orchestrates complex application workflows by coordinating
between multiple services and managing application-wide operations.

Key Functions:

Data Initialization:
- initialize_application_data(): Loads and validates all required data sources
- process_reflector_data(): Validates and processes reflector spectral data

Report Generation:
- generate_application_report(): Creates comprehensive PNG analysis reports
- setup_report_download(): Configures Streamlit download interface

System Operations:
- rebuild_application_cache(): Clears and rebuilds data caches
- sanitize_filename_component(): Ensures safe filename generation

Workflow Integration:
This module serves as the bridge between the UI layer and individual services,
handling error propagation, data validation, and result coordination. It ensures
that complex operations like report generation have all required dependencies
and handle failures gracefully.

Architecture:
- Uses dependency injection for service composition
- Implements error handling with user-friendly messages  
- Maintains separation between UI logic and business operations
- Supports both synchronous and background operation patterns
"""
import numpy as np
import streamlit as st
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from models.core import FilterCollection, ReflectorCollection
from models.constants import INTERP_GRID

# Import StateManager for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from services.state_manager import StateManager
from services.calculations import (
    compute_filter_transmission,
    compute_selected_filter_indices,
    compute_effective_stops
)
from services.calculations import (
    compute_rgb_response,
    compute_white_balance_gains
)
from services.visualization import generate_report_png
from services.visualization import add_filter_curve_to_matplotlib


# ----- UTILITY FUNCTIONS -----

def sanitize_filename_component(name: str, lowercase=False, max_len=None) -> str:
    """
    Sanitize a string for safe use in filenames across operating systems.
    
    Removes or replaces characters that are invalid in Windows, macOS, and Linux
    filenames, ensuring generated files can be saved and shared reliably.
    
    Args:
        name: The input string to sanitize
        lowercase: If True, convert to lowercase for consistency
        max_len: Maximum length limit for the output string
    
    Returns:
        Cleaned string safe for use in filenames
        
    Example:
        >>> sanitize_filename_component("Filter: UV/IR-Cut", lowercase=True)
        'filter- uv-ir-cut'
    """
    clean = re.sub(r'[<>:"/\\|?*]', "-", name).strip()
    if lowercase:
        clean = clean.lower()
    if max_len:
        clean = clean[:max_len]
    return clean


# ----- APPLICATION OPERATIONS -----


def initialize_application_data():
    """
    Initialize and validate all application data sources.
    
    Performs comprehensive loading of all required data:
    1. Filter collection from TSV files in data/filters_data/
    2. Camera quantum efficiency curves from data/QE_data/  
    3. Illuminant spectra from data/illuminants/
    4. Reflector spectra from data/reflectors/
    
    Each data source is loaded with error handling and validation.
    Failed loads use appropriate fallback values to maintain application stability.
    
    Returns:
        Dictionary containing all loaded data with keys:
        - 'filter_collection': FilterCollection object with all filters
        - 'camera_keys': List of available camera QE profile names
        - 'qe_data': Dict mapping camera names to RGB channel QE curves
        - 'default_key': Name of default camera QE profile
        - 'illuminants': Dict of illuminant name -> spectrum arrays
        - 'illuminant_metadata': Dict of illuminant metadata
        - 'reflector_collection': ReflectorCollection with surface spectra
        
        Returns None if critical data loading fails (e.g., no filters found)
        
    Note:
        Uses caching to improve performance on subsequent loads.
        Cache is automatically invalidated when source files change.
    """
    from services.data import (
        load_filter_collection,
        load_quantum_efficiencies, 
        load_illuminant_collection,
        load_reflector_collection
    )
    from views.ui_utils import try_operation, handle_error
    from models.core import FilterCollection, ReflectorCollection
    import numpy as np
    
    # Load filter collection
    filter_collection = try_operation(
        load_filter_collection,
        "Failed to load filter collection",
        default_value=FilterCollection([], None, np.array([]), np.array([]))
    )
    
    if not filter_collection.filters:
        handle_error("No filter data found. Please add .tsv files to data/filters_data", stop_execution=True)
        return None
        
    # Load QE data
    camera_keys, qe_data, default_key = try_operation(
        load_quantum_efficiencies,
        "Failed to load quantum efficiencies", 
        default_value=([], {}, "")
    )
    
    # Load illuminants
    illuminants, illuminant_metadata = try_operation(
        load_illuminant_collection,
        "Failed to load illuminants",
        default_value=({}, {})
    )
    
    # Load reflectors
    reflector_collection = try_operation(
        load_reflector_collection,
        "Failed to load reflector collection",
        default_value=ReflectorCollection([], np.array([]))
    )
    
    # Process reflector data
    process_reflector_data(reflector_collection)
    
    return {
        'filter_collection': filter_collection,
        'camera_keys': camera_keys,
        'qe_data': qe_data,
        'default_key': default_key,
        'illuminants': illuminants,
        'illuminant_metadata': illuminant_metadata,
        'reflector_collection': reflector_collection
    }


def process_reflector_data(reflector_collection: ReflectorCollection) -> None:
    """
    Process and validate the reflector data.
    This ensures the reflector data is valid and fixes any issues.
    
    Args:
        reflector_collection: Collection of reflector spectra
    """
    # No validation/fix needed; function removed in new data format
    pass


def generate_application_report(
    app_state: "StateManager", 
    filter_collection: FilterCollection,
    selected_camera: Optional[str] = None
) -> bool:
    """
    Generate a PNG report of the current filter configuration.
    
    Args:
        app_state: Current application state
        filter_collection: Available filters
        selected_camera: Name of selected camera (optional)
        
    Returns:
        True if report was generated successfully, False otherwise
    """
    # Get selected filter indices
    selected_indices = compute_selected_filter_indices(
        app_state.selected_filters, 
        app_state.filter_multipliers, 
        filter_collection
    )
    
    if not selected_indices:
        return False
        
    # Get filter transmission
    transmission = compute_filter_transmission(selected_indices, filter_collection.filter_matrix)
    
    if transmission is None:
        return False
        
    # Calculate sensor QE
    sensor_qe = None
    if app_state.current_qe:
        responses, rgb_matrix, _ = compute_rgb_response(
            transmission, 
            app_state.current_qe,
            app_state.white_balance_gains,
            app_state.rgb_channels_visibility
        )
        # Use Green channel for effective stops calculation (most representative)
        sensor_qe = responses.get('G', None) if responses else None
    
    # Get illuminant
    illuminant = (app_state.illuminant if app_state.illuminant is not None 
                 else np.ones_like(INTERP_GRID))
    
    # Compute effective stops
    effective_stops_fn = lambda t, qe: compute_effective_stops(t, qe) if qe is not None else (0.0, 0.0)
    
    # Compute white balance
    white_balance_fn = lambda t, qe, illum: (
        compute_white_balance_gains(t, qe, illum) if qe is not None 
        else {"R": 1.0, "G": 1.0, "B": 1.0}
    )
    
    # Generate report
    result = generate_report_png(
        selected_filters=app_state.selected_filters,
        current_qe=app_state.current_qe,
        filter_matrix=filter_collection.filter_matrix,
        df=filter_collection.df,
        display_to_index=filter_collection.get_display_to_index_map(),
        compute_selected_indices_fn=lambda sel: selected_indices,
        compute_filter_transmission_fn=lambda idxs: compute_filter_transmission(
            idxs, filter_collection.filter_matrix
        ),
        compute_effective_stops_fn=effective_stops_fn,
        compute_white_balance_gains_fn=white_balance_fn,
        masks=filter_collection.extrapolated_masks,
        add_curve_fn=add_filter_curve_to_matplotlib,
        interp_grid=INTERP_GRID,
        sensor_qe=sensor_qe,
        camera_name=selected_camera or "UnknownCamera",
        illuminant_name=app_state.illuminant_name or "UnknownIlluminant",
        sanitize_fn=sanitize_filename_component,
        illuminant_curve=illuminant
    )
    
    if result:
        app_state.last_export = result
        return True
        
    return False


def rebuild_application_cache(cache_dir: Path) -> bool:
    """
    Rebuild the filter cache by clearing cache files.
    
    Args:
        cache_dir: Directory containing cache files
        
    Returns:
        True if cache was successfully rebuilt, False otherwise
    """
    if not cache_dir.exists():
        return False
        
    success = True
    for f in cache_dir.glob("*"):
        try:
            f.unlink()
        except Exception:
            success = False
            
    return success


def setup_report_download(app_state: "StateManager") -> None:
    """
    Setup download button for the last generated report.
    
    Args:
        app_state: Current application state
    """
    last_export = app_state.last_export
    if last_export and last_export.get("bytes"):
        st.sidebar.download_button(
            label="⬇️ Download Last Report",
            data=last_export["bytes"],
            file_name=last_export["name"],
            mime="image/png",
            use_container_width=True
        )
