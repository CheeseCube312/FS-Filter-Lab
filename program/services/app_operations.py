"""
High-level application operations for FS FilterLab.

This module orchestrates complex application workflows by coordinating
between multiple services and managing application-wide operations.

Key Functions:

Data Initialization:
- initialize_application_data(): Loads and validates all required data sources

Report Generation:
- generate_application_report(): Creates comprehensive PNG analysis reports

System Operations:
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
# Standard library imports
import logging
import re
from typing import Any, Dict, List, Optional

# Third-party imports
import numpy as np

# Local imports
from models.core import AppData, FilterCollection, ReflectorCollection
from models.constants import INTERP_GRID, DATA_FOLDERS
from services.data import (
    load_filter_collection,
    load_quantum_efficiencies, 
    load_illuminant_collection,
    load_reflector_collection,
    load_film_profiles,
    build_sensor_profiles,
    create_empty_filter_collection,
    create_empty_reflector_collection
)

logger = logging.getLogger(__name__)


def _try_operation(operation, error_message, default_value=None):
    """Execute an operation with error handling via logging.
    
    Pure Python wrapper for safe operation execution with error handling.
    Services use this to safely execute operations without UI dependencies.
    """
    try:
        return operation()
    except Exception as e:
        logger.error(f"{error_message}: {e}")
        return default_value
from services.calculations import (
    compute_filter_transmission,
    compute_selected_filter_indices,
    compute_effective_stops,
    compute_white_balance_gains
)
from services.visualization import generate_report_png
from services.state_manager import AppState


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
    1. Filter collection from TSV files in program/data/filters_data/
    2. Camera quantum efficiency curves from program/data/QE_data/  
    3. Illuminant spectra from program/data/illuminants/
    4. Reflector spectra from program/data/reflectors/
    
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
    # Load filter collection
    filter_collection = _try_operation(
        load_filter_collection,
        "Failed to load filter collection",
        default_value=create_empty_filter_collection()
    )
    
    if not filter_collection.filters:
        logger.error(f"No filter data found. Please add .tsv files to {DATA_FOLDERS['filters']}")
        return None
        
    # Load QE data
    camera_keys, qe_data, default_key = _try_operation(
        load_quantum_efficiencies,
        "Failed to load quantum efficiencies", 
        default_value=([], {}, "")
    )
    
    # Load illuminants
    illuminants, illuminant_metadata = _try_operation(
        load_illuminant_collection,
        "Failed to load illuminants",
        default_value=({}, {})
    )
    
    # Load reflectors
    reflector_collection = _try_operation(
        load_reflector_collection,
        "Failed to load reflector collection",
        default_value=create_empty_reflector_collection()
    )

    # Load film profiles and merge with QE data into unified sensor_profiles
    film_profiles = _try_operation(
        load_film_profiles,
        "Failed to load film profiles",
        default_value={}
    )
    sensor_profiles = build_sensor_profiles(qe_data, film_profiles)

    return AppData(
        filter_collection=filter_collection,
        camera_keys=camera_keys,
        qe_data=qe_data,
        default_key=default_key,
        illuminants=illuminants,
        illuminant_metadata=illuminant_metadata,
        reflector_collection=reflector_collection,
        sensor_profiles=sensor_profiles,
    )


def generate_application_report(
    app_state: AppState,
    filter_collection: FilterCollection,
    selected_camera: Optional[str] = None,
) -> Optional[Dict]:
    """
    Generate a PNG report of the current filter configuration.
    
    Args:
        app_state: Current application state
        filter_collection: Available filters
        selected_camera: Name of selected camera (optional)
        
    Returns:
        Report result dict if successful, None otherwise.
    """
    # Get selected filter indices
    selected_indices = compute_selected_filter_indices(
        app_state.selected_filters, 
        app_state.filter_multipliers, 
        filter_collection
    )
    
    if not selected_indices:
        return None
        
    # Get filter transmission
    transmission, label, combined = compute_filter_transmission(
        selected_indices, filter_collection.filter_matrix
    )
    if transmission is None:
        return None

    active_trans = combined if combined is not None else transmission
    
    # Resolve sensor QE and illuminant
    sensor_qe = (
        app_state.current_qe.get('G') if app_state.current_qe else None
    )
    illuminant = (
        app_state.illuminant if app_state.illuminant is not None
        else np.ones_like(INTERP_GRID)
    )
    
    # Compute metrics
    avg_trans, stops = (
        compute_effective_stops(active_trans, sensor_qe, illuminant)
        if sensor_qe is not None else (0.0, 0.0)
    )
    wb = (
        compute_white_balance_gains(active_trans, app_state.current_qe, illuminant)
        if app_state.current_qe else {"R": 1.0, "G": 1.0, "B": 1.0}
    )
    
    camera_name = selected_camera or "UnknownCamera"
    illuminant_name = app_state.illuminant_name or "UnknownIlluminant"
    
    result = generate_report_png(
        selected_filters=app_state.selected_filters,
        selected_indices=selected_indices,
        active_transmission=active_trans,
        transmission_label=label,
        combined_transmission=combined,
        effective_stops=stops,
        avg_transmission=avg_trans,
        white_balance_gains=wb,
        current_qe=app_state.current_qe,
        sensor_qe=sensor_qe,
        camera_name=camera_name,
        illuminant_name=illuminant_name,
        filter_df=filter_collection.df,
        display_to_index=filter_collection.get_display_to_index_map(),
        filter_matrix=filter_collection.filter_matrix,
        masks=filter_collection.extrapolated_masks,
        interp_grid=INTERP_GRID,
        sanitize_fn=sanitize_filename_component,
    )

    return result or None
