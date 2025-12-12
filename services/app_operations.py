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
- generate_full_report(): Creates both PNG and TSV reports with matching filenames
- generate_tsv_for_download(): Creates TSV export files for filter stacks

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
# Standard library imports
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING

# Third-party imports
import numpy as np
import pandas as pd
import streamlit as st

# Local imports
from models.core import FilterCollection, ReflectorCollection
from models.constants import INTERP_GRID, CACHE_DIR
from services.data import (
    load_filter_collection,
    load_quantum_efficiencies, 
    load_illuminant_collection,
    load_reflector_collection
)
from views.ui_utils import try_operation, handle_error
from services.calculations import (
    compute_filter_transmission,
    compute_selected_filter_indices,
    compute_effective_stops,
    compute_rgb_response,
    compute_white_balance_gains
)
from services.visualization import (
    add_filter_curve_to_matplotlib, generate_report_png_v2,
    create_report_config, create_filter_data, create_computation_functions, create_sensor_data
)

# Type checking imports
if TYPE_CHECKING:
    from services.state_manager import StateManager


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
    transmission, transmission_label, combined_transmission = compute_filter_transmission(selected_indices, filter_collection.filter_matrix)
    
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
        # Use raw Green channel QE for effective stops calculation
        sensor_qe = app_state.current_qe.get('G', None) if app_state.current_qe else None
    
    # Get illuminant
    illuminant = (app_state.illuminant if app_state.illuminant is not None 
                 else np.ones_like(INTERP_GRID))
    
    # Compute effective stops
    effective_stops_fn = lambda t, qe, illum: compute_effective_stops(t, qe, illum) if qe is not None else (0.0, 0.0)
    
    # Compute white balance
    white_balance_fn = lambda t, qe, illum: (
        compute_white_balance_gains(t, qe, illum) if qe is not None 
        else {"R": 1.0, "G": 1.0, "B": 1.0}
    )
    
    # Generate report using new data class structure (simplified version)
    report_config = create_report_config(
        selected_filters=app_state.selected_filters,
        current_qe=app_state.current_qe,
        camera_name=selected_camera or "UnknownCamera",
        illuminant_name=app_state.illuminant_name or "UnknownIlluminant",
        illuminant_curve=illuminant
    )
    
    filter_data = create_filter_data(
        filter_matrix=filter_collection.filter_matrix,
        df=filter_collection.df,
        display_to_index=filter_collection.get_display_to_index_map(),
        masks=filter_collection.extrapolated_masks,
        interp_grid=INTERP_GRID
    )
    
    computation_fns = create_computation_functions(
        compute_selected_indices_fn=lambda sel: selected_indices,
        compute_filter_transmission_fn=lambda idxs: compute_filter_transmission(
            idxs, filter_collection.filter_matrix
        ),
        compute_effective_stops_fn=effective_stops_fn,
        compute_white_balance_gains_fn=white_balance_fn,
        add_curve_fn=add_filter_curve_to_matplotlib,
        sanitize_fn=sanitize_filename_component
    )
    
    sensor_data = create_sensor_data(sensor_qe=sensor_qe)
    
    # Use new simplified interface (4 parameters instead of 17)
    result = generate_report_png_v2(
        report_config=report_config,
        filter_data=filter_data,
        computation_fns=computation_fns,
        sensor_data=sensor_data
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


def _create_tsv_data(
    app_state: "StateManager",
    filter_collection: FilterCollection
) -> Optional[str]:
    """
    Create TSV data for filter stack export.
    
    Args:
        app_state: Current application state
        filter_collection: Available filters
        
    Returns:
        TSV content string or None if export fails
    """
    # Get selected filter indices
    selected_indices = compute_selected_filter_indices(
        app_state.selected_filters,
        app_state.filter_multipliers,
        filter_collection
    )
    
    if not selected_indices:
        return None
    
    # Get filter transmission data
    transmission, _, combined_transmission = compute_filter_transmission(
        selected_indices,
        filter_collection.filter_matrix
    )
    
    if transmission is None:
        return None
    
    # Use combined transmission if available, otherwise single filter transmission
    active_transmission = combined_transmission if combined_transmission is not None else transmission
    
    # Build metadata for the combined filter stack
    counts = {}
    for filter_name in app_state.selected_filters:
        if filter_name in filter_collection.get_display_to_index_map():
            idx = filter_collection.get_display_to_index_map()[filter_name]
            filter_row = filter_collection.df.iloc[idx]
            
            key = (filter_row['Manufacturer'], filter_row['Filter Number'], filter_row['Filter Name'])
            multiplier = app_state.filter_multipliers.get(filter_name, 1)
            counts[key] = counts.get(key, 0) + multiplier
    
    # Create metadata string with "+" separator
    metadata_parts = []
    for (manufacturer, filter_number, filter_name), count in counts.items():
        if count > 1:
            part = f"{manufacturer} {filter_number} ({filter_name}) x{count}"
        else:
            part = f"{manufacturer} {filter_number} ({filter_name})"
        metadata_parts.append(part)
    
    combined_metadata = " + ".join(metadata_parts)
    
    # Get the first filter's hex color
    first_filter_idx = selected_indices[0]
    first_filter_row = filter_collection.df.iloc[first_filter_idx]
    hex_color = first_filter_row.get('Hex Color', '#808080')
    
    # Filter to only include wavelengths where we have actual filter data
    # The interpolation function fills values outside filter range with NaN
    # So we only include wavelengths where transmission is not NaN (indicating actual filter coverage)
    transmission_percentage = active_transmission * 100
    
    # Create mask for wavelengths with actual filter data (not NaN)
    meaningful_data_mask = ~np.isnan(active_transmission)
    
    # If no valid data found, include all data (fallback - shouldn't happen)
    if not np.any(meaningful_data_mask):
        meaningful_data_mask = np.ones_like(active_transmission, dtype=bool)
    
    # Create filtered arrays only for meaningful wavelengths
    valid_wavelengths = INTERP_GRID[meaningful_data_mask].astype(int)
    valid_transmission = np.round(transmission_percentage[meaningful_data_mask], 3)
    
    # Build TSV content manually to avoid pandas formatting issues
    num_rows = len(valid_wavelengths)
    header = "Wavelength\tTransmittance\thex_color\tManufacturer\tName\tFilter Number"
    
    # Build data rows
    rows = [header]
    for i in range(num_rows):
        wavelength = valid_wavelengths[i]
        transmission = valid_transmission[i]
        color = hex_color if i == 0 else ""
        manufacturer = combined_metadata if i == 0 else ""
        name = "Combined Filter Stack" if i == 0 else ""
        filter_num = f"STACK_{len(selected_indices)}" if i == 0 else ""
        
        row = f"{wavelength}\t{transmission}\t{color}\t{manufacturer}\t{name}\t{filter_num}"
        rows.append(row)
    
    tsv_content = "\n".join(rows)
    
    return tsv_content


def generate_tsv_for_download(
    app_state: "StateManager",
    filter_collection: FilterCollection
) -> bool:
    """
    Generate TSV data and prepare it for download (does not save to disk).
    
    Args:
        app_state: Current application state
        filter_collection: Available filters
        
    Returns:
        True if generation was successful, False otherwise
    """
    tsv_content = _create_tsv_data(app_state, filter_collection)
    if not tsv_content:
        return False
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"FilterStack_{timestamp}.tsv"
    app_state.last_tsv_export = {
        'bytes': tsv_content.encode('utf-8'),
        'name': filename,
        'timestamp': timestamp
    }
    
    return True


def generate_full_report(
    app_state: "StateManager",
    filter_collection: FilterCollection,
    selected_camera: Optional[str] = None
) -> bool:
    """
    Generate both PNG and TSV reports and save them to the output folder with matching filter names.
    
    Args:
        app_state: Current application state
        filter_collection: Available filters
        selected_camera: Name of selected camera (optional)
        
    Returns:
        True if at least one report was generated successfully, False otherwise
    """
    png_success = False
    tsv_success = False
    
    # Generate PNG report (this already saves to output folder with filter-based name)
    if generate_application_report(app_state, filter_collection, selected_camera):
        if app_state.last_export and app_state.last_export.get("name"):
            png_success = True
            
            # Extract base name from PNG (remove .png extension) for TSV
            png_name = app_state.last_export["name"]
            base_filename = png_name.replace(".png", "")
            
            # Generate TSV export with same base name
            tsv_content = _create_tsv_data(app_state, filter_collection)
            if tsv_content:
                # Create output directory (same as PNG)
                camera_name = selected_camera or "UnknownCamera"
                illuminant_name = app_state.illuminant_name or "UnknownIlluminant"
                output_dir = Path("output") / sanitize_filename_component(camera_name) / sanitize_filename_component(illuminant_name)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save TSV to output folder with same base name as PNG
                tsv_path = output_dir / f"{base_filename}.tsv"
                with open(tsv_path, "w", encoding='utf-8') as f:
                    f.write(tsv_content)
                
                # Also store for download
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                app_state.last_tsv_export = {
                    'bytes': tsv_content.encode('utf-8'),
                    'name': f"{base_filename}.tsv",
                    'timestamp': timestamp
                }
                tsv_success = True
    
    return png_success or tsv_success
