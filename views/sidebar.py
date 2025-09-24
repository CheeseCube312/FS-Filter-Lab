"""
Sidebar UI components for FS FilterLab.

This module provides UI components for the sidebar, including filter selection,
filter multipliers, and extras (illuminant, QE, target).
"""
import streamlit as st
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from models.core import FilterCollection, TargetProfile, ReflectorCollection
from models.constants import CACHE_DIR
from services.app_operations import setup_report_download
from views.ui_utils import try_operation, handle_error


def filter_selection(
    filter_collection: FilterCollection, 
    app_state
) -> List[str]:
    """
    UI component for filter selection in the sidebar.
    
    Args:
        filter_collection: Collection of available filters
        state_manager: Unified state manager
    
    Returns:
        List of selected filter display names
    """
    # --- Prepare default value for widget (including pending selections) ---
    current_selection = app_state.selected_filters
    if "_pending_selected_filters" in st.session_state:
        pending = st.session_state.pop("_pending_selected_filters")
        current_selection = list(set(current_selection + pending))

    # --- Widget logic ---
    filter_display_names = filter_collection.get_display_names()
    all_options = sorted(set(filter_display_names) | set(current_selection))

    selected = st.sidebar.multiselect(
        "Select filters to plot",
        options=all_options,
        default=current_selection,
        key="selected_filters",
    )

    # --- Advanced search toggle ---
    show_advanced_search = st.sidebar.checkbox(
        "Show Advanced Search", 
        key="show_advanced_search"
    )

    return selected


def filter_multipliers(selected: List[str]) -> Dict[str, int]:
    """
    UI component for filter multiplier controls in the sidebar.
    
    Args:
        selected: List of selected filter display names
    
    Returns:
        Dictionary mapping filter names to their multiplier counts
    """
    filter_multipliers = {}
    if selected:
        with st.sidebar.expander("Set Filter Stack Counts", expanded=False):
            for name in selected:
                filter_multipliers[name] = st.number_input(
                    f"{name}",
                    min_value=1,
                    max_value=5,
                    value=st.session_state.get(f"mult_{name}", 1),
                    step=1,
                    key=f"mult_{name}"
                )
    return filter_multipliers


def extras(
    illuminants: Dict[str, np.ndarray],
    illuminant_metadata: Dict[str, str],
    camera_keys: List[str],
    qe_data: Dict[str, Dict[str, np.ndarray]],
    default_camera_key: str,
    filter_collection: FilterCollection,
    reflector_collection: Any
) -> Tuple[Optional[str], Optional[np.ndarray], Optional[Dict[str, np.ndarray]], Optional[str], Optional[TargetProfile], Optional[int]]:
    """
    UI component for extras (illuminant, QE, target) in the sidebar.
    
    Args:
        illuminants: Dictionary of illuminant curves by name
        illuminant_metadata: Dictionary of illuminant descriptions by name
        camera_keys: List of camera keys
        qe_data: Dictionary of QE data by camera key
        default_camera_key: Default camera key
        filter_collection: Collection of available filters
        reflector_collection: Collection of available reflectors
    
    Returns:
        Tuple of (illuminant_name, illuminant, qe, camera_name, target_profile, selected_reflector_idx)
    """
    with st.sidebar.expander("Extras", expanded=False):
        # --- Illuminant Selector ---
        if illuminants:
            illum_names = list(illuminants.keys())
            from models.constants import DEFAULT_ILLUMINANT
            default_idx = illum_names.index(DEFAULT_ILLUMINANT) if DEFAULT_ILLUMINANT in illum_names else 0
            selected_illum_name = st.selectbox("Scene Illuminant", illum_names, index=default_idx)
            selected_illum = illuminants[selected_illum_name]
        else:
            handle_error("⚠️ No illuminants found.")
            selected_illum_name, selected_illum = None, None

        # --- QE Profile Selector ---
        default_idx = camera_keys.index(default_camera_key) + 1 if default_camera_key in camera_keys else 0
        selected_camera = st.selectbox("Sensor QE Profile", ["None"] + camera_keys, index=default_idx)
        current_qe = qe_data.get(selected_camera) if selected_camera != "None" else None

        # --- Target Profile Selector ---
        filter_display_names = filter_collection.get_display_names()
        display_to_index = filter_collection.get_display_to_index_map()
        
        target_options = ["None"] + list(filter_display_names)
        default_target = "None"
        target_selection = st.selectbox(
            "Reference Target",
            options=target_options,
            index=target_options.index(default_target),
            key="target_profile_selection"
        )

        target_profile = None
        if target_selection != "None":
            target_index = display_to_index[target_selection]
            filter_obj = filter_collection.filters[target_index]
            
            target_profile = TargetProfile(
                name=filter_obj.name,
                values=filter_obj.transmission * 100,  # Convert to percentage
                valid=~np.isnan(filter_obj.transmission)
            )

        # --- Reflector Spectrum Selector ---
        selected_reflector_idx = None
        if reflector_collection and hasattr(reflector_collection, "reflectors") and reflector_collection.reflectors:
            reflector_names = [r.name for r in reflector_collection.reflectors]
            selected_reflector_idx = st.selectbox(
                "Surface Reflectance Spectrum",
                options=list(range(len(reflector_names))),
                format_func=lambda i: reflector_names[i],
                index=0,
                key="selected_reflector_idx"
            )
        else:
            st.info("No reflectance spectra found.")

    return selected_illum_name, selected_illum, current_qe, selected_camera, target_profile, selected_reflector_idx


def settings_panel(app_state) -> Tuple[bool, bool, Dict[str, bool]]:
    """
    Settings panel in the sidebar.
    
    Args:
        state_manager: Unified state manager
        
    Returns:
        Tuple of (rebuild_cache, show_import, rgb_channels)
    """
    with st.sidebar.expander("Settings", expanded=False):
        # RGB channel toggles
        st.markdown("**Sensor-Weighted Response Channels**")
        rgb_channels = {}
        for channel in ["R", "G", "B"]:
            rgb_channels[channel] = st.checkbox(
                f"{channel} Channel", 
                key=f"show_{channel}",
                value=True  # Default to True for first run
            )
        
        # Display options
        st.markdown("**Display Options**")
        log_view = st.checkbox(
            "Show stop-view (logarithmic)", 
            help="Display transmission in camera stops (logarithmic scale) instead of percentage",
            key="sidebar_log_view_toggle"
        )
        
        # Update log_view state directly from widget
        # No manual state management needed - use st.session_state["sidebar_log_view_toggle"] instead
        
        # Rebuild Cache button
        if st.button("🔄 Rebuild Filter Cache"):
            return True, False, rgb_channels
            
        # Import data button
        if st.button("WebPlotDigitizer .csv importers"):
            return False, True, rgb_channels
            
    return False, False, rgb_channels


def reflector_preview(pixels: np.ndarray, reflector_names: Optional[List[str]] = None) -> None:
    """
    Display reflector color preview in the sidebar.
    
    Args:
        pixels: Array of RGB pixel values, shape (n, m, 3)
        reflector_names: List of reflector names (optional)
    """
    # Ensure pixels is a valid RGB array
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        handle_error("Invalid pixel array format")
        return
    
    # Display in the sidebar
    st.sidebar.subheader("Vegetation Color Preview")
    
    # Normalize pixels for display (RGB values need to be in [0.0, 1.0] range)
    max_val = np.max(pixels)
    if max_val > 0:
        pixels_normalized = np.clip(pixels / max_val, 0.0, 1.0)
    else:
        pixels_normalized = pixels
    
    # Display the image in the sidebar
    st.sidebar.image(pixels_normalized, width=300, channels="RGB", output_format="PNG")


def single_reflector_preview(
    pixel_color: np.ndarray, 
    reflector_name: str,
    global_max: float = None
) -> None:
    """
    Display single reflector color preview in the sidebar.
    
    Args:
        pixel_color: Single RGB color as 1x1x3 array
        reflector_name: Name of the reflector
        global_max: Global maximum value for consistent scaling (optional)
    """
    # Ensure pixel_color is a valid RGB array
    if pixel_color.ndim != 3 or pixel_color.shape[2] != 3:
        handle_error("Invalid pixel color format")
        return
    
    # Display in the sidebar
    st.sidebar.subheader("Surface Preview")
    
    # Use global max if provided, otherwise normalize individually
    max_val = global_max if global_max is not None and global_max > 0 else np.max(pixel_color)
    
    # Normalize pixel for display (RGB values need to be in [0.0, 1.0] range)
    if max_val > 0:
        pixel_normalized = np.clip(pixel_color / max_val, 0.0, 1.0)
    else:
        pixel_normalized = pixel_color
    
    # Display the single color as a larger image
    st.sidebar.image(pixel_normalized, width=200, channels="RGB", output_format="PNG")
    st.sidebar.caption(f"Selected: {reflector_name}")


def render_sidebar(app_state, data):
    """
    Render the complete sidebar with all controls.
    
    Args:
        app_state: Application state object
        data: Dictionary containing loaded application data
        
    Returns:
        Dictionary containing sidebar actions to be processed
    """
    import streamlit as st
    from views.ui_utils import try_operation, handle_error
    from services.app_operations import generate_application_report, setup_report_download, rebuild_application_cache
    from models.constants import CACHE_DIR
    
    st.sidebar.header("Filter Plotter")
    
    # Extract data
    filter_collection = data['filter_collection'] 
    camera_keys = data['camera_keys']
    qe_data = data['qe_data']
    default_key = data['default_key']
    illuminants = data['illuminants']
    illuminant_metadata = data['illuminant_metadata']
    reflector_collection = data['reflector_collection']
    
    # Filter selection (using existing functions from this module)
    selected_filters = filter_selection(filter_collection, app_state)
    filter_multipliers_dict = filter_multipliers(selected_filters)
    
    # Note: selected_filters is managed by the widget, no need to set app_state.selected_filters
    # Update only the filter multipliers (not managed by a widget)
    app_state.filter_multipliers = filter_multipliers_dict
    
    # QE, Illuminant, Target selection (using existing function from this module)
    selected_illum_name, selected_illum, current_qe, selected_camera, target_profile, selected_reflector_idx = extras(
        illuminants, illuminant_metadata,
        camera_keys, qe_data, default_key,
        filter_collection,
        reflector_collection
    )
    
    # Update QE and illuminant in state directly
    app_state.current_qe = current_qe
    app_state.selected_camera = selected_camera
    app_state.illuminant = selected_illum
    app_state.illuminant_name = selected_illum_name
    app_state.target_profile = target_profile
    # Note: selected_reflector_idx is stored in session state by the widget, no need to store in app_state
    
    # Collect actions to return
    actions = {}
    
    # Report generation
    if st.sidebar.button("📄 Generate Report (PNG)"):
        actions['generate_report'] = selected_camera
    
    # Show download button if a report is ready
    setup_report_download(app_state)
    
    # Settings Panel (using existing function from this module)
    rebuild_cache, show_import, rgb_channels = settings_panel(app_state)
    
    # No need to manually update RGB channels - they're widget-controlled now
    app_state.show_import_data = show_import
    
    if rebuild_cache:
        actions['rebuild_cache'] = True
        
    return actions
