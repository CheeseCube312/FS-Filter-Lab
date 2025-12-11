"""
Sidebar UI components for FS FilterLab.

This module provides UI components for the sidebar, including filter selection,
filter multipliers, and extras (illuminant, QE, target).
"""
# Third-party imports
import streamlit as st
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# Local imports
from models.constants import (
    CACHE_DIR, DEFAULT_ILLUMINANT, UI_BUTTONS, UI_SECTIONS, UI_LABELS, 
    UI_INFO_MESSAGES, UI_WARNING_MESSAGES, UI_HELP_TEXT
)
from models.core import FilterCollection, TargetProfile, ReflectorCollection
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
    # Note: Pending selections are handled directly through session_state for widget compatibility

    # --- Widget logic ---
    filter_display_names = filter_collection.get_display_names()
    all_options = sorted(set(filter_display_names) | set(current_selection))

    selected = st.sidebar.multiselect(
        UI_LABELS['select_filters'],
        options=all_options,
        default=current_selection,
        key="selected_filters",
    )

<<<<<<< Updated upstream
    # --- Advanced search toggle ---
    show_advanced_search = st.sidebar.checkbox(
        UI_SECTIONS['show_advanced_search'], 
        key="show_advanced_search"
    )
    
    # --- Channel mixer toggle ---
    show_channel_mixer = st.sidebar.checkbox(
        UI_SECTIONS['show_channel_mixer'],
        key="show_channel_mixer",
        help=UI_HELP_TEXT['channel_mixer']
    )

=======
>>>>>>> Stashed changes
    return selected


def filter_multipliers(selected: List[str], app_state) -> Dict[str, int]:
    """
    UI component for filter multiplier controls in the sidebar.
    
    Args:
        selected: List of selected filter display names
        app_state: Application state manager
    
    Returns:
        Dictionary mapping filter names to their multiplier counts
    """
    filter_multipliers_dict = {}
    if selected:
        with st.sidebar.expander(UI_LABELS['set_filter_counts'], expanded=False):
            for name in selected:
                filter_multipliers_dict[name] = st.number_input(
                    f"{name}",
                    min_value=1,
                    max_value=5,
                    value=app_state.filter_multipliers.get(name, 1),
                    step=1,
                    key=f"mult_{name}"
                )
    return filter_multipliers_dict


def analysis_setup(
    illuminants: Dict[str, np.ndarray],
    illuminant_metadata: Dict[str, str],
    camera_keys: List[str],
    qe_data: Dict[str, Dict[str, np.ndarray]],
    default_camera_key: str,
    filter_collection: FilterCollection,
    reflector_collection: Any
) -> Tuple[Optional[str], Optional[np.ndarray], Optional[Dict[str, np.ndarray]], Optional[str], Optional[TargetProfile], Optional[int]]:
    """
    UI component for analysis setup (illuminant, QE, target) in the sidebar.
    
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
<<<<<<< Updated upstream
    with st.sidebar.expander(UI_SECTIONS['extras'], expanded=False):
        # --- Illuminant Selector ---
        if illuminants:
            illum_names = list(illuminants.keys())
            default_idx = illum_names.index(DEFAULT_ILLUMINANT) if DEFAULT_ILLUMINANT in illum_names else 0
            selected_illum_name = st.selectbox(UI_LABELS['scene_illuminant'], illum_names, index=default_idx)
            selected_illum = illuminants[selected_illum_name]
        else:
            handle_error(UI_WARNING_MESSAGES['no_illuminants'])
            selected_illum_name, selected_illum = None, None

        # --- QE Profile Selector ---
        default_idx = camera_keys.index(default_camera_key) + 1 if default_camera_key in camera_keys else 0
        selected_camera = st.selectbox(UI_LABELS['sensor_qe_profile'], ["None"] + camera_keys, index=default_idx)
        current_qe = qe_data.get(selected_camera) if selected_camera != "None" else None
=======
    # --- Illuminant Selector ---
    if illuminants:
        illum_names = list(illuminants.keys())
        default_idx = illum_names.index(DEFAULT_ILLUMINANT) if DEFAULT_ILLUMINANT in illum_names else 0
        selected_illum_name = st.selectbox(UI_LABELS['scene_illuminant'], illum_names, index=default_idx)
        selected_illum = illuminants[selected_illum_name]
    else:
        handle_error(UI_WARNING_MESSAGES['no_illuminants'])
        selected_illum_name, selected_illum = None, None

    # --- QE Profile Selector ---
    default_idx = camera_keys.index(default_camera_key) + 1 if default_camera_key in camera_keys else 0
    selected_camera = st.selectbox(UI_LABELS['sensor_qe_profile'], ["None"] + camera_keys, index=default_idx)
    current_qe = qe_data.get(selected_camera) if selected_camera != "None" else None
>>>>>>> Stashed changes

    # --- Target Profile Selector ---
    filter_display_names = filter_collection.get_display_names()
    display_to_index = filter_collection.get_display_to_index_map()
    
    target_options = ["None"] + list(filter_display_names)
    default_target = "None"
    target_selection = st.selectbox(
        UI_LABELS['reference_target'],
        options=target_options,
        index=target_options.index(default_target),
        key="target_profile_selection"
    )

    target_profile = None
    if target_selection != "None":
        target_index = display_to_index[target_selection]
        filter_obj = filter_collection.filters[target_index]
        
<<<<<<< Updated upstream
        target_options = ["None"] + list(filter_display_names)
        default_target = "None"
        target_selection = st.selectbox(
            UI_LABELS['reference_target'],
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
            # Add "None" option to allow hiding the single reflector preview
            options = ["None"] + list(range(len(reflector_names)))
            format_func = lambda idx: "None" if idx == "None" else reflector_names[idx]
            
            selection = st.selectbox(
                UI_LABELS['surface_reflectance'],
                options=options,
                format_func=format_func,
                index=0,  # Default to "None"
                key="selected_reflector_idx"
            )
            
            # Convert "None" selection to None, keep numeric indices as-is
            selected_reflector_idx = None if selection == "None" else selection
        else:
            from views.ui_utils import show_info_message
            show_info_message(UI_INFO_MESSAGES['no_reflectors'])
=======
        target_profile = TargetProfile(
            name=filter_obj.name,
            values=filter_obj.transmission * 100,  # Convert to percentage
            valid=~np.isnan(filter_obj.transmission)
        )

    # --- Reflector Spectrum Selector ---
    selected_reflector_idx = None
    if reflector_collection and hasattr(reflector_collection, "reflectors") and reflector_collection.reflectors:
        reflector_names = [r.name for r in reflector_collection.reflectors]
        # Add "None" option to allow hiding the single reflector preview
        options = ["None"] + list(range(len(reflector_names)))
        format_func = lambda idx: "None" if idx == "None" else reflector_names[idx]
        
        selection = st.selectbox(
            UI_LABELS['surface_reflectance'],
            options=options,
            format_func=format_func,
            index=0,  # Default to "None"
            key="selected_reflector_idx"
        )
        
        # Convert "None" selection to None, keep numeric indices as-is
        selected_reflector_idx = None if selection == "None" else selection
    else:
        from views.ui_utils import show_info_message
        show_info_message(UI_INFO_MESSAGES['no_reflectors'])
>>>>>>> Stashed changes

    return selected_illum_name, selected_illum, current_qe, selected_camera, target_profile, selected_reflector_idx


def settings_panel(app_state) -> Tuple[bool, bool, Dict[str, bool]]:
    """
    Settings panel in the sidebar.
    
    Args:
        state_manager: Unified state manager
        
    Returns:
        Tuple of (rebuild_cache, show_import, rgb_channels)
    """
    with st.sidebar.expander(UI_SECTIONS['settings'], expanded=False):
        # RGB channel toggles
        st.markdown(f"**{UI_SECTIONS['sensor_response_channels']}**")
        rgb_channels = {}
        for channel in ["R", "G", "B"]:
            rgb_channels[channel] = st.checkbox(
                f"{channel} Channel", 
                key=f"show_{channel}",
                value=True  # Default to True for first run
            )
        
        # Display options
        st.markdown(f"**{UI_SECTIONS['display_options']}**")
        log_view = st.checkbox(
            UI_LABELS['stop_view_toggle'], 
            help=UI_HELP_TEXT['stop_view'],
            key="sidebar_log_view_toggle"
        )
        
        # Log view state is managed by the widget and accessed through app_state.log_view
        
<<<<<<< Updated upstream
        # Rebuild Cache button
        rebuild_cache = st.button(UI_BUTTONS['rebuild_cache'])
            
        # Import data toggle button using app_state
        if app_state.show_import_data:
            if st.button(UI_BUTTONS['close_importers']):
                st.session_state['show_import_data'] = False
                st.rerun()
        else:
            if st.button(UI_BUTTONS['csv_importers']):
                st.session_state['show_import_data'] = True
                st.rerun()
        
        show_import = app_state.show_import_data
            
    return rebuild_cache, show_import, rgb_channels
=======
        show_import = app_state.show_import_data
            
    return False, show_import, rgb_channels
>>>>>>> Stashed changes


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
    st.sidebar.subheader(UI_SECTIONS['vegetation_preview'])
    
    # Use camera-realistic normalization with independent channel saturation
    from services.visualization import prepare_rgb_for_display
    pixels_normalized = prepare_rgb_for_display(pixels, auto_exposure=True)
    
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
    st.sidebar.subheader(UI_SECTIONS['surface_preview'])
    
    # Use camera-realistic normalization
    from services.visualization import prepare_rgb_for_display
    
    if global_max is not None and global_max > 0:
        # Use consistent scaling with vegetation preview
        # Apply the same exposure scaling, then camera-realistic saturation
        pixel_normalized = prepare_rgb_for_display(
            pixel_color, 
            saturation_level=global_max, 
            auto_exposure=False
        )
    else:
        # Independent normalization when no global reference
        pixel_normalized = prepare_rgb_for_display(pixel_color, auto_exposure=True)
    
    # Display the single color as a larger image
    st.sidebar.image(pixel_normalized, width=200, channels="RGB", output_format="PNG")
    st.sidebar.caption(f"Selected: {reflector_name}")


def render_sidebar(app_state, data):
    """
    Render the application sidebar with controls and settings.
<<<<<<< Updated upstream
    
    Args:
        app_state: Application state manager
        data: Application data dictionary
        
    Returns:
        Dictionary of user actions from sidebar interactions
    """
    """
    Render the complete sidebar with all controls.
=======
>>>>>>> Stashed changes
    
    Args:
        app_state: Application state manager
        data: Application data dictionary
        
    Returns:
        Dictionary of user actions from sidebar interactions
    """
    import streamlit as st
    
    from models.constants import CACHE_DIR
    from views.ui_utils import try_operation, handle_error
    from services.app_operations import (
        generate_application_report, generate_full_report, 
        rebuild_application_cache
    )
    
    st.sidebar.header(UI_SECTIONS['filter_plotter'])
    
    # Extract data
    filter_collection = data['filter_collection'] 
    camera_keys = data['camera_keys']
    qe_data = data['qe_data']
    default_key = data['default_key']
    illuminants = data['illuminants']
    illuminant_metadata = data['illuminant_metadata']
    reflector_collection = data['reflector_collection']
    
    # ========== 1. FILTER SELECTION & CONFIGURATION (Always visible) ==========
    selected_filters = filter_selection(filter_collection, app_state)
    filter_multipliers_dict = filter_multipliers(selected_filters, app_state)
<<<<<<< Updated upstream
    
    # Filter multipliers are not widget-controlled, so we update them in app_state
=======
>>>>>>> Stashed changes
    app_state.filter_multipliers = filter_multipliers_dict
    
    # ========== 2. ANALYSIS SETUP (Collapsed by default) ==========
    with st.sidebar.expander(UI_SECTIONS['analysis_setup'], expanded=False):
        selected_illum_name, selected_illum, current_qe, selected_camera, target_profile, selected_reflector_idx = analysis_setup(
            illuminants, illuminant_metadata,
            camera_keys, qe_data, default_key,
            filter_collection, reflector_collection
        )
    
    # Update state
    app_state.current_qe = current_qe
    app_state.selected_camera = selected_camera
    app_state.illuminant = selected_illum
    app_state.illuminant_name = selected_illum_name
    app_state.target_profile = target_profile
<<<<<<< Updated upstream
    # Reflector selection is widget-managed and accessed directly from session_state
=======
>>>>>>> Stashed changes
    
    # ========== 3. DISPLAY & VISUALIZATION (Collapsed by default) ==========
    with st.sidebar.expander(UI_SECTIONS['display_visualization'], expanded=False):
        # Advanced search toggle
        show_advanced_search = st.checkbox(
            UI_SECTIONS['show_advanced_search'], 
            key="show_advanced_search"
        )
        
        # Channel mixer toggle
        show_channel_mixer = st.checkbox(
            UI_SECTIONS['show_channel_mixer'],
            key="show_channel_mixer",
            help=UI_HELP_TEXT['channel_mixer']
        )
        
        # RGB channel toggles
        st.markdown(f"**{UI_SECTIONS['sensor_response_channels']}**")
        rgb_channels = {}
        for channel in ["R", "G", "B"]:
            rgb_channels[channel] = st.checkbox(
                f"{channel} Channel", 
                key=f"show_{channel}",
                value=True
            )
        
        # Stop/Log view toggle
        log_view = st.checkbox(
            UI_LABELS['stop_view_toggle'], 
            help=UI_HELP_TEXT['stop_view'],
            key="sidebar_log_view_toggle"
        )
    
    # ========== 4. EXPORT & REPORTS (Collapsed by default) ==========
    actions = {}
    with st.sidebar.expander(UI_SECTIONS['export_reports'], expanded=False):
        # Generate full report (both PNG and TSV to output folder)
        if st.button(UI_BUTTONS['generate_full_report']):
            actions['generate_full_report'] = selected_camera
        
        st.markdown("**Individual Downloads:**")
        
        # PNG Report download - generate and download in one step
        if hasattr(app_state, 'last_export') and app_state.last_export and app_state.last_export.get('bytes'):
            st.download_button(
                label="Download PNG Report",
                data=app_state.last_export['bytes'],
                file_name=app_state.last_export['name'],
                mime="image/png"
            )
        else:
            if st.button("Download PNG Report"):
                actions['generate_report'] = selected_camera
        
        # TSV Export download - generate and download in one step
        if hasattr(app_state, 'last_tsv_export') and app_state.last_tsv_export and app_state.last_tsv_export.get('bytes'):
            st.download_button(
                label="Download Filter Stack TSV",
                data=app_state.last_tsv_export['bytes'],
                file_name=app_state.last_tsv_export['name'],
                mime="text/tab-separated-values"
            )
        else:
            if st.button("Download Filter Stack TSV"):
                actions['export_tsv'] = True
    
<<<<<<< Updated upstream
    # Report generation
    if st.sidebar.button(UI_BUTTONS['generate_report']):
        actions['generate_report'] = selected_camera
    
    # Show download button if a report is ready
    setup_report_download(app_state)
    
    # Settings Panel (using existing function from this module)
    rebuild_cache, show_import, rgb_channels = settings_panel(app_state)
    
    # RGB channels and import data state are widget-controlled and accessed through app_state
    
    if rebuild_cache:
        actions['rebuild_cache'] = True
=======
    # ========== 5. DATA MANAGEMENT (Collapsed by default) ==========
    with st.sidebar.expander(UI_SECTIONS['data_management'], expanded=False):
        # Rebuild Cache button
        if st.button(UI_BUTTONS['rebuild_cache']):
            actions['rebuild_cache'] = True
            
        # Import data toggle
        if app_state.show_import_data:
            if st.button(UI_BUTTONS['close_importers']):
                st.session_state['show_import_data'] = False
                st.rerun()
        else:
            if st.button(UI_BUTTONS['csv_importers']):
                st.session_state['show_import_data'] = True
                st.rerun()
    
    # ========== REFLECTOR PREVIEW (Always visible when selected) ==========
    # This stays outside expanders and shows when a reflector is selected
    # The reflector preview logic will be handled in the main content rendering
>>>>>>> Stashed changes
        
    return actions
