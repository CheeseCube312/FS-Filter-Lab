"""
Sidebar UI components for FS FilterLab.

This module provides UI components for the sidebar, including filter selection,
filter multipliers, and extras (illuminant, QE, target).
"""
# Third-party imports
import streamlit as st
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from services.state_manager import StateManager

# Local imports
from models.constants import (
    CACHE_DIR, DEFAULT_ILLUMINANT, UI_BUTTONS, UI_SECTIONS, UI_LABELS, 
    UI_INFO_MESSAGES, UI_WARNING_MESSAGES, UI_HELP_TEXT, ACTION_TYPES
)
from models.core import FilterCollection, TargetProfile, ReflectorCollection
from views.ui_utils import try_operation, handle_error, show_info_message


def filter_selection(
    filter_collection: FilterCollection, 
    app_state
) -> List[str]:
    """
    UI component for filter selection.
    
    Args:
        filter_collection: Collection of available filters
        app_state: Application state manager
    
    Returns:
        List of selected filter display names
    """
    # --- Prepare default value for widget (including pending selections) ---
    current_selection = app_state.selected_filters
    
    # Check for pending selections from advanced search
    pending_selections = st.session_state.get("_pending_selected_filters", [])
    if pending_selections:
        # Add pending selections to current selection and clear them
        current_selection = list(set(current_selection + pending_selections))
        st.session_state.pop("_pending_selected_filters", None)

    # --- Widget logic ---
    filter_display_names = filter_collection.get_display_names()
    all_options = sorted(set(filter_display_names) | set(current_selection))

    selected = st.sidebar.multiselect(
        UI_LABELS['select_filters'],
        options=all_options,
        default=current_selection,
        key="selected_filters",
    )
    return selected


def filter_multipliers(selected: List[str], app_state) -> Dict[str, int]:
    """
    UI component for filter multiplier controls.
    
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
    filter_collection: FilterCollection
) -> Tuple[Optional[str], Optional[np.ndarray], Optional[Dict[str, np.ndarray]], Optional[str], Optional[TargetProfile]]:
    """
    UI component for analysis setup (illuminant, QE, target).
    
    Args:
        illuminants: Dictionary of illuminant curves by name
        illuminant_metadata: Dictionary of illuminant descriptions by name
        camera_keys: List of camera keys
        qe_data: Dictionary of QE data by camera key
        default_camera_key: Default camera key
        filter_collection: Collection of available filters
    
    Returns:
        Tuple of (illuminant_name, illuminant, qe, camera_name, target_profile)
    """
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
        
        target_profile = TargetProfile(
            name=filter_obj.name,
            values=filter_obj.transmission,  # Keep 0-1 scale consistent with filters
            valid=~np.isnan(filter_obj.transmission)
        )

    return selected_illum_name, selected_illum, current_qe, selected_camera, target_profile


def render_sidebar(app_state: "StateManager", data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Render the application sidebar with controls and settings.
    
    Args:
        app_state: Application state manager
        data: Application data dictionary
        
    Returns:
        Dictionary of user actions from sidebar interactions
    """
    import streamlit as st
    
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
    app_state.filter_multipliers = filter_multipliers_dict
    
    # ========== 2. ANALYSIS SETUP (Collapsed by default) ==========
    with st.sidebar.expander(UI_SECTIONS['analysis_setup'], expanded=False):
        selected_illum_name, selected_illum, current_qe, selected_camera, target_profile = analysis_setup(
            illuminants, illuminant_metadata,
            camera_keys, qe_data, default_key,
            filter_collection
        )
    
    # Update state
    app_state.current_qe = current_qe
    app_state.selected_camera = selected_camera
    app_state.illuminant = selected_illum
    app_state.illuminant_name = selected_illum_name
    app_state.target_profile = target_profile
    
    # ========== 3. DISPLAY & VISUALIZATION (Collapsed by default) ==========
    with st.sidebar.expander(UI_SECTIONS['display_visualization'], expanded=False):
        # Check for signal to close advanced search
        if st.session_state.get("_close_advanced_search", False):
            st.session_state["show_advanced_search"] = False
            st.session_state.pop("_close_advanced_search", None)
        
        # Advanced filter search toggle
        show_advanced_search = st.checkbox(
            UI_SECTIONS['show_advanced_search'], 
            key="show_advanced_search"
        )
        
        # Advanced reflector search toggle
        # Check for Done button click before creating checkbox
        if st.session_state.get("close_reflector_search", False):
            st.session_state["close_reflector_search"] = False
            if "show_reflector_search" in st.session_state:
                del st.session_state["show_reflector_search"]
        
        show_reflector_search = st.checkbox(
            UI_SECTIONS['show_reflector_search'],
            key="show_reflector_search"
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
            actions[ACTION_TYPES['generate_full_report']] = selected_camera
        
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
                actions[ACTION_TYPES['generate_report']] = selected_camera
        
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
                actions[ACTION_TYPES['export_tsv']] = True
    
    # ========== 5. DATA MANAGEMENT (Collapsed by default) ==========
    with st.sidebar.expander(UI_SECTIONS['data_management'], expanded=False):
        # Rebuild Cache button
        if st.button(UI_BUTTONS['rebuild_cache']):
            actions[ACTION_TYPES['rebuild_cache']] = True
            
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
        
    return actions
