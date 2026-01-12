"""
Main content display components for FS FilterLab.

This module consolidates all main content area UI components including
data displays, charts, and visualizations.
"""
# Standard library imports
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from services.state_manager import StateManager

# Third-party imports
import streamlit as st
import numpy as np

# Local imports
from models.constants import (UI_INFO_MESSAGES, UI_WARNING_MESSAGES, UI_CHART_TITLES, 
                             UI_SECTIONS, UI_LABELS, UI_BUTTONS, INTERP_GRID, METADATA_FIELDS,
                             VEGETATION_PREVIEW, SURFACE_COLOR_METADATA)
from models.core import TargetProfile
from services.calculations import (
    format_transmission_metrics, format_deviation_metrics, 
    calculate_transmission_deviation_metrics, format_white_balance_data,
    compute_selected_filter_indices, compute_effective_stops,
    compute_reflector_color, compute_reflector_preview_colors,
    compute_single_reflector_color, is_reflector_data_valid,
    check_reflector_wavelength_validity, compute_filter_transmission,
    compute_rgb_response, compute_white_balance_gains, compute_active_transmission
)
from services.visualization import (
    create_illuminant_figure, create_leaf_reflectance_figure, 
    create_single_reflectance_figure, prepare_rgb_for_display,
    create_filter_response_plot, create_sensor_response_plot
)
from views.ui_utils import (
    show_warning_message, show_info_message, format_error_message,
    reflector_preview, single_reflector_preview, render_color_swatch_from_rgb
)
from views.forms import import_data_form, advanced_filter_search, advanced_reflector_search
from views.channel_mixer import render_channel_mixer_panel, render_compact_channel_mixer_status

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _has_sufficient_reflectors(reflector_collection) -> bool:
    """Check if reflector collection has sufficient data for vegetation preview."""
    return (reflector_collection and 
            hasattr(reflector_collection, 'reflector_matrix') and 
            len(reflector_collection.reflector_matrix) >= VEGETATION_PREVIEW['required_count'])


def _is_valid_reflector_selection(selected_idx, reflector_collection) -> bool:
    """Validate reflector selection index."""
    if not reflector_collection or not hasattr(reflector_collection, 'reflector_matrix'):
        return False
    
    return (selected_idx is not None and 
            selected_idx != "None" and 
            isinstance(selected_idx, int) and 
            0 <= selected_idx < len(reflector_collection.reflector_matrix))


# ============================================================================
# CHART RENDERING UTILITIES
# ============================================================================

def render_chart(
    fig: Any, 
    title: Optional[str] = None, 
    description: Optional[str] = None,
    width: str = 'stretch',
    height: Optional[int] = None
) -> None:
    """Render a chart with optional title and description."""
    if title:
        st.subheader(title)
        
    # Apply height if specified
    if height and hasattr(fig, "update_layout"):
        fig.update_layout(height=height)
        
    # Display the chart
    st.plotly_chart(fig, width=width)
    
    if description:
        st.markdown(description)




# ============================================================================
# DATA DISPLAY COMPONENTS
# ============================================================================

def transmission_metrics(
    trans: np.ndarray, 
    label: str, 
    sensor_qe: Optional[np.ndarray],
    illuminant: Optional[np.ndarray] = None
) -> None:
    """Display transmission metrics (light loss)."""
    
    # Check if we have valid data
    valid = ~np.isnan(trans)
    if not valid.any():
        message = format_error_message('compute_failed', 
                                     metric='average transmission', 
                                     item=label, 
                                     reason='insufficient data')
        show_warning_message(message)
        return
    
    # Check if we have sensor QE data
    if sensor_qe is None:
        message = format_error_message('compute_failed', 
                                     metric='light loss', 
                                     item=label, 
                                     reason='no sensor QE data')
        show_warning_message(message)
        return
    
    # Calculate effective stops with illuminant weighting
    avg_trans, effective_stops = compute_effective_stops(trans, sensor_qe, illuminant)
    
    # Format metrics
    metrics = format_transmission_metrics(trans, label, avg_trans, effective_stops)
    
    # Display results
    st.markdown(
        f"**Estimated light loss ({metrics['label']}):** "
        f"{metrics['effective_stops']} stops  \n"
        f"(Avg transmission: {metrics['avg_transmission_pct']})"
    )


def deviation_metrics(
    transmission: np.ndarray, 
    combined: Optional[np.ndarray], 
    target_profile: Optional[TargetProfile]
) -> None:
    """Display deviation metrics from target profile."""
    
    if target_profile is None:
        return
    
    # Use combined transmission if available
    trans_for_dev = combined if combined is not None else transmission
    
    # Calculate metrics
    metrics = calculate_transmission_deviation_metrics(trans_for_dev, target_profile)
    
    if not metrics:
        if target_profile:
            show_info_message(UI_INFO_MESSAGES['no_target_overlap'])
        return    # Format metrics for display
    formatted = format_deviation_metrics(metrics, target_profile)
    
    # Display metrics
    st.markdown(
        f"**Deviation from target ({formatted['target_name']}):**  \n"
        f"- MAE: `{formatted['mae']}`  \n"
        f"- Bias: `{formatted['bias']}`  \n"
        f"- Max Dev: `{formatted['max_dev']}`  \n"
        f"- RMSE: `{formatted['rmse']}`"
    )


def white_balance_display(
    white_balance_gains: Dict[str, float],
    selected_filters: List[str],
    wb_reference_surface: Optional[str] = None
) -> None:
    """Display white balance gains in the UI.
    
    Args:
        white_balance_gains: Dictionary of WB gains by channel
        selected_filters: List of selected filter names
        wb_reference_surface: Source file of reference surface (if any)
    """
    
    # Format white balance data
    wb_data = format_white_balance_data(white_balance_gains, selected_filters)
    
    # Add a note based on mode
    if wb_reference_surface:
        mode_note = " (from surface reference)"
    elif not wb_data["has_filters"]:
        mode_note = " (No filter selected)"
    else:
        mode_note = ""
    
    st.markdown(
        f"**White Balance Gains{mode_note}:** (Green = 1.000):  \n"
        f"R: {wb_data['intensities']['R']}   "
        f"G: {wb_data['intensities']['G']}   "
        f"B: {wb_data['intensities']['B']}"
    )


def _render_default_reflector_list(app_state, data) -> None:
    """Render the default reflector list with computed surface colors."""
    
    reflector_collection = data['reflector_collection']
    
    # Get default reflector files
    default_files = app_state.get_default_reflector_files()
    if not default_files:
        return
    
    # Find matching reflectors by source file
    default_reflectors = []
    for idx, reflector in enumerate(reflector_collection.reflectors):
        source_file = reflector.metadata.get('source_file', '')
        if source_file in default_files:
            default_reflectors.append((idx, reflector))
    
    if not default_reflectors:
        return
    
    # Get current filter transmission
    combined_trans = app_state.combined_transmission
    if combined_trans is None:
        combined_trans = np.ones_like(INTERP_GRID, dtype=float)
    
    with st.expander(UI_SECTIONS['default_reflector_list'], expanded=True):
        # Reset WB button at the top
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button(
                UI_BUTTONS['reset_white_balance'], 
                help="Reset to standard white balance (no surface reference)"
            ):
                app_state.reset_white_balance()
                st.success("White balance reset to standard computation")
                st.rerun()
        
        with col2:
            st.markdown("*Click 'WB from Surface' next to any reflector to use it as white balance reference*")
        
        st.markdown("---")  # Separator line
        
        for idx, reflector in default_reflectors:
            # Compute color for this reflector
            rgb_color = None
            if app_state.current_qe and app_state.illuminant is not None:
                raw_rgb = compute_reflector_color(
                    reflector.values,
                    combined_trans,
                    app_state.current_qe,
                    app_state.illuminant,
                    app_state.channel_mixer,
                    app_state.white_balance_gains
                )
                raw_rgb = np.nan_to_num(raw_rgb)
                # Normalize for display (like other RGB calculations in the app)
                rgb_color = prepare_rgb_for_display(raw_rgb, auto_exposure=True)
            
            # Build display
            cols = st.columns([1, 5, 1])
            
            with cols[0]:
                # Color swatch
                render_color_swatch_from_rgb(rgb_color, size=40, border=True)
            
            with cols[1]:
                # Name and metadata
                st.markdown(f"**{reflector.name}**")
                meta = reflector.metadata
                
                # Build prioritized list of fields to display
                # API attribution fields always come first
                display_fields = SURFACE_COLOR_METADATA['api_attribution_fields'].copy()
                
                # Add user-selected relevant metadata, or fallback to common fields
                relevant_metadata_str = meta.get(METADATA_FIELDS['relevant_metadata'], '')
                if relevant_metadata_str:
                    display_fields.extend(col.strip() for col in relevant_metadata_str.split('|') if col.strip())
                else:
                    display_fields.extend(SURFACE_COLOR_METADATA['fallback_fields'])
                
                # Remove duplicates while preserving order, then collect non-empty values
                seen = set()
                meta_lines = []
                for field in display_fields:
                    if field in seen:
                        continue
                    seen.add(field)
                    value = meta.get(field, '').strip()
                    if value:
                        meta_lines.append(f"**{field}:** {value}")
                
                # Display metadata or fallback message
                if meta_lines:
                    st.caption("  \n".join(meta_lines))
                elif meta.get('source_folder', ''):
                    st.caption(f"**Source:** {meta['source_folder']}")
                else:
                    st.caption("*No additional metadata available*")
            
            with cols[2]:
                # White balance and remove buttons
                source_file = reflector.metadata.get('source_file', '')
                
                # White balance button - only enabled if we have QE and illuminant data
                wb_disabled = (app_state.current_qe is None or app_state.illuminant is None)
                
                if st.button(
                    UI_BUTTONS['white_balance_from_surface'], 
                    key=f"wb_default_{idx}",
                    disabled=wb_disabled,
                    help="Use this surface as white balance reference"
                ):
                    # Set white balance from this reflector (stores source_file for recalc on filter change)
                    app_state.set_white_balance_from_surface(reflector.values, combined_trans, source_file)
                    st.success(f"White balance set from: {reflector.name}")
                    st.rerun()
                
                # Remove button
                if st.button("×", key=f"remove_default_{idx}", help="Remove from defaults"):
                    app_state.remove_from_default_reflectors(source_file)
                    st.rerun()


def raw_qe_and_illuminant(app_state, data) -> None:
    """Display raw quantum efficiency and illuminant data charts."""
    
    # Extract needed data
    reflector_collection = data['reflector_collection']
    illuminant_metadata = data['illuminant_metadata']
    
    with st.expander(UI_SECTIONS['reflectance_illuminant_curves']):
        # Show leaf reflectance spectra if available
        if (_has_sufficient_reflectors(reflector_collection)):
            
            fig_leaves = create_leaf_reflectance_figure(
                INTERP_GRID, 
                reflector_collection.reflector_matrix, 
                reflector_collection
            )
            
            if fig_leaves is not None:
                render_chart(fig_leaves, UI_CHART_TITLES['leaf_reflectance'])
            else:
                show_info_message(UI_INFO_MESSAGES['leaf_data_required'])
        
        # Show selected reflectance spectrum if available
        selected_reflector_idx = app_state.get_selected_reflector_idx() if hasattr(app_state, 'get_selected_reflector_idx') else st.session_state.get("selected_reflector_idx", None)
        if (_is_valid_reflector_selection(selected_reflector_idx, reflector_collection)):
            
            fig_single = create_single_reflectance_figure(
                INTERP_GRID,
                reflector_collection.reflector_matrix,
                reflector_collection,
                selected_reflector_idx
            )
            
            if fig_single is not None:
                render_chart(fig_single)
        
        # Show illuminant curve
        if app_state.illuminant is not None and app_state.illuminant_name is not None:
            fig_illum = create_illuminant_figure(INTERP_GRID, app_state.illuminant, app_state.illuminant_name)
            
            # Description for the illuminant
            if app_state.illuminant_name in illuminant_metadata:
                description = f"**Description:** {illuminant_metadata[app_state.illuminant_name]}"
                st.markdown(description)
                
            render_chart(
                fig_illum, 
                f"Illuminant: {app_state.illuminant_name}"
            )
        else:
            show_info_message(UI_INFO_MESSAGES['no_illuminant'])


def filter_response_display(fig) -> None:
    """Display filter response chart in an expandable section."""
    render_chart(fig, title=UI_CHART_TITLES['combined_filter_response'])


def sensor_response_display(fig) -> None:
    """Display sensor response chart in an expandable section."""
    
    # White balance toggle is widget-controlled and accessed through app_state
    st.checkbox(
        UI_LABELS['apply_white_balance'], 
        key="apply_white_balance_toggle"
    )
    
    # Display the chart
    render_chart(fig, title=UI_CHART_TITLES['sensor_weighted_response'])


def render_main_content(app_state: "StateManager", data: Dict[str, Any]) -> None:
    """
    Render the main content area of the application.
    
    Args:
        app_state: Application state manager
        data: Application data dictionary containing collections and calculations
    """
    
    # Extract data  
    filter_collection = data['filter_collection']
    illuminant_metadata = data['illuminant_metadata']
    reflector_collection = data['reflector_collection']
    
    # Show import dialog if needed
    if app_state.show_import_data:
        import_data_form()
    
    # Compute selected filter indices
    selected_indices = compute_selected_filter_indices(
        app_state.selected_filters,
        app_state.filter_multipliers,
        filter_collection
    )
    
    # Header
    st.markdown("""
    <div style='display: flex; justify-content: space-between; align-items: center;'>
        <h4 style='margin: 0;'>FS FilterLab</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Filter transmission plots and metrics
    if selected_indices:
        _render_filter_analysis(app_state, filter_collection, selected_indices)
    
    # Sensor response and white balance
    if app_state.current_qe:
        _render_sensor_analysis(app_state, data, selected_indices)
    
    # Raw QE and illuminant curves
    raw_qe_and_illuminant(app_state, data)
    
    # Default reflector list display
    if reflector_collection and hasattr(reflector_collection, 'df') and len(reflector_collection.df) > 0:
        _render_default_reflector_list(app_state, data)
    
    # Advanced filter search UI
    if app_state.show_advanced_search:
        advanced_filter_search(filter_collection.df, filter_collection.filter_matrix)
    
    # Advanced reflector search UI
    if st.session_state.get("show_reflector_search", False):
        advanced_reflector_search(reflector_collection.df, reflector_collection.reflector_matrix, app_state)
    
    # Channel mixer UI
    if app_state.show_channel_mixer:
        app_state.channel_mixer = render_channel_mixer_panel(app_state.channel_mixer)


def _render_filter_analysis(app_state, filter_collection, selected_indices):
    """Render filter analysis plots and metrics."""
    
    # Calculate transmission and combined transmission
    trans, label, combined = compute_filter_transmission(
        selected_indices,
        filter_collection.filter_matrix
    )
    
    # Update combined transmission in state
    app_state.combined_transmission = combined if combined is not None else trans
    
    # Calculate sensor QE for display (RGB response)
    if app_state.current_qe:
        responses, rgb_matrix, _ = compute_rgb_response(
            trans, 
            app_state.current_qe,
            app_state.white_balance_gains,
            app_state.rgb_channels_visibility,
            app_state.channel_mixer  # Pass channel mixer settings
        )
    
    # Display transmission metrics using raw QE data and illuminant
    raw_qe = app_state.current_qe.get('G') if app_state.current_qe else None
    transmission_metrics(trans, label, raw_qe, app_state.illuminant)
    
    # Create and display filter response plot
    filter_names = [filter.name for filter in filter_collection.filters]
    filter_hex_colors = [filter.hex_color for filter in filter_collection.filters]
    
    fig = create_filter_response_plot(
        interp_grid=INTERP_GRID,
        filter_matrix=filter_collection.filter_matrix,
        masks=filter_collection.extrapolated_masks,
        selected_indices=selected_indices,
        combined=combined,
        target_profile=app_state.target_profile,
        log_stops=app_state.log_view,
        filter_names=filter_names,
        filter_hex_colors=filter_hex_colors
    )
    filter_response_display(fig)
    
    # Display deviation metrics
    deviation_metrics(trans, combined, app_state.target_profile)


def _compute_white_balance(app_state, trans_interp, reflector_collection=None) -> Dict[str, float]:
    """Compute and update white balance gains.
    
    If a reference surface is set (from 'WB from Surface' button), recalculates
    WB from that surface using the current transmission. Otherwise computes
    standard white balance from the filter transmission.
    
    Args:
        app_state: Application state manager
        trans_interp: Current filter transmission (combined)
        reflector_collection: Optional reflector collection for surface WB lookup
        
    Returns:
        Dictionary of white balance gains by channel
    """
    from services.calculations import compute_white_balance_gains_from_surface
    
    wb_gains = app_state.white_balance_gains  # Current gains
    if app_state.current_qe and app_state.illuminant is not None:
        # Check if there's a reference surface set for WB
        wb_ref_surface = app_state.wb_reference_surface
        
        if wb_ref_surface and reflector_collection:
            # Recalculate WB from the reference surface with current transmission
            # Find the reflector by source_file
            ref_reflector = None
            for reflector in reflector_collection.reflectors:
                if reflector.metadata.get('source_file', '') == wb_ref_surface:
                    ref_reflector = reflector
                    break
            
            if ref_reflector is not None:
                # Recalculate WB from the reference surface
                wb_gains = compute_white_balance_gains_from_surface(
                    ref_reflector.values, trans_interp, 
                    app_state.current_qe, app_state.illuminant
                )
                app_state.white_balance_gains = wb_gains
            else:
                # Reference surface no longer available, fall back to standard WB
                wb_gains = compute_white_balance_gains(trans_interp, app_state.current_qe, app_state.illuminant)
                app_state.white_balance_gains = wb_gains
                app_state.wb_reference_surface = None  # Clear invalid reference
        else:
            # No reference surface - compute standard white balance
            wb_gains = compute_white_balance_gains(trans_interp, app_state.current_qe, app_state.illuminant)
            app_state.white_balance_gains = wb_gains
    
    return wb_gains


def _render_sensor_response_plot(app_state, trans_interp, wb_gains) -> None:
    """Create and display the sensor response plot."""
    
    fig_response = create_sensor_response_plot(
        interp_grid=INTERP_GRID,
        transmission=trans_interp,
        qe_data=app_state.current_qe,
        visible_channels=app_state.rgb_channels_visibility,
        white_balance_gains=wb_gains,
        apply_white_balance=app_state.apply_white_balance,
        target_profile=app_state.target_profile,
        channel_mixer=app_state.channel_mixer
    )
    
    sensor_response_display(fig_response)


def _render_vegetation_preview(app_state, trans_interp, reflector_collection) -> Optional[np.ndarray]:
    """Render vegetation color preview and return pixels for normalization.
    
    Returns:
        RGB pixel array for normalization if successful, None otherwise.
    """
    
    reflector_matrix = reflector_collection.reflector_matrix
    
    if len(reflector_matrix) >= VEGETATION_PREVIEW['required_count']:
        pixels = compute_reflector_preview_colors(
            reflector_matrix, trans_interp, app_state.current_qe, 
            app_state.illuminant, reflector_collection, app_state.channel_mixer,
            app_state.white_balance_gains
        )
        
        if pixels is not None:
            reflector_preview(pixels)
            return pixels
        else:
            show_warning_message(UI_WARNING_MESSAGES['vegetation_preview_required'])
    else:
        show_warning_message(UI_WARNING_MESSAGES['vegetation_preview_required'])
    
    return None


def _render_single_reflector_preview(app_state, trans_interp, reflector_collection, pixels) -> None:
    """Render single reflector preview if one is selected."""
    
    reflector_matrix = reflector_collection.reflector_matrix
    selected_reflector_idx = app_state.get_selected_reflector_idx() if hasattr(app_state, 'get_selected_reflector_idx') else st.session_state.get("selected_reflector_idx", None)
    
    # Validate selection
    if (_is_valid_reflector_selection(selected_reflector_idx, reflector_collection)):
        
        single_color = compute_single_reflector_color(
            reflector_matrix, selected_reflector_idx, trans_interp, 
            app_state.current_qe, app_state.illuminant, app_state.channel_mixer,
            app_state.white_balance_gains
        )
        
        if single_color is not None:
            reflector_name = reflector_collection.reflectors[selected_reflector_idx].name
            
            # Determine global normalization scale
            if pixels is not None:
                global_max = np.max(pixels)
            else:
                # Compute global max from available reflectors
                all_colors = []
                for i in range(min(len(reflector_matrix), VEGETATION_PREVIEW['required_count'])):
                    color = compute_single_reflector_color(
                        reflector_matrix, i, trans_interp,
                        app_state.current_qe, app_state.illuminant, app_state.channel_mixer,
                        app_state.white_balance_gains
                    )
                    if color is not None:
                        all_colors.append(color)
                global_max = np.max(all_colors) if all_colors else np.max(single_color)
            
            single_reflector_preview(single_color, reflector_name, global_max)
        else:
            show_warning_message("Could not compute reflector color - insufficient data")


def _render_reflector_previews(app_state, trans_interp, reflector_collection) -> None:
    """Render all reflector color previews."""
    
    # Check if we have the basic requirements for reflector previews
    if not (app_state.current_qe is not None and 
            app_state.illuminant is not None and 
            hasattr(reflector_collection, 'reflector_matrix')):
        return
    
    if not is_reflector_data_valid(reflector_collection):
        return
        
    reflector_matrix = reflector_collection.reflector_matrix
    
    if not check_reflector_wavelength_validity(reflector_matrix):
        show_warning_message(UI_WARNING_MESSAGES['incomplete_reflector_data'])
    
    # Render vegetation preview first (returns pixels for normalization)
    pixels = _render_vegetation_preview(app_state, trans_interp, reflector_collection)
    
    # Render single reflector preview using the same normalization
    _render_single_reflector_preview(app_state, trans_interp, reflector_collection, pixels)


def _render_sensor_analysis(app_state, data, selected_indices):
    """Render sensor response analysis and reflector previews."""
    
    # Extract data
    filter_collection = data['filter_collection'] 
    reflector_collection = data['reflector_collection']
    
    # Compute active transmission
    trans_interp = compute_active_transmission(
        app_state.selected_filters, selected_indices, filter_collection.filter_matrix
    )
    
    # Compute and update white balance (pass reflector_collection to recalc from reference surface)
    wb_gains = _compute_white_balance(app_state, trans_interp, reflector_collection)
    
    # Render sensor response plot
    _render_sensor_response_plot(app_state, trans_interp, wb_gains)
    
    # Render reflector previews
    _render_reflector_previews(app_state, trans_interp, reflector_collection)
    
    # Display white balance information
    if app_state.current_qe and app_state.illuminant is not None:
        white_balance_display(
            app_state.white_balance_gains, 
            app_state.selected_filters,
            app_state.wb_reference_surface
        )
    else:
        show_info_message(UI_INFO_MESSAGES['qe_illuminant_required'])
