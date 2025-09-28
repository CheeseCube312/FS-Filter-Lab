"""
Main content display components for FS FilterLab.

This module consolidates all main content area UI components including
data displays, charts, and visualizations.
"""
import streamlit as st
import numpy as np
from typing import Dict, List, Optional, Any, Callable

from models.core import TargetProfile

# ============================================================================
# CHART RENDERING UTILITIES
# ============================================================================

def render_chart(
    fig: Any, 
    title: Optional[str] = None, 
    description: Optional[str] = None,
    use_container_width: bool = True,
    height: Optional[int] = None
) -> None:
    """Render a chart with optional title and description."""
    if title:
        st.subheader(title)
        
    # Apply height if specified
    if height and hasattr(fig, "update_layout"):
        fig.update_layout(height=height)
        
    # Display the chart
    st.plotly_chart(fig, use_container_width=use_container_width)
    
    if description:
        st.markdown(description)


def render_expandable_chart(
    fig: Any,
    title: str,
    expanded: bool = False,
    use_container_width: bool = True,
    height: Optional[int] = None,
    description: Optional[str] = None
) -> None:
    """Render a chart within an expandable section."""
    with st.expander(title, expanded=expanded):
        # Apply height if specified
        if height and hasattr(fig, "update_layout"):
            fig.update_layout(height=height)
            
        # Display the chart
        st.plotly_chart(fig, use_container_width=use_container_width)
        
        if description:
            st.markdown(description)


def render_chart_with_controls(
    fig: Any,
    title: Optional[str] = None,
    control_elements: Dict[str, Callable] = None,
    use_container_width: bool = True,
    height: Optional[int] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """Render a chart with control elements."""
    if title:
        st.subheader(title)
    
    # Handle controls
    control_values = {}
    if control_elements:
        cols = st.columns(len(control_elements))
        for i, (name, render_fn) in enumerate(control_elements.items()):
            with cols[i]:
                control_values[name] = render_fn()
    
    # Apply height if specified
    if height and hasattr(fig, "update_layout"):
        fig.update_layout(height=height)
        
    # Display the chart
    st.plotly_chart(fig, use_container_width=use_container_width)
    
    if description:
        st.markdown(description)
    
    return control_values

# ============================================================================
# DATA DISPLAY COMPONENTS
# ============================================================================

def transmission_metrics(
    trans: np.ndarray, 
    label: str, 
    sensor_qe: np.ndarray
) -> None:
    """Display transmission metrics (light loss)."""
    from services.calculations import format_transmission_metrics
    from services import compute_effective_stops
    
    # Check if we have valid data
    valid = ~np.isnan(trans)
    if not valid.any():
        st.warning(f"Cannot compute average transmission for {label}: insufficient data.")
        return
    
    # Calculate effective stops
    avg_trans, effective_stops = compute_effective_stops(trans, sensor_qe)
    
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
    from services.calculations import format_deviation_metrics, calculate_transmission_deviation_metrics
    
    if target_profile is None:
        return
    
    # Use combined transmission if available
    trans_for_dev = combined if combined is not None else transmission
    
    # Calculate metrics
    metrics = calculate_transmission_deviation_metrics(trans_for_dev, target_profile)
    
    if not metrics:
        if target_profile:
            st.info("ℹ️ No valid overlap with target for deviation calculation.")
        return
    
    # Format metrics for display
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
    selected_filters: List[str]
) -> None:
    """Display white balance gains in the UI."""
    from services.calculations import format_white_balance_data
    
    # Format white balance data
    wb_data = format_white_balance_data(white_balance_gains, selected_filters)
    
    # Add a note if no filters are selected
    no_filter_note = " (No filter selected)" if not wb_data["has_filters"] else ""
    
    st.markdown(
        f"**White Balance Gains{no_filter_note}:** (Green = 1.000):  \n"
        f"R: {wb_data['intensities']['R']}   "
        f"G: {wb_data['intensities']['G']}   "
        f"B: {wb_data['intensities']['B']}"
    )


def raw_qe_and_illuminant(
    interp_grid: np.ndarray,
    qe_data: Optional[Dict[str, np.ndarray]],
    illuminant: Optional[np.ndarray],
    illuminant_name: Optional[str],
    illuminant_metadata: Dict[str, str],
    add_trace_fn: Callable
) -> None:
    """Display raw QE curves and illuminant in Streamlit expander."""
    from services.visualization import create_qe_figure, create_illuminant_figure
    
    with st.expander("Show Raw QE and Illuminant Curves"):
        if qe_data:
            fig_qe = create_qe_figure(interp_grid, qe_data, {"R": True, "G": True, "B": True})
            render_chart(
                fig_qe, 
                "Sensor Quantum Efficiency (QE)"
            )
        else:
            st.info("ℹ️ No QE profile loaded.")

        if illuminant is not None and illuminant_name is not None:
            fig_illum = create_illuminant_figure(interp_grid, illuminant, illuminant_name)
            
            # Description for the illuminant
            description = None
            if illuminant_name in illuminant_metadata:
                description = f"**Description:** {illuminant_metadata[illuminant_name]}"
                st.markdown(description)
                
            render_chart(
                fig_illum, 
                f"Illuminant: {illuminant_name}"
            )
        else:
            st.info("ℹ️ No illuminant loaded.")


def filter_response_display(fig) -> None:
    """Display filter response plot."""
    render_chart(fig, title="Combined Filter Response")


def sensor_response_display(fig, apply_white_balance: bool = False) -> bool:
    """Display sensor response plot with white balance toggle."""
    
    # Put the toggle before the chart to avoid recomputation issues
    new_apply_white_balance = st.checkbox(
        "Apply White Balance to Response", 
        key="apply_white_balance_toggle"  # More descriptive key
    )
    
    # Display the chart
    render_chart(fig, title="Sensor-Weighted Response (QE × Transmission)")
    
    # Return the widget value directly (don't try to manage state manually)
    return new_apply_white_balance


def render_main_content(app_state, data):
    """
    Render the main content area of the application.
    
    Args:
        app_state: Application state object (now using unified StateManager under the hood)
        data: Dictionary containing loaded application data
    """
    import streamlit as st
    
    from models.constants import INTERP_GRID
    from models.core import TargetProfile
    from services.calculations import compute_selected_filter_indices
    from services import (
        compute_filter_transmission,
        compute_active_transmission, 
        compute_rgb_response,
        compute_white_balance_gains
    )
    from services.visualization import (
        create_filter_response_plot,
        create_sensor_response_plot,
        add_filter_curve_to_plotly
    )
    from services.calculations import (
        is_reflector_data_valid,
        check_reflector_wavelength_validity, 
        compute_reflector_preview_colors,
        compute_single_reflector_color
    )
    from views.forms import import_data_form, advanced_filter_search
    from views.channel_mixer import render_channel_mixer_panel, render_compact_channel_mixer_status
    from views.ui_utils import show_warning_message, show_info_message
    
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
    raw_qe_and_illuminant(
        interp_grid=INTERP_GRID,
        qe_data=app_state.current_qe,
        illuminant=app_state.illuminant,
        illuminant_name=app_state.illuminant_name,
        illuminant_metadata=illuminant_metadata,
        add_trace_fn=add_filter_curve_to_plotly
    )
    
    # Advanced search UI
    if app_state.show_advanced_search:
        advanced_filter_search(filter_collection.df, filter_collection.filter_matrix)
    
    # Channel mixer UI
    if app_state.show_channel_mixer:
        app_state.channel_mixer = render_channel_mixer_panel(app_state.channel_mixer)


def _render_filter_analysis(app_state, filter_collection, selected_indices):
    """Render filter analysis plots and metrics."""
    from models.constants import INTERP_GRID
    from services import compute_filter_transmission, compute_rgb_response, create_filter_response_plot
    
    # Calculate transmission and combined transmission
    trans, label, combined = compute_filter_transmission(
        selected_indices,
        filter_collection.filter_matrix
    )
    
    # Update combined transmission in state
    app_state.combined_transmission = combined if combined is not None else trans
    
    # Calculate sensor QE
    if app_state.current_qe:
        responses, rgb_matrix, _ = compute_rgb_response(
            trans, 
            app_state.current_qe,
            app_state.white_balance_gains,
            app_state.rgb_channels_visibility,
            app_state.channel_mixer  # Pass channel mixer settings
        )
        # Use Green channel for effective stops calculation (most representative)
        sensor_qe = responses.get('G', None) if responses else None
    else:
        sensor_qe = None
    
    # Display transmission metrics
    transmission_metrics(trans, label, sensor_qe)
    
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


def _render_sensor_analysis(app_state, data, selected_indices):
    """Render sensor response analysis and reflector previews."""
    import streamlit as st
    
    from models.constants import INTERP_GRID
    from services import (
        compute_active_transmission,
        compute_white_balance_gains,
        create_sensor_response_plot
    )
    from services.calculations import (
        is_reflector_data_valid,
        check_reflector_wavelength_validity,
        compute_reflector_preview_colors,
        compute_single_reflector_color
    )
    from views.sidebar import reflector_preview, single_reflector_preview
    from views.ui_utils import show_warning_message, show_info_message
    
    # Extract data
    filter_collection = data['filter_collection'] 
    reflector_collection = data['reflector_collection']
    
    # Compute active transmission
    trans_interp = compute_active_transmission(
        app_state.selected_filters, selected_indices, filter_collection.filter_matrix
    )
    
    # Compute white balance
    wb_gains = app_state.white_balance_gains  # Default gains
    if app_state.selected_filters and app_state.current_qe and app_state.illuminant is not None:
        wb_gains = compute_white_balance_gains(trans_interp, app_state.current_qe, app_state.illuminant)
        app_state.white_balance_gains = wb_gains  # Update state with computed gains
    elif app_state.current_qe and app_state.illuminant is not None:
        # Even with no filters selected, compute white balance with 100% transmission
        wb_gains = compute_white_balance_gains(trans_interp, app_state.current_qe, app_state.illuminant)
        app_state.white_balance_gains = wb_gains  # Update state with computed gains
    
    # Create sensor response plot
    fig_response = create_sensor_response_plot(
        interp_grid=INTERP_GRID,
        transmission=trans_interp,
        qe_data=app_state.current_qe,
        visible_channels=app_state.rgb_channels_visibility,
        white_balance_gains=wb_gains,
        apply_white_balance=app_state.apply_white_balance,
        target_profile=app_state.target_profile,
        channel_mixer=app_state.channel_mixer  # Pass channel mixer settings
    )
    
    # Display sensor response with white balance toggle
    new_apply_white_balance = sensor_response_display(fig_response, app_state.apply_white_balance)
    # Don't manually set state - it's widget-controlled now
    
    # Reflector color previews
    if (app_state.current_qe is not None and app_state.illuminant is not None and 
        hasattr(reflector_collection, 'reflector_matrix')):
        
        if is_reflector_data_valid(reflector_collection):
            reflector_matrix = reflector_collection.reflector_matrix
            
            if not check_reflector_wavelength_validity(reflector_matrix):
                show_warning_message("Some reflector data appears incomplete. Check data files.")
            
            # Vegetation preview (2x2 grid) - only with hardcoded leaf files
            if len(reflector_matrix) >= 4:
                pixels = compute_reflector_preview_colors(
                    reflector_matrix, trans_interp, app_state.current_qe, 
                    app_state.illuminant, reflector_collection, app_state.channel_mixer
                )
                
                if pixels is not None:
                    reflector_preview(pixels)
                else:
                    show_warning_message(
                        "⚠️ Vegetation Color Preview requires these exact files in data/reflectors/plant/:\n" +
                        "• Leaf_1_reflectance_extrapolated_1100.tsv\n" +
                        "• Leaf_2_reflectance_extrapolated_1100.tsv\n" +
                        "• Leaf_3_reflectance_extrapolated_1100.tsv\n" +
                        "• Leaf_4_reflectance_extrapolated_1100.tsv"
                    )
            else:
                show_warning_message(
                    "⚠️ Vegetation Color Preview requires these exact files in data/reflectors/plant/:\n" +
                    "• Leaf_1_reflectance_extrapolated_1100.tsv\n" +
                    "• Leaf_2_reflectance_extrapolated_1100.tsv\n" +
                    "• Leaf_3_reflectance_extrapolated_1100.tsv\n" +
                    "• Leaf_4_reflectance_extrapolated_1100.tsv"
                )
                
            # Single reflector preview - get selected index from session state
            selected_reflector_idx = st.session_state.get("selected_reflector_idx", None)
            
            # Handle both None and "None" string values, and ensure it's a valid integer index
            if (selected_reflector_idx is not None and 
                selected_reflector_idx != "None" and 
                isinstance(selected_reflector_idx, int) and 
                selected_reflector_idx < len(reflector_matrix)):
                
                single_color = compute_single_reflector_color(
                    reflector_matrix, selected_reflector_idx, trans_interp, 
                    app_state.current_qe, app_state.illuminant, app_state.channel_mixer
                )
                
                if single_color is not None:
                    # Get reflector name
                    reflector_name = reflector_collection.reflectors[selected_reflector_idx].name
                    
                    # Use the same global normalization scale as vegetation preview if available
                    if pixels is not None:
                        global_max = np.max(pixels)
                    else:
                        # If no vegetation preview, compute global max from all available reflectors
                        all_colors = []
                        for i in range(min(len(reflector_matrix), 4)):  # Check up to 4 reflectors
                            color = compute_single_reflector_color(
                                reflector_matrix, i, trans_interp,
                                app_state.current_qe, app_state.illuminant, app_state.channel_mixer
                            )
                            if color is not None:
                                all_colors.append(color)
                        global_max = np.max(all_colors) if all_colors else np.max(single_color)
                    
                    single_reflector_preview(single_color, reflector_name, global_max)
                else:
                    show_info_message("Unable to compute color for selected surface")
    
    # White balance display
    if app_state.current_qe and app_state.illuminant is not None:
        white_balance_display(app_state.white_balance_gains, app_state.selected_filters)
    else:
        show_info_message("Select a QE & illuminant profile to compute white balance.")
