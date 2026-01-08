"""
Visualization utilities for FS FilterLab.

This module provides all visualization functionality including:
- Interactive plotting with Plotly
- Static report generation with Matplotlib  
- Chart creation and styling utilities
- PNG report generation
- Channel mixer visualization support
"""
# Standard library imports
import io
import os
from typing import List, Dict, Optional, Any, Callable, Tuple

# Third-party imports
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import plotly.graph_objects as go

# Local imports
from models.core import ChannelMixerSettings
from models.constants import (
    CHART_HEIGHTS, CHART_LINE_STYLES, CHART_COLORS, PLOT_LAYOUT, 
    SENSOR_RESPONSE_DEFAULTS, MPL_STYLE_CONFIG, REPORT_CONFIG,
    ReportConfig, FilterData, ComputationFunctions, SensorData
)

# =============================================================================
# CONSTANTS
# =============================================================================

# Standard RGB color mappings from constants
COLOR_MAP = CHART_COLORS['rgb_colors']

# =============================================================================
# SHARED UTILITY FUNCTIONS
# =============================================================================

def _calculate_channel_responses(
    transmission: np.ndarray,
    qe_data: Dict[str, np.ndarray],
    visible_channels: Dict[str, bool],
    white_balance_gains: Dict[str, float],
    apply_white_balance: bool,
    channel_mixer: Optional[ChannelMixerSettings] = None
) -> Dict[str, np.ndarray]:
    """
    Calculate channel responses with optional white balancing and channel mixing.
    
    Args:
        transmission: Filter transmission data
        qe_data: Dictionary of QE data by channel
        visible_channels: Dictionary of channel visibility flags
        white_balance_gains: Dictionary of white balance gains by channel
        apply_white_balance: Whether to apply white balance
        channel_mixer: Optional channel mixer settings
        
    Returns:
        Dictionary of channel responses
    """
    # Compute all responses
    responses = {}
    for channel, qe_curve in qe_data.items():
        if not visible_channels.get(channel, True):
            continue
            
        # Calculate response and convert to percentage scale for display
        response = transmission * qe_curve * 100
        
        # Apply white balance if requested
        if apply_white_balance:
            wb_gain = white_balance_gains.get(channel, 1.0)
            response = response / wb_gain
        
        responses[channel] = response
    
    # Apply channel mixing if enabled
    if channel_mixer is not None and channel_mixer.enabled and responses:
        from services.channel_mixer import apply_channel_mixing_to_responses
        responses = apply_channel_mixing_to_responses(responses, channel_mixer)
    
    return responses


def _create_line_style(color: str, style: str = 'default', dash: str = None) -> dict:
    """
    Create a standardized line style dictionary.
    
    Args:
        color: Line color
        style: Line style type ('default', 'thick', 'sparkline')
        dash: Optional dash pattern ('dash', 'dot', etc.)
        
    Returns:
        Dictionary with line styling parameters
    """
    line_dict = {
        'color': color,
        'width': CHART_LINE_STYLES[style].get('width', 2) if isinstance(CHART_LINE_STYLES[style], dict) else CHART_LINE_STYLES[style]
    }
    
    if dash:
        line_dict['dash'] = dash
    
    return line_dict


def _calculate_spectral_colors(
    wavelengths: np.ndarray, 
    r_channel: np.ndarray, 
    g_channel: np.ndarray, 
    b_channel: np.ndarray,
    saturation_scaling_factor: float = 5.0, 
    min_saturation: float = 0.15
) -> np.ndarray:
    """
    Calculate spectral colors for visualization.
    
    Args:
        wavelengths: Array of wavelength values
        r_channel: Red channel response at each wavelength
        g_channel: Green channel response at each wavelength
        b_channel: Blue channel response at each wavelength
        saturation_scaling_factor: Controls color saturation intensity
        min_saturation: Minimum saturation to maintain some color
        
    Returns:
        RGB matrix of shape (len(wavelengths), 3) with values in range [0, 1]
    """
    # Stack RGB channels and handle NaN/infinity values
    rgb_matrix = np.stack([r_channel, g_channel, b_channel], axis=1)
    rgb_matrix = np.nan_to_num(rgb_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Calculate channel sum and valid signals
    rgb_sum = np.sum(rgb_matrix, axis=1, keepdims=True)
    valid_mask = rgb_sum.flatten() > 0.001
    
    # Only process valid signals
    if not np.any(valid_mask):
        return np.zeros_like(rgb_matrix)
    
    # Normalize colors by intensity to get color direction
    norm_rgb = np.zeros_like(rgb_matrix)
    norm_rgb[valid_mask] = rgb_matrix[valid_mask] / rgb_sum[valid_mask]
    
    # Calculate color saturation (max channel difference)
    # Simple approach: average absolute difference between channels
    max_channel = np.max(norm_rgb, axis=1, keepdims=True)
    min_channel = np.min(norm_rgb, axis=1, keepdims=True)
    color_saturation = np.clip((max_channel - min_channel) * saturation_scaling_factor, 0, 1)
    
    # Apply minimum saturation
    saturation = min_saturation + (1.0 - min_saturation) * color_saturation
    
    # Calculate relative brightness for each wavelength
    brightness = np.sum(rgb_matrix, axis=1, keepdims=True)
    max_brightness = np.max(brightness) if np.max(brightness) > 0 else 1.0
    brightness_factor = brightness / max_brightness
    
    # Apply brightness to normalized colors
    rgb_final = norm_rgb * brightness_factor
    
    # Normalize to use full dynamic range
    max_val = np.max(rgb_final) if np.max(rgb_final) > 0 else 1.0
    if max_val > 0:
        rgb_final = rgb_final / max_val
    
    return np.clip(rgb_final, 0.0, 1.0)


def prepare_rgb_for_display(
    rgb_values: np.ndarray,
    saturation_level: float = 1.0,
    auto_exposure: bool = True
) -> np.ndarray:
    """
    Prepare RGB values for display with camera-realistic normalization.
    
    This function mimics real camera sensor behavior where each channel
    saturates independently, rather than scaling all channels together.
    
    Args:
        rgb_values: RGB array of any shape with last dimension = 3
        saturation_level: Maximum sensor response level (default 1.0)
        auto_exposure: If True, scale to use full dynamic range
        
    Returns:
        RGB array normalized for display in range [0.0, 1.0]
    """
    # Handle negative values by clamping to zero (like real sensors)
    rgb_clamped = np.maximum(rgb_values, 0.0)
    
    if auto_exposure:
        # Auto-exposure: scale to use full dynamic range without clipping
        # Find the maximum value that would cause any channel to saturate
        max_val = np.max(rgb_clamped)
        if max_val > 0:
            # Scale so the brightest pixel reaches saturation_level
            exposure_scale = saturation_level / max_val
            rgb_exposed = rgb_clamped * exposure_scale
        else:
            rgb_exposed = rgb_clamped
    else:
        rgb_exposed = rgb_clamped
    
    # Apply sensor saturation: each channel clips independently
    rgb_saturated = np.minimum(rgb_exposed, saturation_level)
    
    # Final normalization for display (0-1 range)
    if saturation_level > 0:
        rgb_normalized = rgb_saturated / saturation_level
    else:
        rgb_normalized = np.zeros_like(rgb_saturated)
    
    return np.clip(rgb_normalized, 0.0, 1.0)


# =============================================================================
# MATPLOTLIB VISUALIZATION
# =============================================================================

def _create_filter_combo_info(selected_filters: List[str], df: Any, display_to_index: Dict[str, int]) -> Tuple[List, str]:
    """
    Create sorted filter combination information.
    
    Returns:
        Tuple of (combo_list, combo_name_string)
    """
    combo = []
    for name in sorted(selected_filters):
        idx = display_to_index.get(name)
        row = df.iloc[idx]
        combo.append((row['Manufacturer'], row['Filter Number'], row))
    combo_name = ", ".join(f"{m} {n}" for m, n, _ in combo)
    return combo, combo_name


def _add_filter_swatches_section(ax0, selected_filters: List[str], df: Any, display_to_index: Dict[str, int]):
    """Add filter color swatches and labels to the report."""
    ax0.axis('off')
    y0 = 0.9
    counts = {f: selected_filters.count(f) for f in set(selected_filters)}
    
    for filter_name, filter_count in counts.items():
        filter_index = display_to_index[filter_name]
        filter_row = df.iloc[filter_index]
        filter_color = filter_row.get('Hex Color', '#000000')
        
        rect = Rectangle((0.0, y0-0.15), 0.03, 0.1, transform=ax0.transAxes,
                         facecolor=filter_color, edgecolor='black', lw=REPORT_CONFIG['swatch_line_width'])
        ax0.add_patch(rect)
        
        ax0.text(0.03, y0-0.1, f"{filter_row['Manufacturer']} – {filter_row['Filter Name']} (#{filter_row['Filter Number']}) ×{filter_count}",
                 transform=ax0.transAxes, fontsize=REPORT_CONFIG['font_sizes']['filter_label'], va='center')
        y0 -= 0.15


def _add_light_loss_section(ax1, label: str, stops: float, avg_trans: float):
    """Add light loss information section to the report."""
    ax1.axis('off')
    ax1.text(0.01, 0.7, 'Estimated Light Loss:', fontsize=REPORT_CONFIG['font_sizes']['section_header'], fontweight='bold')
    ax1.text(0.01, 0.3, f"{label} → {stops:.2f} stops (Avg: {avg_trans*100:.1f}%)", fontsize=REPORT_CONFIG['font_sizes']['section_header'])


def _add_transmission_plot_section(ax2, selected_indices: List[int], df: Any, filter_matrix: np.ndarray, 
                                  masks: np.ndarray, add_curve_fn: Callable, interp_grid: np.ndarray, 
                                  active_trans: np.ndarray):
    """Add transmission plot section to the report."""
    for filter_index in selected_indices:
        filter_row = df.iloc[filter_index]
        transmission_pct = np.clip(filter_matrix[filter_index], 1e-6, 1.0) * 100
        filter_mask = masks[filter_index]
        add_curve_fn(ax2, interp_grid, transmission_pct, filter_mask,
                     filter_row['Filter Name'], filter_row.get('Hex Color', '#000000'))
    
    if len(selected_indices) > 1:
        ax2.plot(interp_grid, active_trans * 100, color='black', lw=REPORT_CONFIG['combined_line_width'], label='Combined Filter')
    
    ax2.set_title('Filter Transmission (%)')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Transmission (%)')
    ax2.set_xlim(interp_grid.min(), interp_grid.max())
    ax2.set_ylim(0, 100)


def _add_white_balance_section(ax3, wb: Dict[str, float]):
    """Add white balance gains section to the report."""
    ax3.axis('off')
    ax3.text(0.01, 0.6, 'White Balance Gains (Green = 1):', fontsize=REPORT_CONFIG['font_sizes']['section_header'], fontweight='bold')
    
    # Convert gains back to raw intensities (relative to green)
    intensities = {
        'R': 1.0 / wb['R'] if wb['R'] != 0 else 0.0,
        'G': 1.0,
        'B': 1.0 / wb['B'] if wb['B'] != 0 else 0.0
    }
    
    ax3.text(0.01, 0.4, f"R: {intensities['R']:.3f}   G: {intensities['G']:.3f}   B: {intensities['B']:.3f}", fontsize=REPORT_CONFIG['font_sizes']['section_header'])


def _add_sensor_response_section(ax4, current_qe: Dict[str, np.ndarray], wb: Dict[str, float], 
                               active_trans: np.ndarray, interp_grid: np.ndarray, 
                               camera_name: str, illuminant_name: str):
    """Add sensor-weighted response section to the report."""
    maxresp = 0
    stack = {}
    
    # Plot in correct RGB order
    for ch in ['R', 'G', 'B']:
        qe = current_qe.get(ch)
        if qe is None:
            continue
        gains = wb.get(ch, 1.0)
        resp = np.nan_to_num(active_trans * qe / gains) * 100
        ax4.plot(interp_grid, resp, label=f"{ch} Channel", lw=REPORT_CONFIG['channel_line_width'], color=COLOR_MAP[ch])
        maxresp = max(maxresp, np.nanmax(resp))
        stack[ch] = resp

    # Spectrum strip
    rgb_matrix = np.stack([
        stack.get('R', np.zeros_like(active_trans)),
        stack.get('G', np.zeros_like(active_trans)),
        stack.get('B', np.zeros_like(active_trans))
    ], axis=1)
    mv = np.nanmax(rgb_matrix)
    if mv > 0:
        rgb_matrix /= mv
    
    extent = [interp_grid.min(), interp_grid.max(), maxresp * 1.02, maxresp * 1.07]
    ax4.imshow(rgb_matrix[np.newaxis, :, :], aspect='auto', extent=extent)

    ax4.set_title('Sensor-Weighted Response (White-Balanced)', fontsize=REPORT_CONFIG['font_sizes']['title'], fontweight='bold')
    subtitle = f"Quantum Efficiency: {camera_name or 'None'}   |   Illuminant: {illuminant_name or 'None'}"
    ax4.text(0.5, 0.98, subtitle, transform=ax4.transAxes, ha='center', va='bottom', fontsize=REPORT_CONFIG['font_sizes']['subtitle'])
    ax4.set_xlabel('Wavelength (nm)')
    ax4.set_ylabel('Response (%)')
    ax4.set_xlim(interp_grid.min(), interp_grid.max())
    ax4.set_ylim(0, extent[3] * 1.02)
    ax4.legend(loc='upper right', fontsize=REPORT_CONFIG['font_sizes']['legend'], bbox_to_anchor=(1.0, 0.95))


def _save_report_to_file(fig, buf: io.BytesIO, fname: str, camera_name: str, illuminant_name: str, sanitize_fn: Callable):
    """Save the report figure to file and return data."""
    # Save to buffer
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    # Save to program/output/[QE]/[Illuminant] folder
    output_dir = os.path.join(OUTPUT_FOLDERS['reports'], sanitize_fn(camera_name), sanitize_fn(illuminant_name))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, fname)
    with open(output_path, "wb") as f:
        f.write(buf.getvalue())

    st.success(f"Report generated: {fname}")
    return {'bytes': buf.getvalue(), 'name': fname}


def setup_matplotlib_style():
    """
    Configure matplotlib with consistent styling for report generation.
    
    Attempts to use modern seaborn styles with fallback options,
    then applies custom configuration from constants.
    """
    # Try different seaborn style variants in order of preference
    style_options = ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "whitegrid", "default"]
    
    for style in style_options:
        try:
            plt.style.use(style)
            break
        except OSError:
            continue
    
    # Apply custom configuration
    plt.rcParams.update(MPL_STYLE_CONFIG)


def add_filter_curve_to_matplotlib(ax, x, y, mask, label, color):
    """
    Add a filter transmission curve to a matplotlib axes.
    
    Args:
        ax: Matplotlib axes object
        x: Wavelength data
        y: Transmission data
        mask: Boolean mask for extrapolated regions
        label: Curve label
        color: Curve color
    """
    # Plot main curve
    ax.plot(x, y, color=color, linewidth=CHART_LINE_STYLES['standard_width'], label=label)
    
    # Add extrapolated regions if mask exists
    if mask is not None and np.any(mask):
        extrap_y = y.copy()
        extrap_y[~mask] = np.nan
        ax.plot(x, extrap_y, color=color, linewidth=CHART_LINE_STYLES['standard_width'], 
                linestyle=CHART_LINE_STYLES['extrapolated_style'], alpha=CHART_LINE_STYLES['extrapolated_alpha'])


def create_report_config(
    selected_filters: List[str],
    current_qe: Dict[str, np.ndarray], 
    camera_name: str,
    illuminant_name: str,
    illuminant_curve: np.ndarray
) -> ReportConfig:
    """Helper function to create ReportConfig from individual parameters."""
    return ReportConfig(
        selected_filters=selected_filters,
        current_qe=current_qe,
        camera_name=camera_name,
        illuminant_name=illuminant_name,
        illuminant_curve=illuminant_curve
    )

def create_filter_data(
    filter_matrix: np.ndarray,
    df: Any,
    display_to_index: Dict[str, int],
    masks: np.ndarray,
    interp_grid: np.ndarray
) -> FilterData:
    """Helper function to create FilterData from individual parameters.""" 
    return FilterData(
        filter_matrix=filter_matrix,
        df=df,
        display_to_index=display_to_index,
        masks=masks,
        interp_grid=interp_grid
    )

def create_computation_functions(
    compute_selected_indices_fn: Callable[[List[str]], List[int]],
    compute_filter_transmission_fn: Callable[[List[int]], Tuple[np.ndarray, str, np.ndarray]],
    compute_effective_stops_fn: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], Tuple[float, float]],
    compute_white_balance_gains_fn: Callable[[np.ndarray, Dict[str, np.ndarray], np.ndarray], Dict[str, float]],
    add_curve_fn: Callable,
    sanitize_fn: Callable[[str], str]
) -> ComputationFunctions:
    """Helper function to create ComputationFunctions from individual parameters."""
    return ComputationFunctions(
        compute_selected_indices_fn=compute_selected_indices_fn,
        compute_filter_transmission_fn=compute_filter_transmission_fn,
        compute_effective_stops_fn=compute_effective_stops_fn,
        compute_white_balance_gains_fn=compute_white_balance_gains_fn,
        add_curve_fn=add_curve_fn,
        sanitize_fn=sanitize_fn
    )

def create_sensor_data(sensor_qe: np.ndarray) -> SensorData:
    """Helper function to create SensorData from individual parameters."""
    return SensorData(sensor_qe=sensor_qe)

def generate_report_png_v2(
    report_config: ReportConfig,
    filter_data: FilterData, 
    computation_fns: ComputationFunctions,
    sensor_data: SensorData
) -> Dict[str, Any]:
    """
    Generate a PNG report with simplified parameter structure using data classes.
    
    This is the refactored version that reduces parameter count from 17 to 4
    by using data classes to group related parameters.
    """
    return generate_report_png(
        selected_filters=report_config.selected_filters,
        current_qe=report_config.current_qe,
        filter_matrix=filter_data.filter_matrix,
        df=filter_data.df,
        display_to_index=filter_data.display_to_index,
        compute_selected_indices_fn=computation_fns.compute_selected_indices_fn,
        compute_filter_transmission_fn=computation_fns.compute_filter_transmission_fn,
        compute_effective_stops_fn=computation_fns.compute_effective_stops_fn,
        compute_white_balance_gains_fn=computation_fns.compute_white_balance_gains_fn,
        masks=filter_data.masks,
        add_curve_fn=computation_fns.add_curve_fn,
        interp_grid=filter_data.interp_grid,
        sensor_qe=sensor_data.sensor_qe,
        camera_name=report_config.camera_name,
        illuminant_name=report_config.illuminant_name,
        sanitize_fn=computation_fns.sanitize_fn,
        illuminant_curve=report_config.illuminant_curve
    )

def generate_report_png(
    selected_filters: List[str],
    current_qe: Dict[str, np.ndarray],
    filter_matrix: np.ndarray,
    df: Any,
    display_to_index: Dict[str, int],
    compute_selected_indices_fn: Callable[[List[str]], List[int]],
    compute_filter_transmission_fn: Callable[[List[int]], Tuple[np.ndarray, str, np.ndarray]],
    compute_effective_stops_fn: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], Tuple[float, float]],
    compute_white_balance_gains_fn: Callable[[np.ndarray, Dict[str, np.ndarray], np.ndarray], Dict[str, float]],
    masks: np.ndarray,
    add_curve_fn: Callable,
    interp_grid: np.ndarray,
    sensor_qe: np.ndarray,
    camera_name: str,
    illuminant_name: str,
    sanitize_fn: Callable[[str], str],
    illuminant_curve: np.ndarray
) -> Dict[str, Any]:
    """
    Generate a PNG report of the current filter configuration.
    
    This function creates a comprehensive multi-panel report showing filter properties,
    transmission curves, light loss calculations, and sensor responses.
    """
    # Validation
    if not selected_filters:
        st.warning("⚠️ No filters selected—nothing to export.")
        return {}

    # Prepare filter combination data
    combo, combo_name = _create_filter_combo_info(selected_filters, df, display_to_index)
    
    # Resolve and validate indices
    selected_indices = compute_selected_indices_fn(selected_filters)
    if not selected_indices:
        st.warning("⚠️ Invalid filter selection—cannot resolve indices.")
        return {}

    # Compute filter characteristics
    trans, label, combined = compute_filter_transmission_fn(selected_indices)
    active_trans = combined if combined is not None else trans
    avg_trans, stops = compute_effective_stops_fn(active_trans, sensor_qe, illuminant_curve)
    wb = compute_white_balance_gains_fn(active_trans, current_qe, illuminant_curve)

    # Create figure with layout
    setup_matplotlib_style()
    fig = plt.figure(figsize=REPORT_CONFIG['figure_size'], dpi=REPORT_CONFIG['dpi'], constrained_layout=False)
    gs = GridSpec(5, 1, figure=fig, height_ratios=PLOT_LAYOUT['grid_height_ratios'])

    # Build report sections
    _add_filter_swatches_section(fig.add_subplot(gs[0]), selected_filters, df, display_to_index)
    _add_light_loss_section(fig.add_subplot(gs[1]), label, stops, avg_trans)
    _add_transmission_plot_section(fig.add_subplot(gs[2]), selected_indices, df, filter_matrix, 
                                  masks, add_curve_fn, interp_grid, active_trans)
    _add_white_balance_section(fig.add_subplot(gs[3]), wb)
    _add_sensor_response_section(fig.add_subplot(gs[4]), current_qe, wb, active_trans, 
                               interp_grid, camera_name, illuminant_name)

    # Finalize layout
    fig.suptitle("Filter Report", fontsize=REPORT_CONFIG['font_sizes']['main_title'], fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    # Save and return
    buf = io.BytesIO()
    fname = sanitize_fn(f"{camera_name}_{illuminant_name}_{combo_name}") + '.png'
    return _save_report_to_file(fig, buf, fname, camera_name, illuminant_name, sanitize_fn)


# =============================================================================
# PLOTLY VISUALIZATION
# =============================================================================

def apply_plotly_default_style(fig, title, x_title="Wavelength (nm)", y_title="Response", height=None):
    """Apply consistent default styling to Plotly figures."""
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        height=height or CHART_HEIGHTS['standard_plot'],
        hovermode='x unified'
    )
    return fig


def add_filter_curve_to_plotly(fig, x, y, mask, label, color):
    """
    Add a filter transmission curve to a Plotly figure.
    
    Args:
        fig: Plotly figure object
        x: Wavelength data
        y: Transmission data
        mask: Boolean mask for extrapolated regions
        label: Curve label
        color: Curve color
    """
    # Convert log scale if needed
    y_display = y.copy()
    
    # Add main curve
    fig.add_trace(go.Scatter(
        x=x,
        y=y_display,
        mode='lines',
        name=label,
        line=_create_line_style(color),
        showlegend=True
    ))
    
    # Add extrapolated regions if mask exists
    if mask is not None and np.any(mask):
        extrap_y = y_display.copy()
        extrap_y[~mask] = np.nan
        
        fig.add_trace(go.Scatter(
            x=x,
            y=extrap_y,
            mode='lines',
            name=f'{label} (extrapolated)',
            line=_create_line_style(color, dash='dot'),
            showlegend=False
        ))


def _add_spectrum_strip_to_plot(
    fig: go.Figure,
    wavelengths: np.ndarray,
    rgb_matrix: np.ndarray,
    relative_brightness: np.ndarray,
    y_position: float
) -> None:
    """
    Add a spectrum strip to a plotly figure.
    
    Args:
        fig: The plotly figure to add the spectrum strip to
        wavelengths: Array of wavelength values
        rgb_matrix: RGB color matrix of shape (len(wavelengths), 3)
        relative_brightness: Relative brightness at each wavelength
        y_position: Y-position where to place the spectrum strip
    """
    # Convert RGB values to integers for color strings
    rgb_colors_int = np.clip(rgb_matrix * 255.0, 0, 255).astype(int)
    
    # Create color strings for plotly
    colors = [
        f'rgb({int(r)},{int(g)},{int(b)})' 
        for r, g, b in rgb_colors_int
    ]
    
    # Create hover text with brightness information
    brightness_pcts = (relative_brightness * 100).astype(int)
    hover_texts = [
        f"Wavelength: {wl} nm<br>Relative brightness: {br_pct}%" 
        for wl, br_pct in zip(wavelengths, brightness_pcts)
    ]
    
    # Add the spectrum strip to the figure
    fig.add_trace(go.Scatter(
        x=wavelengths,
        y=[y_position] * len(wavelengths),
        mode='markers',
        marker=dict(
            size=9,
            color=colors,
            line=dict(width=0),
            symbol='square'
        ),
        text=hover_texts,
        hoverinfo='text',
        showlegend=False,
        name='Spectrum'
    ))


def _update_response_plot_layout(
    fig: go.Figure,
    has_spectrum_strip: bool,
    is_white_balanced: bool,
    is_channel_mixed: bool
) -> None:
    """
    Update the layout of a sensor response plot.
    
    Args:
        fig: The plotly figure to update
        has_spectrum_strip: Whether the plot has a spectrum strip
        is_white_balanced: Whether white balancing is applied
        is_channel_mixed: Whether channel mixing is applied
    """
    # Update title to reflect current state
    wb_status = " (White Balanced)" if is_white_balanced else ""
    mixer_status = " (Channel Mixed)" if is_channel_mixed else ""
    title = f"Sensor Response{wb_status}{mixer_status}"
    
    # Determine plot height based on contents
    plot_height = CHART_HEIGHTS['plot_with_spectrum'] if has_spectrum_strip else CHART_HEIGHTS['standard_plot']
    
    # Apply consistent styling
    apply_plotly_default_style(fig, title, y_title="Response (%)", height=plot_height)


def create_filter_response_plot(
    interp_grid: np.ndarray,
    filter_matrix: np.ndarray,
    masks: np.ndarray,
    selected_indices: List[int],
    combined: Optional[np.ndarray],
    target_profile: Optional[Any],
    log_stops: bool,
    filter_names: List[str],
    filter_hex_colors: List[str]
) -> go.Figure:
    """Create a plotly figure showing filter response curves."""
    fig = go.Figure()
    
    # Add individual filter curves
    for curve_index, filter_index in enumerate(selected_indices):
        transmission = filter_matrix[filter_index]
        mask = masks[filter_index] if masks is not None else np.zeros_like(transmission, dtype=bool)
        name = filter_names[filter_index]
        color = filter_hex_colors[filter_index]
        
        # Convert to log scale or percentage scale for display
        y_data = -np.log2(np.maximum(transmission, 1e-6)) if log_stops else transmission * 100
        
        add_filter_curve_to_plotly(fig, interp_grid, y_data, mask, name, color)
    
    # Add combined transmission if available
    if combined is not None:
        y_data = -np.log2(np.maximum(combined, 1e-6)) if log_stops else combined * 100
        fig.add_trace(go.Scatter(
            x=interp_grid,
            y=y_data,
            mode='lines',
            name='Combined',
            line=_create_line_style(CHART_COLORS['text'], 'thick'),
            showlegend=True
        ))
    
    # Add target profile if available
    if target_profile and hasattr(target_profile, 'values') and hasattr(target_profile, 'valid'):
        valid_mask = target_profile.valid
        target_values = target_profile.values
        if log_stops:
            # Convert fraction to stops
            target_values = -np.log2(np.maximum(target_values, 1e-6))
        else:
            # Convert to percentage scale for display
            target_values = target_values * 100
            
        fig.add_trace(go.Scatter(
            x=interp_grid[valid_mask],
            y=target_values[valid_mask],
            mode='lines',
            name='Target',
            line=_create_line_style(CHART_COLORS['warning'], dash='dash'),
            showlegend=True
        ))
    
    # Update layout
    y_title = 'Transmission (stops)' if log_stops else 'Transmission (%)'
    fig = apply_plotly_default_style(fig, "Filter Response", y_title=y_title)
    
    # Invert y-axis for log view (high transmission = low stops = bottom of plot)
    if log_stops:
        fig.update_layout(yaxis={'autorange': 'reversed'})
    
    return fig


def create_sensor_response_plot(
    interp_grid: np.ndarray,
    transmission: np.ndarray,
    qe_data: Dict[str, np.ndarray],
    visible_channels: Dict[str, bool],
    white_balance_gains: Dict[str, float],
    apply_white_balance: bool,
    target_profile: Optional[Any] = None,
    channel_mixer: Optional[ChannelMixerSettings] = None
) -> go.Figure:
    """Create a plotly figure showing sensor response with optional channel mixing."""
    fig = go.Figure()
    
    # Calculate channel responses
    responses = _calculate_channel_responses(
        transmission, qe_data, visible_channels, 
        white_balance_gains, apply_white_balance, channel_mixer
    )
    
    # Plot each channel response
    for channel, response in responses.items():
        fig.add_trace(go.Scatter(
            x=interp_grid,
            y=response,
            mode='lines',
            name=f'{channel} Channel',
            line=_create_line_style(COLOR_MAP.get(channel, 'gray')),
            showlegend=True
        ))
    
    # Add spectrum strip if we have RGB data
    if len(responses) >= 3 and 'R' in responses and 'G' in responses and 'B' in responses:
        # Get channel responses (camera sensitivity to each wavelength)
        r_channel = responses.get('R', np.zeros_like(interp_grid))
        g_channel = responses.get('G', np.zeros_like(interp_grid))
        b_channel = responses.get('B', np.zeros_like(interp_grid))
        
        # Create RGB matrix from responses and process it for display
        rgb_matrix = _calculate_spectral_colors(
            interp_grid, 
            r_channel, g_channel, b_channel,
            saturation_scaling_factor=SENSOR_RESPONSE_DEFAULTS['saturation_scaling_factor'], 
            min_saturation=SENSOR_RESPONSE_DEFAULTS['min_saturation']
        )
        
        # Calculate brightness for hover information
        brightness = np.sum(rgb_matrix, axis=1)
        max_brightness = np.max(brightness) if np.max(brightness) > 0 else 1.0
        relative_brightness = brightness / max_brightness
        
        # Find the maximum response for positioning the spectrum strip
        max_response = 1.0
        for response in responses.values():
            if response is not None and len(response) > 0:
                clean_response = np.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)
                response_max = np.max(clean_response)
                if np.isfinite(response_max) and response_max > max_response:
                    max_response = response_max
        
        # Calculate spectrum strip position
        spectrum_y_pos = max_response * SENSOR_RESPONSE_DEFAULTS['spectrum_strip_position_pct']
        
        # Add spectrum strip to plot
        _add_spectrum_strip_to_plot(
            fig, 
            interp_grid, 
            rgb_matrix, 
            relative_brightness, 
            spectrum_y_pos
        )
    
    # Add target profile if provided
    if target_profile is not None:
        fig.add_trace(go.Scatter(
            x=interp_grid,
            y=target_profile.values,
            mode='lines',
            name=f'Target: {target_profile.name}',
            line=_create_line_style(CHART_COLORS['text'], dash='dash'),
            showlegend=True
        ))
    
    # Update layout with appropriate title and settings
    _update_response_plot_layout(
        fig, 
        len(responses) >= 3,
        apply_white_balance, 
        channel_mixer and channel_mixer.enabled
    )
    
    return fig


def _create_standard_plotly_figure(
    x_data: np.ndarray,
    y_data_dict: Dict[str, np.ndarray],
    title: str,
    y_title: str,
    color_map: Dict[str, str],
    visible_channels: Dict[str, bool] = None,
    height: int = None
) -> go.Figure:
    """
    Create a standard Plotly figure with multiple data series.
    
    Args:
        x_data: X-axis data (usually wavelength)
        y_data_dict: Dictionary of data series {name: y_values}
        title: Chart title
        y_title: Y-axis title
        color_map: Dictionary mapping series names to colors
        visible_channels: Optional visibility filter for channels
        height: Optional chart height
        
    Returns:
        Configured Plotly figure
    """
    fig = go.Figure()
    
    for name, y_data in y_data_dict.items():
        # Skip if visibility filter exists and channel is hidden
        if visible_channels and not visible_channels.get(name, True):
            continue
            
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines',
            name=name,
            line=_create_line_style(color_map.get(name, 'gray'))
        ))
    
    apply_plotly_default_style(fig, title, y_title=y_title, height=height or CHART_HEIGHTS['default'])
    return fig


def create_qe_figure(
    interp_grid: np.ndarray,
    qe_data: Dict[str, np.ndarray],
    visible_channels: Dict[str, bool],
    height: int = None
) -> go.Figure:
    """Create a QE response figure."""
    # Add channel suffix for display names
    display_data = {f'{channel} QE': curve for channel, curve in qe_data.items()}
    
    return _create_standard_plotly_figure(
        x_data=interp_grid,
        y_data_dict=display_data,
        title="Quantum Efficiency",
        y_title="QE",
        color_map=COLOR_MAP,
        visible_channels={f'{k} QE': v for k, v in visible_channels.items()},
        height=height
    )


def create_illuminant_figure(
    interp_grid: np.ndarray,
    illuminant: np.ndarray,
    illuminant_name: str,
    height: int = None
) -> go.Figure:
    """Create an illuminant figure."""
    return _create_standard_plotly_figure(
        x_data=interp_grid,
        y_data_dict={illuminant_name: illuminant},
        title="Illuminant Spectrum",
        y_title="Relative Power",
        color_map={illuminant_name: CHART_COLORS['illuminant']},
        height=height
    )


def create_leaf_reflectance_figure(
    interp_grid: np.ndarray,
    reflector_matrix: np.ndarray,
    reflector_collection: Any,
    height: int = None
) -> Optional[go.Figure]:
    """Create a figure showing the four default vegetation reflectance spectra."""
    from services.calculations import find_vegetation_preview_reflectors
    
    if not reflector_collection or not hasattr(reflector_collection, 'reflectors'):
        return None
        
    leaf_indices = find_vegetation_preview_reflectors(reflector_collection)
    if leaf_indices is None:
        return None
    
    fig = go.Figure()
    
    # Define colors for the four default spectra
    leaf_colors = CHART_COLORS['leaf_colors']
    
    for i, leaf_idx in enumerate(leaf_indices):
        reflector_data = reflector_matrix[leaf_idx]
        # Use the actual reflector name from the collection
        leaf_name = reflector_collection.reflectors[leaf_idx].name
        
        fig.add_trace(go.Scatter(
            x=interp_grid,
            y=reflector_data,
            mode='lines',
            name=leaf_name,
            line=_create_line_style(leaf_colors[i])
        ))
    
    apply_plotly_default_style(fig, "Default Vegetation Reflectance Spectra", y_title="Reflectance", 
                              height=height or CHART_HEIGHTS['default'])
    
    return fig


def create_single_reflectance_figure(
    interp_grid: np.ndarray,
    reflector_matrix: np.ndarray,
    reflector_collection: Any,
    selected_reflector_idx: int,
    height: int = None
) -> Optional[go.Figure]:
    """Create a figure showing a single selected reflectance spectrum."""
    if (not reflector_collection or 
        not hasattr(reflector_collection, 'reflectors') or
        selected_reflector_idx >= len(reflector_matrix)):
        return None
    
    reflector_data = reflector_matrix[selected_reflector_idx]
    reflector_name = reflector_collection.reflectors[selected_reflector_idx].name
    
    return _create_standard_plotly_figure(
        x_data=interp_grid,
        y_data_dict={reflector_name: reflector_data},
        title=f"Reflectance: {reflector_name}",
        y_title="Reflectance",
        color_map={reflector_name: CHART_COLORS['single_reflector']},
        height=height
    )


def create_sparkline_plot(
    x_data: np.ndarray,
    y_data: np.ndarray,
    color: str = "blue",
    height: int = None,
    width: int = 300
) -> go.Figure:
    """Create a simple sparkline plot for inline display."""
    if height is None:
        height = CHART_HEIGHTS['sparkline']
    fig = go.Figure()
    
    # Convert transmission values to percentage for display
    # Convert internal 0-1 scale to 0-100% for display
    y_data_pct = y_data * 100
    
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data_pct,
        mode='lines',
        line=_create_line_style(color, 'sparkline'),
        showlegend=False
    ))
    
    # Calculate reasonable tick values for wavelength axis
    x_min, x_max = int(min(x_data)), int(max(x_data))
    x_range = x_max - x_min
    
    # Determine appropriate tick interval based on range
    if x_range > 500:
        x_tick_interval = 200
    elif x_range > 200:
        x_tick_interval = 100
    else:
        x_tick_interval = 50
    
    x_ticks = list(range(
        x_min + x_tick_interval - (x_min % x_tick_interval),
        x_max,
        x_tick_interval
    ))
    
    fig.update_layout(
        height=height,
        width=width,
        margin=dict(l=40, r=10, t=10, b=30),  # Increased margins for axes
        showlegend=False,
        xaxis=dict(
            showgrid=True,
            gridcolor=CHART_COLORS['grid'],
            showticklabels=True,
            tickvals=x_ticks,
            title=dict(
                text="Wavelength (nm)",
                font=dict(size=10)
            ),
            tickfont=dict(size=8)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=CHART_COLORS['grid'],
            showticklabels=True,
            title=dict(
                text="Transmission (%)",
                font=dict(size=10)
            ),
            tickfont=dict(size=8),
            range=[0, 100]  # Set y-axis range to 0-100%
        ),
        paper_bgcolor=CHART_COLORS['transparent'],
        plot_bgcolor=CHART_COLORS['transparent']
    )
    
    return fig