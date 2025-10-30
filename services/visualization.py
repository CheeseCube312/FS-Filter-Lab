"""
Visualization utilities for FS FilterLab.

This module provides all visualization functionality including:
- Interactive plotting with Plotly
- Static report generation with Matplotlib  
- Chart creation and styling utilities
- PNG report generation
- Channel mixer visualization support
"""
import io
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import streamlit as st
from typing import List, Dict, Optional, Any, Callable, Tuple

from models.core import ChannelMixerSettings
from services.channel_mixer import apply_channel_mixing_to_matrix

# =============================================================================
# CONSTANTS
# =============================================================================

# Standard RGB color mappings
COLOR_MAP = {
    'R': 'red', 
    'G': 'green', 
    'B': 'blue'
}

# Plot styling constants
PLOT_HEIGHT_DEFAULT = 400
PLOT_HEIGHT_WITH_SPECTRUM = 450

# Default matplotlib style configuration
MPL_STYLE_CONFIG = {
    "font.family": "DejaVu Sans",
    "axes.facecolor": "white",
    "axes.edgecolor": "#CCCCCC",
    "axes.grid": True,
    "grid.color": "#EEEEEE",
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "text.color": "#333333",
    "axes.labelcolor": "#333333",
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.frameon": False,
    "legend.fontsize": 8,
}

# =============================================================================
# SHARED UTILITY FUNCTIONS
# =============================================================================

def apply_color_matrix(rgb_values, matrix):
    """
    Apply a 3x3 color transformation matrix to RGB values.
    
    Args:
        rgb_values: RGB array of shape (..., 3)
        matrix: 3x3 color transformation matrix
        
    Returns:
        Transformed RGB values
    """
    # Reshape input to 2D array (n_pixels, 3)
    original_shape = rgb_values.shape
    pixels = rgb_values.reshape(-1, 3)
    
    # Apply matrix transformation
    transformed = np.dot(pixels, matrix.T)
    
    # Reshape back to original dimensions
    return transformed.reshape(original_shape)


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
            
        # Calculate response
        response = transmission * qe_curve
        
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

def setup_matplotlib_style():
    """Set up matplotlib style for report generation."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")
    plt.rcParams.update(MPL_STYLE_CONFIG)


def add_filter_curve_to_matplotlib(ax, x, y, mask, label, color):
    """Add a filter curve to a matplotlib axes."""
    # Plot main curve
    ax.plot(x, y, color=color, linewidth=2, label=label)
    
    # Add extrapolated regions if mask exists
    if mask is not None and np.any(mask):
        extrap_y = y.copy()
        extrap_y[~mask] = np.nan
        ax.plot(x, extrap_y, color=color, linewidth=2, linestyle='--', alpha=0.7)


def generate_report_png(
    selected_filters: List[str],
    current_qe: Dict[str, np.ndarray],
    filter_matrix: np.ndarray,
    df: Any,
    display_to_index: Dict[str, int],
    compute_selected_indices_fn: Callable[[List[str]], List[int]],
    compute_filter_transmission_fn: Callable[[List[int]], Tuple[np.ndarray, str, np.ndarray]],
    compute_effective_stops_fn: Callable[[np.ndarray, np.ndarray], Tuple[float, float]],
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
    
    Args:
        selected_filters: List of selected filter display names
        current_qe: Dictionary of QE data by channel
        filter_matrix: Matrix of filter transmissions
        df: DataFrame with filter metadata
        display_to_index: Mapping from display names to indices
        compute_selected_indices_fn: Function to compute selected indices
        compute_filter_transmission_fn: Function to compute filter transmission
        compute_effective_stops_fn: Function to compute effective stops
        compute_white_balance_gains_fn: Function to compute white balance gains
        masks: Matrix of extrapolation masks
        add_curve_fn: Function to add curves to matplotlib plot
        interp_grid: Wavelength grid for x-axis
        sensor_qe: Mean sensor QE values
        camera_name: Name of selected camera
        illuminant_name: Name of selected illuminant
        sanitize_fn: Function to sanitize filenames
        illuminant_curve: Illuminant curve
        
    Returns:
        Dictionary with report bytes and filename
    """
    # Guard
    if not selected_filters:
        st.warning("⚠️ No filters selected—nothing to export.")
        return {}

    # Sort combo name
    combo = []
    for name in sorted(selected_filters):
        idx = display_to_index.get(name)
        row = df.iloc[idx]
        combo.append((row['Manufacturer'], row['Filter Number'], row))
    combo_name = ", ".join(f"{m} {n}" for m, n, _ in combo)

    # Resolve indices
    selected_indices = compute_selected_indices_fn(selected_filters)
    if not selected_indices:
        st.warning("⚠️ Invalid filter selection—cannot resolve indices.")
        return {}

    # Compute curves
    trans, label, combined = compute_filter_transmission_fn(selected_indices)
    active_trans = combined if combined is not None else trans
    avg_trans, stops = compute_effective_stops_fn(active_trans, sensor_qe)
    wb = compute_white_balance_gains_fn(active_trans, current_qe, illuminant_curve)

    # Style & figure
    setup_matplotlib_style()
    fig = plt.figure(figsize=(8, 14), dpi=150, constrained_layout=False)
    gs = GridSpec(5, 1, figure=fig, height_ratios=[1.2, 0.6, 3.2, 0.8, 3.2])

    # 1: Filter swatches
    ax0 = fig.add_subplot(gs[0])
    ax0.axis('off')
    y0 = 0.9
    counts = {f: selected_filters.count(f) for f in set(selected_filters)}
    for name, cnt in counts.items():
        idx = display_to_index[name]
        row = df.iloc[idx]
        hexc = row.get('Hex Color', '#000000')
        rect = Rectangle((0.0, y0-0.15), 0.03, 0.1, transform=ax0.transAxes,
                         facecolor=hexc, edgecolor='black', lw=0.5)
        ax0.add_patch(rect)
        ax0.text(0.03, y0-0.1, f"{row['Manufacturer']} – {row['Filter Name']} (#{row['Filter Number']}) ×{cnt}",
                 transform=ax0.transAxes, fontsize=10, va='center')
        y0 -= 0.15

    # 2: Light loss
    ax1 = fig.add_subplot(gs[1])
    ax1.axis('off')
    ax1.text(0.01, 0.7, 'Estimated Light Loss:', fontsize=12, fontweight='bold')
    ax1.text(0.01, 0.3, f"{label} → {stops:.2f} stops (Avg: {avg_trans*100:.1f}%)", fontsize=12)

    # 3: Transmission plot
    ax2 = fig.add_subplot(gs[2])
    for idx in selected_indices:
        row = df.iloc[idx]
        y = np.clip(filter_matrix[idx], 1e-6, 1.0) * 100
        mask = masks[idx]
        add_curve_fn(ax2, interp_grid, y, mask,
                     f"{row['Filter Name']} ({row['Filter Number']})", row.get('Hex Color', '#000000'))
    if len(selected_indices) > 1:
        ax2.plot(interp_grid, active_trans * 100, color='black', lw=2.5, label='Combined Filter')
    ax2.set_title('Filter Transmission (%)')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Transmission (%)')
    ax2.set_xlim(interp_grid.min(), interp_grid.max())
    ax2.set_ylim(0, 100)

    # 4: WB multipliers (convert gains to intensities)
    ax3 = fig.add_subplot(gs[3])
    ax3.axis('off')
    ax3.text(0.01, 0.6, 'White Balance Gains (Green = 1):', fontsize=12, fontweight='bold')

    # Convert gains back to raw intensities (relative to green)
    intensities = {
        'R': 1.0 / wb['R'] if wb['R'] != 0 else 0.0,
        'G': 1.0,
        'B': 1.0 / wb['B'] if wb['B'] != 0 else 0.0
    }

    ax3.text(0.01, 0.4, f"R: {intensities['R']:.3f}   G: {intensities['G']:.3f}   B: {intensities['B']:.3f}", fontsize=12)

    # 5: Sensor-weighted response
    ax4 = fig.add_subplot(gs[4])
    maxresp = 0
    stack = {}
    # Plot in correct RGB order
    for ch in ['R', 'G', 'B']:
        qe = current_qe.get(ch)
        if qe is None:
            continue
        gains = wb.get(ch, 1.0)
        resp = np.nan_to_num(active_trans * (qe / 100)) * 100 / gains
        ax4.plot(
            interp_grid,
            resp,
            label=f"{ch} Channel",
            lw=2,
            color=COLOR_MAP[ch]
        )
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

    ax4.set_title('Sensor-Weighted Response (White-Balanced)', fontsize=14, fontweight='bold')
    subtitle = f"Quantum Efficiency: {camera_name or 'None'}   |   Illuminant: {illuminant_name or 'None'}"
    ax4.text(0.5, 0.98, subtitle, transform=ax4.transAxes, ha='center', va='bottom', fontsize=8)
    ax4.set_xlabel('Wavelength (nm)')
    ax4.set_ylabel('Response (%)')
    ax4.set_xlim(interp_grid.min(), interp_grid.max())
    ax4.set_ylim(0, extent[3] * 1.02)
    ax4.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1.0, 0.95))

    # Finalize
    fig.suptitle(f"Filter Report", fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    # Filename
    fname = sanitize_fn(f"{camera_name}_{illuminant_name}_{combo_name}") + '.png'

    # Save to /outputs/[QE]/[Illuminant] folder
    output_dir = os.path.join("output", sanitize_fn(camera_name), sanitize_fn(illuminant_name))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, fname)
    with open(output_path, "wb") as f:
        f.write(buf.getvalue())

    # Return the report data
    st.success(f"✔️ Report generated: {fname}")
    return {'bytes': buf.getvalue(), 'name': fname}


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
        height=height or PLOT_HEIGHT_DEFAULT,
        hovermode='x unified'
    )
    return fig


def add_filter_curve_to_plotly(fig, x, y, mask, label, color):
    """Add a filter curve to an existing plotly figure."""
    # Convert log scale if needed
    y_display = y.copy()
    
    # Add main curve
    fig.add_trace(go.Scatter(
        x=x,
        y=y_display,
        mode='lines',
        name=label,
        line=dict(color=color, width=2),
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
            line=dict(color=color, width=2, dash='dot'),
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
    plot_height = PLOT_HEIGHT_WITH_SPECTRUM if has_spectrum_strip else PLOT_HEIGHT_DEFAULT
    
    # Apply consistent styling
    apply_plotly_default_style(fig, title, height=plot_height)


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
    for i, idx in enumerate(selected_indices):
        transmission = filter_matrix[idx]
        mask = masks[idx] if masks is not None else np.zeros_like(transmission, dtype=bool)
        name = filter_names[idx]
        color = filter_hex_colors[idx]
        
        # Convert to log scale if needed
        y_data = -np.log2(np.maximum(transmission, 1e-6)) if log_stops else transmission
        
        add_filter_curve_to_plotly(fig, interp_grid, y_data, mask, name, color)
    
    # Add combined transmission if available
    if combined is not None:
        y_data = -np.log2(np.maximum(combined, 1e-6)) if log_stops else combined
        fig.add_trace(go.Scatter(
            x=interp_grid,
            y=y_data,
            mode='lines',
            name='Combined',
            line=dict(color='black', width=3),
            showlegend=True
        ))
    
    # Add target profile if available
    if target_profile and hasattr(target_profile, 'values') and hasattr(target_profile, 'valid'):
        valid_mask = target_profile.valid
        target_values = target_profile.values
        if log_stops:
            # Convert percentage to fraction, then to stops
            target_values = -np.log2(np.maximum(target_values / 100.0, 1e-6))
            
        fig.add_trace(go.Scatter(
            x=interp_grid[valid_mask],
            y=target_values[valid_mask],
            mode='lines',
            name='Target',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=True
        ))
    
    # Update layout
    y_title = "Light Loss (stops)" if log_stops else "Transmission"
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
    target_profile: Optional[Any],
    channel_mixer: Optional[ChannelMixerSettings] = None,
    # Added parameters for configurability
    spectrum_strip_height_pct: float = 0.05,
    spectrum_strip_position_pct: float = 1.02,
    saturation_scaling_factor: float = 5.0,
    min_saturation: float = 0.15
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
            line=dict(color=COLOR_MAP.get(channel, 'gray'), width=2),
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
            saturation_scaling_factor=saturation_scaling_factor, 
            min_saturation=min_saturation
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
        spectrum_y_pos = max_response * spectrum_strip_position_pct
        
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
            line=dict(color='black', width=2, dash='dash'),
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


def create_qe_figure(
    interp_grid: np.ndarray,
    qe_data: Dict[str, np.ndarray],
    visible_channels: Dict[str, bool],
    height: int = 300
) -> go.Figure:
    """Create a QE response figure."""
    fig = go.Figure()
    
    for channel, curve in qe_data.items():
        if visible_channels.get(channel, True):
            fig.add_trace(go.Scatter(
                x=interp_grid,
                y=curve,
                mode='lines',
                name=f'{channel} QE',
                line=dict(color=COLOR_MAP.get(channel, 'gray'), width=2)
            ))
    
    apply_plotly_default_style(fig, "Quantum Efficiency", y_title="QE", height=height)
    
    return fig


def create_illuminant_figure(
    interp_grid: np.ndarray,
    illuminant: np.ndarray,
    illuminant_name: str,
    height: int = 300
) -> go.Figure:
    """Create an illuminant figure."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=interp_grid,
        y=illuminant,
        mode='lines',
        name=illuminant_name,
        line=dict(color='orange', width=2)
    ))
    
    apply_plotly_default_style(fig, "Illuminant Spectrum", y_title="Relative Power", height=height)
    
    return fig


def create_sparkline_plot(
    x_data: np.ndarray,
    y_data: np.ndarray,
    color: str = "blue",
    height: int = 150,  # Increased height for better visibility
    width: int = 300
) -> go.Figure:
    """Create a small sparkline plot with grid and axes."""
    fig = go.Figure()
    
    # Convert transmission values to percentage for display
    y_data_pct = y_data * 100
    
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data_pct,
        mode='lines',
        line=dict(color=color, width=2),
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
            gridcolor='rgba(200,200,200,0.4)',
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
            gridcolor='rgba(200,200,200,0.4)',
            showticklabels=True,
            title=dict(
                text="Transmission (%)",
                font=dict(size=10)
            ),
            tickfont=dict(size=8),
            range=[0, 100]  # Set y-axis range to 0-100%
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig