"""
Visualization utilities for FS FilterLab.

This module provides all visualization functionality including:
- Interactive plotting with Plotly
- Static report generation with Matplotlib  
- Chart creation and styling utilities
- PNG report generation
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

# ============================================================================
# MATPLOTLIB REPORT GENERATION
# ============================================================================

def setup_matplotlib_style():
    """Set up matplotlib style for report generation."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")
    plt.rcParams.update({
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
    })


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
    color_map = {'R': 'red', 'G': 'green', 'B': 'blue'}
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
            color=color_map[ch]
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


def add_filter_curve_to_matplotlib(ax, x, y, mask, label, color):
    """Add a filter curve to a matplotlib axes."""
    # Plot main curve
    ax.plot(x, y, color=color, linewidth=2, label=label)
    
    # Add extrapolated regions if mask exists
    if mask is not None and np.any(mask):
        extrap_y = y.copy()
        extrap_y[~mask] = np.nan
        ax.plot(x, extrap_y, color=color, linewidth=2, linestyle='--', alpha=0.7)


# ============================================================================
# PLOTLY INTERACTIVE PLOTTING
# ============================================================================

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
    layout_update = {
        'title': "Filter Response",
        'xaxis_title': "Wavelength (nm)",
        'yaxis_title': y_title,
        'template': "plotly_white",
        'height': 400,
        'hovermode': 'x unified'
    }
    
    # Invert y-axis for log view (high transmission = low stops = bottom of plot)
    if log_stops:
        layout_update['yaxis'] = {'autorange': 'reversed'}
    
    fig.update_layout(**layout_update)
    
    return fig


def create_sensor_response_plot(
    interp_grid: np.ndarray,
    transmission: np.ndarray,
    qe_data: Dict[str, np.ndarray],
    visible_channels: Dict[str, bool],
    white_balance_gains: Dict[str, float],
    apply_white_balance: bool,
    target_profile: Optional[Any]
) -> go.Figure:
    """Create a plotly figure showing sensor response."""
    fig = go.Figure()
    
    colors = {'R': 'red', 'G': 'green', 'B': 'blue'}
    
    for channel, qe_curve in qe_data.items():
        if not visible_channels.get(channel, True):
            continue
            
        # Calculate response
        response = transmission * qe_curve
        
        # Apply white balance if requested
        if apply_white_balance:
            wb_gain = white_balance_gains.get(channel, 1.0)
            # White balance gains represent response ratios, so we need the inverse to balance
            response = response / wb_gain
        
        fig.add_trace(go.Scatter(
            x=interp_grid,
            y=response,
            mode='lines',
            name=f'{channel} Channel',
            line=dict(color=colors.get(channel, 'gray'), width=2),
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title="Sensor Response",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Response",
        template="plotly_white",
        height=400,
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


def create_sparkline_plot(
    x_data: np.ndarray,
    y_data: np.ndarray,
    color: str = "blue",
    height: int = 100,
    width: int = 300
) -> go.Figure:
    """Create a small sparkline plot."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='lines',
        line=dict(color=color, width=2),
        showlegend=False
    ))
    
    fig.update_layout(
        height=height,
        width=width,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
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
    
    colors = {'R': 'red', 'G': 'green', 'B': 'blue'}
    
    for channel, curve in qe_data.items():
        if visible_channels.get(channel, True):
            fig.add_trace(go.Scatter(
                x=interp_grid,
                y=curve,
                mode='lines',
                name=f'{channel} QE',
                line=dict(color=colors.get(channel, 'gray'), width=2)
            ))
    
    fig.update_layout(
        title="Quantum Efficiency",
        xaxis_title="Wavelength (nm)",
        yaxis_title="QE",
        height=height,
        template="plotly_white"
    )
    
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
    
    fig.update_layout(
        title="Illuminant Spectrum",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Relative Power",
        height=height,
        template="plotly_white"
    )
    
    return fig
