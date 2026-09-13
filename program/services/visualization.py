"""
Visualization utilities for FS FilterLab.

Plotly chart creation, Matplotlib PNG report generation, and styling helpers.
"""
# Standard library imports
import io
import os
from typing import List, Dict, Optional, Any, Callable, Tuple

# Third-party imports
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import plotly.graph_objects as go

# Local imports
from models.core import ChannelMixerSettings, SensorProfile
from models.constants import (
    CHART_HEIGHTS, CHART_LINE_STYLES, CHART_COLORS, PLOT_LAYOUT,
    SENSOR_RESPONSE_DEFAULTS, MPL_STYLE_CONFIG, REPORT_CONFIG, OUTPUT_FOLDERS,
    SPARKLINE_CONFIG, SENSOR_TYPE_LABELS
)
from services.channel_mixer import apply_channel_mixing_to_responses
from services.spectra import invert_for_display

logger = logging.getLogger(__name__)

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
    """Calculate per-channel responses with optional white balancing and mixing."""
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
        responses = apply_channel_mixing_to_responses(responses, channel_mixer)
    
    return responses


def _create_line_style(color: str, style: str = 'default', dash: str = None) -> dict:
    """Create a standardized line style dictionary."""
    line_dict = {
        'color': color,
        'width': CHART_LINE_STYLES[style]['width'],
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
    """Compute per-wavelength display colors; returns an (n, 3) RGB matrix in [0, 1]."""
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
    """Normalize RGB for display, mimicking per-channel sensor saturation."""
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
    """Create sorted (manufacturer, number, row) combo info and joined name."""
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


def setup_matplotlib_style():
    """Configure matplotlib: best-available seaborn grid style + custom config."""
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
    """Add a filter transmission curve (plus dashed extrapolation) to matplotlib axes."""
    mpl_width = CHART_LINE_STYLES['matplotlib']['width']
    extrap = CHART_LINE_STYLES['extrapolated']

    # Plot main curve
    ax.plot(x, y, color=color, linewidth=mpl_width, label=label)
    
    # Add extrapolated regions if mask exists
    if mask is not None and np.any(mask):
        extrap_y = y.copy()
        extrap_y[~mask] = np.nan
        ax.plot(x, extrap_y, color=color, linewidth=mpl_width, 
                linestyle=extrap['style'], alpha=extrap['alpha'])


def generate_report_png(
    selected_filters: List[str],
    selected_indices: List[int],
    active_transmission: np.ndarray,
    transmission_label: str,
    combined_transmission: Optional[np.ndarray],
    effective_stops: float,
    avg_transmission: float,
    white_balance_gains: Dict[str, float],
    current_qe: Optional[Dict[str, np.ndarray]],
    sensor_qe: Optional[np.ndarray],
    camera_name: str,
    illuminant_name: str,
    filter_df: Any,
    display_to_index: Dict[str, int],
    filter_matrix: np.ndarray,
    masks: np.ndarray,
    interp_grid: np.ndarray,
    sanitize_fn: Callable[[str], str],
) -> Dict[str, Any]:
    """Generate a multi-panel PNG report of the current filter configuration.

    All computations (transmission, stops, white balance) are done by the
    caller; this function only renders. Returns {'bytes', 'name'} or {}.
    """
    if not selected_filters:
        logger.warning("No filters selected — nothing to export.")
        return {}

    # Prepare filter combination data
    combo, combo_name = _create_filter_combo_info(
        selected_filters, filter_df, display_to_index
    )

    # Create figure with layout
    setup_matplotlib_style()
    fig = plt.figure(figsize=REPORT_CONFIG['figure_size'], dpi=REPORT_CONFIG['dpi'], constrained_layout=False)
    gs = GridSpec(5, 1, figure=fig, height_ratios=PLOT_LAYOUT['grid_height_ratios'])

    # Build report sections
    _add_filter_swatches_section(
        fig.add_subplot(gs[0]), selected_filters,
        filter_df, display_to_index
    )
    _add_light_loss_section(fig.add_subplot(gs[1]), transmission_label, effective_stops, avg_transmission)
    _add_transmission_plot_section(
        fig.add_subplot(gs[2]), selected_indices, filter_df,
        filter_matrix, masks,
        add_filter_curve_to_matplotlib, interp_grid, active_transmission
    )
    _add_white_balance_section(fig.add_subplot(gs[3]), white_balance_gains)
    _add_sensor_response_section(
        fig.add_subplot(gs[4]), current_qe, white_balance_gains,
        active_transmission, interp_grid,
        camera_name, illuminant_name
    )

    # Finalize layout
    fig.suptitle("Filter Report", fontsize=REPORT_CONFIG['font_sizes']['main_title'], fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    # Render to PNG bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    png_bytes = buf.getvalue()

    fname = sanitize_fn(
        f"{camera_name}_{illuminant_name}_{combo_name}"
    ) + '.png'

    # Save to output folder
    output_dir = os.path.join(
        OUTPUT_FOLDERS['reports'],
        sanitize_fn(camera_name),
        sanitize_fn(illuminant_name),
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, fname), "wb") as f:
        f.write(png_bytes)

    logger.info("Report generated: %s", fname)
    return {'bytes': png_bytes, 'name': fname}


# =============================================================================
# PLOTLY VISUALIZATION
# =============================================================================

def _apply_plotly_default_style(fig, title, x_title="Wavelength (nm)", y_title="Response", height=None):
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


def _add_filter_curve_to_plotly(fig, x, y, mask, label, color):
    """Add a filter transmission curve (plus dotted extrapolation) to a Plotly figure."""
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
    """Add a colored spectrum strip at *y_position* with wavelength/brightness hovers."""
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


def _add_response_spectrum_strip(
    fig: go.Figure,
    interp_grid: np.ndarray,
    responses: Dict[str, np.ndarray],
) -> None:
    """Compute spectral colors from RGB responses and add a spectrum strip to *fig*."""
    rgb_matrix = _calculate_spectral_colors(
        interp_grid,
        responses['R'], responses['G'], responses['B'],
        saturation_scaling_factor=SENSOR_RESPONSE_DEFAULTS['saturation_scaling_factor'],
        min_saturation=SENSOR_RESPONSE_DEFAULTS['min_saturation'],
    )

    # Brightness for hover information
    brightness = np.sum(rgb_matrix, axis=1)
    max_brightness = np.max(brightness) if np.max(brightness) > 0 else 1.0
    relative_brightness = brightness / max_brightness

    # Find the maximum response for positioning the strip
    max_response = 1.0
    for response in responses.values():
        if response is not None and len(response) > 0:
            clean = np.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)
            peak = np.max(clean)
            if np.isfinite(peak) and peak > max_response:
                max_response = peak

    spectrum_y_pos = max_response * SENSOR_RESPONSE_DEFAULTS['spectrum_strip_position_pct']
    _add_spectrum_strip_to_plot(fig, interp_grid, rgb_matrix, relative_brightness, spectrum_y_pos)


def _update_response_plot_layout(
    fig: go.Figure,
    has_spectrum_strip: bool,
    is_white_balanced: bool,
    is_channel_mixed: bool,
    profile: Optional['SensorProfile'] = None,
) -> None:
    """Set sensor-response plot title (WB/mixer status, film label) and height."""
    wb_status = " (White Balanced)" if is_white_balanced else ""
    mixer_status = " (Channel Mixed)" if is_channel_mixed else ""

    if profile is not None and profile.is_film:
        type_label = SENSOR_TYPE_LABELS.get(profile.sensor_type, "Film")
        title = f"{type_label} Response{wb_status}{mixer_status}"
    else:
        title = f"Sensor Response{wb_status}{mixer_status}"

    plot_height = CHART_HEIGHTS['plot_with_spectrum'] if has_spectrum_strip else CHART_HEIGHTS['standard_plot']
    _apply_plotly_default_style(fig, title, y_title="Response (%)", height=plot_height)


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
    for filter_index in selected_indices:
        transmission = filter_matrix[filter_index]
        mask = masks[filter_index] if masks is not None else np.zeros_like(transmission, dtype=bool)
        name = filter_names[filter_index]
        color = filter_hex_colors[filter_index]
        
        # Convert to log scale or percentage scale for display
        y_data = -np.log2(np.maximum(transmission, 1e-6)) if log_stops else transmission * 100
        
        _add_filter_curve_to_plotly(fig, interp_grid, y_data, mask, name, color)
    
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
    if target_profile and hasattr(target_profile, 'transmission'):
        valid_mask = ~np.isnan(target_profile.transmission)
        target_values = target_profile.transmission
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
    fig = _apply_plotly_default_style(fig, "Filter Response", y_title=y_title)
    
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
    channel_mixer: Optional[ChannelMixerSettings] = None,
    sensor_profile: Optional['SensorProfile'] = None,
    film_emulation_channels: Optional[Dict[str, np.ndarray]] = None,
    film_emulation_label: str = "Film Target",
    film_emulation_is_bw: bool = False,
) -> go.Figure:
    """Create a plotly figure showing sensor/film response with optional channel mixing."""
    fig = go.Figure()

    is_bw = sensor_profile is not None and sensor_profile.sensor_type == 'bw'
    invert = sensor_profile is not None and sensor_profile.invert_for_display

    # Calculate channel responses
    responses = _calculate_channel_responses(
        transmission, qe_data, visible_channels,
        white_balance_gains, apply_white_balance, channel_mixer
    )

    # Apply inversion for colour-negative film
    if invert:
        responses = invert_for_display(responses)

    if is_bw:
        # B&W: collapse three identical traces into a single grey luminance trace
        lum = responses.get('R', responses.get('G', responses.get('B')))
        if lum is not None:
            fig.add_trace(go.Scatter(
                x=interp_grid,
                y=lum,
                mode='lines',
                name='Luminance',
                line=_create_line_style('gray'),
                showlegend=True,
            ))
    else:
        # Colour sensor or film: plot R / G / B channels
        for channel, response in responses.items():
            fig.add_trace(go.Scatter(
                x=interp_grid,
                y=response,
                mode='lines',
                name=f'{channel} Channel',
                line=_create_line_style(COLOR_MAP.get(channel, 'gray')),
                showlegend=True,
            ))

    # Add spectrum strip if we have full RGB data (skip for B&W)
    has_rgb = not is_bw and all(ch in responses for ch in ('R', 'G', 'B'))
    if has_rgb:
        _add_response_spectrum_strip(fig, interp_grid, responses)

    # Film emulation target overlay — dashed, desaturated, independently scaled
    if film_emulation_channels:
        # Find peak of primary sensor traces for shared y-scale reference
        primary_peak = max(
            (np.nanmax(r) for r in responses.values() if len(r) > 0),
            default=1.0
        ) or 1.0

        if film_emulation_is_bw:
            lum = next(iter(film_emulation_channels.values()))
            film_peak = np.nanmax(lum) if len(lum) > 0 else 1.0
            scale = primary_peak / film_peak if film_peak > 0 else 1.0
            fig.add_trace(go.Scatter(
                x=interp_grid,
                y=lum * scale,
                mode='lines',
                name=f'{film_emulation_label} — Lum',
                line={'color': '#888888', 'width': 1.5},
                opacity=0.55,
                showlegend=True,
            ))
        else:
            # Desaturated versions of the standard RGB colors
            _DESATURATED = {'R': '#d87f7f', 'G': '#7faf7f', 'B': '#7f9fd8'}
            film_peak = max(
                (np.nanmax(v) for v in film_emulation_channels.values() if len(v) > 0),
                default=1.0
            ) or 1.0
            scale = primary_peak / film_peak
            for ch, vals in film_emulation_channels.items():
                fig.add_trace(go.Scatter(
                    x=interp_grid,
                    y=vals * scale,
                    mode='lines',
                    name=f'{film_emulation_label} — {ch}',
                    line={'color': _DESATURATED.get(ch, '#aaaaaa'), 'width': 1.5},
                    opacity=0.55,
                    showlegend=True,
                ))

    # Add target profile if provided
    if target_profile is not None:
        fig.add_trace(go.Scatter(
            x=interp_grid,
            y=target_profile.transmission,
            mode='lines',
            name=f'Target: {target_profile.name}',
            line=_create_line_style(CHART_COLORS['text'], dash='dash'),
            showlegend=True,
        ))

    _update_response_plot_layout(
        fig,
        has_rgb,
        apply_white_balance,
        channel_mixer and channel_mixer.enabled,
        profile=sensor_profile,
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
    """Create a standard multi-series Plotly figure with optional channel visibility."""
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
    
    _apply_plotly_default_style(fig, title, y_title=y_title, height=height or CHART_HEIGHTS['default'])
    return fig


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


def create_sparkline_plot(
    x_data: np.ndarray,
    y_data: np.ndarray,
    color: str = "blue",
    height: int = None,
    width: int = None
) -> go.Figure:
    """Create a simple sparkline plot for inline display."""
    if height is None:
        height = CHART_HEIGHTS['sparkline']
    if width is None:
        width = SPARKLINE_CONFIG['default_width']
    fig = go.Figure()
    
    # Convert internal 0-1 transmission to 0-100% for display
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
    tick_cfg = SPARKLINE_CONFIG['wavelength_tick_intervals']
    if x_range > tick_cfg['large']['threshold']:
        x_tick_interval = tick_cfg['large']['interval']
    elif x_range > tick_cfg['medium']['threshold']:
        x_tick_interval = tick_cfg['medium']['interval']
    else:
        x_tick_interval = tick_cfg['small']['interval']
    
    x_ticks = list(range(
        x_min + x_tick_interval - (x_min % x_tick_interval),
        x_max,
        x_tick_interval
    ))
    
    font_sizes = SPARKLINE_CONFIG['font_sizes']
    fig.update_layout(
        height=height,
        width=width,
        margin=SPARKLINE_CONFIG['margins'],
        showlegend=False,
        xaxis=dict(
            showgrid=True,
            gridcolor=CHART_COLORS['grid'],
            showticklabels=True,
            tickvals=x_ticks,
            title=dict(
                text="Wavelength (nm)",
                font=dict(size=font_sizes['axis_title'])
            ),
            tickfont=dict(size=font_sizes['tick_label'])
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=CHART_COLORS['grid'],
            showticklabels=True,
            title=dict(
                text="Transmission (%)",
                font=dict(size=font_sizes['axis_title'])
            ),
            tickfont=dict(size=font_sizes['tick_label']),
            range=[0, 100]  # Set y-axis range to 0-100%
        ),
        paper_bgcolor=CHART_COLORS['transparent'],
        plot_bgcolor=CHART_COLORS['transparent']
    )
    
    return fig