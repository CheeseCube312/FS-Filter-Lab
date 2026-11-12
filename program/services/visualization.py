"""
Visualization utilities for FS FilterLab.

Plotly chart creation for UI views, plus shared styling helpers.
"""
# Standard library imports
import io
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

# Local imports
from models.core import ChannelMixerSettings, FilterCollection, SensorProfile
from models.constants import (
    CHART_HEIGHTS, CHART_LINE_STYLES, CHART_COLORS, EPSILON, GRID_HEIGHT_RATIOS,
    INTERP_GRID, MPL_STYLE_CONFIG, OUTPUT_FOLDERS, REPORT_CONFIG,
    SENSOR_RESPONSE_DEFAULTS, SENSOR_TYPE_LABELS, SPECTRUM_STRIP,
    VISIBLE_MAX_NM, VISIBLE_MIN_NM,
)
from services.calculations import (
    compute_filter_transmission,
    compute_selected_filter_indices,
    compute_effective_stops,
    photopic_relative_brightness,
    spectral_locus_linear_srgb,
    srgb_encode,
)
from services.spectra import apply_channel_mixing_to_responses
from services.state_manager import AppState

logger = logging.getLogger(__name__)


def _peak_of_curves(responses: Dict[str, np.ndarray]) -> float:
    """Highest finite value across all curves, floored at 1.0."""
    peak = 1.0
    for curve in responses.values():
        if curve is None or len(curve) == 0:
            continue
        clean = np.nan_to_num(curve, nan=0.0, posinf=0.0, neginf=0.0)
        m = float(np.max(clean))
        if np.isfinite(m) and m > peak:
            peak = m
    return peak


def _invert_negative(
    responses: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Colour-negative film: flip the latent image for display.

    The latent image of a negative stores dye densities, so the visualisation
    applies a peak-relative inversion (peak - response) to render a positive.
    Peak-relative rather than absolute so it works at any display scale.
    """
    max_val = _peak_of_curves(responses)
    return {channel: max_val - curve for channel, curve in responses.items()}


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
    # A cleared sensor select leaves current_qe None: render an empty chart
    # (the presenter shows its info message) instead of crashing.
    for channel, qe_curve in (qe_data or {}).items():
        if not visible_channels.get(channel, True):
            continue
            
        # Calculate response and convert to percentage scale for display
        response = transmission * qe_curve * 100
        
        # Apply white balance if requested (a zero/near-zero gain is floored
        # at EPSILON so it cannot produce inf responses)
        if apply_white_balance:
            wb_gain = white_balance_gains.get(channel, 1.0)
            response = response / max(wb_gain, EPSILON)
        
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


def _add_filter_curve_to_plotly(fig, x, y, mask, label, color, emphasize=True):
    """Add a filter transmission curve (plus dotted extrapolation) to a Plotly figure.

    ``emphasize=False`` renders a faint underlay stroke (multi-filter stacks):
    the filter's own contribution stays visible while the bold Combined line
    carries the reading.
    """
    y_display = y.copy()
    opacity = 1.0 if emphasize else 0.35
    main_style = _create_line_style(color) if emphasize else dict(color=color, width=1.4)

    # Add main curve
    fig.add_trace(go.Scatter(
        x=x,
        y=y_display,
        mode='lines',
        name=label,
        line=main_style,
        opacity=opacity,
        showlegend=True
    ))

    # Add extrapolated regions if mask exists
    if mask is not None and np.any(mask):
        extrap_y = y_display.copy()
        extrap_y[~mask] = np.nan
        extrap_style = (
            _create_line_style(color, dash='dot')
            if emphasize else dict(color=color, width=1.4, dash='dot')
        )

        fig.add_trace(go.Scatter(
            x=x,
            y=extrap_y,
            mode='lines',
            name=f'{label} (extrapolated)',
            line=extrap_style,
            opacity=opacity,
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
        name='This camera'
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

    max_response = _peak_of_curves(responses)
    spectrum_y_pos = max_response * SENSOR_RESPONSE_DEFAULTS['spectrum_strip_position_pct']
    _add_spectrum_strip_to_plot(fig, interp_grid, rgb_matrix, relative_brightness, spectrum_y_pos)


def _human_vision_strip_colors(interp_grid: np.ndarray) -> Tuple[List[str], List[str]]:
    """
    Per-wavelength (color, hover) pairs for the human-vision spectrum strip.

    The strip is a hue legend: visible wavelengths get the CIE-1931 2-deg
    spectral locus normalized per wavelength (strongest channel at full
    scale, R:G:B ratios preserved) so every core hue stays readable for
    identifying what a stack blocks.  The edges of the visible band taper
    linearly to black over SPECTRUM_STRIP['edge_taper_nm'], marking where
    human vision fades out (and where the CIE fit is least reliable).
    Photopic V(lambda) is not applied to the pixels — its true value stays
    in the hover text instead.  The strip covers only the visible band:
    UV and IR get empty entries (not drawn) — with no human-visible color
    to show, the legend simply fades out at the band edges.
    """
    colors = [''] * len(interp_grid)
    hovers = [''] * len(interp_grid)
    visible_idx = np.flatnonzero(
        (interp_grid >= VISIBLE_MIN_NM) & (interp_grid <= VISIBLE_MAX_NM))

    if len(visible_idx):
        vis_wl = interp_grid[visible_idx]
        locus = spectral_locus_linear_srgb(vis_wl)
        # Per-wavelength chroma: divide each row by its own peak channel, not
        # a global peak, so core hues all read at full scale.
        row_peak = np.maximum(locus.max(axis=1, keepdims=True), EPSILON)
        hue = np.clip(locus / row_peak, 0.0, 1.0)
        # Edges of vision taper to dark: linear fade at each band end.
        taper_nm = SPECTRUM_STRIP['edge_taper_nm']
        taper = np.clip(np.minimum(
            (vis_wl - VISIBLE_MIN_NM) / taper_nm,
            (VISIBLE_MAX_NM - vis_wl) / taper_nm), 0.0, 1.0)
        encoded = srgb_encode(np.clip(hue * taper[:, np.newaxis], 0.0, 1.0))
        v_lambda = photopic_relative_brightness(vis_wl)
        for k, idx in enumerate(visible_idx):
            colors[idx] = 'rgb({},{},{})'.format(
                *(int(round(c * 255)) for c in encoded[k]))
            hovers[idx] = (f"{interp_grid[idx]} nm · human vision · "
                           f"{v_lambda[k] * 100:.1f}% brightness")

    return colors, hovers


def _add_human_spectrum_strip(
    fig: go.Figure,
    interp_grid: np.ndarray,
    y_position: float,
) -> None:
    """
    Human-vision hue reference band at *y_position*: one square marker per
    nanometre of the visible band, colored by its perceived hue; the band's
    edges taper to dark and true photopic brightness stays in the hover
    text.  UV / IR are not drawn — outside human vision, so the strip just
    ends.
    """
    visible = (interp_grid >= VISIBLE_MIN_NM) & (interp_grid <= VISIBLE_MAX_NM)
    grid = interp_grid[visible]
    colors, hovers = _human_vision_strip_colors(grid)
    fig.add_trace(go.Scatter(
        x=grid,
        y=[y_position] * len(grid),
        mode='markers',
        marker=dict(
            size=SPECTRUM_STRIP['marker_size'],
            color=colors,
            line=dict(width=0),
            symbol='square',
        ),
        text=hovers,
        hoverinfo='text',
        showlegend=False,
        name='Human vision',
    ))


def _update_response_plot_layout(
    fig: go.Figure,
    is_white_balanced: bool,
    is_channel_mixed: bool,
    profile: Optional['SensorProfile'] = None,
) -> None:
    """Set sensor-response plot title (WB/mixer status, film label) and height.

    The human-vision strip is always drawn, so the taller spectrum layout is
    always used.
    """
    wb_status = " (White Balanced)" if is_white_balanced else ""
    mixer_status = " (Channel Mixed)" if is_channel_mixed else ""

    if profile is not None and profile.is_film:
        type_label = SENSOR_TYPE_LABELS.get(profile.sensor_type, "Film")
        title = f"{type_label} Response{wb_status}{mixer_status}"
    else:
        title = f"Sensor Response{wb_status}{mixer_status}"

    _apply_plotly_default_style(
        fig, title, y_title="Response (%)",
        height=CHART_HEIGHTS['plot_with_spectrum'])


def create_filter_response_plot(
    interp_grid: np.ndarray,
    filter_matrix: np.ndarray,
    masks: np.ndarray,
    selected_indices: List[int],
    combined: Optional[np.ndarray],
    log_stops: bool,
    filter_names: List[str],
    filter_hex_colors: List[str],
    stack_entries: Optional[List[Tuple[str, str, int, int]]] = None,
) -> go.Figure:
    """Create a plotly figure showing filter response curves.

    stack_entries: presenter-prepared ``[(name, hex_color, matrix_index,
    multiplier)]`` in stack order. When given, each unique filter renders as
    a faint trace of its multiplied curve (``row ** multiplier``, i.e. T^N)
    so the bold Combined line reads as their product; repeated entries
    collapse into one trace. None keeps the legacy per-index rendering
    (individual curves bold when no Combined line exists, faint when it does).
    """
    fig = go.Figure()
    displayed_curves: List[np.ndarray] = []

    if stack_entries is not None:
        seen: set = set()
        for name, color, filter_index, multiplier in stack_entries:
            if filter_index in seen:
                continue
            seen.add(filter_index)
            label = name if multiplier == 1 else f"{name} ×{multiplier}"
            transmission = filter_matrix[filter_index] ** multiplier
            mask = masks[filter_index] if masks is not None else np.zeros_like(transmission, dtype=bool)

            # Convert to log scale or percentage scale for display
            y_data = -np.log2(np.maximum(transmission, 1e-6)) if log_stops else transmission * 100

            displayed_curves.append(y_data)
            _add_filter_curve_to_plotly(
                fig, interp_grid, y_data, mask, label, color,
                emphasize=combined is None)
    else:
        # Add individual filter curves
        for filter_index in selected_indices:
            transmission = filter_matrix[filter_index]
            mask = masks[filter_index] if masks is not None else np.zeros_like(transmission, dtype=bool)
            name = filter_names[filter_index]
            color = filter_hex_colors[filter_index]

            # Convert to log scale or percentage scale for display
            y_data = -np.log2(np.maximum(transmission, 1e-6)) if log_stops else transmission * 100

            displayed_curves.append(y_data)
            _add_filter_curve_to_plotly(
                fig, interp_grid, y_data, mask, name, color,
                emphasize=combined is None)
    
    # Add combined transmission if available
    if combined is not None:
        y_data = -np.log2(np.maximum(combined, 1e-6)) if log_stops else combined * 100
        displayed_curves.append(y_data)
        fig.add_trace(go.Scatter(
            x=interp_grid,
            y=y_data,
            mode='lines',
            name='Combined',
            line=_create_line_style(CHART_COLORS['text'], 'thick'),
            showlegend=True
        ))
    
    # Update layout (taller: the human-vision strip needs headroom above the curves)
    y_title = 'Transmission (stops)' if log_stops else 'Transmission (%)'
    fig = _apply_plotly_default_style(
        fig, "Filter Response", y_title=y_title,
        height=CHART_HEIGHTS['plot_with_spectrum'])

    # Human-vision reference strip above the curves: perceived hue per
    # wavelength (CIE-1931 observer; visible band only, edges taper to black)
    strip_peak = _peak_of_curves(dict(enumerate(displayed_curves)))
    _add_human_spectrum_strip(
        fig, interp_grid, strip_peak * SPECTRUM_STRIP['solo_position_pct'])
    
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
    channel_mixer: Optional[ChannelMixerSettings] = None,
    sensor_profile: Optional['SensorProfile'] = None,
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
        responses = _invert_negative(responses)

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

    # Camera-capture strip if we have full RGB data (skip for B&W)
    has_rgb = not is_bw and all(ch in responses for ch in ('R', 'G', 'B'))
    if has_rgb:
        _add_response_spectrum_strip(fig, interp_grid, responses)

    # Human-vision reference strip stacked above the camera strip: perceived
    # hue per wavelength, for direct capture comparison
    peak_response = _peak_of_curves(responses)
    _add_human_spectrum_strip(
        fig, interp_grid,
        peak_response * SENSOR_RESPONSE_DEFAULTS['human_strip_position_pct'],
    )

    _update_response_plot_layout(
        fig,
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
                               camera_name: str, illuminant_name: str,
                               apply_white_balance: bool = False):
    """Add sensor-weighted response section to the report."""
    maxresp = 0
    stack = {}
    
    # Plot in correct RGB order
    for ch in ['R', 'G', 'B']:
        qe = current_qe.get(ch)
        if qe is None:
            continue
        # Unmeasured wavelengths stay NaN: the plot shows gaps, never zeros
        resp = active_trans * qe * 100
        if apply_white_balance:
            resp = resp / max(wb.get(ch, 1.0), EPSILON)
        ax4.plot(interp_grid, resp, label=f"{ch} Channel", lw=REPORT_CONFIG['channel_line_width'], color=COLOR_MAP[ch])
        if np.isfinite(resp).any():
            maxresp = max(maxresp, np.nanmax(resp))
        stack[ch] = resp

    # Spectrum strip
    rgb_matrix = np.stack([
        stack.get('R', np.zeros_like(active_trans)),
        stack.get('G', np.zeros_like(active_trans)),
        stack.get('B', np.zeros_like(active_trans))
    ], axis=1)
    finite = rgb_matrix[np.isfinite(rgb_matrix)]
    mv = finite.max() if finite.size else 0.0
    if mv > 0:
        rgb_matrix /= mv
    
    extent = [interp_grid.min(), interp_grid.max(), maxresp * 1.02, maxresp * 1.07]
    ax4.imshow(rgb_matrix[np.newaxis, :, :], aspect='auto', extent=extent)

    suffix = " (White-Balanced)" if apply_white_balance else " (Raw)"
    ax4.set_title(f'Sensor-Weighted Response{suffix}', fontsize=REPORT_CONFIG['font_sizes']['title'], fontweight='bold')
    subtitle = f"Quantum Efficiency: {camera_name or 'None'}   |   Illuminant: {illuminant_name or 'None'}"
    ax4.text(0.5, 0.98, subtitle, transform=ax4.transAxes, ha='center', va='bottom', fontsize=REPORT_CONFIG['font_sizes']['subtitle'])
    ax4.set_xlabel('Wavelength (nm)')
    ax4.set_ylabel('Response (%)')
    ax4.set_xlim(interp_grid.min(), interp_grid.max())
    ax4.set_ylim(0, extent[3] * 1.02)
    ax4.legend(loc='upper right', fontsize=REPORT_CONFIG['font_sizes']['legend'], bbox_to_anchor=(1.0, 0.95))


def _setup_matplotlib_style():
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


def _add_filter_curve_to_matplotlib(ax, x, y, mask, label, color):
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
    effective_stops: float,
    avg_transmission: float,
    white_balance_gains: Dict[str, float],
    current_qe: Optional[Dict[str, np.ndarray]],
    camera_name: str,
    illuminant_name: str,
    filter_df: Any,
    display_to_index: Dict[str, int],
    filter_matrix: np.ndarray,
    masks: np.ndarray,
    interp_grid: np.ndarray,
    sanitize_fn: Callable[[str], str],
    apply_white_balance: bool = False,
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
    _setup_matplotlib_style()
    fig = plt.figure(figsize=REPORT_CONFIG['figure_size'], dpi=REPORT_CONFIG['dpi'], constrained_layout=False)
    gs = GridSpec(5, 1, figure=fig, height_ratios=GRID_HEIGHT_RATIOS)

    # Build report sections
    _add_filter_swatches_section(
        fig.add_subplot(gs[0]), selected_filters,
        filter_df, display_to_index
    )
    _add_light_loss_section(fig.add_subplot(gs[1]), transmission_label, effective_stops, avg_transmission)
    _add_transmission_plot_section(
        fig.add_subplot(gs[2]), selected_indices, filter_df,
        filter_matrix, masks,
        _add_filter_curve_to_matplotlib, interp_grid, active_transmission
    )
    _add_white_balance_section(fig.add_subplot(gs[3]), white_balance_gains)
    _add_sensor_response_section(
        fig.add_subplot(gs[4]), current_qe, white_balance_gains,
        active_transmission, interp_grid,
        camera_name, illuminant_name, apply_white_balance
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
    return {'bytes': png_bytes, 'name': fname, 'path': os.path.join(output_dir, fname)}


def generate_application_report(
    app_state: AppState,
    filter_collection: FilterCollection,
    selected_camera: Optional[str] = None,
) -> Optional[Dict]:
    """
    Generate a PNG report of the current filter configuration.

    Args:
        app_state: Current application state
        filter_collection: Available filters
        selected_camera: Name of selected camera (optional)

    Returns:
        Report result dict if successful, None otherwise.
    """
    # Get selected filter indices
    selected_indices = compute_selected_filter_indices(
        app_state.selected_filters,
        app_state.filter_multipliers,
        filter_collection
    )

    if not selected_indices:
        return None

    # Get filter transmission
    transmission, label = compute_filter_transmission(
        selected_indices, filter_collection.filter_matrix
    )

    # Resolve sensor QE and illuminant
    sensor_qe = (
        app_state.current_qe.get('G') if app_state.current_qe else None
    )
    illuminant = (
        app_state.illuminant if app_state.illuminant is not None
        else np.ones_like(INTERP_GRID)
    )

    # Compute metrics
    avg_trans, stops = (
        compute_effective_stops(transmission, sensor_qe, illuminant)
        if sensor_qe is not None else (0.0, 0.0)
    )
    # Mirror the on-screen numbers exactly: the action layer keeps
    # white_balance_gains in sync (standard gains on stack/sensor/illuminant
    # changes, surface-referenced gains after WB-from-surface). Recomputing
    # standard gains here would silently override a surface reference.
    wb = dict(app_state.white_balance_gains)

    camera_name = selected_camera or "UnknownCamera"
    illuminant_name = app_state.illuminant_name or "UnknownIlluminant"

    result = generate_report_png(
        selected_filters=app_state.selected_filters,
        selected_indices=selected_indices,
        active_transmission=transmission,
        transmission_label=label,
        effective_stops=stops,
        avg_transmission=avg_trans,
        white_balance_gains=wb,
        apply_white_balance=app_state.apply_white_balance,
        current_qe=app_state.current_qe,
        camera_name=camera_name,
        illuminant_name=illuminant_name,
        filter_df=filter_collection.df,
        display_to_index=filter_collection.get_display_to_index_map(),
        filter_matrix=filter_collection.filter_matrix,
        masks=filter_collection.extrapolated_masks,
        interp_grid=INTERP_GRID,
        sanitize_fn=sanitize_filename_component,
    )

    return result or None