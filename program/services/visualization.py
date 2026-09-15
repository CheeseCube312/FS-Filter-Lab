"""
Visualization utilities for FS FilterLab.

Plotly chart creation for UI views, plus shared styling helpers.
"""
# Standard library imports
from typing import List, Dict, Optional

# Third-party imports
import numpy as np
import plotly.graph_objects as go

# Local imports
from models.core import ChannelMixerSettings, SensorProfile
from models.constants import (
    CHART_HEIGHTS, CHART_LINE_STYLES, CHART_COLORS, EPSILON,
    SENSOR_RESPONSE_DEFAULTS, SENSOR_TYPE_LABELS
)
from services.spectra import apply_channel_mixing_to_responses


def _invert_negative(responses: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Colour-negative film: flip the latent image for display.

    The latent image of a negative stores dye densities, so the visualisation
    applies a peak-relative inversion (peak - response) to render a positive.
    Peak-relative rather than absolute so it works at any display scale.
    """
    max_val = max(
        (np.nanmax(curve) for curve in responses.values() if len(curve) > 0),
        default=1.0
    )
    max_val = max_val if max_val > 0 else 1.0
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
