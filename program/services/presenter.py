"""
Presenter service for FS FilterLab.

Bridges the gap between raw business logic (calculations, data) and the UI views.
Prepares ready-to-render data bundles so views contain zero computation logic.

This module is framework-agnostic — it depends only on models and services,
never on any UI toolkit.
"""
# Standard library imports
import colorsys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Local imports
from models.constants import CHART_COLORS, INTERP_GRID, SENSOR_TYPE_LABELS, UI_INFO_MESSAGES
from models.core import AppData, FilterCollection
from services.calculations import (
    compute_filter_transmission,
    compute_selected_filter_indices,
    compute_effective_stops,
    compute_rgb_response,
    compute_active_transmission,
    format_transmission_metrics,
    format_white_balance_data,
    compute_reflector_color,
    compute_white_balance_gains,
)
from services.visualization import (
    create_filter_response_plot,
    create_illuminant_figure,
    create_sensor_response_plot,
    prepare_rgb_for_display,
)
from services.spectra import sensor_channel_curves
from services.state_manager import AppState


# =============================================================================
# VIEW MODEL DATA CLASSES
# =============================================================================

@dataclass
class TransmissionMetricsVM:
    """View model for transmission/light loss metrics display."""
    label: str
    avg_transmission_pct: str
    effective_stops: str
    has_valid_data: bool = True
    error_message: Optional[str] = None


@dataclass
class WhiteBalanceVM:
    """View model for white balance display."""
    gains: Dict[str, float]
    intensities: Dict[str, str]  # Pre-formatted strings
    has_filters: bool
    mode_note: str  # e.g. " (from surface reference)" or ""


@dataclass
class FilterAnalysisVM:
    """View model containing everything the filter analysis view needs to render."""
    transmission: np.ndarray
    label: str
    combined: Optional[np.ndarray]
    effective_transmission: np.ndarray
    metrics: Optional[TransmissionMetricsVM]
    filter_response_fig: go.Figure


@dataclass  
class SensorAnalysisVM:
    """View model containing everything the sensor analysis view needs to render."""
    sensor_response_fig: go.Figure
    wb_display: Optional[WhiteBalanceVM]
    has_qe_and_illuminant: bool
    info_message: Optional[str] = None  # Shown when QE+illuminant not available


# =============================================================================
# FILTER ANALYSIS PRESENTER
# =============================================================================

def prepare_filter_analysis(
    app_state: AppState,
    filter_collection: FilterCollection,
    selected_indices: Optional[List[int]] = None,
    log_stops: Optional[bool] = None,
) -> FilterAnalysisVM:
    """
    Prepare all data needed to render the filter analysis section.
    
    Computes transmission, updates state, calculates metrics,
    and creates the Plotly chart — returning everything as a view model.
    
    Args:
        app_state: Application state manager
        filter_collection: The loaded filter collection
        selected_indices: Indices of selected filters in the collection
        
    Returns:
        FilterAnalysisVM with all data needed for display
    """
    if selected_indices is None:
        selected_indices = compute_selected_filter_indices(
            app_state.selected_filters,
            app_state.filter_multipliers,
            filter_collection,
        )

    # Calculate transmission
    trans, label, combined = compute_filter_transmission(
        selected_indices,
        filter_collection.filter_matrix
    )
    effective_trans = combined if combined is not None else trans
    
    # Update state
    app_state.combined_transmission = effective_trans
    
    # Trigger RGB response computation (needed for downstream effects)
    if app_state.current_qe:
        compute_rgb_response(
            trans,
            app_state.current_qe,
            app_state.white_balance_gains,
            app_state.rgb_channels_visibility,
            app_state.channel_mixer
        )
    
    # Calculate transmission metrics
    metrics = _prepare_transmission_metrics(trans, label, app_state)
    
    # Create filter response chart
    filter_names = [f.name for f in filter_collection.filters]
    filter_hex_colors = [f.hex_color for f in filter_collection.filters]
    
    fig = create_filter_response_plot(
        interp_grid=INTERP_GRID,
        filter_matrix=filter_collection.filter_matrix,
        masks=filter_collection.extrapolated_masks,
        selected_indices=selected_indices,
        combined=combined,
        target_profile=app_state.target_profile,
        log_stops=log_stops if log_stops is not None else app_state.log_view,
        filter_names=filter_names,
        filter_hex_colors=filter_hex_colors
    )
    
    return FilterAnalysisVM(
        transmission=trans,
        label=label,
        combined=combined,
        effective_transmission=effective_trans,
        metrics=metrics,
        filter_response_fig=fig
    )


def _prepare_transmission_metrics(
    trans: np.ndarray,
    label: str,
    app_state: AppState
) -> Optional[TransmissionMetricsVM]:
    """Prepare transmission metrics view model."""
    valid = ~np.isnan(trans)
    if not valid.any():
        return TransmissionMetricsVM(
            label=label,
            avg_transmission_pct="N/A",
            effective_stops="N/A",
            has_valid_data=False,
            error_message=f"Cannot compute average transmission for {label}: insufficient data"
        )
    
    raw_qe = app_state.current_qe.get('G') if app_state.current_qe else None
    if raw_qe is None:
        return TransmissionMetricsVM(
            label=label,
            avg_transmission_pct="N/A",
            effective_stops="N/A",  
            has_valid_data=False,
            error_message=f"Cannot compute light loss for {label}: no sensor QE data"
        )
    
    avg_trans, effective_stops = compute_effective_stops(trans, raw_qe, app_state.illuminant)
    metrics = format_transmission_metrics(trans, label, avg_trans, effective_stops)
    
    return TransmissionMetricsVM(
        label=metrics['label'],
        avg_transmission_pct=metrics['avg_transmission_pct'],
        effective_stops=metrics['effective_stops'],
        has_valid_data=True
    )


# =============================================================================
# SENSOR ANALYSIS PRESENTER  
# =============================================================================

def _compute_film_emulation_target(
    app_state: AppState,
    filter_collection: FilterCollection
) -> Tuple[Optional[Dict[str, np.ndarray]], str, bool]:
    """
    Compute the film-emulation overlay data for the sensor response chart.

    The film target is modelled as an independent capture: the film profile's
    sensitivity curves under the film-emulation filter stack (independent of
    the main stack) and the current illuminant, in percent display scale,
    optionally white-balanced against the film target itself.  The physics of
    the response chain lives in :func:`services.spectra.sensor_channel_curves`.

    Args:
        app_state: Application state manager
        filter_collection: Collection of available filters

    Returns:
        Tuple of (response curves per channel or None, display label, is B&W)
    """
    film_profile = app_state.film_emulation_profile
    if film_profile is None or app_state.illuminant is None:
        return None, "Film Target", False

    illuminant = app_state.illuminant
    film_channels_raw = film_profile.get_channels()

    # Compute independent film-target filter transmission
    emul_filter_names = app_state.film_emulation_filters
    if emul_filter_names:
        emul_indices = compute_selected_filter_indices(
            emul_filter_names, {}, filter_collection
        )
        film_filter = compute_active_transmission(
            emul_filter_names, emul_indices, filter_collection.filter_matrix
        )
    else:
        film_filter = np.ones_like(illuminant)

    # Compute illuminant-weighted sensitivity per channel (percent display scale)
    film_resp = sensor_channel_curves(
        film_filter, film_channels_raw, illuminant, display_scale=100
    )

    # Optionally apply independent WB to the target
    if app_state.film_emulation_wb:
        em_gains = compute_white_balance_gains(film_filter, film_channels_raw, illuminant)
        app_state.film_emulation_wb_gains = em_gains
        for channel in film_resp:
            gain = em_gains.get(channel, 1.0)
            film_resp[channel] = (
                film_resp[channel] / gain if gain > 0 else film_resp[channel]
            )

    return film_resp, film_profile.display_name, film_profile.sensor_type == 'bw'


def prepare_sensor_analysis(
    app_state: AppState,
    data: AppData,
    selected_indices: Optional[List[int]] = None,
    visible_channels: Optional[Dict[str, bool]] = None,
) -> SensorAnalysisVM:
    """
    Prepare all data needed to render the sensor analysis section.
    
    Computes active transmission, white balance, creates sensor response 
    chart, and formats WB display data — returning everything as a view model.
    
    Args:
        app_state: Application state manager
        data: Typed application data container (AppData)
        selected_indices: Indices of selected filters
        
    Returns:
        SensorAnalysisVM with all data needed for display
    """
    if selected_indices is None:
        selected_indices = compute_selected_filter_indices(
            app_state.selected_filters,
            app_state.filter_multipliers,
            data.filter_collection,
        )

    filter_collection = data.filter_collection
    reflector_collection = data.reflector_collection
    
    # Compute active transmission
    trans_interp = compute_active_transmission(
        app_state.selected_filters,
        selected_indices,
        filter_collection.filter_matrix
    )
    
    # Compute and update white balance
    wb_gains = app_state.recompute_white_balance(trans_interp, reflector_collection)

    # Compute film emulation target channels
    film_emulation_channels, film_emulation_label, film_emulation_is_bw = (
        _compute_film_emulation_target(app_state, filter_collection)
    )

    # Create sensor response chart
    fig_response = create_sensor_response_plot(
        interp_grid=INTERP_GRID,
        transmission=trans_interp,
        qe_data=app_state.current_qe,
        visible_channels=(
            visible_channels if visible_channels is not None
            else app_state.rgb_channels_visibility
        ),
        white_balance_gains=wb_gains,
        apply_white_balance=app_state.apply_white_balance,
        target_profile=app_state.target_profile,
        channel_mixer=app_state.channel_mixer,
        sensor_profile=app_state.active_profile,
        film_emulation_channels=film_emulation_channels,
        film_emulation_label=film_emulation_label,
        film_emulation_is_bw=film_emulation_is_bw,
    )
    
    # Prepare white balance display
    has_qe_and_illuminant = (app_state.current_qe is not None 
                             and app_state.illuminant is not None)
    
    wb_display = None
    info_message = None
    
    if has_qe_and_illuminant:
        wb_data = format_white_balance_data(wb_gains, app_state.selected_filters)
        
        if app_state.wb_reference_surface:
            mode_note = " (from surface reference)"
        elif not wb_data["has_filters"]:
            mode_note = " (No filter selected)"
        else:
            mode_note = ""
        
        wb_display = WhiteBalanceVM(
            gains=wb_gains,
            intensities=wb_data['intensities'],
            has_filters=wb_data['has_filters'],
            mode_note=mode_note
        )
    else:
        info_message = UI_INFO_MESSAGES['qe_illuminant_required']
    
    return SensorAnalysisVM(
        sensor_response_fig=fig_response,
        wb_display=wb_display,
        has_qe_and_illuminant=has_qe_and_illuminant,
        info_message=info_message
    )


# =============================================================================
# ADVANCED SEARCH PRESENTER
# =============================================================================

def _filter_by_manufacturer(df: pd.DataFrame, manufacturers: List[str]) -> pd.DataFrame:
    return df if not manufacturers else df[df["Manufacturer"].isin(manufacturers)]


def _filter_by_trans_at_wavelength(
    df: pd.DataFrame, interp_grid: np.ndarray, matrix: np.ndarray,
    wavelength: int, min_t: float = 0.0, max_t: float = 1.0,
) -> Tuple[pd.DataFrame, np.ndarray]:
    idx = np.where(interp_grid == wavelength)[0]
    if idx.size == 0:
        return df, np.zeros(len(df))
    si = idx[0]
    di = df.index.to_numpy()
    tv = matrix[di, si]
    mask = (tv >= min_t) & (tv <= max_t)
    return df.iloc[mask], tv[mask]


def _filter_by_multiple_wavelength_criteria(
    df: pd.DataFrame, interp_grid: np.ndarray, matrix: np.ndarray,
    criteria: List[Dict],
) -> pd.DataFrame:
    """Narrow *df* to filters matching ALL wavelength/transmission criteria (AND)."""
    if not criteria:
        return df
    filtered_df = df
    for criterion in criteria:
        # UI percentages -> 0-1 fractions
        filtered_df, _ = _filter_by_trans_at_wavelength(
            filtered_df, interp_grid, matrix,
            criterion["wavelength"],
            criterion["trans_min"] / 100.0,
            criterion["trans_max"] / 100.0,
        )
    return filtered_df


def _owned_only(df: pd.DataFrame, owned: set) -> pd.DataFrame:
    """Restrict *df* to rows whose display name is in the owned set."""
    def _display_name(row):
        return f"{row['Filter Name']} ({row['Filter Number']}, {row['Manufacturer']})"
    return df[df.apply(lambda row: _display_name(row) in owned, axis=1)]


def _sort_filters(
    df: pd.DataFrame, sort_choice: str, criteria: List[Dict],
    interp_grid: np.ndarray, matrix: np.ndarray,
) -> pd.DataFrame:
    """Sort search results: hue (Hex-Rainbow), Trans @ first-criterion λ, Name, Number."""
    if sort_choice == "Hex-Rainbow":
        def _hue(hex_color):
            h = hex_color.lstrip("#")
            try:
                r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
                return colorsys.rgb_to_hls(r, g, b)[0]
            except Exception:
                return 0.0
        tmp = df.copy()
        tmp["_hue"] = tmp["Hex Color"].apply(_hue)
        return tmp.sort_values("_hue").drop(columns=["_hue"])
    if sort_choice.startswith("Trans"):
        # Sort by transmission at first criterion wavelength (no-op if off-grid)
        if criteria:
            idx = np.where(interp_grid == criteria[0]["wavelength"])[0]
            if idx.size > 0:
                tmp = df.copy()
                tmp["_t"] = matrix[df.index.to_numpy(), idx[0]]
                return tmp.sort_values("_t", ascending=False).drop(columns=["_t"])
        return df
    if sort_choice == "Filter Name":
        return df.sort_values("Filter Name")
    return df.sort_values("Filter Number")


def search_filters(
    df: pd.DataFrame,
    filter_matrix: np.ndarray,
    app_state: AppState,
    interp_grid: np.ndarray = INTERP_GRID,
) -> pd.DataFrame:
    """
    Run the advanced filter search owned by app state: owned-only toggle,
    manufacturer selection, wavelength criteria (AND), and sort choice.

    Pure compute — no UI. Returns the filtered/sorted DataFrame.
    """
    filtered = df
    if app_state.my_filters_only:
        filtered = _owned_only(filtered, app_state.get_my_filters())
    filtered = _filter_by_manufacturer(filtered, app_state.advanced_search_manufacturers)
    filtered = _filter_by_multiple_wavelength_criteria(
        filtered, interp_grid, filter_matrix, app_state.advanced_search_wavelength_criteria
    )
    return _sort_filters(
        filtered, app_state.advanced_search_sort,
        app_state.advanced_search_wavelength_criteria, interp_grid, filter_matrix,
    )


def search_reflectors(
    df: pd.DataFrame, orgs: List[str], pkgs: List[str], tgts: List[str],
) -> pd.DataFrame:
    """Filter the reflector table by organization / package title / target type."""
    filtered = df.copy()
    if orgs:
        filtered = filtered[filtered["Organization"].isin(orgs)]
    if pkgs:
        filtered = filtered[filtered["Package Title"].isin(pkgs)]
    if tgts:
        filtered = filtered[filtered["Target Type"].isin(tgts)]
    return filtered


# =============================================================================
# FILTER SHELF ROWS (quick + advanced search → renderable rows)
# =============================================================================

@dataclass
class FilterRowVM:
    """One search-result row for the Filters shelf / stack quick-add."""
    display_name: str
    manufacturer: str
    hex_color: str
    is_owned: bool
    in_stack: bool
    multiplier: int
    sparkline: go.Figure


def _create_filter_sparkline(transmission_row: np.ndarray, hex_color: str) -> go.Figure:
    """Tiny inline transmission preview for shelf rows (no axes, decimated)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=transmission_row[::2],
        mode="lines",
        line=dict(color=hex_color or "#888888", width=1.5),
        hoverinfo="skip",
    ))
    fig.update_layout(
        width=96, height=30, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False, range=[0, 1.05]),
        showlegend=False,
    )
    return fig


def _quick_search_df(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Case-insensitive substring match on filter name / number."""
    if not query:
        return df
    q = query.strip().lower()
    mask = (
        df["Filter Name"].astype(str).str.lower().str.contains(q, na=False)
        | df["Filter Number"].astype(str).str.lower().str.contains(q, na=False)
    )
    return df[mask]


def prepare_filter_rows(
    app_state: AppState,
    data: AppData,
    query: str = "",
    use_advanced: bool = False,
    limit: int = 60,
) -> List[FilterRowVM]:
    """
    Run the shelf query (quick substring or advanced criteria — presenter-owned
    either way) and shape renderable rows with sparkline figures.
    """
    collection = data.filter_collection
    if use_advanced:
        result_df = search_filters(collection.df, collection.filter_matrix, app_state)
    else:
        result_df = _quick_search_df(collection.df, query)
        if app_state.my_filters_only:
            result_df = _owned_only(result_df, app_state.get_my_filters())

    owned = app_state.get_my_filters()
    display_to_index = collection.get_display_to_index_map()
    rows: List[FilterRowVM] = []
    for _, r in result_df.head(limit).iterrows():
        display = f"{r['Filter Name']} ({r['Filter Number']}, {r['Manufacturer']})"
        idx = display_to_index.get(display)
        trans_row = collection.filter_matrix[idx] if idx is not None else None
        rows.append(FilterRowVM(
            display_name=display,
            manufacturer=str(r["Manufacturer"]),
            hex_color=str(r["Hex Color"]),
            is_owned=display in owned,
            in_stack=display in app_state.selected_filters,
            multiplier=app_state.filter_multipliers.get(display, 1),
            sparkline=(
                _create_filter_sparkline(trans_row, str(r["Hex Color"]))
                if trans_row is not None else go.Figure()
            ),
        ))
    return rows


# =============================================================================
# REFERENCE TAB (read-only raw curves)
# =============================================================================

@dataclass
class ReferenceVM:
    """Read-only reference curves for the Reference tab (UX_SPEC.md §3.2.3)."""
    sections: List[Tuple[str, Optional[go.Figure]]]
    note: Optional[str] = None


def _create_channel_curves_figure(
    curves: Dict[str, np.ndarray], title: str, y_title: str,
) -> go.Figure:
    """Presenter-built raw per-channel curves figure (read-only lookup)."""
    fig = go.Figure()
    for channel in ("R", "G", "B"):
        curve = curves.get(channel)
        if curve is None:
            continue
        fig.add_trace(go.Scatter(
            x=INTERP_GRID,
            y=np.asarray(curve) * 100.0,
            mode="lines",
            name=channel,
            line=dict(color=CHART_COLORS["rgb_colors"].get(channel, "gray"), width=2),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Wavelength (nm)",
        yaxis_title=y_title,
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def prepare_reference(app_state: AppState, data: AppData) -> ReferenceVM:
    """Raw sensitivity / illuminant / film-target curves — demoted lookup view."""
    sections: List[Tuple[str, Optional[go.Figure]]] = []
    note: Optional[str] = None
    profile = app_state.active_profile

    if profile and app_state.current_qe:
        type_label = SENSOR_TYPE_LABELS.get(profile.sensor_type, str(profile.sensor_type))
        sections.append((
            f"Raw sensitivity — {profile.display_name} ({type_label})",
            _create_channel_curves_figure(
                app_state.current_qe, "Raw channel sensitivity", "Sensitivity (%)"
            ),
        ))
    else:
        note = "Select a sensor to see its raw sensitivity curves."

    if app_state.illuminant is not None:
        sections.append((
            f"Illuminant — {app_state.illuminant_name or 'Illuminant'}",
            create_illuminant_figure(
                INTERP_GRID,
                app_state.illuminant,
                app_state.illuminant_name or "Illuminant",
            ),
        ))

    film_profile = app_state.film_emulation_profile
    if film_profile is not None and film_profile is not profile:
        sections.append((
            f"Film target — {film_profile.display_name}",
            _create_channel_curves_figure(
                film_profile.get_channels(), "Film target sensitivity", "Sensitivity (%)"
            ),
        ))

    if not sections:
        note = note or "No reference data available."
    return ReferenceVM(sections=sections, note=note)
