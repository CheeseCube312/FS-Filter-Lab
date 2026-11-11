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
from models.constants import (
    CHART_COLORS,
    EPSILON,
    INTERP_GRID,
    SCENE_COVERAGE_LIMIT,
    SENSOR_TYPE_LABELS,
)
from models.core import AppData, FilterCollection
from services.calculations import (
    compute_filter_transmission,
    compute_selected_filter_indices,
    compute_effective_stops,
    effective_transmission,
    compute_reflector_color,
    compute_white_reference_color,
    reference_color_to_display,
    apply_camera_look,
    filter_display_color,
    srgb_to_hex,
)
from services.visualization import (
    create_filter_response_plot,
    create_illuminant_figure,
    create_sensor_response_plot,
)
from services.state_manager import AppState

QE_ILLUMINANT_REQUIRED_MSG = "Select a QE & illuminant profile to compute white balance."


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
    mode_note: str  # e.g. " (from surface reference)" or ""


@dataclass
class FilterAnalysisVM:
    """View model containing everything the filter analysis view needs to render."""
    metrics: Optional[TransmissionMetricsVM]
    filter_response_fig: go.Figure


@dataclass  
class SensorAnalysisVM:
    """View model containing everything the sensor analysis view needs to render."""
    sensor_response_fig: go.Figure
    wb_display: Optional[WhiteBalanceVM]
    info_message: Optional[str] = None  # Shown when QE+illuminant not available


@dataclass
class HeaderStatusVM:
    """One-line app summary for the page header (segments joined for display)."""
    segments: List[str]

    @property
    def text(self) -> str:
        return "  ·  ".join(self.segments)


# =============================================================================
# FILTER ANALYSIS PRESENTER
# =============================================================================

def prepare_filter_analysis(
    app_state: AppState,
    filter_collection: FilterCollection,
    log_stops: Optional[bool] = None,
) -> FilterAnalysisVM:
    """Prepare all data needed to render the filter analysis section."""
    selected_indices = compute_selected_filter_indices(
        app_state.selected_filters,
        app_state.filter_multipliers,
        filter_collection,
    )

    # Calculate transmission
    trans, label = compute_filter_transmission(
        selected_indices,
        filter_collection.filter_matrix
    )
    combined = trans if label == "Combined" else None

    # Calculate transmission metrics
    metrics = _prepare_transmission_metrics(trans, label, app_state)

    # Create filter response chart
    filter_names = [f.name for f in filter_collection.filters]
    # Curve strokes trust metadata, but derive from the spectrum when a file
    # ships no (valid) color — same rule as the shelf swatches (identity hues
    # only for curves actually drawn, so cost stays negligible).
    selected = set(int(i) for i in (selected_indices or []))
    filter_hex_colors = [
        filter_display_color(filter_collection.filter_matrix[i], f.hex_color)
        if i in selected else f.hex_color
        for i, f in enumerate(filter_collection.filters)
    ]

    fig = create_filter_response_plot(
        interp_grid=INTERP_GRID,
        filter_matrix=filter_collection.filter_matrix,
        masks=filter_collection.extrapolated_masks,
        selected_indices=selected_indices,
        combined=combined,
        log_stops=bool(log_stops),
        filter_names=filter_names,
        filter_hex_colors=filter_hex_colors
    )

    return FilterAnalysisVM(metrics=metrics, filter_response_fig=fig)


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

    return TransmissionMetricsVM(
        label=label,
        avg_transmission_pct=f"{avg_trans * 100:.1f}%",
        effective_stops=f"{effective_stops:.2f}",
        has_valid_data=True
    )


# =============================================================================
# HEADER STATUS LINE (read-only summary of the current analysis state)
# =============================================================================

def prepare_header_status(app_state: AppState, data: AppData) -> HeaderStatusVM:
    """
    Compact status line for the page header: receiver · illuminant · stack · WB.

    Mirrors panel content cheaply — transmission numbers reuse
    ``_prepare_transmission_metrics`` (no figures are built) and the WB surface
    name resolves the same way the Scene Preview card does. Read-only.
    """
    profile = app_state.active_profile
    if profile is not None:
        type_label = SENSOR_TYPE_LABELS.get(profile.sensor_type, str(profile.sensor_type))
        segments = [f"{profile.display_name} ({type_label})"]
    else:
        segments = ["No receiver"]
    segments.append(app_state.illuminant_name or "No illuminant")

    if app_state.selected_filters:
        selected_indices = compute_selected_filter_indices(
            app_state.selected_filters,
            app_state.filter_multipliers,
            data.filter_collection,
        )
        trans, _label = compute_filter_transmission(
            selected_indices, data.filter_collection.filter_matrix
        )
        metrics = _prepare_transmission_metrics(trans, _label, app_state)
        n = len(app_state.selected_filters)
        stack = f"{n} filter" + ("" if n == 1 else "s")
        if metrics.has_valid_data:
            stack += f" · {metrics.effective_stops} stops · {metrics.avg_transmission_pct}"
        segments.append(stack)
    else:
        segments.append("Stack empty")

    if not app_state.apply_white_balance:
        segments.append("WB off")
    elif app_state.wb_reference_surface:
        name = next(
            (r.name for r in data.reflector_collection.reflectors
             if r.metadata.get("source_file", "") == app_state.wb_reference_surface),
            None,
        )
        segments.append(f"WB: {name or app_state.wb_reference_surface}")
    else:
        segments.append("WB: standard")

    return HeaderStatusVM(segments=segments)


# =============================================================================
# SENSOR ANALYSIS PRESENTER  
# =============================================================================

def prepare_sensor_analysis(
    app_state: AppState,
    data: AppData,
    visible_channels: Optional[Dict[str, bool]] = None,
) -> SensorAnalysisVM:
    """Prepare the sensor response chart and white-balance display."""
    filter_collection = data.filter_collection
    # Effective transmission of the selected filter stack
    trans_interp = effective_transmission(
        app_state.selected_filters,
        app_state.filter_multipliers,
        filter_collection,
    )

    # WB gains are maintained by the action layer; presenters only read.
    wb_gains = app_state.white_balance_gains

    # Create sensor response chart
    fig_response = create_sensor_response_plot(
        interp_grid=INTERP_GRID,
        transmission=trans_interp,
        qe_data=app_state.current_qe,
        visible_channels=visible_channels
        or {"R": True, "G": True, "B": True},
        white_balance_gains=wb_gains,
        apply_white_balance=app_state.apply_white_balance,
        channel_mixer=app_state.channel_mixer,
        sensor_profile=app_state.active_profile,
    )

    # Prepare white balance display
    wb_display = None
    info_message = None

    if app_state.current_qe is not None and app_state.illuminant is not None:
        if app_state.wb_reference_surface:
            mode_note = " (from surface reference)"
        elif not app_state.selected_filters:
            mode_note = " (No filter selected)"
        else:
            mode_note = ""

        wb_display = WhiteBalanceVM(gains=wb_gains, mode_note=mode_note)
    else:
        info_message = QE_ILLUMINANT_REQUIRED_MSG

    return SensorAnalysisVM(
        sensor_response_fig=fig_response,
        wb_display=wb_display,
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
) -> pd.DataFrame:
    idx = np.where(interp_grid == wavelength)[0]
    if idx.size == 0:
        return df
    si = idx[0]
    di = df.index.to_numpy()
    tv = matrix[di, si]
    return df.iloc[(tv >= min_t) & (tv <= max_t)]


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
        filtered_df = _filter_by_trans_at_wavelength(
            filtered_df, interp_grid, matrix,
            criterion["wavelength"],
            criterion["trans_min"] / 100.0,
            criterion["trans_max"] / 100.0,
        )
    return filtered_df


def _row_display_name(row) -> str:
    """DataFrame-row form of Filter.__str__ (models.core) — the shelf join key."""
    return f"{row['Filter Name']} ({row['Filter Number']}, {row['Manufacturer']})"


def _owned_only(df: pd.DataFrame, owned: set) -> pd.DataFrame:
    """Restrict *df* to rows whose display name is in the owned set."""
    return df[df.apply(lambda row: _row_display_name(row) in owned, axis=1)]


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
        filtered, INTERP_GRID, filter_matrix, app_state.advanced_search_wavelength_criteria
    )
    return _sort_filters(
        filtered, app_state.advanced_search_sort,
        app_state.advanced_search_wavelength_criteria, INTERP_GRID, filter_matrix,
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


@dataclass
class FilterShelfVM:
    """Filters shelf results: the capped rows plus the uncapped match count."""
    rows: List[FilterRowVM]
    total_matches: int


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
    """Tokenized case-insensitive AND-match on name / number / manufacturer.

    Every whitespace-separated token must appear in at least one of the three
    fields, so "lee oklahoma" finds Lee Filters' "Oklahoma Yellow" (brand
    token + name token in one query). Tokens are matched literally, never as
    regex, so queries with metacharacters like "(" are safe.
    """
    tokens = str(query).lower().split()
    if not tokens:
        return df
    haystack = (
        df["Filter Name"].astype(str) + "\x1f"
        + df["Filter Number"].astype(str) + "\x1f"
        + df["Manufacturer"].astype(str)
    ).str.lower()
    mask = pd.Series(True, index=df.index)
    for token in tokens:
        mask &= haystack.str.contains(token, regex=False, na=False)
    return df[mask]


def prepare_filter_rows(
    app_state: AppState,
    data: AppData,
    query: str = "",
    use_advanced: bool = False,
    limit: int = 60,
) -> FilterShelfVM:
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
        display = _row_display_name(r)
        idx = display_to_index.get(display)
        trans_row = collection.filter_matrix[idx] if idx is not None else None
        color = filter_display_color(trans_row, str(r["Hex Color"]))
        rows.append(FilterRowVM(
            display_name=display,
            manufacturer=str(r["Manufacturer"]),
            hex_color=color,
            is_owned=display in owned,
            in_stack=display in app_state.selected_filters,
            multiplier=app_state.filter_multipliers.get(display, 1),
            sparkline=(
                _create_filter_sparkline(trans_row, color)
                if trans_row is not None else go.Figure()
            ),
        ))
    return FilterShelfVM(rows=rows, total_matches=int(len(result_df)))


# =============================================================================
# REFERENCE TAB (read-only raw curves)
# =============================================================================

@dataclass
class ReferenceVM:
    """Read-only reference curves for the Reference tab."""
    sections: List[Tuple[str, go.Figure]]
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
    """Raw sensitivity / illuminant curves — demoted lookup view."""
    sections: List[Tuple[str, go.Figure]] = []
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

    if not sections:
        note = note or "No reference data available."
    return ReferenceVM(sections=sections, note=note)


# =============================================================================
# SCENE PREVIEW (per-surface render swatches + camera-look toggle)
# =============================================================================

@dataclass
class SceneSwatchVM:
    """One surface's render swatch, following the camera-look toggle."""
    name: str
    source_file: str
    display_hex: str
    display_linear: Tuple[float, float, float]
    partial_data: bool
    coverage_note: str
    sparkline: go.Figure
    is_wb_ref: bool


@dataclass
class ScenePreviewVM:
    """Everything the Scene Preview card needs to render."""
    swatches: List[SceneSwatchVM]
    look_on: bool
    inverts_for_display: bool
    warning: Optional[str] = None
    wb_source_name: Optional[str] = None


def _surface_render_pair(
    app_state: AppState, reflectance: np.ndarray,
    transmission: np.ndarray, white_ref: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """(plain, camera) white-referenced linear colors for one surface.

    ``plain`` is the neutral ratio render; ``camera`` adds the default-on
    camera look (filmic tone curve + chroma, with negative-film inversion
    for ``invert_for_display`` profiles) when ``scene_look`` is enabled.
    """
    surf = compute_reflector_color(
        reflectance, transmission, app_state.current_qe, app_state.illuminant,
        channel_mixer=app_state.channel_mixer,
        white_balance_gains=app_state.white_balance_gains,
    )
    plain = reference_color_to_display(surf, white_ref)
    if app_state.scene_look:
        invert = bool(
            app_state.active_profile
            and app_state.active_profile.invert_for_display
        )
        return plain, apply_camera_look(plain, invert=invert)
    return plain, plain


def _missing_band_note(missing: np.ndarray) -> str:
    '''"a–b nm, c–d nm" for each contiguous run of no-data wavelengths.'''
    idx = np.where(missing)[0]
    runs: List[Tuple[int, int]] = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i == prev + 1:
            prev = i
            continue
        runs.append((start, prev))
        start = prev = i
    runs.append((start, prev))
    return ", ".join(
        f"{int(INTERP_GRID[a])}–{int(INTERP_GRID[b])} nm" for a, b in runs
    )


def _scene_coverage_flag(
    values: np.ndarray, transmission: np.ndarray,
    qe: Dict[str, np.ndarray], illuminant: np.ndarray,
) -> Tuple[bool, str]:
    """Partial-data flag + missing-band note for one surface.

    Weights every wavelength by the chain's spectral weight (illuminant x
    stack transmission x mean QE); when more than SCENE_COVERAGE_LIMIT of
    that weight lands on wavelengths without reflectance data (NaN), the
    surface is flagged and the note lists the missing bands.
    """
    reflectance = np.asarray(values, dtype=float)
    missing = ~np.isfinite(reflectance)
    if not missing.any():
        return False, ""
    qe_mean = np.mean([
        np.nan_to_num(np.asarray(qe.get(ch), dtype=float))
        for ch in ("R", "G", "B") if qe.get(ch) is not None
    ], axis=0)
    weight = np.nan_to_num(
        np.asarray(illuminant, dtype=float)
        * np.asarray(transmission, dtype=float)
        * qe_mean
    )
    total = float(weight.sum())
    fraction = float(weight[missing].sum()) / total if total > EPSILON else 0.0
    if fraction <= SCENE_COVERAGE_LIMIT:
        return False, ""
    return True, _missing_band_note(missing)


def prepare_scene_preview(app_state: AppState, data: AppData) -> ScenePreviewVM:
    """Render swatches for every surface selected in the instrument."""
    qe = app_state.current_qe
    inverts = bool(
        app_state.active_profile and app_state.active_profile.invert_for_display
    )
    if not qe or app_state.illuminant is None:
        return ScenePreviewVM(
            swatches=[], look_on=app_state.scene_look,
            inverts_for_display=inverts, warning=QE_ILLUMINANT_REQUIRED_MSG,
        )

    transmission = effective_transmission(
        app_state.selected_filters, app_state.filter_multipliers,
        data.filter_collection,
    )
    white_ref = compute_white_reference_color(
        transmission, qe, app_state.illuminant,
        white_balance_gains=app_state.white_balance_gains,
        channel_mixer=app_state.channel_mixer,
    )

    by_source = {
        r.metadata.get("source_file", ""): r
        for r in data.reflector_collection.reflectors
    }
    swatches: List[SceneSwatchVM] = []
    wb_source_name: Optional[str] = None
    for src in app_state.selected_surfaces:
        spectrum = by_source.get(src)
        if spectrum is None:
            continue
        plain, camera = _surface_render_pair(
            app_state, spectrum.values, transmission, white_ref
        )
        color = camera if app_state.scene_look else plain
        partial_data, coverage_note = _scene_coverage_flag(
            spectrum.values, transmission, qe, app_state.illuminant
        )
        swatches.append(SceneSwatchVM(
            name=spectrum.name,
            source_file=src,
            display_hex=srgb_to_hex(color),
            display_linear=(float(color[0]), float(color[1]), float(color[2])),
            partial_data=partial_data,
            coverage_note=coverage_note,
            sparkline=_create_reflector_sparkline(spectrum.values),
            is_wb_ref=(app_state.wb_reference_surface == src),
        ))
        if app_state.wb_reference_surface == src:
            wb_source_name = spectrum.name

    return ScenePreviewVM(
        swatches=swatches, look_on=app_state.scene_look,
        inverts_for_display=inverts, wb_source_name=wb_source_name,
    )

# =============================================================================
# REFLECTORS SHELF (library search + white-referenced preview rows)
# =============================================================================

@dataclass
class ReflectorRowVM:
    """One library row in the Reflectors shelf."""
    name: str
    source_file: str
    organization: str
    target_type: str
    coverage_note: str
    sparkline: go.Figure
    is_default: bool
    is_surface: bool
    is_wb_ref: bool
    preview_hex: str


@dataclass
class ReflectorsShelfVM:
    rows: List[ReflectorRowVM]
    total_matches: int


def _coverage_note(values: np.ndarray) -> str:
    """Usefulness badge: wavelength span + mean reflectance of valid data."""
    valid = np.isfinite(values) & (values > 0.02)
    if not valid.any():
        return "no data"
    idx = np.where(valid)[0]
    lo, hi = int(INTERP_GRID[idx[0]]), int(INTERP_GRID[idx[-1]])
    mean_pct = float(np.nanmean(values[valid])) * 100.0
    return f"{lo}-{hi} nm, mean {mean_pct:.0f}%"


def _create_reflector_sparkline(reflectance_row: np.ndarray) -> go.Figure:
    """Tiny inline reflectance preview for shelf rows (no axes, decimated).

    Mirrors the filter sparkline; NaN entries stay NaN so honest out-of-range
    gaps render as line breaks (plotly's default connectgaps behaviour).
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=np.asarray(reflectance_row)[::2],
        mode="lines",
        line=dict(color="#888888", width=1.5),
        hoverinfo="skip",
    ))
    fig.update_layout(
        width=96, height=30, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False, range=[0, 1.05]),
        showlegend=False,
    )
    return fig


def prepare_reflectors_shelf(
    app_state: AppState,
    data: AppData,
    orgs: Optional[List[str]] = None,
    target_types: Optional[List[str]] = None,
    limit: int = 40,
) -> ReflectorsShelfVM:
    """Search the reflector library and shape rows with sparklines + live previews."""
    collection = data.reflector_collection
    result_df = search_reflectors(collection.df, orgs or [], [], target_types or [])

    qe = app_state.current_qe
    has_chain = bool(qe) and app_state.illuminant is not None
    transmission = effective_transmission(
        app_state.selected_filters, app_state.filter_multipliers,
        data.filter_collection,
    )
    white_ref = None
    if has_chain:
        white_ref = compute_white_reference_color(
            transmission, qe, app_state.illuminant,
            white_balance_gains=app_state.white_balance_gains,
            channel_mixer=app_state.channel_mixer,
        )

    source_to_name: Dict[str, str] = {}
    for r in collection.reflectors:
        source_to_name.setdefault(r.metadata.get("source_file", ""), r.name)

    rows: List[ReflectorRowVM] = []
    for idx, row in result_df.head(limit).iterrows():
        values = collection.reflector_matrix[idx]
        name = str(row.get("Name", ""))
        src = str(row.get("Source File", "")) if "Source File" in result_df.columns else ""
        if not src:
            src = next((s for s, n in source_to_name.items() if n == name), "")
        preview_hex = "#777777"
        if has_chain:
            plain, look = _surface_render_pair(
                app_state, values, transmission, white_ref
            )
            preview_hex = srgb_to_hex(look if app_state.scene_look else plain)
        rows.append(ReflectorRowVM(
            name=name,
            source_file=src,
            organization=str(row.get("Organization", "")),
            target_type=str(row.get("Target Type", "")),
            coverage_note=_coverage_note(values),
            sparkline=_create_reflector_sparkline(values),
            is_default=app_state.is_default_reflector(src),
            is_surface=(src in app_state.selected_surfaces),
            is_wb_ref=(app_state.wb_reference_surface == src),
            preview_hex=preview_hex,
        ))

    return ReflectorsShelfVM(rows=rows, total_matches=len(result_df))
