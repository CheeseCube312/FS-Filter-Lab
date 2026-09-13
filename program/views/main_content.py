"""
Main content area for FS FilterLab.

Charts, metrics, reflector list, and analysis panels.
"""
import logging
from typing import Any

import numpy as np
from nicegui import ui

from models.constants import (
    INTERP_GRID, UI_INFO_MESSAGES, UI_CHART_TITLES,
    UI_SECTIONS, UI_LABELS, METADATA_FIELDS, SURFACE_COLOR_METADATA, UI_HELP_TEXT,
)
from models.core import AppData
from services.calculations import (
    compute_selected_filter_indices,
    compute_reflector_color,
)
from services.presenter import (
    prepare_filter_analysis, prepare_sensor_analysis,
)
from services.visualization import (
    create_illuminant_figure, prepare_rgb_for_display,
)
from views.ui_utils import (
    inline_warning, inline_info, rgb_to_hex,
    format_reflector_metadata, stateful_expansion,
)

logger = logging.getLogger(__name__)


# ============================================================================
# METRIC DISPLAYS
# ============================================================================

def _transmission_metrics(metrics) -> None:
    """Display light-loss metrics from a TransmissionMetricsVM."""
    if metrics is None:
        return
    if not metrics.has_valid_data:
        if metrics.error_message:
            inline_warning(metrics.error_message)
        return
    ui.markdown(
        f"**Estimated light loss ({metrics.label}):** "
        f"{metrics.effective_stops} stops  \n"
        f"(Avg transmission: {metrics.avg_transmission_pct})"
    )


def _white_balance_display(wb_display) -> None:
    """Display white balance gains from a WhiteBalanceVM."""
    if wb_display is None:
        return
    ui.markdown(
        f"**White Balance Gains{wb_display.mode_note}:** (Green = 1.000):  \n"
        f"R: {wb_display.intensities['R']}   "
        f"G: {wb_display.intensities['G']}   "
        f"B: {wb_display.intensities['B']}"
    )


# ============================================================================
# CHART RENDERING
# ============================================================================

def _render_chart(fig, title: str = None) -> None:
    if title:
        ui.label(title).classes("text-lg font-bold mt-2")
    ui.plotly(fig).classes("w-full")


# ============================================================================
# DEFAULT REFLECTOR LIST
# ============================================================================

def _render_default_reflector_list(app_state, data: AppData, on_change) -> None:
    reflector_collection = data.reflector_collection
    default_files = app_state.get_default_reflector_files()
    if not default_files:
        return

    # Collect matching reflectors
    default_reflectors = []
    for idx, reflector in enumerate(reflector_collection.reflectors):
        sf = reflector.metadata.get("source_file", "")
        if sf in default_files:
            default_reflectors.append((idx, reflector))

    if not default_reflectors:
        return

    combined_trans = app_state.combined_transmission
    if combined_trans is None:
        combined_trans = np.ones_like(INTERP_GRID, dtype=float)

    with stateful_expansion(UI_SECTIONS["default_reflector_list"], "default_reflector_list", app_state, default_open=True):
        with ui.column().classes("w-full gap-1").style("max-height:500px;overflow-y:auto"):
            for idx, reflector in default_reflectors:
                _render_reflector_row(
                    reflector, combined_trans, app_state, on_change,
                )


def _render_reflector_row(reflector, combined_trans, app_state, on_change) -> None:
    """Render a single reflector row: swatch, name/metadata, WB button, remove."""
    rgb_color = None
    if app_state.current_qe and app_state.illuminant is not None:
        raw_rgb = compute_reflector_color(
            reflector.values, combined_trans,
            app_state.current_qe, app_state.illuminant,
            app_state.channel_mixer, app_state.white_balance_gains,
        )
        rgb_color = prepare_rgb_for_display(np.nan_to_num(raw_rgb), auto_exposure=True)

    source_file = reflector.metadata.get("source_file", "")
    is_active_wb = source_file == app_state.wb_reference_surface
    wb_disabled = app_state.current_qe is None or app_state.illuminant is None

    with ui.row().classes("w-full items-center gap-2 py-1"):
        # Swatch
        hex_c = rgb_to_hex(rgb_color) if rgb_color is not None else "#808080"
        ui.element('div').style(
            f'width:36px;height:36px;border-radius:4px;'
            f'background-color:{hex_c};border:1px solid #ccc;flex-shrink:0'
        )
        # Name + metadata
        with ui.column().classes("flex-grow gap-0"):
            ui.label(reflector.name).classes("font-medium text-sm")
            summary = format_reflector_metadata(
                reflector.metadata,
                api_fields=SURFACE_COLOR_METADATA["api_attribution_fields"],
                fallback_fields=SURFACE_COLOR_METADATA["fallback_fields"],
                relevant_meta_key=METADATA_FIELDS["relevant_metadata"],
            )
            if summary:
                ui.label(summary).classes("text-xs text-gray-500")

        # WB button (3-state: disabled / active / normal)
        if wb_disabled:
            ui.button("WB").props("dense outline size=sm disable")
        elif is_active_wb:
            def _make_reset(sf=source_file):
                def _reset():
                    app_state.reset_white_balance()
                    on_change()
                return _reset
            ui.button("● WB", on_click=_make_reset(source_file)).props(
                "dense flat size=sm color=primary"
            ).tooltip("Currently used as WB reference — click to reset")
        else:
            def _make_set_wb(r=reflector, ct=combined_trans, sf=source_file):
                def _set():
                    app_state.set_white_balance_from_surface(r.values, ct, sf)
                    on_change()
                return _set
            ui.button("WB", on_click=_make_set_wb()).props("dense outline size=sm").tooltip(
                "Set as white balance reference"
            )

        # Remove button
        def _make_remove(sf=source_file):
            def _rem():
                app_state.remove_from_default_reflectors(sf)
                on_change()
            return _rem
        ui.button("×", on_click=_make_remove()).props("dense outline size=sm color=negative")


# ============================================================================
# RAW QE / ILLUMINANT CURVES
# ============================================================================

def _raw_qe_and_illuminant(app_state, data: AppData) -> None:
    illuminant_metadata = data.illuminant_metadata

    with stateful_expansion(UI_SECTIONS["reflectance_illuminant_curves"], "reflectance_illuminant_curves", app_state):
        if app_state.illuminant is not None and app_state.illuminant_name is not None:
            if app_state.illuminant_name in illuminant_metadata:
                ui.markdown(f"**Description:** {illuminant_metadata[app_state.illuminant_name]}")
            fig = create_illuminant_figure(INTERP_GRID, app_state.illuminant, app_state.illuminant_name)
            _render_chart(fig, f"Illuminant: {app_state.illuminant_name}")
        else:
            inline_info(UI_INFO_MESSAGES["no_illuminant"])


# ============================================================================
# MAIN RENDERER
# ============================================================================

def render_main_content(
    container: ui.column,
    app_state,
    data: AppData,
    on_change,
) -> None:
    """Build the main content area inside *container*.

    Called once at startup and again whenever state changes
    (the container is cleared + rebuilt).
    """
    filter_collection = data.filter_collection
    reflector_collection = data.reflector_collection

    container.clear()

    with container:
        # Header
        ui.label("FS FilterLab").classes("text-2xl font-bold")

        # Compute indices
        selected_indices = compute_selected_filter_indices(
            app_state.selected_filters,
            app_state.filter_multipliers,
            filter_collection,
        )

        # Filter analysis
        if selected_indices:
            vm = prepare_filter_analysis(app_state, filter_collection, selected_indices)
            _transmission_metrics(vm.metrics)
            _render_chart(vm.filter_response_fig, UI_CHART_TITLES["combined_filter_response"])
        else:
            # No filters selected - reset combined_transmission to None
            app_state.combined_transmission = None

        # Sensor analysis
        if app_state.current_qe:
            vm_sensor = prepare_sensor_analysis(app_state, data, selected_indices)

            # WB toggle with context-sensitive tooltip
            _active_profile = app_state.active_profile
            _wb_tooltip = (
                UI_HELP_TEXT["apply_white_balance_film"]
                if _active_profile is not None and _active_profile.is_film
                else UI_HELP_TEXT["apply_white_balance_sensor"]
            )
            with ui.row().classes("items-center gap-1"):
                ui.switch(
                    UI_LABELS["apply_white_balance"],
                    value=app_state.apply_white_balance,
                    on_change=lambda e: (
                        setattr(app_state, "apply_white_balance", e.value),
                        on_change(),
                    ),
                ).props("dense")
                ui.icon("info_outline").classes("text-grey-6 text-sm cursor-help") \
                    .tooltip(_wb_tooltip)

            _render_chart(vm_sensor.sensor_response_fig, UI_CHART_TITLES["sensor_weighted_response"])

            if vm_sensor.wb_display:
                _white_balance_display(vm_sensor.wb_display)
            elif vm_sensor.info_message:
                inline_info(vm_sensor.info_message)
        else:
            ui.label(UI_INFO_MESSAGES["select_qe_prompt"]).classes(
                "text-sm italic text-gray-500 mt-4"
            )

        # Raw QE / illuminant
        _raw_qe_and_illuminant(app_state, data)

        # Default reflector list
        if reflector_collection and len(reflector_collection.df) > 0:
            _render_default_reflector_list(app_state, data, on_change)

        # Channel mixer
        if app_state.show_channel_mixer:
            from views.channel_mixer import render_channel_mixer_panel
            with ui.card().classes("w-full"):
                render_channel_mixer_panel(app_state, on_change=on_change)

        # Import dialog
        if app_state.show_import_data:
            from views.forms import render_import_dialog
            render_import_dialog(app_state, on_change)

        # Advanced filter search
        if app_state.show_advanced_search:
            from views.forms import render_advanced_filter_search
            render_advanced_filter_search(
                filter_collection.df, filter_collection.filter_matrix, app_state, on_change,
            )

        # Advanced reflector search
        if app_state.show_reflector_search:
            from views.forms import render_advanced_reflector_search
            render_advanced_reflector_search(
                reflector_collection.df, reflector_collection.reflector_matrix,
                app_state, on_change,
            )

        # My Filters management panel
        if app_state.show_my_filters_manager:
            from views.forms import render_my_filters_manager
            render_my_filters_manager(filter_collection, app_state, on_change)
