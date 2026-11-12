"""Transmission instrument panel.

Renders the combined stack transmission chart + metrics row. The linear/
stops scale is chart-local view state (page_state), passed into the
presenter per call.
"""
from typing import Any, Dict

from nicegui import ui

from services.presenter import prepare_filter_analysis


def render_transmission(ctx: Dict[str, Any]) -> None:
    app_state = ctx["app_state"]
    data = ctx["data"]
    page_state = ctx["page_state"]
    log_stops = page_state["trans_log_stops"]

    with ui.row().classes("w-full items-center justify-between"):
        ui.label("Transmission").classes("text-h6")
        ui.toggle(
            {"linear": "Linear", "stops": "Stops"},
            value="stops" if log_stops else "linear",
            on_change=lambda e: _set_scale(ctx, e.value),
        ).props("dense")

    if not app_state.selected_filters:
        ui.label(
            "No filters stacked yet — add filters from the Filters shelf "
            "(right) to see the combined stack."
        ).classes("text-body2 text-grey-6")
        return

    vm = prepare_filter_analysis(
        app_state,
        data.filter_collection,
        log_stops=log_stops,
    )

    metrics = vm.metrics
    if metrics:
        with ui.row().classes("w-full items-center gap-6 q-mb-xs"):
            ui.label(f"Stack: {metrics.label}").classes("text-body2 text-grey-7")
            ui.label(f"Avg transmission: {metrics.avg_transmission_pct}").classes("text-body2")
            ui.label(f"Effective stops: {metrics.effective_stops}").classes("text-body2")
            if not metrics.has_valid_data and metrics.error_message:
                ui.icon("warning", color="amber-8").tooltip(metrics.error_message)

    ui.plotly(vm.filter_response_fig).classes("w-full")


def _set_scale(ctx: Dict[str, Any], value: Any) -> None:
    ctx["page_state"]["trans_log_stops"] = (value == "stops")
    ctx["refresh_panel"]("transmission")
