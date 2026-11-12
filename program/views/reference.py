"""Reference fold-out: raw QE / illuminant curves (read-only)."""
from typing import Any, Dict

from nicegui import ui

from services.presenter import prepare_reference


def render_reference(ctx: Dict[str, Any]) -> None:
    vm = prepare_reference(ctx["app_state"], ctx["data"])
    # Collapsed by default: lookup material, not a primary readout. The
    # open/closed state survives panel rebuilds via page_state (same
    # pattern as the Channel mixer expansion in sensor_response).
    with ui.expansion(
        "Raw sensitivity & illuminant", icon="insights",
        value=ctx["page_state"].setdefault("ref_open", False),
    ).classes("w-full"):
        if vm.note:
            ui.label(vm.note).classes("text-body2 text-grey-6")
        for title, fig in vm.sections:
            ui.label(title).classes("text-subtitle1 q-mt-sm")
            ui.plotly(fig).classes("w-full")
