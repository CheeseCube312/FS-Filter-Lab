"""Reference tab: raw QE / illuminant / film-target curves (§3.2.3, read-only)."""
from typing import Any, Dict

from nicegui import ui

from services.presenter import prepare_reference


def render_reference(ctx: Dict[str, Any]) -> None:
    vm = prepare_reference(ctx["app_state"], ctx["data"])
    if vm.note:
        ui.label(vm.note).classes("text-body2 text-grey-6")
    for title, fig in vm.sections:
        if fig is None:
            continue
        ui.label(title).classes("text-subtitle1 q-mt-sm")
        ui.plotly(fig).classes("w-full")
