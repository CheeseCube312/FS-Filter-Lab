"""Sensor response instrument: per-channel curves, WB, channel toggles (§3.2.2)."""
from typing import Any, Dict

from nicegui import ui

from services.presenter import prepare_sensor_analysis

_CHANNELS = ("R", "G", "B")


def render_sensor_response(ctx: Dict[str, Any]) -> None:
    app_state = ctx["app_state"]
    data = ctx["data"]
    page_state = ctx["page_state"]
    visible = page_state.setdefault("vis_channels", {"R": True, "G": True, "B": True})

    with ui.row().classes("w-full items-center justify-between"):
        ui.label("Sensor response").classes("text-h6")
        with ui.row().classes("items-center gap-2"):
            for ch in _CHANNELS:
                ui.checkbox(
                    ch, value=visible.get(ch, True),
                    on_change=lambda e, c=ch: _toggle_channel(ctx, c, e.value),
                ).props("dense")
            ui.switch(
                "WB", value=app_state.apply_white_balance,
                on_change=lambda e: ctx["dispatch"]("set_wb_enabled", {"value": e.value}),
            ).props("dense")

    vm = prepare_sensor_analysis(app_state, data, visible_channels=visible)

    wb = vm.wb_display
    if wb:
        with ui.row().classes("w-full items-center gap-4 q-mb-xs"):
            gains = "  ".join(f"{c} ×{wb.gains.get(c, 1.0):.2f}" for c in _CHANNELS)
            ui.label(f"WB gains: {gains}").classes("text-body2 text-grey-7")
            if wb.mode_note:
                ui.label(wb.mode_note).classes("text-caption text-amber-8")
    elif vm.info_message:
        ui.label(vm.info_message).classes("text-body2 text-grey-6")

    ui.plotly(vm.sensor_response_fig).classes("w-full")


def _toggle_channel(ctx: Dict[str, Any], channel: str, value: bool) -> None:
    ctx["page_state"]["vis_channels"][channel] = bool(value)
    ctx["refresh_panel"]("sensor_response")
