"""Sensor response instrument: per-channel curves, WB, channel mixer."""
from typing import Any, Dict

from nicegui import ui

from services.presenter import prepare_sensor_analysis

_CHANNELS = ("R", "G", "B")


def render_sensor_response(ctx: Dict[str, Any]) -> None:
    app_state = ctx["app_state"]
    page_state = ctx["page_state"]
    visible = page_state.setdefault("vis_channels", {"R": True, "G": True, "B": True})

    with ui.row().classes("w-full items-center justify-between"):
        ui.label("Sensor response").classes("text-h6")
        with ui.row().classes("items-center gap-2"):
            for ch in _CHANNELS:
                ui.checkbox(
                    ch, value=visible[ch],
                    on_change=lambda e, c=ch: _toggle_channel(ctx, c, e.value),
                ).props("dense")
            ui.switch(
                "WB", value=app_state.apply_white_balance,
                on_change=lambda e: ctx["dispatch"]("set_wb_enabled", {"value": e.value}),
            ).props("dense")

    # Only the chart block lives in the nested refreshable region (see
    # shell.build_page): mixer-gain edits refresh it without rebuilding the
    # mixer inputs below, so typing is never destroyed mid-edit.
    ctx["render_sensor_chart"]()

    _render_mixer(ctx)


def render_sensor_chart(ctx: Dict[str, Any]) -> None:
    """WB readout + response figure: the gain-dependent sensor-chart region."""
    app_state = ctx["app_state"]
    data = ctx["data"]
    page_state = ctx["page_state"]
    visible = page_state.setdefault("vis_channels", {"R": True, "G": True, "B": True})

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
    ctx["page_state"]["vis_channels"][channel] = value
    # Only the chart block depends on channel visibility — refreshing the
    # whole panel would needlessly rebuild the mixer inputs.
    ctx["refresh_panel"]("sensor_response_chart")


def _render_mixer(ctx: Dict[str, Any]) -> None:
    page_state = ctx["page_state"]
    mixer = ctx["app_state"].channel_mixer
    # Expansion open-state lives in page_state (like the Filters shelf's
    # adv_open/mf_open): re-renders must not snap a user-opened panel shut
    # just because the mixer is currently disabled.
    with ui.expansion(
        "Channel mixer", icon="tune",
        value=page_state.setdefault("mixer_open", mixer.enabled),
    ).classes("w-full") as mixer_panel:
        ui.switch(
            "Enable channel mixing",
            value=mixer.enabled,
            on_change=lambda e: ctx["dispatch"]("set_mixer_enabled", {"value": e.value}),
        )
        ui.label(
            "Output channel = linear mix of input R/G/B (relative intensities)."
        ).classes("text-caption text-grey-7")
        for row_label, prefix in (("R", "red"), ("G", "green"), ("B", "blue")):
            with ui.row().classes("w-full items-center no-wrap"):
                ui.label(f"{row_label} out").classes("text-body2 w-16")
                for in_label in ("R", "G", "B"):
                    field = f"{prefix}_{in_label.lower()}"
                    ui.number(
                        value=getattr(mixer, field), min=-2.0, max=2.0, step=0.05,
                        format="%.2f", on_change=lambda e, f=field: ctx["dispatch"](
                            "set_mixer_gain", {"field": f, "value": e.value}
                        ),
                    ).props(f"outlined dense label='{in_label}' debounce=400").classes("w-24")
        ui.button(
            "Reset mixer", icon="restart_alt",
            on_click=lambda: ctx["dispatch"]("reset_mixer"),
        ).props("flat dense")
    mixer_panel.on_value_change(
        lambda e: page_state.__setitem__("mixer_open", e.value)
    )
