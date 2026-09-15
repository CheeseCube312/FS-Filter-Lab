"""Film Match instrument: target overlay, deviation readout, mixer (§3.2.3/§7.2)."""
from typing import Any, Dict

from nicegui import ui

from services.presenter import prepare_film_match

_MIXER_FIELDS = [
    ("red", "R", ("red_r", "red_g", "red_b")),
    ("green", "G", ("green_r", "green_g", "green_b")),
    ("blue", "B", ("blue_r", "blue_g", "blue_b")),
]


def render_film_match(ctx: Dict[str, Any]) -> None:
    app_state = ctx["app_state"]
    data = ctx["data"]

    ui.label("Film Match").classes("text-h6")

    films = sorted(k for k, p in data.sensor_profiles.items() if p.is_film)
    ui.select(
        (["None"] + films) if films else ["None"],
        value=(app_state.film_emulation_profile_name or "None"),
        label="Film target",
        on_change=lambda e: ctx["dispatch"]("select_film_target", {"name": e.value}),
    ).classes("w-full max-w-md").props("outlined dense")

    ui.switch(
        "White-balance film target on itself",
        value=app_state.film_emulation_wb,
        on_change=lambda e: ctx["dispatch"]("set_film_wb", {"value": e.value}),
    )

    display_names = [str(f) for f in data.filter_collection.filters]
    ui.select(
        display_names, multiple=True,
        value=list(app_state.film_emulation_filters),
        label="Filters on film target (independent stack)",
        on_change=lambda e: ctx["dispatch"](
            "set_film_target_filters", {"names": e.value}
        ),
    ).classes("w-full max-w-xl").props("outlined dense use-chips use-input hide-selected")

    vm = prepare_film_match(app_state, data)
    if vm.note:
        ui.label(vm.note).classes("text-body2 text-grey-6")
    if vm.overlay_fig is not None:
        ui.plotly(vm.overlay_fig).classes("w-full")

    if vm.deviations:
        ui.label("Deviation (achieved vs target, peak-normalized)").classes(
            "text-subtitle1 q-mt-md"
        )
        with ui.row().classes("w-full items-center gap-6 q-py-xs"):
            ui.label("").classes("w-6")
            ui.label("Peak λ").classes("text-caption text-grey-7 w-20")
            ui.label("Peak ratio").classes("text-caption text-grey-7 w-20")
            ui.label("RMS shape dev.").classes("text-caption text-grey-7")
        colors = {"R": "red", "G": "green", "B": "blue"}
        for dev in vm.deviations:
            with ui.row().classes("w-full items-center gap-6"):
                ui.icon("circle", color=colors.get(dev.channel, "grey")).classes("text-caption w-6")
                ui.label(dev.channel).classes("text-body2 w-6")
                ui.label(dev.peak_wavelength).classes("text-body2 w-20")
                ui.label(dev.peak_ratio).classes("text-body2 w-20")
                ui.label(dev.rms_deviation).classes("text-body2")

    _render_mixer(ctx)


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
        for row_name, row_label, fields in _MIXER_FIELDS:
            with ui.row().classes("w-full items-center no-wrap"):
                ui.label(f"{row_label} out").classes("text-body2 w-16")
                for field, in_label in zip(fields, ("R", "G", "B")):
                    ui.number(
                        value=getattr(mixer, field), min=-2.0, max=2.0, step=0.05,
                        format="%.2f", on_change=lambda e, f=field: ctx["dispatch"](
                            "set_mixer_gain", {"field": f, "value": e.value}
                        ),
                    ).props(f"outlined dense label='{in_label}'").classes("w-24")
        ui.button(
            "Reset mixer", icon="restart_alt",
            on_click=lambda: ctx["dispatch"]("reset_mixer"),
        ).props("flat dense")
    mixer_panel.on_value_change(
        lambda e: page_state.__setitem__("mixer_open", e.value)
    )
