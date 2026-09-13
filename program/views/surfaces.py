"""Surfaces instrument: white-referenced color chip grid (§3.2.3, §7.1)."""
from typing import Any, Dict

from nicegui import ui

from services.presenter import prepare_surfaces


def render_surfaces(ctx: Dict[str, Any]) -> None:
    app_state = ctx["app_state"]
    vm = prepare_surfaces(app_state, ctx["data"])

    with ui.row().classes("w-full items-center justify-between"):
        ui.label("Surfaces").classes("text-h6")
        if vm.wb_source_name:
            ui.badge(f"WB: {vm.wb_source_name}", color="positive")

    if vm.warning:
        ui.label(vm.warning).classes("text-body2 text-grey-6")
        return

    if not vm.chips:
        ui.label(
            "No surfaces selected — add reflectors from the Reflectors shelf "
            "(right drawer) to preview how materials render under the current stack."
        ).classes("text-body2 text-grey-6")
        return

    with ui.row().classes("w-full flex-wrap gap-2"):
        for chip in vm.chips:
            _render_chip(ctx, chip)


def _render_chip(ctx: Dict[str, Any], chip) -> None:
    with ui.card().classes("items-center q-pa-sm"):
        ui.element("div").style(
            f"background: {chip.hex_display}; width: 64px; height: 44px; "
            "border: 1px solid #999; border-radius: 4px;"
        ).tooltip(
            "linear R/G/B: "
            f"{chip.linear_rgb[0]:.3f} / {chip.linear_rgb[1]:.3f} / {chip.linear_rgb[2]:.3f}"
        )
        ui.label(chip.name).classes("text-caption ellipsis").style("max-width: 120px")
        ui.label(chip.hex_display).classes("text-caption text-grey-6")
        with ui.row().classes("q-pa-none gap-1"):
            if chip.is_wb_ref:
                ui.button(
                    "WB ●", icon="sync",
                    on_click=lambda: ctx["dispatch"]("reset_surface_wb"),
                ).props("dense unelevated color=positive size=sm").tooltip(
                    "Active WB reference — click to reset"
                )
            else:
                ui.button(
                    "WB", icon="sync",
                    on_click=lambda c=chip: ctx["dispatch"](
                        "wb_from_surface", {"source_file": c.source_file}
                    ),
                ).props("dense flat size=sm").tooltip("White-balance on this surface")
            ui.button(
                icon="close",
                on_click=lambda c=chip: ctx["dispatch"](
                    "remove_surface", {"source_file": c.source_file}
                ),
            ).props("dense flat round size=sm color=negative").tooltip("Remove surface")
