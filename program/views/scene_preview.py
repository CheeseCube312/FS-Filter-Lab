"""Scene Preview card: per-surface render swatches + camera-look toggle."""
from typing import Any, Dict

from nicegui import ui

from services.presenter import prepare_scene_preview


def render_scene_preview(ctx: Dict[str, Any]) -> None:
    vm = prepare_scene_preview(ctx["app_state"], ctx["data"])

    with ui.row().classes("w-full items-center justify-between"):
        with ui.row().classes("items-center gap-2 q-pa-none"):
            ui.label("Scene Preview").classes("text-h6")
            if vm.wb_source_name:
                ui.badge(f"WB: {vm.wb_source_name}", color="positive")
        # Buried render-mode toggle: the camera look (filmic tone + chroma).
        ui.button(
            icon="photo_camera",
            on_click=lambda: ctx["dispatch"](
                "set_scene_look", {"value": not vm.look_on}),
        ).props(
            "flat dense round size=sm color={}".format(
                "primary" if vm.look_on else "grey-6")
        ).tooltip(
            "Camera look is ON — filmic tone curve + chroma boost"
            + (" · negative film inverted" if vm.inverts_for_display else "")
            if vm.look_on
            else "Camera look is OFF — plain white-referenced render"
        )

    if vm.warning:
        ui.label(vm.warning).classes("text-body2 text-grey-6")
        return

    if not vm.swatches:
        with ui.row().classes("w-full items-center gap-3"):
            ui.label(
                "No scene surfaces — load the curated default set or add "
                "reflectors from the Reflectors shelf (right drawer)."
            ).classes("text-body2 text-grey-6")
            ui.button(
                "Load default scene set", icon="park",
                on_click=lambda: ctx["dispatch"]("load_default_scene"),
            ).props("dense unelevated").tooltip(
                "Add the curated default reflectors as scene surfaces")
        return

    with ui.row().classes("w-full flex-wrap gap-3"):
        for swatch in vm.swatches:
            _render_swatch(ctx, swatch)


def _render_swatch(ctx: Dict[str, Any], swatch) -> None:
    with ui.card().classes("items-center q-pa-sm"):
        # The swatch is its own popover trigger; it shows the active render —
        # plain or camera look, per the card's camera-look toggle.
        with ui.button().props("flat dense").style(
                "min-width: 68px; min-height: 46px; padding: 0; "
                "border-radius: 4px;"):
            ui.element("div").style(
                f"background: {swatch.display_hex}; width: 64px; "
                "height: 44px; border: 1px solid #999; border-radius: 4px;"
            )
            with ui.menu():
                ui.label(swatch.name).classes(
                    "text-body2 text-bold q-px-md q-pt-sm")
                ui.label(
                    "linear R/G/B: "
                    f"{swatch.display_linear[0]:.3f} / "
                    f"{swatch.display_linear[1]:.3f} / "
                    f"{swatch.display_linear[2]:.3f}"
                ).classes("text-caption q-px-md")
                if swatch.partial_data:
                    ui.label(
                        f"Partial data — no reflectance: {swatch.coverage_note}"
                    ).classes("text-caption text-orange-8 q-px-md")
                with ui.row().classes("q-px-md q-py-sm gap-1"):
                    if swatch.is_wb_ref:
                        ui.button(
                            "WB ●", icon="sync",
                            on_click=lambda: ctx["dispatch"]("reset_surface_wb"),
                        ).props("dense unelevated color=positive size=sm").tooltip(
                            "Active WB reference — click to reset")
                    else:
                        ui.button(
                            "WB", icon="sync",
                            on_click=lambda s=swatch: ctx["dispatch"](
                                "wb_from_surface",
                                {"source_file": s.source_file}),
                        ).props("dense flat size=sm").tooltip(
                            "White-balance on this surface")
                    ui.button(
                        icon="close",
                        on_click=lambda s=swatch: ctx["dispatch"](
                            "remove_surface", {"source_file": s.source_file}),
                    ).props("dense flat round size=sm color=negative").tooltip(
                        "Remove surface")
        ui.plotly(swatch.sparkline).tooltip(
            "Raw reflectance 300–1100 nm — line breaks mark no-data gaps")
        ui.label(swatch.name).classes("text-caption ellipsis").style(
            "max-width: 140px")
        if swatch.partial_data:
            ui.badge("partial data", color="warning").tooltip(
                f"No reflectance data: {swatch.coverage_note}")