"""Shell: page scaffolding, panel registry, targeted refresh."""
from typing import Any, Callable, Dict

from nicegui import ui

from models.constants import FEATURES
from services.presenter import prepare_header_status
from views.actions import handle_app_action
from views.reference import render_reference
from views.sensor_response import render_sensor_chart, render_sensor_response
from views.scene_preview import render_scene_preview
from views.setup_rail import render_setup_rail
from views.shelf_data import render_data_shelf
from views.shelf_filters import render_filters_shelf
from views.shelf_reflectors import render_reflectors_shelf
from views.transmission import render_transmission


def build_page(app_state, data) -> None:
    page_state: Dict[str, Any] = {
        "trans_log_stops": False,
        "filter_query": "",
        "adv_open": False,
        "mf_open": False,
    }
    ctx: Dict[str, Any] = {"app_state": app_state, "data": data, "page_state": page_state}
    registry: Dict[str, Callable[[], None]] = {}

    # --- Refreshable panels: one registration per render ---------------------
    panels: Dict[str, Callable[[], None]] = {}

    def _panel(key: str, render: Callable[[Dict[str, Any]], None]) -> None:
        @ui.refreshable
        def panel():
            render(ctx)
        panels[key] = panel            # layout renders it into its slot
        registry[key] = panel.refresh  # dirty-key dispatch re-renders it

    _panel("setup", render_setup_rail)
    if FEATURES["reflectance"]:
        _panel("scene_preview", render_scene_preview)
    _panel("transmission", render_transmission)
    _panel("sensor_response", render_sensor_response)
    _panel("reference", render_reference)
    _panel("shelf_filters", render_filters_shelf)
    if FEATURES["reflectance"]:
        _panel("shelf_reflectors", render_reflectors_shelf)
    _panel("shelf_data", render_data_shelf)

    # Nested region inside the sensor_response panel: only the chart block
    # (WB readout + response curves). set_mixer_gain refreshes this key
    # instead of the whole panel so the mixer inputs are never rebuilt
    # mid-typing.
    @ui.refreshable
    def sensor_chart():
        render_sensor_chart(ctx)
    registry["sensor_response_chart"] = sensor_chart.refresh
    ctx["render_sensor_chart"] = sensor_chart

    # --- Targeted refresh + dispatch ------------------------------------------
    def refresh_panel(key: str) -> None:
        if key == "*":
            for refresh in registry.values():
                refresh()
            return
        refresh = registry.get(key)
        if refresh:
            refresh()

    def dispatch(action: str, payload=None) -> None:
        dirty = handle_app_action(action, payload, ctx)
        for key in sorted(dirty):
            refresh_panel(key)

    ctx["dispatch"] = dispatch
    ctx["refresh_panel"] = refresh_panel

    # Header status strip: a read-only one-liner mirroring the analysis state
    # (receiver · illuminant · stack · WB). Registered like a panel so any
    # optical-chain change re-renders it.
    @ui.refreshable
    def header_status():
        status = prepare_header_status(app_state, data)
        ui.label(status.text).classes("text-caption text-white opacity-80")
    registry["header_status"] = header_status.refresh

    # Reset asks for confirmation: it clears the analysis chain (stack, WB,
    # mixer) while keeping loaded data, My Filters and surfaces.
    def _confirm_reset() -> None:
        reset_confirm.close()
        dispatch("reset_all")

    with ui.dialog() as reset_confirm, ui.card():
        ui.label("Reset the analysis chain?").classes("text-h6")
        ui.label(
            "Clears the filter stack, white balance and channel mixer. "
            "Loaded data, My Filters and selected surfaces are kept."
        ).classes("text-body2 text-grey-7")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=reset_confirm.close).props("flat")
            ui.button("Reset", icon="restart_alt", color="negative",
                      on_click=_confirm_reset)

    # --- Layout --------------------------------------------------------------
    ui.add_head_html("""
    <style>
        body { font-family: Inter, system-ui, sans-serif; }
        .q-menu .q-item__label {
            overflow: hidden; text-overflow: ellipsis;
            white-space: nowrap; max-width: 400px;
        }
        .q-menu { max-width: 460px !important; }
    </style>
    """)

    with ui.header().classes("items-center justify-between px-4 py-2 bg-blue-800"):
        ui.label("FS FilterLab").classes("text-white text-lg font-bold")
        header_status()
        with ui.row().classes("items-center gap-2"):
            ui.button("Report", icon="picture_as_pdf", on_click=lambda: dispatch("run_report"))\
                .props("flat dense color=white")
            ui.button("Reset", icon="restart_alt", on_click=reset_confirm.open)\
                .props("flat dense color=white")

    with ui.left_drawer(value=True, fixed=True, bordered=True).classes("bg-grey-1 p-3").props("behavior=desktop width=320"):
        panels["setup"]()

    with ui.right_drawer(bordered=True).classes("bg-grey-1 p-3").props("behavior=desktop width=390") as shelf_drawer:
        with ui.tabs() as shelf_tabs:
            ui.tab("filters", "Filters")
            if FEATURES["reflectance"]:
                ui.tab("reflectors", "Reflectors")
            ui.tab("data", "Data")
        with ui.tab_panels(shelf_tabs, value="filters").classes("w-full"):
            with ui.tab_panel("filters"):
                panels["shelf_filters"]()
            if FEATURES["reflectance"]:
                with ui.tab_panel("reflectors"):
                    panels["shelf_reflectors"]()
            with ui.tab_panel("data"):
                panels["shelf_data"]()
    shelf_drawer.props("breakpoint=0")

    with ui.column().classes("q-px-lg q-py-md w-full max-w-[1100px]"):
        panels["transmission"]()
        panels["sensor_response"]()
        if FEATURES["reflectance"]:
            panels["scene_preview"]()
        panels["reference"]()

