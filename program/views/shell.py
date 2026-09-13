"""Shell: page scaffolding, PanelRegistry, targeted refresh (UX_SPEC.md §3/§5)."""
from typing import Any, Callable, Dict

from nicegui import ui

from views.actions import handle_app_action
from views.reference import render_reference
from views.sensor_response import render_sensor_response
from views.setup_rail import render_setup_rail
from views.shelf_filters import render_filters_shelf
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

    # --- Refreshable panels -------------------------------------------------
    @ui.refreshable
    def setup_panel():
        render_setup_rail(ctx)

    @ui.refreshable
    def transmission_panel():
        render_transmission(ctx)

    @ui.refreshable
    def sensor_response_panel():
        render_sensor_response(ctx)

    @ui.refreshable
    def surfaces_panel():
        _placeholder("Surfaces grid lands in slice U4 (white-referenced chips, §7.1).")

    @ui.refreshable
    def film_match_panel():
        _placeholder("Film Match lands in slice U5 (target overlay + mixer, §7.2).")

    @ui.refreshable
    def reference_panel():
        render_reference(ctx)

    @ui.refreshable
    def filters_shelf_panel():
        render_filters_shelf(ctx)

    @ui.refreshable
    def reflectors_shelf_panel():
        _placeholder("Reflectors shelf lands in slice U4 (library + ECOSIS preview import).")

    @ui.refreshable
    def data_shelf_panel():
        _placeholder("Data shelf lands in slice U4 (importers; digitized-CSV fast path).")

    # --- Targeted refresh + dispatch (§5 contract) ---------------------------
    def refresh_panel(key: str) -> None:
        if key == "*":
            for refresh in list(registry.values()):
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

    registry.update({
        "setup": setup_panel.refresh,
        "transmission": transmission_panel.refresh,
        "sensor_response": sensor_response_panel.refresh,
        "surfaces": surfaces_panel.refresh,
        "film_match": film_match_panel.refresh,
        "reference": reference_panel.refresh,
        "shelf_filters": filters_shelf_panel.refresh,
        "shelf_reflectors": reflectors_shelf_panel.refresh,
        "shelf_data": data_shelf_panel.refresh,
    })

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
        with ui.row().classes("items-center gap-2"):
            ui.button("Report", icon="picture_as_pdf", on_click=lambda: dispatch("run_report"))\
                .props("flat dense color=white")
            ui.button("Reset", icon="restart_alt", on_click=lambda: dispatch("reset_all"))\
                .props("flat dense color=white")

    with ui.left_drawer(value=True, fixed=True, bordered=True).classes("bg-grey-1 p-3").props("behavior=desktop width=320"):
        setup_panel()

    with ui.right_drawer(bordered=True).classes("bg-grey-1 p-3").props("behavior=desktop width=390") as shelf_drawer:
        with ui.tabs() as shelf_tabs:
            ui.tab("filters", "Filters")
            ui.tab("reflectors", "Reflectors")
            ui.tab("data", "Data")
        with ui.tab_panels(shelf_tabs, value="filters").classes("w-full"):
            with ui.tab_panel("filters"):
                filters_shelf_panel()
            with ui.tab_panel("reflectors"):
                reflectors_shelf_panel()
            with ui.tab_panel("data"):
                data_shelf_panel()
    shelf_drawer.props("breakpoint=0")

    with ui.column().classes("q-px-lg q-py-md w-full max-w-[1100px]"):
        transmission_panel()
        sensor_response_panel()
        with ui.tabs() as canvas_tabs:
            ui.tab("surfaces", "Surfaces")
            ui.tab("film", "Film Match")
            ui.tab("reference", "Reference")
        with ui.tab_panels(canvas_tabs, value="surfaces").classes("w-full"):
            with ui.tab_panel("surfaces"):
                surfaces_panel()
            with ui.tab_panel("film"):
                film_match_panel()
            with ui.tab_panel("reference"):
                reference_panel()


def _placeholder(message: str) -> None:
    with ui.card().classes("w-full"):
        ui.label(message).classes("text-body2 text-grey-6")
