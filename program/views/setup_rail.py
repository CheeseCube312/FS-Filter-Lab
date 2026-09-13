"""Setup rail: sensor / film / illuminant + stack editor (UX_SPEC.md §3.1)."""
from typing import Any, Dict

from nicegui import ui

from services.presenter import prepare_filter_rows


def render_setup_rail(ctx: Dict[str, Any]) -> None:
    app_state = ctx["app_state"]
    data = ctx["data"]

    ui.label("Setup").classes("text-h6 q-mb-sm")

    profiles = data.sensor_profiles
    sensors = sorted(k for k, p in profiles.items() if not p.is_film)
    films = sorted(k for k, p in profiles.items() if p.is_film)
    current = app_state.selected_camera
    current_is_film = bool(app_state.active_profile and app_state.active_profile.is_film)

    # --- Receiver selection: sensor OR film (single active_profile slot) ---
    ui.select(
        sensors if sensors else ["No sensors loaded"],
        value=(current if current and not current_is_film else None),
        label="Sensor",
        on_change=lambda e: ctx["dispatch"]("select_sensor", {"name": e.value}),
    ).classes("w-full").props('clearable outlined dense')

    ui.select(
        (["None"] + films) if films else ["None"],
        value=(current if current_is_film else "None"),
        label="Film",
        on_change=lambda e: ctx["dispatch"]("select_film", {"name": e.value}),
    ).classes("w-full").props("outlined dense")

    ui.select(
        sorted(data.illuminants.keys()),
        value=app_state.illuminant_name,
        label="Illuminant",
        on_change=lambda e: ctx["dispatch"]("select_illuminant", {"name": e.value}),
    ).classes("w-full").props("outlined dense")

    ui.switch(
        "Owned filters only",
        value=app_state.my_filters_only,
        on_change=lambda e: ctx["dispatch"]("set_owned_only", {"value": e.value}),
    )

    # --- Stack editor ---
    ui.label("Filter stack").classes("text-subtitle1 q-mt-md")
    if not app_state.selected_filters:
        ui.label("Stack is empty.").classes("text-body2 text-grey-6")
    else:
        for name in list(app_state.selected_filters):
            _render_stack_row(ctx, name)
        ui.button(
            "Clear stack",
            icon="delete_sweep",
            on_click=lambda: ctx["dispatch"]("clear_stack"),
        ).props("flat dense color=negative")

    # --- Quick add ---
    ui.label("Quick add").classes("text-subtitle1 q-mt-md")
    search = ui.input(placeholder="Search by name or number…").classes("w-full").props("dense outlined")
    search.props('debounce="300"')
    results = ui.column().classes("w-full gap-1")
    search.on_value_change(
        lambda e: _render_quick_results(ctx, results, str(e.value or ""))
    )


def _render_stack_row(ctx: Dict[str, Any], name: str) -> None:
    app_state = ctx["app_state"]
    with ui.row().classes("w-full items-center no-wrap q-py-xs"):
        ui.label(name).classes("text-body2 col ellipsis")
        ui.select(
            [1, 2, 3, 4, 5],
            value=app_state.filter_multipliers.get(name, 1),
            on_change=lambda e, n=name: ctx["dispatch"]("set_multiplier", {"name": n, "value": e.value}),
        ).classes("w-16").props("dense outlined")
        ui.button(
            icon="close",
            on_click=lambda n=name: ctx["dispatch"]("remove_filter", {"name": n}),
        ).props("flat dense round size=sm color=negative")


def _render_quick_results(ctx: Dict[str, Any], container: ui.column, query: str) -> None:
    app_state = ctx["app_state"]
    data = ctx["data"]
    container.clear()
    if not query.strip():
        return
    rows = prepare_filter_rows(app_state, data, query=query, limit=8)
    with container:
        if not rows:
            ui.label("No matches.").classes("text-body2 text-grey-6")
        for row in rows:
            with ui.row().classes("w-full items-center no-wrap q-py-xs"):
                ui.element("div").style(
                    f"background: {row.hex_color}; width: 14px; height: 14px; "
                    "border: 1px solid #999; flex: none;"
                )
                ui.label(row.display_name).classes("text-body2 col ellipsis")
                if row.is_owned:
                    ui.icon("star", color="amber-6").classes("text-caption")
                ui.button(
                    icon="add",
                    on_click=lambda r=row: ctx["dispatch"]("add_filter", {"name": r.display_name}),
                ).props("flat dense round size=sm").tooltip("Add to stack")
