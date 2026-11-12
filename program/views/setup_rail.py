"""Setup rail: sensor / film / illuminant + stack editor."""
from typing import Any, Dict

from nicegui import ui

from services.presenter import (
    prepare_receiver_options,
    prepare_stack_rows,
    StackRowVM,
)


def render_setup_rail(ctx: Dict[str, Any]) -> None:
    app_state = ctx["app_state"]
    data = ctx["data"]

    ui.label("Setup").classes("text-h6 q-mb-sm")

    # --- Receiver selection: ONE dropdown covering sensors AND film ----------
    # Both feed the same active_profile slot (select_sensor looks up by
    # display name across every profile). NiceGUI validates the select value
    # against the top-level keys of an options dict — grouped/nested options
    # are unsupported — so the presenter hands over one flat {name: label}
    # dict (films carry a " (film)" label suffix) with the current selection
    # already validated: a stale name renders as empty instead of raising.
    receiver = prepare_receiver_options(app_state, data)
    ui.select(
        receiver.options,
        value=receiver.current,
        label="Receiver",
        on_change=lambda e: ctx["dispatch"]("select_sensor", {"name": e.value}),
    ).classes("w-full").props('clearable outlined dense')

    ui.select(
        sorted(data.illuminants.keys()),
        value=app_state.illuminant_name,
        label="Illuminant",
        on_change=lambda e: ctx["dispatch"]("select_illuminant", {"name": e.value}),
    ).classes("w-full").props("outlined dense")

    # --- Stack editor ---
    ui.label("Filter stack").classes("text-subtitle1 q-mt-md")
    if not app_state.selected_filters:
        ui.label("Stack is empty.").classes("text-body2 text-grey-6")
    else:
        for row in prepare_stack_rows(app_state, data.filter_collection):
            _render_stack_row(ctx, row)
        ui.button(
            "Clear stack",
            icon="delete_sweep",
            on_click=lambda: ctx["dispatch"]("clear_stack"),
        ).props("flat dense color=negative")


def _render_stack_row(ctx: Dict[str, Any], row: StackRowVM) -> None:
    with ui.row().classes("w-full items-center no-wrap q-py-xs"):
        ui.element("div").style(
            f"background: {row.hex_color}; width: 14px; height: 14px; "
            "border: 1px solid #999; flex: none;"
        ).tooltip(row.display_name)
        ui.label(row.display_name).classes("text-body2 col ellipsis")
        ui.select(
            [1, 2, 3, 4, 5],
            value=row.multiplier,
            on_change=lambda e, n=row.display_name: ctx["dispatch"](
                "set_multiplier", {"name": n, "value": e.value},
            ),
        ).classes("w-16").props("dense outlined").tooltip(
            f"Stacked ×{row.multiplier} — each pass multiplies the filter's "
            f"transmission (T^{row.multiplier})"
        )
        ui.button(
            icon="close",
            on_click=lambda n=row.display_name: ctx["dispatch"](
                "remove_filter", {"name": n},
            ),
        ).props("flat dense round size=sm color=negative")

