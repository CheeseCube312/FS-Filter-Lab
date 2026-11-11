"""Filters shelf: quick + advanced search, results, My Filters."""
from typing import Any, Dict

from nicegui import ui

from services.presenter import prepare_filter_rows

_RESULT_LIMIT = 60


def render_filters_shelf(ctx: Dict[str, Any]) -> None:
    app_state = ctx["app_state"]
    page_state = ctx["page_state"]

    ui.label("Filters").classes("text-h6")

    search = ui.input(
        placeholder="Search by name, number, or brand…  e.g. “lee oklahoma”",
        value=page_state.get("filter_query", ""),
    ).classes("w-full").props('outlined dense debounce="300"')

    ui.switch(
        "Owned only",
        value=app_state.my_filters_only,
        on_change=lambda e: ctx["dispatch"]("set_owned_only", {"value": e.value}),
    )

    with ui.expansion(
        "Advanced search", icon="tune", value=page_state.get("adv_open", False)
    ).classes("w-full") as adv:
        _render_advanced(ctx)
    adv.on_value_change(lambda e: page_state.__setitem__("adv_open", e.value))

    @ui.refreshable
    def results_section():
        _render_results(ctx)

    results_section()
    search.on_value_change(
        lambda e: (
            page_state.__setitem__("filter_query", str(e.value or "")),
            results_section.refresh(),
        )
    )

    owned = sorted(app_state.get_my_filters())
    with ui.expansion(
        f"My Filters ({len(owned)})", icon="stars", value=page_state.get("mf_open", False)
    ).classes("w-full") as mf:
        _render_my_filters_manager(ctx, owned)
    mf.on_value_change(lambda e: page_state.__setitem__("mf_open", e.value))


def _manufacturer_names(data) -> list:
    return sorted(set(data.filter_collection.df["Manufacturer"].astype(str)))


def _render_advanced(ctx: Dict[str, Any]) -> None:
    app_state = ctx["app_state"]
    data = ctx["data"]

    manufacturers = _manufacturer_names(data)
    ui.select(
        manufacturers,
        multiple=True,
        value=list(app_state.advanced_search_manufacturers),
        label="Manufacturers",
        on_change=lambda e: ctx["dispatch"](
            "set_advanced_manufacturers", {"value": e.value}
        ),
    ).classes("w-full").props("outlined dense use-chips")

    ui.label("Transmission criteria (ALL must match)").classes(
        "text-caption text-grey-7 q-mt-sm"
    )
    criteria = app_state.advanced_search_wavelength_criteria
    for i, crit in enumerate(criteria):
        with ui.row().classes("w-full items-center no-wrap"):
            ui.number(
                value=crit["wavelength"], min=300, max=1100, format="%.0f",
                on_change=lambda e, i=i: ctx["dispatch"](
                    "update_advanced_criterion",
                    {"index": i, "field": "wavelength", "value": e.value},
                ),
            ).props("outlined dense label='λ (nm)'").classes("w-28")
            ui.number(
                value=crit["trans_min"], min=0, max=100, format="%.0f",
                on_change=lambda e, i=i: ctx["dispatch"](
                    "update_advanced_criterion",
                    {"index": i, "field": "trans_min", "value": e.value},
                ),
            ).props("outlined dense label='min %'").classes("w-20")
            ui.number(
                value=crit["trans_max"], min=0, max=100, format="%.0f",
                on_change=lambda e, i=i: ctx["dispatch"](
                    "update_advanced_criterion",
                    {"index": i, "field": "trans_max", "value": e.value},
                ),
            ).props("outlined dense label='max %'").classes("w-20")
            if len(criteria) > 1:
                ui.button(
                    icon="close",
                    on_click=lambda idx=i: ctx["dispatch"](
                        "remove_advanced_criterion", {"index": idx}
                    ),
                ).props("flat dense round size=sm color=negative")
    ui.button(
        "Add criterion", icon="add",
        on_click=lambda: ctx["dispatch"]("add_advanced_criterion", {}),
    ).props("flat dense")

    ui.select(
        ["Filter Number", "Filter Name", "Trans @ λ", "Hex-Rainbow"],
        value=app_state.advanced_search_sort,
        label="Sort",
        on_change=lambda e: ctx["dispatch"](
            "set_advanced_sort", {"value": str(e.value)}
        ),
    ).classes("w-full").props("outlined dense")

    ui.button(
        "Apply", icon="search",
        on_click=lambda: ctx["refresh_panel"]("shelf_filters"),
    ).props("outlined dense")


def _render_results(ctx: Dict[str, Any]) -> None:
    app_state = ctx["app_state"]
    data = ctx["data"]
    page_state = ctx["page_state"]

    use_advanced = page_state.get("adv_open", False)
    query = page_state.get("filter_query", "")

    if not use_advanced and not query.strip():
        ui.label("Type to search the filter library, or open Advanced search.").classes(
            "text-body2 text-grey-6"
        )
        return

    shelf = prepare_filter_rows(
        app_state,
        data,
        query=query,
        use_advanced=use_advanced,
        limit=_RESULT_LIMIT,
    )

    if not shelf.rows:
        ui.label("No matches — adjust the search or criteria.").classes(
            "text-body2 text-grey-6"
        )
        return

    match_note = f"{shelf.total_matches} matches"
    if shelf.total_matches > len(shelf.rows):
        match_note += f" — showing first {len(shelf.rows)}"
    ui.label(match_note).classes("text-caption text-grey-6")

    for row in shelf.rows:
        _render_row(ctx, row)


def _render_row(ctx: Dict[str, Any], row) -> None:
    with ui.row().classes("w-full items-center no-wrap q-py-xs"):
        ui.element("div").style(
            f"background: {row.hex_color}; width: 18px; height: 18px; "
            "border: 1px solid #999; flex: none;"
        )
        ui.plotly(row.sparkline)
        with ui.column().classes("col gap-0"):
            ui.label(row.display_name).classes("text-body2 ellipsis")
            ui.label(row.manufacturer).classes("text-caption text-grey-6")
        ui.button(
            icon="star" if row.is_owned else "star_border",
            color="amber-6",
            on_click=lambda r=row: ctx["dispatch"]("toggle_my_filter", {"name": r.display_name}),
        ).props("flat dense round size=sm").tooltip("Owned")
        if row.in_stack:
            ui.button(
                icon="check_circle", color="positive",
                on_click=lambda r=row: ctx["dispatch"]("remove_filter", {"name": r.display_name}),
            ).props("flat dense round size=sm").tooltip(
                f"In stack ×{row.multiplier} — click to remove"
            )
        else:
            ui.button(
                icon="add_circle",
                on_click=lambda r=row: ctx["dispatch"]("add_filter", {"name": r.display_name}),
            ).props("flat dense round size=sm").tooltip("Add to stack")


def _render_my_filters_manager(ctx: Dict[str, Any], owned) -> None:
    app_state = ctx["app_state"]
    data = ctx["data"]
    manufacturers = _manufacturer_names(data)

    ui.label("Manage by manufacturer").classes("text-caption text-grey-7")
    for m in manufacturers:
        with ui.row().classes("w-full items-center no-wrap"):
            ui.label(m).classes("text-body2 col ellipsis")
            ui.button(
                icon="library_add",
                on_click=lambda mm=m: ctx["dispatch"](
                    "add_my_filter_manufacturer", {"manufacturer": mm}
                ),
            ).props("flat dense round size=sm").tooltip("Mark all as owned")
            ui.button(
                icon="library_remove",
                on_click=lambda mm=m: ctx["dispatch"](
                    "remove_my_filter_manufacturer", {"manufacturer": mm}
                ),
            ).props("flat dense round size=sm").tooltip("Unmark all")

    ui.separator()
    if not owned:
        ui.label("No owned filters yet — star results above.").classes(
            "text-body2 text-grey-6"
        )
        return
    with ui.scroll_area().classes("w-full").style("max-height: 240px"):
        for name in owned[:200]:
            with ui.row().classes("w-full items-center no-wrap"):
                ui.label(name).classes("text-body2 col ellipsis")
                ui.button(
                    icon="close",
                    on_click=lambda n=name: ctx["dispatch"]("toggle_my_filter", {"name": n}),
                ).props("flat dense round size=sm color=negative")
    if len(owned) > 200:
        ui.label(f"…and {len(owned) - 200} more.").classes("text-caption text-grey-6")
    ui.button(
        "Clear all", icon="delete_sweep",
        on_click=lambda: ctx["dispatch"]("clear_my_filters"),
    ).props("flat dense color=negative")

