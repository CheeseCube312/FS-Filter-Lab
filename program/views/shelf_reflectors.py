"""Reflectors shelf: library search, spectrum sparklines, add-as-surface."""
from typing import Any, Dict

from nicegui import ui

from services.presenter import prepare_reflectors_shelf

_ROW_LIMIT = 40


def render_reflectors_shelf(ctx: Dict[str, Any]) -> None:
    app_state = ctx["app_state"]
    data = ctx["data"]
    page_state = ctx["page_state"]

    ui.label("Reflectors").classes("text-h6")

    orgs_sel = page_state.setdefault("refl_orgs", [])
    targets_sel = page_state.setdefault("refl_targets", [])

    @ui.refreshable
    def results_section():
        vm = prepare_reflectors_shelf(
            app_state, data, orgs=orgs_sel, target_types=targets_sel, limit=_ROW_LIMIT,
        )
        tail = f" — showing first {_ROW_LIMIT}" if vm.total_matches > _ROW_LIMIT else ""
        ui.label(f"{vm.total_matches} matches{tail}").classes("text-caption text-grey-6")
        if not vm.rows:
            ui.label("No reflectors match the current filters.").classes(
                "text-body2 text-grey-6"
            )
        for row in vm.rows:
            _render_row(ctx, row)

    df = data.reflector_collection.df
    full_orgs = _column_values(df, "Organization")
    full_targets = _column_values(df, "Target Type")

    ui.select(
        full_orgs, multiple=True, value=orgs_sel, label="Organization",
        on_change=lambda e: (_sync(orgs_sel, e.value), results_section.refresh()),
    ).classes("w-full").props("outlined dense use-chips")
    ui.select(
        full_targets, multiple=True, value=targets_sel, label="Target type",
        on_change=lambda e: (_sync(targets_sel, e.value), results_section.refresh()),
    ).classes("w-full").props("outlined dense use-chips")

    results_section()


def _sync(target_list: list, value) -> None:
    target_list[:] = [str(v) for v in (value or [])]


def _column_values(df, column) -> list:
    """Distinct sorted values; [] when the stock TSV lacks the column."""
    return sorted(set(df[column].astype(str))) if column in df.columns else []


def _render_row(ctx: Dict[str, Any], row) -> None:
    with ui.row().classes("w-full items-center no-wrap q-py-xs"):
        ui.element("div").style(
            f"background: {row.preview_hex}; width: 18px; height: 18px; "
            "border: 1px solid #999; flex: none;"
        ).tooltip(f"Preview under current stack: {row.preview_hex}")
        ui.plotly(row.sparkline).tooltip(
            "Raw reflectance 300–1100 nm — line breaks mark no-data gaps"
        )
        with ui.column().classes("col gap-0"):
            ui.label(row.name).classes("text-body2 ellipsis")
            ui.label(
                f"{row.organization} · {row.coverage_note}"
                + (f" · {row.target_type}" if row.target_type else "")
            ).classes("text-caption text-grey-6 ellipsis")
        if row.is_default:
            ui.icon("verified", color="teal").tooltip("Default reflector")
        if row.is_wb_ref:
            ui.badge("WB", color="positive")
        if row.is_surface:
            ui.button(
                icon="check_circle", color="positive",
                on_click=lambda r=row: ctx["dispatch"](
                    "remove_surface", {"source_file": r.source_file}
                ),
            ).props("flat dense round size=sm").tooltip("Added as surface — click to remove")
        else:
            ui.button(
                icon="add_circle",
                on_click=lambda r=row: ctx["dispatch"](
                    "add_surface", {"source_file": r.source_file}
                ),
            ).props("flat dense round size=sm").tooltip("Add as surface")

