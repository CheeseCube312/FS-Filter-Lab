"""Reflectors shelf: library search, surface/WB actions, ECOSIS import (§3.3)."""
from typing import Any, Dict

from nicegui import ui

from services.presenter import prepare_ecosis_preview, prepare_reflectors_shelf

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
    full_orgs = (
        sorted(set(df["Organization"].astype(str)))
        if "Organization" in df.columns else []
    )
    full_targets = (
        sorted(set(df["Target Type"].astype(str)))
        if "Target Type" in df.columns else []
    )

    ui.select(
        full_orgs, multiple=True, value=orgs_sel, label="Organization",
        on_change=lambda e: (_sync(orgs_sel, e.value), results_section.refresh()),
    ).classes("w-full").props("outlined dense use-chips")
    ui.select(
        full_targets, multiple=True, value=targets_sel, label="Target type",
        on_change=lambda e: (_sync(targets_sel, e.value), results_section.refresh()),
    ).classes("w-full").props("outlined dense use-chips")

    results_section()

    with ui.expansion("ECOSIS import (preview before commit)", icon="upload").classes("w-full"):
        render_ecosis_card(ctx, page_state)


def _sync(target_list: list, value) -> None:
    target_list[:] = [str(v) for v in (value or [])]

def _render_row(ctx: Dict[str, Any], row) -> None:
    with ui.row().classes("w-full items-center no-wrap q-py-xs"):
        ui.element("div").style(
            f"background: {row.preview_hex}; width: 18px; height: 18px; "
            "border: 1px solid #999; flex: none;"
        ).tooltip(f"Preview under current stack: {row.preview_hex}")
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
        ui.button(
            icon="sync",
            color="positive" if row.is_wb_ref else None,
            on_click=lambda r=row: (
                ctx["dispatch"]("reset_surface_wb") if r.is_wb_ref
                else ctx["dispatch"]("wb_from_surface", {"source_file": r.source_file})
            ),
        ).props("flat dense round size=sm").tooltip(
            "Active WB reference — click to reset"
            if row.is_wb_ref else "White-balance on this surface"
        )
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


def render_ecosis_card(ctx: Dict[str, Any], page_state: Dict[str, Any]) -> None:
    ui.label(
        "Upload an ECOSIS multi-spectrum CSV. Nothing is written to disk until "
        "you confirm the import."
    ).classes("text-caption text-grey-7")

    @ui.refreshable
    def preview_section():
        pvm = page_state.get("ecosis_preview")
        if pvm is None:
            return
        if pvm.error:
            ui.label(f"Could not parse file: {pvm.error}").classes("text-body2 text-negative")
            return
        with ui.card().classes("w-full q-pa-sm"):
            ui.label(f"{pvm.filename}: {pvm.n_rows} rows × {pvm.n_cols} columns").classes(
                "text-body2"
            )
            if pvm.organizations:
                shown = ", ".join(pvm.organizations[:6]) + (
                    "…" if len(pvm.organizations) > 6 else ""
                )
                ui.label(f"Organizations: {shown}").classes("text-caption text-grey-6")
            tail = "…" if pvm.n_rows > 8 else ""
            ui.label("Samples: " + ", ".join(pvm.sample_names) + tail).classes(
                "text-caption text-grey-6"
            )
            ui.button(
                f"Import {pvm.n_rows} spectra", icon="download_done",
                on_click=lambda: ctx["dispatch"]("import_ecosis", {
                    "content": page_state.get("ecosis_content"),
                    "filename": pvm.filename,
                }),
            ).props("unelevated dense color=positive")

    preview_section()

    ui.upload(
        auto_upload=True,
        on_upload=lambda e: _handle_upload(ctx, e, preview_section),
    ).classes("w-full max-w-xs").props("accept=.csv label='Choose ECOSIS CSV'")

    # Persistent per-card import status, shared with the Data shelf cards.
    # Function-local import: shelf_data imports this module at module level.
    from views.shelf_data import _status_line

    _status_line(ctx["app_state"], "ecosis")


def _handle_upload(ctx: Dict[str, Any], event, preview_section) -> None:
    content = event.content.read()
    ctx["page_state"]["ecosis_content"] = content
    ctx["page_state"]["ecosis_preview"] = prepare_ecosis_preview(content, event.name)
    preview_section.refresh()

