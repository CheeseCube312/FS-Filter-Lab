"""
Forms for FS FilterLab.

Provides UI components for:
- Advanced filter search with metadata filtering
- Advanced reflector search with organization/package filtering  
- Import dialogs for filters, illuminants, camera QE, film, and reflectance data

All import dialogs support searchable dropdowns for large datasets (1000+ items).
"""
import functools
import io
import logging
import os
import tempfile
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from nicegui import ui, events

from models.constants import (
    INTERP_GRID, UI_BUTTONS, UI_LABELS, UI_SECTIONS, UI_WARNING_MESSAGES,
    OUTPUT_FOLDERS, IMPORT_TABS, IMPORT_DATA_TYPES, IMPORT_CATEGORIES,
    IMPORT_ECOSIS_MODES, SENSOR_TYPE_LABELS,
)
from views.ui_utils import (
    is_dark_color, is_valid_hex_color, handle_error,
    show_success_message,
)
from services.presenter import search_filters, search_reflectors
from services.visualization import create_sparkline_plot

logger = logging.getLogger(__name__)


# ============================================================================
# HELPERS
# ============================================================================

@functools.lru_cache(maxsize=512)
def _cached_sparkline(wl_tuple, trans_tuple, color: str, height: int, width: int):
    """LRU-cached sparkline (tuples are hashable)."""
    return create_sparkline_plot(
        np.array(wl_tuple), np.array(trans_tuple), color=color,
        height=height, width=width,
    )


def _sparkline_fig(wavelengths, transmission, color, height=150, width=300):
    return _cached_sparkline(tuple(wavelengths), tuple(transmission), color, height, width)


# ============================================================================
# ADVANCED FILTER SEARCH
# ============================================================================

def render_advanced_filter_search(
    df: pd.DataFrame,
    filter_matrix: np.ndarray,
    app_state,
    on_change: Callable,
) -> None:
    """Render the advanced filter search panel with multiple wavelength criteria."""

    with ui.card().classes("w-full"):
        ui.label(UI_SECTIONS["advanced_filter_search"]).classes("text-lg font-bold")
        ui.label(UI_LABELS["search_by_manufacturer"]).classes("text-sm text-gray-500")

        # Manufacturer filter
        with ui.row().classes("w-full items-end gap-2"):
            ui.select(
                options=sorted(df["Manufacturer"].unique().tolist()),
                label="Manufacturer",
                value=app_state.advanced_search_manufacturers,
                multiple=True,
                on_change=lambda e: setattr(app_state, "advanced_search_manufacturers", list(e.value) if e.value else []),
            ).props("dense outlined use-chips").classes("flex-grow")

        ui.separator().classes("my-2")
        
        # Wavelength criteria section; ctx bundles the args row callbacks need
        results_container_ref = [None]
        ctx = (df, filter_matrix, on_change, results_container_ref)
        
        with ui.row().classes("w-full items-center gap-2"):
            ui.label("Wavelength Criteria (AND logic)").classes("text-sm font-semibold")
            ui.button("Add Criterion", on_click=lambda: _add_wavelength_criterion(
                app_state, criteria_container, ctx
            )).props("dense outline size=sm color=primary")
        
        criteria_container = ui.column().classes("w-full gap-1")
        
        # Render existing criteria
        _set_criteria(
            app_state, app_state.advanced_search_wavelength_criteria,
            criteria_container, ctx,
        )
        
        ui.separator().classes("my-2")
        
        # Sort and apply
        with ui.row().classes("w-full items-end gap-2"):
            ui.select(
                options=["Filter Number", "Filter Name", "Hex-Rainbow", "Trans @ λ"],
                label="Sort by", value=app_state.advanced_search_sort,
                on_change=lambda e: setattr(app_state, "advanced_search_sort", e.value),
            ).props("dense outlined").style("width:160px")

            ui.button(UI_BUTTONS["apply"], on_click=lambda: _apply_filter_search(
                df, filter_matrix, app_state, on_change, results_container_ref[0]
            )).props("dense")

        ui.separator().classes("my-2")
        
        # Results container at the end
        results_container = ui.column().classes("w-full")
        results_container_ref[0] = results_container

        # Initial search with saved state
        _apply_filter_search(df, filter_matrix, app_state, on_change, results_container)


def _render_criterion_row(idx: int, criterion: Dict, app_state, criteria_container, ctx):
    """Render a single wavelength criterion row."""
    with ui.row().classes("w-full items-center gap-2"):
        ui.number(
            label="λ (nm)", value=criterion["wavelength"], min=300, max=1100, step=5,
            on_change=lambda e, i=idx: _update_criterion(i, "wavelength", int(e.value), app_state),
        ).props("dense outlined").style("width:110px")
        
        ui.number(
            label="Min %", value=criterion["trans_min"], min=0, max=100, step=1,
            on_change=lambda e, i=idx: _update_criterion(i, "trans_min", int(e.value), app_state),
        ).props("dense outlined").style("width:90px")
        
        ui.number(
            label="Max %", value=criterion["trans_max"], min=0, max=100, step=1,
            on_change=lambda e, i=idx: _update_criterion(i, "trans_max", int(e.value), app_state),
        ).props("dense outlined").style("width:90px")
        
        # Show remove button only if more than one criterion
        if len(app_state.advanced_search_wavelength_criteria) > 1:
            ui.button("X", on_click=lambda i=idx: _remove_wavelength_criterion(
                i, app_state, criteria_container, ctx
            )).props("dense outline size=sm color=negative").style("width:40px")


def _update_criterion(idx: int, field: str, value, app_state):
    """Update a single field in a wavelength criterion."""
    criteria = app_state.advanced_search_wavelength_criteria.copy()
    if 0 <= idx < len(criteria):
        criteria[idx][field] = value
        app_state.advanced_search_wavelength_criteria = criteria


def _set_criteria(app_state, criteria, criteria_container, ctx):
    """Store criteria in state and re-render the criteria rows."""
    app_state.advanced_search_wavelength_criteria = criteria
    criteria_container.clear()
    with criteria_container:
        for idx, criterion in enumerate(criteria):
            _render_criterion_row(idx, criterion, app_state, criteria_container, ctx)


def _add_wavelength_criterion(app_state, criteria_container, ctx):
    """Append a criterion copying the last one's values, then re-render."""
    criteria = app_state.advanced_search_wavelength_criteria.copy()
    last = criteria[-1] if criteria else {"wavelength": 550, "trans_min": 0, "trans_max": 100}
    criteria.append(dict(last))
    _set_criteria(app_state, criteria, criteria_container, ctx)


def _remove_wavelength_criterion(idx: int, app_state, criteria_container, ctx):
    """Remove a criterion (keeping at least one), then re-render."""
    criteria = app_state.advanced_search_wavelength_criteria.copy()
    if 0 <= idx < len(criteria) and len(criteria) > 1:
        criteria.pop(idx)
        _set_criteria(app_state, criteria, criteria_container, ctx)


def _apply_filter_search(
    df, filter_matrix, app_state, on_change, results_container
):
    """Execute filter search and populate results container."""
    filtered = search_filters(df, filter_matrix, app_state)

    results_container.clear()
    with results_container:
        ui.label(f"{len(filtered)} filters found").classes("font-bold text-sm")
        ui.separator()

        selected_indices = set()

        for row_idx, row in filtered.iterrows():
            hex_c = row["Hex Color"]
            if not is_valid_hex_color(hex_c):
                hex_c = "#888888"
            name = row["Filter Name"]
            number = row["Filter Number"]
            brand = row["Manufacturer"]
            text_color = "#fff" if is_dark_color(hex_c) else "#111"

            # Mutable holder so the toggle closure can reference the detail
            # column that is created *after* the colored row element.
            dc_holder = [None]

            def _make_detail_toggle(dc_h=dc_holder, ri=row_idx, hc=hex_c):
                rendered = [False]
                def _toggle(e):
                    dc = dc_h[0]
                    if dc is None:
                        return
                    dc.set_visibility(e.value)
                    if e.value and not rendered[0]:
                        rendered[0] = True
                        with dc:
                            fig = _sparkline_fig(INTERP_GRID, filter_matrix[ri, :], hc, height=280, width=900)
                            ui.plotly(fig).classes("w-full")

                            def _make_sel(idx=ri):
                                def _t(ev):
                                    if ev.value:
                                        selected_indices.add(idx)
                                    else:
                                        selected_indices.discard(idx)
                                return _t
                            ui.checkbox("Select this filter", on_change=_make_sel(ri)).props("dense")
                return _toggle

            # Colored header row — full-width background matches filter hex color
            with ui.row().classes("w-full items-center gap-2 py-1 px-2").style(
                f"background-color:{hex_c};border-radius:6px"
            ):
                ui.label(f"{number} — {name} — {brand}").classes(
                    "flex-grow text-sm font-semibold"
                ).style(f"color:{text_color}")
                ui.switch("Details", on_change=_make_detail_toggle()).props("dense").style(
                    f"color:{text_color}"
                )

            # Detail container lives outside (below) the colored row
            dc = ui.column().classes("w-full gap-1")
            dc.set_visibility(False)
            dc_holder[0] = dc

        ui.separator()
        with ui.row().classes("gap-2"):
            def _done():
                sel_display = [
                    f"{filtered.loc[i, 'Filter Name']} "
                    f"({filtered.loc[i, 'Filter Number']}, "
                    f"{filtered.loc[i, 'Manufacturer']})"
                    for i in selected_indices if i in filtered.index
                ]
                existing = list(app_state.selected_filters)
                app_state.selected_filters = list(set(existing + sel_display))
                app_state.show_advanced_search = False
                on_change()

            ui.button(UI_BUTTONS["done"], on_click=_done).props("dense")
            ui.button(
                UI_BUTTONS["cancel"],
                on_click=lambda: (setattr(app_state, "show_advanced_search", False), on_change()),
            ).props("dense flat")


# ============================================================================
# ADVANCED REFLECTOR SEARCH
# ============================================================================

def render_advanced_reflector_search(
    df: pd.DataFrame,
    reflector_matrix: np.ndarray,
    app_state,
    on_change: Callable,
) -> None:
    """Render the advanced reflector search panel."""

    with ui.card().classes("w-full"):
        ui.label(UI_SECTIONS["advanced_reflector_search"]).classes("text-lg font-bold")
        ui.label(UI_LABELS["filter_reflectors"]).classes("text-sm text-gray-500")

        orgs = sorted([x for x in df["Organization"].unique() if x])
        packages = sorted([x for x in df["Package Title"].unique() if x])
        targets = sorted([x for x in df["Target Type"].unique() if x])

        sel_orgs = [[]];  sel_pkgs = [[]];  sel_tgts = [[]]

        with ui.row().classes("w-full items-end gap-2"):
            ui.select(
                options=orgs, label="Organization", multiple=True,
                on_change=lambda e: sel_orgs.__setitem__(0, list(e.value) if e.value else []),
            ).props("dense outlined use-chips").classes("flex-grow")
            ui.select(
                options=packages, label="Package/Dataset", multiple=True,
                on_change=lambda e: sel_pkgs.__setitem__(0, list(e.value) if e.value else []),
            ).props("dense outlined use-chips").classes("flex-grow")
            ui.select(
                options=targets, label="Target Type", multiple=True,
                on_change=lambda e: sel_tgts.__setitem__(0, list(e.value) if e.value else []),
            ).props("dense outlined use-chips").classes("flex-grow")

            ui.button(UI_BUTTONS["apply"], on_click=lambda: _apply_reflector_search(
                df, reflector_matrix, app_state, on_change, refl_results,
                sel_orgs[0], sel_pkgs[0], sel_tgts[0],
            )).props("dense")

        refl_results = ui.column().classes("w-full mt-2")

        _apply_reflector_search(df, reflector_matrix, app_state, on_change, refl_results, [], [], [])

        ui.separator()
        ui.button(
            UI_BUTTONS["done"],
            on_click=lambda: (setattr(app_state, "show_reflector_search", False), on_change()),
        ).props("dense")


def _apply_reflector_search(df, reflector_matrix, app_state, on_change, container, orgs, pkgs, tgts):
    filtered = search_reflectors(df, orgs, pkgs, tgts)

    container.clear()
    with container:
        ui.label(f"{len(filtered)} reflectors found").classes("font-bold text-sm")
        ui.separator()

        for row_idx, row in filtered.iterrows():
            sf = row["Source File"]
            is_def = app_state.is_default_reflector(sf)
            name = row["Name"]
            org = row.get("Organization", "") or row.get("Source Folder", "")
            tt = row.get("Target Type", "")

            with ui.row().classes("w-full items-center gap-2 py-1"):
                with ui.column().classes("flex-grow gap-0"):
                    ui.label(name).classes("font-medium text-sm")
                    meta_parts = [x for x in [org, tt] if x]
                    if meta_parts:
                        ui.label(" | ".join(meta_parts)).classes("text-xs text-gray-500")

                # Details toggle — lazy-renders sparkline on first open
                detail_container = ui.column().classes("w-full gap-1")
                detail_container.set_visibility(False)

                def _make_detail_toggle(dc=detail_container, ri=row_idx):
                    rendered = [False]
                    def _toggle(e):
                        dc.set_visibility(e.value)
                        if e.value and not rendered[0]:
                            rendered[0] = True
                            with dc:
                                fig = _sparkline_fig(INTERP_GRID, reflector_matrix[ri, :], "#4CAF50")
                                ui.plotly(fig).classes("w-full").style("height:120px")
                    return _toggle

                ui.switch("Details", on_change=_make_detail_toggle()).props("dense")

                if is_def:
                    def _make_rem(s=sf):
                        def _r():
                            app_state.remove_from_default_reflectors(s)
                            on_change()
                        return _r
                    ui.button("X", on_click=_make_rem()).props("dense outline size=sm color=negative").tooltip(
                        "Remove from defaults"
                    )
                else:
                    def _make_add(s=sf):
                        def _a():
                            app_state.add_to_default_reflectors(s)
                            on_change()
                        return _a
                    ui.button("+", on_click=_make_add()).props("dense outline size=sm").tooltip(
                        "Add to defaults"
                    )


# ============================================================================
# IMPORT DIALOG
# ============================================================================

def render_import_dialog(app_state, on_change: Callable) -> None:
    """Render the data import tabs."""

    with ui.card().classes("w-full"):
        ui.label(UI_SECTIONS["import_data"]).classes("text-lg font-bold")

        with ui.tabs().classes("w-full") as tabs:
            tab_f = ui.tab(IMPORT_TABS["filters"])
            tab_i = ui.tab(IMPORT_TABS["illuminants"])
            tab_q = ui.tab(IMPORT_TABS["camera_qe"])
            tab_film = ui.tab(IMPORT_TABS["film"])
            tab_r = ui.tab(IMPORT_TABS["reflectance_ecosis"])

        with ui.tab_panels(tabs).classes("w-full"):
            with ui.tab_panel(tab_f):
                _import_filter_panel(on_change)
            with ui.tab_panel(tab_i):
                _import_illuminant_panel(on_change)
            with ui.tab_panel(tab_q):
                _import_qe_panel(on_change)
            with ui.tab_panel(tab_film):
                _import_film_panel(on_change)
            with ui.tab_panel(tab_r):
                _import_reflectance_panel(on_change)


# -- Reusable import helpers --

def _file_upload(label: str) -> List:
    """Render a CSV upload widget and return mutable [bytes|None, filename] state."""
    file_state: List = [None, ""]

    async def _on_upload(e: events.UploadEventArguments):
        file_state[0] = await e.file.read()
        file_state[1] = e.file.name

    ui.upload(label=label, on_upload=_on_upload, auto_upload=True).props(
        "accept=.csv dense"
    ).classes("w-full")

    return file_state

def _make_import_button(
    file_state: List,
    import_fn: Callable,
    button_label: str,
    on_change: Callable,
    success_message: str,
    build_args: Callable,
) -> None:
    """Create the import button that reads file_state and calls import_fn."""
    def _import():
        if file_state[0] is None:
            handle_error(UI_LABELS["upload_file_first"], "warning")
            return
        buf = io.BytesIO(file_state[0])
        buf.name = file_state[1]
        args = build_args(buf)
        success, msg = import_fn(*args)
        if success:
            show_success_message(success_message)
            on_change()
        else:
            handle_error(f"Import failed: {msg}")

    ui.button(button_label, on_click=_import).props("dense color=primary")


# -- Individual import panels --

def _import_filter_panel(on_change: Callable) -> None:
    from services.importing import import_filter_from_csv

    file_state = _file_upload(UI_LABELS["upload_csv_wl_trans"])

    manufacturer = ui.input("Manufacturer", value="Custom").props("dense outlined")
    filter_name = ui.input("Filter Name", value="Custom Filter").props("dense outlined")
    filter_number = ui.input("Filter Number", value="001").props("dense outlined")
    hex_color = ui.color_input("Color", value="#808080").props("dense outlined")

    def _build_args(buf):
        meta = {
            "manufacturer": manufacturer.value.strip(),
            "filter_name": filter_name.value.strip(),
            "filter_number": filter_number.value.strip(),
            "hex_color": hex_color.value or "#808080",
        }
        return (buf, meta, True, True)

    _make_import_button(
        file_state, import_filter_from_csv,
        UI_BUTTONS["import_filter"], on_change,
        "Filter imported successfully!", _build_args,
    )


def _import_illuminant_panel(on_change: Callable) -> None:
    from services.importing import import_illuminant_from_csv

    file_state = _file_upload(UI_LABELS["upload_csv_wl_power"])

    name_input = ui.input("Illuminant Name", value="Custom Illuminant").props("dense outlined")

    _make_import_button(
        file_state, import_illuminant_from_csv,
        UI_BUTTONS["import_illuminant"], on_change,
        "Illuminant imported!",
        lambda buf: (buf, name_input.value.strip()),
    )


def _import_film_panel(on_change: Callable) -> None:
    from services.importing import import_film_from_csv

    # Film type options — only types that make sense for import
    film_type_options = {
        'bw': SENSOR_TYPE_LABELS['bw'],
        'color_slide': SENSOR_TYPE_LABELS['color_slide'],
        'color_negative': SENSOR_TYPE_LABELS['color_negative'],
    }

    film_type = ui.select(
        options=film_type_options,
        label="Film Type *",
        value='bw',
    ).props("dense outlined")

    manufacturer = ui.input("Manufacturer *", value="").props("dense outlined")
    name_input = ui.input("Film Name / Model *", value="").props("dense outlined")
    iso_input = ui.input("ISO (optional)", value="").props("dense outlined")
    process_input = ui.input("Process (optional, e.g. E-6, D-76 1:1)", value="").props("dense outlined")
    notes_input = ui.textarea("Notes (optional)", value="").props("dense outlined").classes("w-full")

    # File label updates dynamically based on film type
    file_label_ref = ["Upload CSV (Wavelength, Sensitivity)"]
    file_label_el = ui.label(file_label_ref[0]).classes("text-xs text-gray-500 mt-1")

    def _update_label(e):
        if e.value == 'bw':
            file_label_ref[0] = "Upload CSV (Wavelength, Sensitivity)"
        else:
            file_label_ref[0] = "Upload CSV (R_wl, R_val, G_wl, G_val, B_wl, B_val) — same X-Y pair format as Camera QE"
        file_label_el.set_text(file_label_ref[0])

    film_type.on_value_change(_update_label)

    log_sens_toggle = ui.switch(
        "Data is in Log Sensitivity format",
        value=False,
    ).props("dense")
    ui.label(
        "Enable if Y-axis values are log\u2081\u2080(Sensitivity), typically ranging from \u22121 to 3. "
        "The importer will convert via 10\u02e3 and normalise so the most sensitive channel peaks at 100%."
    ).classes("text-xs text-grey-6")

    file_state = _file_upload("Upload CSV")

    def _import():
        buf = file_state[0]
        if buf is None:
            handle_error("Please upload a CSV file.")
            return
        buf = io.BytesIO(buf)
        buf.name = file_state[1]
        m = manufacturer.value.strip()
        n = name_input.value.strip()
        ft = film_type.value
        if not m:
            handle_error("Manufacturer is required.")
            return
        if not n:
            handle_error("Film name is required.")
            return

        success, msg = import_film_from_csv(
            uploaded_file=buf,
            manufacturer=m,
            name=n,
            film_type=ft,
            iso=iso_input.value.strip(),
            process=process_input.value.strip(),
            notes=notes_input.value.strip(),
            log_sensitivity=log_sens_toggle.value,
        )
        if success:
            show_success_message(msg or "Film profile imported!")
            on_change()
        else:
            handle_error(f"Import failed: {msg}")

    ui.button("Import Film Profile", on_click=_import).props("dense color=primary").classes("mt-2")


def _import_qe_panel(on_change: Callable) -> None:
    from services.importing import import_qe_from_csv

    file_state = _file_upload(UI_LABELS["upload_csv_wl_rgb"])

    brand = ui.input("Brand", value="Custom").props("dense outlined")
    model = ui.input("Model", value="Custom Model").props("dense outlined")

    _make_import_button(
        file_state, import_qe_from_csv,
        UI_BUTTONS["import_camera_qe"], on_change,
        "Camera QE imported!",
        lambda buf: (buf, brand.value.strip(), model.value.strip()),
    )


def _import_reflectance_panel(on_change: Callable) -> None:
    from services.importing import (
        import_reflectance_absorption_from_csv,
        import_ecosis_csv,
        get_ecosis_csv_metadata_columns,
    )

    ui.label(UI_SECTIONS["import_reflectance_absorption"]).classes("font-bold text-sm")

    # -- Mode containers (only one visible at a time) --
    single_section = ui.column().classes("w-full gap-2")
    ecosis_section = ui.column().classes("w-full gap-2")
    ecosis_section.set_visibility(False)

    def _on_mode_change(e):
        is_single = "Single" in e.value
        single_section.set_visibility(is_single)
        ecosis_section.set_visibility(not is_single)

    ui.radio(
        IMPORT_ECOSIS_MODES,
        value=IMPORT_ECOSIS_MODES[0],
        on_change=_on_mode_change,
    ).props("dense inline")

    # -- Single spectrum --
    with single_section:
        file_state = _file_upload(UI_LABELS["upload_csv"])

        sname = ui.input("Spectrum Name", value="Custom Spectrum").props("dense outlined")
        stype = ui.select(IMPORT_DATA_TYPES, label="Data Type", value="Reflectance").props("dense outlined")
        scat = ui.select(IMPORT_CATEGORIES, label="Category", value="Plant").props("dense outlined")

        _make_import_button(
            file_state, import_reflectance_absorption_from_csv,
            UI_BUTTONS["import_single_spectrum"], on_change,
            "Spectrum imported!",
            lambda buf: (buf, {"name": sname.value.strip(), "data_type": stype.value,
                               "category": scat.value, "description": ""}, True, True),
        )

    # -- ECOSIS multi-spectrum --
    with ecosis_section:
        ecosis_state: Dict = {"content": None, "name": "", "tmp_path": None, "meta_cols": []}
        meta_cols_container = ui.column().classes("w-full")
        name_col = [None]
        relevant_cols: List = [[]]

        async def _up_e(e: events.UploadEventArguments):
            ecosis_state["content"] = await e.file.read()
            ecosis_state["name"] = e.file.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(ecosis_state["content"])
                ecosis_state["tmp_path"] = tmp.name
            ecosis_state["meta_cols"] = get_ecosis_csv_metadata_columns(ecosis_state["tmp_path"])
            meta_cols_container.clear()
            with meta_cols_container:
                if ecosis_state["meta_cols"]:
                    ui.label(UI_SECTIONS["column_selection"]).classes("font-bold text-sm")
                    ui.select(
                        options=["None"] + ecosis_state["meta_cols"],
                        label=UI_LABELS["choose_name_column"],
                        value="None",
                        on_change=lambda ev: name_col.__setitem__(0, ev.value if ev.value != "None" else None),
                    ).props("dense outlined").classes("w-full")
                    ui.select(
                        options=ecosis_state["meta_cols"],
                        label=UI_LABELS["relevant_metadata"],
                        multiple=True,
                        on_change=lambda ev: relevant_cols.__setitem__(0, list(ev.value) if ev.value else []),
                    ).props("dense outlined use-chips").classes("w-full")

        ui.upload(label=UI_LABELS["upload_ecosis_csv"], on_upload=_up_e, auto_upload=True).props(
            "accept=.csv dense"
        ).classes("w-full")
        api_url = ui.input(
            UI_LABELS["ecosis_api_url"], placeholder="https://ecosis.org/api/package/..."
        ).props("dense outlined").classes("w-full")

        def _import_ecosis():
            if ecosis_state["tmp_path"] is None:
                handle_error(UI_LABELS["upload_file_first"], "warning")
                return
            csv_fn = os.path.splitext(ecosis_state["name"])[0]
            csv_fn = "".join(c for c in csv_fn if c.isalnum() or c in " _-").strip().replace(" ", "_")
            out_dir = os.path.join(OUTPUT_FOLDERS["ecosis"], csv_fn)
            os.makedirs(out_dir, exist_ok=True)
            try:
                files = import_ecosis_csv(
                    ecosis_state["tmp_path"], out_dir,
                    api_url.value.strip() or None,
                    name_col[0],
                    relevant_cols[0],
                )
                os.unlink(ecosis_state["tmp_path"])
                show_success_message(f"Imported {len(files)} spectra!")
                on_change()
            except Exception as exc:
                handle_error(f"ECOSIS import failed: {exc}")

        ui.button(UI_BUTTONS["import_ecosis_file"], on_click=_import_ecosis).props("dense color=primary")


# ============================================================================
# MY FILTERS MANAGEMENT PANEL
# ============================================================================

def render_my_filters_manager(
    filter_collection,
    app_state,
    on_change: Callable,
) -> None:
    """Render the owned filters management panel.
    
    Shows individual search, manufacturer bulk controls, and owned/total counts.
    """
    with ui.card().classes("w-full"):
        ui.label(UI_SECTIONS["my_filters_manager"]).classes("text-lg font-bold")
        
        owned = app_state.get_my_filters()
        total = len(filter_collection.filters)
        owned_count = len(owned)
        
        # Summary + mfr container ref for refreshing
        summary_ref = [None]
        mfr_results_ref = [None]
        add_selector_ref = [None]
        remove_selector_ref = [None]

        summary_container = ui.row().classes("w-full items-center gap-2 mb-2")
        summary_ref[0] = summary_container
        with summary_container:
            ui.label(f"Total: {owned_count}/{total} filters owned").classes("text-sm text-gray-600")

        def _refresh_all():
            """Refresh summary, manufacturer list, and selectors."""
            current_owned = app_state.get_my_filters()
            # Update summary
            sc = summary_ref[0]
            if sc:
                sc.clear()
                with sc:
                    ui.label(f"Total: {len(current_owned)}/{total} filters owned").classes("text-sm text-gray-600")
            # Update manufacturer rows
            _refresh_manufacturer_list(
                mfr_results_ref[0], filter_collection, app_state, on_change, _refresh_all
            )
            # Update filter selector options
            asr = add_selector_ref[0]
            if asr:
                asr.options = sorted(filter_collection.get_display_names())
                asr.update()
            # Update remove selector options (only owned filters)
            rsr = remove_selector_ref[0]
            if rsr:
                rsr.options = sorted(current_owned)
                rsr.update()
        
        # ---- Add filters ----
        ui.label("Add Filters").classes("text-sm font-semibold mt-1")
        
        def _on_filter_add(e):
            """Add selected filter to owned list."""
            if e.value:
                filter_name = e.value
                # Clear selection first
                e.sender.value = None
                # Add to owned list
                app_state.add_to_my_filters(filter_name)
                # Notify before refresh (to avoid deleted context)
                ui.notify(f"Added: {filter_name}", type="positive")
                # Refresh UI
                _refresh_all()
                on_change()
        
        all_filters = sorted(filter_collection.get_display_names())
        add_selector = ui.select(
            options=all_filters,
            label="Search and select filter to add",
            on_change=_on_filter_add,
        ).props("use-chips dense outlined use-input input-debounce=300 clearable").classes("w-full")
        add_selector_ref[0] = add_selector
        
        ui.separator().classes("my-2")
        
        # ---- Remove filters ----
        ui.label("Remove Filters").classes("text-sm font-semibold")
        
        def _on_filter_remove(e):
            """Remove selected filter from owned list."""
            if e.value:
                filter_name = e.value
                # Clear selection first
                e.sender.value = None
                # Remove from owned list
                app_state.remove_from_my_filters(filter_name)
                # Notify before refresh (to avoid deleted context)
                ui.notify(f"Removed: {filter_name}", type="info")
                # Refresh UI
                _refresh_all()
                on_change()
        
        owned_filters = sorted(owned)
        remove_selector = ui.select(
            options=owned_filters,
            label="Search and select filter to remove",
            on_change=_on_filter_remove,
        ).props("use-chips dense outlined use-input input-debounce=300 clearable").classes("w-full")
        remove_selector_ref[0] = remove_selector
        
        ui.separator().classes("my-2")
        
        # ---- Manufacturer bulk controls ----
        ui.label("Bulk Add/Remove by Manufacturer").classes("text-sm font-semibold")
        
        mfr_results_container = ui.column().classes("w-full gap-0")
        mfr_results_ref[0] = mfr_results_container
        
        _render_manufacturer_list(
            mfr_results_container, filter_collection, app_state, on_change, _refresh_all
        )
        
        ui.separator().classes("my-2")
        ui.button(
            UI_BUTTONS["done"],
            on_click=lambda: (setattr(app_state, "show_my_filters_manager", False), on_change()),
        ).props("dense")


def _render_manufacturer_list(container, filter_collection, app_state, on_change, refresh_fn):
    """Render the full manufacturer list into container."""
    owned = app_state.get_my_filters()
    manufacturers = {}
    for f in filter_collection.filters:
        m = f.manufacturer
        if m not in manufacturers:
            manufacturers[m] = {"total": 0, "owned": 0}
        manufacturers[m]["total"] += 1
        if str(f) in owned:
            manufacturers[m]["owned"] += 1
    
    with container:
        for mfr in sorted(manufacturers.keys()):
            stats = manufacturers[mfr]
            _render_manufacturer_row(
                mfr, stats["owned"], stats["total"],
                filter_collection, app_state, on_change, refresh_fn,
            )


def _refresh_manufacturer_list(container, filter_collection, app_state, on_change, refresh_fn):
    """Re-render manufacturer list after ownership changes."""
    if container is None:
        return
    container.clear()
    _render_manufacturer_list(container, filter_collection, app_state, on_change, refresh_fn)


def _render_manufacturer_row(
    manufacturer: str, owned_count: int, total_count: int,
    filter_collection, app_state, on_change: Callable,
    refresh_fn: Callable,
) -> None:
    """Render a single manufacturer row with counts and bulk buttons."""
    all_owned = owned_count == total_count
    none_owned = owned_count == 0
    
    with ui.row().classes("w-full items-center gap-2 py-1"):
        # Manufacturer name and count
        ui.label(manufacturer).classes("font-medium text-sm").style("min-width:200px")
        
        count_color = "text-green-600" if all_owned else ("text-gray-500" if none_owned else "text-orange-500")
        ui.label(f"{owned_count}/{total_count}").classes(f"text-sm {count_color}").style("min-width:60px")
        
        # Add All button
        def _add_all(m=manufacturer):
            added = app_state.add_manufacturer_to_my_filters(m, filter_collection)
            if added:
                ui.notify(f"Added {added} filters from {m}", type="positive")
                refresh_fn()
                on_change()
        
        ui.button(UI_BUTTONS["add_all"], on_click=_add_all).props(
            "dense outline size=sm" + (" disable" if all_owned else "")
        )
        
        # Remove All button
        def _remove_all(m=manufacturer):
            removed = app_state.remove_manufacturer_from_my_filters(m, filter_collection)
            if removed:
                ui.notify(f"Removed {removed} filters from {m}", type="info")
                refresh_fn()
                on_change()
        
        ui.button(UI_BUTTONS["remove_all"], on_click=_remove_all).props(
            "dense outline size=sm color=negative" + (" disable" if none_owned else "")
        )
