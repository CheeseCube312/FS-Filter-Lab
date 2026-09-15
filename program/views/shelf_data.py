"""Data shelf: all data importers as cards, digitized-CSV fast path first (§3.3).

Cards stage an upload in ``page_state`` and only write to disk when the
Import button dispatches. Success rebuilds the whole page through the
import actions; failure re-renders this shelf so the per-card status line
(from ``AppState.last_import``) shows the message.
"""
import zlib
from pathlib import Path
from typing import Any, Callable, Dict

from nicegui import ui

from models.constants import FILM_TYPES

_DIGITIZED_BRAND = "Digitized"

# Display colors auto-assigned to digitized filters (deterministic per name).
_PALETTE = (
    "#d32f2f", "#1976d2", "#388e3c", "#f57c00",
    "#7b1fa2", "#0097a7", "#5d4037", "#455a64",
)

def render_data_shelf(ctx: Dict[str, Any]) -> None:
    ui.label("Data").classes("text-h6")
    ui.label(
        "Import spectra into the libraries. Files are validated before anything "
        "is written; the source shelves pick up new entries immediately."
    ).classes("text-caption text-grey-7")

    _render_quick_digitized(ctx)

    _staged_import_card(
        ctx, key="filter",
        title="Filter CSV (full metadata)", icon="filter_alt",
        note="2 columns: wavelength (nm), transmission (%).",
        upload_label="Choose filter CSV",
        button_label="Import filter",
        fields=_filter_fields, payload_fn=_filter_payload,
    )
    _staged_import_card(
        ctx, key="spectrum",
        title="Surface spectrum CSV (reflectance/absorption)", icon="texture",
        note="2 columns: wavelength (nm), reflectance or absorption (%).",
        upload_label="Choose surface CSV",
        button_label="Import surface spectrum",
        fields=_spectrum_fields, payload_fn=_spectrum_payload,
    )
    _staged_import_card(
        ctx, key="illuminant",
        title="Illuminant CSV", icon="wb_sunny",
        note="2 columns: wavelength (nm), relative power.",
        upload_label="Choose illuminant CSV",
        button_label="Import illuminant",
        fields=_illuminant_fields, payload_fn=_illuminant_payload,
    )
    _staged_import_card(
        ctx, key="film",
        title="Film profile CSV", icon="movie",
        note=(
            "Colour film: 6 columns (R_wl, R_val, G_wl, G_val, B_wl, B_val). "
            "B&W film: 2 columns (Wavelength, Sensitivity)."
        ),
        upload_label="Choose film CSV",
        button_label="Import film profile",
        fields=_film_fields, payload_fn=_film_payload,
    )
    _staged_import_card(
        ctx, key="qe",
        title="Camera QE CSV", icon="photo_camera",
        note="6 columns: R_wl, R_qe, G_wl, G_qe, B_wl, B_qe.",
        upload_label="Choose camera QE CSV",
        button_label="Import camera QE",
        fields=_qe_fields, payload_fn=_qe_payload,
    )


def _status_line(app_state, card: str) -> None:
    """Outcome of this card's last import (ok/time + message), if any."""
    info = app_state.last_import.get(card)
    if not info:
        return
    ok = info.get("ok") == "1"
    color = "text-positive" if ok else "text-negative"
    mark = "✓" if ok else "✗"
    with ui.row().classes("w-full items-center no-wrap q-gutter-x-sm"):
        ui.label(f"{mark} {info.get('when', '')}").classes(f"text-caption {color}")
        ui.label(str(info.get("message", ""))).classes(
            f"text-caption {color} col ellipsis"
        )


def _render_quick_digitized(ctx: Dict[str, Any]) -> None:
    app_state = ctx["app_state"]
    with ui.card().classes("w-full q-pa-sm"):
        ui.label("Quick digitized filter").classes("text-subtitle1")
        ui.label(
            "Fastest path: upload a 2-column CSV (wavelength nm, transmission %) "
            "and it is imported immediately — manufacturer \"Digitized\", name from "
            "the filename, auto color. Find it via the Filters shelf search."
        ).classes("text-caption text-grey-7")
        _status_line(app_state, "filter")
        ui.upload(
            auto_upload=True,
            on_upload=lambda e: _import_digitized(ctx, e),
        ).classes("w-full max-w-xs").props("accept=.csv label='Choose digitized CSV'")


def _import_digitized(ctx: Dict[str, Any], event) -> None:
    stem = Path(event.name).stem
    ctx["dispatch"]("import_filter_csv", {
        "content": event.content.read(),
        "filename": event.name,
        "manufacturer": _DIGITIZED_BRAND,
        "filter_name": _title_from_stem(stem),
        "filter_number": "-",
        "hex_color": _palette_for(stem),
        "extrap_lower": False,
        "extrap_upper": False,
    })


def _title_from_stem(stem: str) -> str:
    """Filename stem -> human filter name: separators become single spaces."""
    name = " ".join(
        part for part in stem.replace("-", " ").replace("_", " ").split() if part
    )
    return name or "Digitized filter"


def _palette_for(stem: str) -> str:
    """Deterministic palette color so same-named files keep the same color."""
    return _PALETTE[zlib.crc32(stem.encode("utf-8")) % len(_PALETTE)]


def _staged_import_card(
    ctx: Dict[str, Any], *, key: str, title: str, icon: str, note: str,
    upload_label: str, button_label: str,
    fields: Callable[[Dict[str, Any]], None],
    payload_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> None:
    """Expansion card: fields + staged upload + Import button (§3.3).

    Nothing reaches disk until Import dispatches; the action marks this
    shelf dirty, so the staged section re-renders afterwards (the slot
    is cleared on success and kept on failure for a metadata fix + retry).
    """
    page_state = ctx["page_state"]
    with ui.expansion(title, icon=icon).classes("w-full"):
        ui.label(note).classes("text-caption text-grey-7")
        _status_line(ctx["app_state"], key)
        values: Dict[str, Any] = {}
        fields(values)

        @ui.refreshable
        def staged() -> None:
            name = page_state.get(f"{key}_name")
            if not name:
                ui.label("No file staged.").classes("text-caption text-grey-5")
                return
            with ui.row().classes("w-full items-center no-wrap q-gutter-x-sm"):
                ui.label(f"{name} staged.").classes("text-body2 col ellipsis")
                ui.button(
                    button_label, icon="download_done",
                    on_click=lambda: _dispatch_staged(ctx, key, payload_fn(values)),
                ).props("unelevated dense color=positive")

        staged()

        def _handle_upload(event) -> None:
            page_state[f"{key}_content"] = event.content.read()
            page_state[f"{key}_name"] = event.name
            staged.refresh()

        ui.upload(auto_upload=True, on_upload=_handle_upload) \
            .classes("w-full max-w-xs").props(f"accept=.csv label='{upload_label}'")


def _dispatch_staged(ctx: Dict[str, Any], key: str, payload: Dict[str, Any]) -> None:
    """Dispatch the staged file; keep it staged when the import fails."""
    page_state = ctx["page_state"]
    payload["content"] = page_state.get(f"{key}_content", b"") or b""
    payload["filename"] = page_state.get(f"{key}_name", "")
    ctx["dispatch"](f"import_{key}_csv", payload)
    if ctx["app_state"].last_import.get(key, {}).get("ok") == "1":
        page_state.pop(f"{key}_content", None)
        page_state.pop(f"{key}_name", None)


def _filter_fields(values: Dict[str, Any]) -> None:
    values["manufacturer"] = ui.input("Manufacturer").classes("w-full").props("outlined dense")
    values["filter_name"] = ui.input("Filter name").classes("w-full").props("outlined dense")
    values["filter_number"] = ui.input("Filter number").classes("w-full").props("outlined dense")
    values["hex_color"] = ui.color_input(value="#1976d2", label="Display color").classes("w-full")
    values["extrap_lower"] = ui.switch("Extrapolate to 300 nm", value=False)
    values["extrap_upper"] = ui.switch("Extrapolate to 1100 nm", value=False)


def _filter_payload(values: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "manufacturer": values["manufacturer"].value,
        "filter_name": values["filter_name"].value,
        "filter_number": values["filter_number"].value,
        "hex_color": values["hex_color"].value,
        "extrap_lower": bool(values["extrap_lower"].value),
        "extrap_upper": bool(values["extrap_upper"].value),
    }


def _spectrum_fields(values: Dict[str, Any]) -> None:
    values["name"] = ui.input("Surface name").classes("w-full").props("outlined dense")
    values["description"] = ui.input("Description (optional)").classes("w-full").props("outlined dense")
    values["category"] = ui.select(
        ["plant", "other"], value="other", label="Category",
    ).classes("w-full").props("outlined dense")
    values["data_type"] = ui.select(
        ["Reflectance", "Absorption"], value="Reflectance", label="Data type",
    ).classes("w-full").props("outlined dense")
    values["extrap_lower"] = ui.switch("Extrapolate at range start", value=False)
    values["extrap_upper"] = ui.switch("Extrapolate at range end", value=False)


def _spectrum_payload(values: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": values["name"].value,
        "description": values["description"].value,
        "category": values["category"].value,
        "data_type": values["data_type"].value,
        "extrap_lower": bool(values["extrap_lower"].value),
        "extrap_upper": bool(values["extrap_upper"].value),
    }


def _illuminant_fields(values: Dict[str, Any]) -> None:
    values["description"] = ui.input("Description").classes("w-full").props("outlined dense")


def _illuminant_payload(values: Dict[str, Any]) -> Dict[str, Any]:
    return {"description": values["description"].value}


def _film_fields(values: Dict[str, Any]) -> None:
    values["manufacturer"] = ui.input("Manufacturer").classes("w-full").props("outlined dense")
    values["name"] = ui.input("Film name").classes("w-full").props("outlined dense")
    values["film_type"] = ui.select(
        list(FILM_TYPES), value="color_negative", label="Film type",
    ).classes("w-full").props("outlined dense")
    values["iso"] = ui.input("ISO (optional)").classes("w-full").props("outlined dense")
    values["process"] = ui.input("Process (optional)").classes("w-full").props("outlined dense")
    values["notes"] = ui.input("Notes (optional)").classes("w-full").props("outlined dense")
    values["log_sensitivity"] = ui.switch("Values are log10 sensitivity", value=False)


def _film_payload(values: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "manufacturer": values["manufacturer"].value,
        "name": values["name"].value,
        "film_type": values["film_type"].value,
        "iso": values["iso"].value,
        "process": values["process"].value,
        "notes": values["notes"].value,
        "log_sensitivity": bool(values["log_sensitivity"].value),
    }


def _qe_fields(values: Dict[str, Any]) -> None:
    values["brand"] = ui.input("Camera brand").classes("w-full").props("outlined dense")
    values["model"] = ui.input("Camera model").classes("w-full").props("outlined dense")


def _qe_payload(values: Dict[str, Any]) -> Dict[str, Any]:
    return {"brand": values["brand"].value, "model": values["model"].value}
