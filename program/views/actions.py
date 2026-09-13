"""
Action dispatcher for FS Filter Lab (view-layer orchestration).

Every UI control dispatches a named action through ``handle_app_action``,
which mutates AppState / calls services and returns the set of **dirty panel
keys** the shell should refresh (UX_SPEC.md §5). No panel rebuilds another
directly; the key ``"*"`` is reserved for full rebuilds (Reset, and
data-invalidating imports in later slices).

This module contains no NiceGUI layout code. The one sanctioned UI call is
``ui.notify`` for user feedback.
"""
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Set

import numpy as np
from nicegui import ui

from models.constants import INTERP_GRID, OUTPUT_FOLDERS
from models.core import AppData
from services.app_operations import generate_application_report
from services.data import load_reflector_collection
from services.importing import import_ecosis_csv
from services.state_manager import AppState

logger = logging.getLogger(__name__)

# Panels that derive from the optical triple (stack x sensor/film x illuminant).
# `setup` is included because selection UI must re-render to show the new value.
_OPTICAL = {"setup", "transmission", "sensor_response", "surfaces", "film_match"}

# Panels that depend on the filter stack (transmission chains).
_STACK = {"setup", "transmission", "surfaces", "film_match"}


def handle_app_action(
    action: str, payload: Optional[Dict[str, Any]], ctx: Dict[str, Any]
) -> Set[str]:
    """Dispatch *action* and return the set of dirty panel keys."""
    app_state: AppState = ctx["app_state"]
    data: AppData = ctx["data"]
    payload = payload or {}
    handler = _HANDLERS.get(action)
    if handler is None:
        logger.warning("Unknown action dispatched: %s", action)
        ui.notify(f"Unknown action: {action}", type="warning")
        return set()
    try:
        return handler(app_state, data, payload)
    except Exception:
        logger.exception("Action %s failed", action)
        ui.notify(f"{action} failed — see terminal log", type="negative")
        return set()


# ---------------------------------------------------------------------------
# Setup rail selections
# ---------------------------------------------------------------------------

def _select_sensor(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    app_state.select_sensor(payload.get("name"), data.sensor_profiles)
    return _OPTICAL


def _select_film(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    """Film select drives the same receiver slot (active_profile)."""
    return _select_sensor(app_state, data, payload)


def _select_illuminant(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    app_state.select_illuminant(payload.get("name"), data.illuminants)
    return _OPTICAL


# ---------------------------------------------------------------------------
# Stack editing
# ---------------------------------------------------------------------------

def _add_filter(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    name = payload.get("name")
    if name and name not in app_state.selected_filters:
        app_state.selected_filters.append(name)
    return {"setup", "transmission", "surfaces", "film_match"}


def _remove_filter(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    name = payload.get("name")
    if name in app_state.selected_filters:
        app_state.selected_filters.remove(name)
        app_state.filter_multipliers.pop(name, None)
    return {"setup", "transmission", "surfaces", "film_match"}


def _set_multiplier(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    name = payload.get("name")
    if name not in app_state.selected_filters:
        return set()
    value = int(payload.get("value", 1))
    if value <= 1:
        app_state.filter_multipliers.pop(name, None)
    else:
        app_state.filter_multipliers[name] = value
    return {"setup", "transmission", "surfaces", "film_match"}


def _clear_stack(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    app_state.selected_filters.clear()
    app_state.filter_multipliers.clear()
    return {"setup", "transmission", "surfaces", "film_match"}

# ---------------------------------------------------------------------------
# My Filters (owned) management
# ---------------------------------------------------------------------------

def _toggle_my_filter(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    name = payload.get("name")
    if not name:
        return set()
    if app_state.is_my_filter(name):
        app_state.remove_from_my_filters(name)
    else:
        app_state.add_to_my_filters(name)
    return {"shelf_filters", "setup"}


def _add_my_filter_manufacturer(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    added = app_state.add_manufacturer_to_my_filters(
        payload.get("manufacturer", ""), data.filter_collection
    )
    if added:
        ui.notify(f"Added {added} filters from {payload.get('manufacturer')}", type="positive")
    return {"shelf_filters", "setup"}


def _remove_my_filter_manufacturer(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    removed = app_state.remove_manufacturer_from_my_filters(
        payload.get("manufacturer", ""), data.filter_collection
    )
    if removed:
        ui.notify(f"Removed {removed} filters from {payload.get('manufacturer')}", type="info")
    return {"shelf_filters", "setup"}


def _clear_my_filters(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    app_state.clear_all_my_filters()
    return {"shelf_filters", "setup"}


def _set_owned_only(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    app_state.my_filters_only = bool(payload.get("value", False))
    return {"shelf_filters", "setup"}


# ---------------------------------------------------------------------------
# White balance / display-level optical toggles
# ---------------------------------------------------------------------------

def _set_wb_enabled(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    app_state.apply_white_balance = bool(payload.get("value", False))
    return {"sensor_response", "surfaces", "film_match"}


# ---------------------------------------------------------------------------
# Header toolbar
# ---------------------------------------------------------------------------

def _run_report(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    if not app_state.selected_filters:
        ui.notify("Add at least one filter before exporting a report", type="warning")
        return set()
    result = generate_application_report(
        app_state, data.filter_collection, app_state.selected_camera
    )
    if result:
        path = result.get("path") or result.get("filepath") or "output"
        ui.notify(f"Report saved: {path}", type="positive")
    else:
        ui.notify("Report generation failed", type="negative")
    return set()


def _reset_all(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    """Restore defaults: empty stack, standard WB, mixer off."""
    app_state.selected_filters.clear()
    app_state.filter_multipliers.clear()
    app_state.reset_white_balance()
    mixer = app_state.channel_mixer
    mixer.enabled = False
    app_state.channel_mixer = mixer
    ui.notify("Reset to defaults", type="info")
    return {"*"}


# ---------------------------------------------------------------------------
# Surfaces instrument (UX_SPEC.md §7.1)
# ---------------------------------------------------------------------------

def _add_surface(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    src = payload.get("source_file")
    if src and src not in app_state.selected_surfaces:
        app_state.selected_surfaces.append(src)
    return {"surfaces"}


def _remove_surface(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    src = payload.get("source_file")
    if src in app_state.selected_surfaces:
        app_state.selected_surfaces.remove(src)
    if app_state.wb_reference_surface == src:
        app_state.reset_white_balance()
        return {"surfaces", "sensor_response"}
    return {"surfaces"}


def _wb_from_surface(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    src = payload.get("source_file")
    spectrum = next(
        (r for r in data.reflector_collection.reflectors
         if r.metadata.get("source_file", "") == src),
        None,
    )
    if spectrum is None:
        ui.notify("Surface not found", type="warning")
        return set()
    transmission = (
        app_state.combined_transmission
        if app_state.combined_transmission is not None
        else np.ones_like(INTERP_GRID, dtype=float)
    )
    app_state.set_white_balance_from_surface(
        reflector=spectrum.values, transmission=transmission, source_file=src
    )
    return {"surfaces", "sensor_response"}


def _reset_surface_wb(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    app_state.reset_white_balance()
    return {"surfaces", "sensor_response"}


# ---------------------------------------------------------------------------
# Data imports (ECOSIS commit path; other importers land with the Data shelf)
# ---------------------------------------------------------------------------

def _import_ecosis(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    content: bytes = payload.get("content") or b""
    filename = payload.get("filename") or "ecosis.csv"
    if not content:
        ui.notify("No ECOSIS file loaded", type="warning")
        return set()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".csv", prefix="ecosis_", delete=False
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        created = import_ecosis_csv(tmp_path, OUTPUT_FOLDERS["ecosis"])
        ui.notify(f"Imported {len(created)} ECOSIS spectra", type="positive")
        reloaded = load_reflector_collection()
        if reloaded is not None:
            data.reflector_collection = reloaded
            app_state.seed_isdefault_reflectors(reloaded)
        return {"shelf_reflectors"}
    except ValueError as exc:
        ui.notify(f"ECOSIS import failed: {exc}", type="negative")
        return set()
    except Exception:
        logger.exception("ECOSIS import failed unexpectedly")
        ui.notify("ECOSIS import failed — see terminal log", type="negative")
        return set()
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


_HANDLERS = {
    "select_sensor": _select_sensor,
    "select_film": _select_film,
    "select_illuminant": _select_illuminant,
    "add_filter": _add_filter,
    "remove_filter": _remove_filter,
    "set_multiplier": _set_multiplier,
    "clear_stack": _clear_stack,
    "toggle_my_filter": _toggle_my_filter,
    "add_my_filter_manufacturer": _add_my_filter_manufacturer,
    "remove_my_filter_manufacturer": _remove_my_filter_manufacturer,
    "clear_my_filters": _clear_my_filters,
    "set_owned_only": _set_owned_only,
    "set_wb_enabled": _set_wb_enabled,
    "add_surface": _add_surface,
    "remove_surface": _remove_surface,
    "wb_from_surface": _wb_from_surface,
    "reset_surface_wb": _reset_surface_wb,
    "import_ecosis": _import_ecosis,
    "run_report": _run_report,
    "reset_all": _reset_all,
}

