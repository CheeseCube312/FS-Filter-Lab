"""
Action dispatcher for FS Filter Lab (view-layer orchestration).

Every UI control dispatches a named action through ``handle_app_action``,
which mutates AppState / calls services and returns the set of **dirty panel
keys** the shell should refresh (UX_SPEC.md §5). No panel rebuilds another
directly; the key ``"*"`` is reserved for full rebuilds (Reset, and
data-invalidating imports — the Data shelf importers).

This module contains no NiceGUI layout code. The one sanctioned UI call is
``ui.notify`` for user feedback.
"""
import io
import logging
import time
from typing import Any, Callable, Dict, Optional, Set

import numpy as np
from nicegui import ui

from models.constants import FILM_TYPES
from models.core import AppData, ChannelMixerSettings
from services.visualization import generate_application_report
from services.data import (
    load_filter_collection,
    load_illuminant_collection,
    load_reflector_collection,
    load_sensor_profiles,
)
from services.importing import (
    import_filter_from_csv,
    import_film_from_csv,
    import_illuminant_from_csv,
    import_qe_from_csv,
    import_reflectance_absorption_from_csv,
)
from services.calculations import effective_transmission
from services.state_manager import AppState

logger = logging.getLogger(__name__)

# Panels that derive from the optical triple (stack x sensor/film x illuminant).
# `setup` re-renders to show the new selection; `shelf_reflectors` previews run
# the §7.1 color chain (transmission x QE x illuminant x WB gains), so they
# react to the same inputs.
_OPTICAL = {
    "setup", "transmission", "sensor_response", "surfaces", "film_match",
    "shelf_reflectors",
}

# Panels that depend on the filter stack (transmission chains): _OPTICAL plus
# the Filters shelf, whose rows mirror in-stack state.
_STACK = _OPTICAL | {"shelf_filters"}


def _effective_transmission(app_state: AppState, data: AppData) -> np.ndarray:
    """Current stack transmission on the interpolation grid (ones if empty)."""
    return effective_transmission(
        app_state.selected_filters, app_state.filter_multipliers,
        data.filter_collection,
    )


def _refresh_white_balance(app_state: AppState, data: AppData) -> None:
    """Recompute stored WB gains from the current optical inputs.

    Single write point for ``white_balance_gains`` outside the surface
    actions: every handler that changes stack / sensor / illuminant calls
    this, so presenters can read the gains without mutating state at render
    time.  The no-QE / no-illuminant guard runs FIRST — handlers must stay
    safe to invoke before data is loaded.
    """
    if not app_state.current_qe or app_state.illuminant is None:
        return
    app_state.recompute_white_balance(
        _effective_transmission(app_state, data), data.reflector_collection
    )


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
    _refresh_white_balance(app_state, data)
    return _OPTICAL


def _select_film(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    """Film select drives the same receiver slot (active_profile)."""
    return _select_sensor(app_state, data, payload)


def _select_illuminant(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    app_state.select_illuminant(payload.get("name"), data.illuminants)
    _refresh_white_balance(app_state, data)
    return _OPTICAL


# ---------------------------------------------------------------------------
# Stack editing
# ---------------------------------------------------------------------------

def _add_filter(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    name = payload.get("name")
    if name and name not in app_state.selected_filters:
        app_state.selected_filters.append(name)
    _refresh_white_balance(app_state, data)
    return _STACK


def _remove_filter(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    name = payload.get("name")
    if name in app_state.selected_filters:
        app_state.selected_filters.remove(name)
        app_state.filter_multipliers.pop(name, None)
    _refresh_white_balance(app_state, data)
    return _STACK


def _set_multiplier(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    name = payload.get("name")
    if name not in app_state.selected_filters:
        return set()
    value = int(payload.get("value", 1))
    if value <= 1:
        app_state.filter_multipliers.pop(name, None)
    else:
        app_state.filter_multipliers[name] = value
    _refresh_white_balance(app_state, data)
    return _STACK


def _clear_stack(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    app_state.selected_filters.clear()
    app_state.filter_multipliers.clear()
    _refresh_white_balance(app_state, data)
    return _STACK

# ---------------------------------------------------------------------------
# My Filters (owned) management
#
# Owned-filter state lives only in the Filters shelf; the setup rail has no
# owned-dependent UI, so these actions never dirty `setup`.
# ---------------------------------------------------------------------------

def _toggle_my_filter(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    name = payload.get("name")
    if not name:
        return set()
    if app_state.is_my_filter(name):
        app_state.remove_from_my_filters(name)
    else:
        app_state.add_to_my_filters(name)
    return {"shelf_filters"}


def _add_my_filter_manufacturer(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    added = app_state.add_manufacturer_to_my_filters(
        payload.get("manufacturer", ""), data.filter_collection
    )
    if added:
        ui.notify(f"Added {added} filters from {payload.get('manufacturer')}", type="positive")
    return {"shelf_filters"}


def _remove_my_filter_manufacturer(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    removed = app_state.remove_manufacturer_from_my_filters(
        payload.get("manufacturer", ""), data.filter_collection
    )
    if removed:
        ui.notify(f"Removed {removed} filters from {payload.get('manufacturer')}", type="info")
    return {"shelf_filters"}


def _clear_my_filters(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    app_state.clear_all_my_filters()
    return {"shelf_filters"}


def _set_owned_only(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    app_state.my_filters_only = bool(payload.get("value", False))
    return {"shelf_filters"}


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
        path = result.get("path") or result.get("name") or "output"
        ui.notify(f"Report saved: {path}", type="positive")
    else:
        ui.notify("Report generation failed", type="negative")
    return set()


def _reset_all(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    """Reset the analysis chain to defaults (UX_SPEC.md §5 ``reset_all``).

    Scope: filter stack, white balance (gains + surface reference), channel
    mixer (enabled state and gains), and the film-emulation target. Kept:
    loaded data, My Filters, the surfaces selection and all display toggles —
    Reset means "start a new analysis", not "wipe the app".
    """
    app_state.selected_filters = []
    app_state.filter_multipliers = {}
    app_state.reset_white_balance()
    app_state.channel_mixer = ChannelMixerSettings()
    app_state.film_emulation_profile = None
    app_state.film_emulation_profile_name = None
    app_state.film_emulation_wb = False
    app_state.film_emulation_filters = []
    ui.notify("Reset to defaults", type="info")
    return {"*"}


# ---------------------------------------------------------------------------
# Advanced filter search (§3.3 Filters shelf)
#
# Field edits are staged in AppState and only reach the results list when
# the user presses Apply (the shelf refreshes itself), so they return an
# empty dirty set; structural changes (add/remove criterion) dirty
# `shelf_filters` so the row controls re-render immediately.
# ---------------------------------------------------------------------------

def _set_advanced_manufacturers(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    app_state.advanced_search_manufacturers = [str(v) for v in (payload.get("value") or [])]
    return set()  # Staged: results update on Apply


def _set_advanced_sort(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    app_state.advanced_search_sort = str(payload.get("value") or "Filter Number")
    return set()  # Staged: results update on Apply


def _add_advanced_criterion(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    app_state.advanced_search_wavelength_criteria.append(
        {"wavelength": 550, "trans_min": 0, "trans_max": 100}
    )
    return {"shelf_filters"}


def _remove_advanced_criterion(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    criteria = app_state.advanced_search_wavelength_criteria
    index = int(payload.get("index", -1))
    if 0 <= index < len(criteria):
        criteria.pop(index)
    return {"shelf_filters"}


_CRITERION_FIELDS = ("wavelength", "trans_min", "trans_max")


def _update_advanced_criterion(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    field = str(payload.get("field") or "")
    index = int(payload.get("index", -1))
    criteria = app_state.advanced_search_wavelength_criteria
    if field not in _CRITERION_FIELDS or not 0 <= index < len(criteria):
        return set()
    default = {"wavelength": 550, "trans_min": 0, "trans_max": 100}[field]
    criteria[index][field] = int(payload.get("value") or default)
    return set()  # Staged: results update on Apply


# ---------------------------------------------------------------------------
# Surfaces instrument (UX_SPEC.md §7.1)
# ---------------------------------------------------------------------------

def _add_surface(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    src = payload.get("source_file")
    if src and src not in app_state.selected_surfaces:
        app_state.selected_surfaces.append(src)
    # The reflectors shelf marks rows already on the surfaces grid.
    return {"surfaces", "shelf_reflectors"}


def _remove_surface(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    src = payload.get("source_file")
    if src in app_state.selected_surfaces:
        app_state.selected_surfaces.remove(src)
    if app_state.wb_reference_surface == src:
        app_state.reset_white_balance()
        return {"surfaces", "sensor_response", "shelf_reflectors"}
    return {"surfaces", "shelf_reflectors"}


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
    app_state.set_white_balance_from_surface(
        reflector=spectrum.values,
        # Always recompute from the live stack (presenters do the same).
        transmission=_effective_transmission(app_state, data),
        source_file=src,
    )
    # The shelf flags the WB-reference row, the sensor chart replots with the
    # new gains.
    return {"surfaces", "sensor_response", "shelf_reflectors"}


def _reset_surface_wb(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    app_state.reset_white_balance()
    return {"surfaces", "sensor_response", "shelf_reflectors"}


# ---------------------------------------------------------------------------
# Film Match instrument (UX_SPEC.md §7.2)
# ---------------------------------------------------------------------------

# The film target renders as dashed overlay curves inside the sensor chart, so
# every film-target change must dirty `sensor_response` too. Channel mixer
# gains also feed the §7.1 color chain (surfaces chips, shelf previews).
_FILM_MATCH = {"film_match", "sensor_response"}
_MIXER = {"film_match", "sensor_response", "surfaces", "shelf_reflectors"}


def _select_film_target(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    app_state.select_film_emulation_target(payload.get("name"), data.sensor_profiles)
    return _FILM_MATCH


def _set_film_wb(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    app_state.film_emulation_wb = bool(payload.get("value", False))
    return _FILM_MATCH


def _set_film_target_filters(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    app_state.film_emulation_filters = [str(v) for v in (payload.get("names") or [])]
    return _FILM_MATCH


def _set_mixer_enabled(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    mixer = app_state.channel_mixer
    mixer.enabled = bool(payload.get("value", False))
    app_state.channel_mixer = mixer
    return _MIXER


def _set_mixer_gain(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    field = payload.get("field")
    if field not in {
        "red_r", "red_g", "red_b", "green_r", "green_g", "green_b",
        "blue_r", "blue_g", "blue_b",
    }:
        return set()
    mixer = app_state.channel_mixer
    setattr(mixer, field, float(payload.get("value") or 0.0))
    app_state.channel_mixer = mixer
    return _MIXER


def _reset_mixer(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    app_state.channel_mixer = ChannelMixerSettings()
    return _MIXER


# ---------------------------------------------------------------------------
# Data imports (UX_SPEC.md §3.3 Data shelf: all importers dispatch from here)
# ---------------------------------------------------------------------------

# Imports write new files into the cached data folders and add library
# entries: rebuild everything (the sanctioned data-invalidating "*" case, §5).
_IMPORT_OK = {"*"}
# Rejected/failed import: only the Data shelf re-renders, to show the status.
_IMPORT_FAILED = {"shelf_data"}


def _record_import(app_state: AppState, card: str, ok: bool, message: str) -> None:
    """Record an import outcome for the Data shelf status lines (§3.3)."""
    app_state.last_import = {
        **app_state.last_import,
        card: {
            "ok": "1" if ok else "0",
            "message": message,
            "when": time.strftime("%H:%M:%S"),
        },
    }


def _reload_reflectors(state: AppState, data: AppData) -> None:
    """Reload the reflector library from disk after a successful import."""
    reloaded = load_reflector_collection()
    if reloaded is not None:
        data.reflector_collection = reloaded
        state.seed_isdefault_reflectors(reloaded)


def _run_csv_import(
    app_state: AppState,
    data: AppData,
    card: str,
    content: bytes,
    run_import: Callable[[io.BytesIO], Any],
    reload_data: Optional[Callable[[AppState, AppData], None]] = None,
) -> Set[str]:
    """Run the importer over *content*, reload the library on success.

    CSV importers take a file-like object and return (ok, message). The ok
    path reloads library data via *reload_data* and returns the full-rebuild
    dirty set; any failure records the message for the shelf status line.
    """
    if not content:
        _record_import(app_state, card, False, "No CSV file staged.")
        ui.notify("No CSV file staged", type="warning")
        return _IMPORT_FAILED
    try:
        ok, message = run_import(io.BytesIO(content))
    except Exception:
        logger.exception("%s import failed unexpectedly", card)
        _record_import(app_state, card, False, "Import failed — see terminal log")
        ui.notify(f"{card} import failed — see terminal log", type="negative")
        return _IMPORT_FAILED
    if ok:
        if reload_data is not None:
            reload_data(app_state, data)
        _record_import(app_state, card, True, message)
        ui.notify(f"{card} import succeeded", type="positive")
        return _IMPORT_OK
    _record_import(app_state, card, False, message)
    ui.notify(f"{card} import failed: {message}", type="negative")
    return _IMPORT_FAILED

def _import_filter_csv(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    meta = {
        "manufacturer": str(payload.get("manufacturer") or "").strip(),
        "filter_name": str(payload.get("filter_name") or "").strip(),
        "filter_number": str(payload.get("filter_number") or "").strip(),
        "hex_color": str(payload.get("hex_color") or "").strip(),
    }
    missing = [key for key, value in meta.items() if not value]
    if missing:
        message = f"Missing required metadata: {', '.join(missing)}"
        _record_import(app_state, "filter", False, message)
        ui.notify(message, type="warning")
        return _IMPORT_FAILED
    return _run_csv_import(
        app_state, data, "filter", payload.get("content") or b"",
        lambda fh: import_filter_from_csv(
            fh, meta,
            bool(payload.get("extrap_lower")), bool(payload.get("extrap_upper")),
        ),
        lambda _state, d: setattr(d, "filter_collection", load_filter_collection()),
    )


def _import_illuminant_csv(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    description = str(payload.get("description") or "").strip()
    if not description:
        _record_import(app_state, "illuminant", False, "Missing required metadata: description")
        ui.notify("Illuminant description is required", type="warning")
        return _IMPORT_FAILED

    def reload(_state: AppState, d: AppData) -> None:
        illuminants, metadata = load_illuminant_collection()
        d.illuminants = illuminants
        d.illuminant_metadata = metadata

    return _run_csv_import(
        app_state, data, "illuminant", payload.get("content") or b"",
        lambda fh: import_illuminant_from_csv(fh, description),
        reload,
    )


def _import_qe_csv(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    brand = str(payload.get("brand") or "").strip()
    model = str(payload.get("model") or "").strip()
    missing = [key for key, value in (("brand", brand), ("model", model)) if not value]
    if missing:
        message = f"Missing required metadata: {', '.join(missing)}"
        _record_import(app_state, "qe", False, message)
        ui.notify(message, type="warning")
        return _IMPORT_FAILED

    def reload(_state: AppState, d: AppData) -> None:
        d.sensor_profiles, d.default_key = load_sensor_profiles()

    return _run_csv_import(
        app_state, data, "qe", payload.get("content") or b"",
        lambda fh: import_qe_from_csv(fh, brand, model),
        reload,
    )


def _import_film_csv(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    manufacturer = str(payload.get("manufacturer") or "").strip()
    name = str(payload.get("name") or "").strip()
    film_type = str(payload.get("film_type") or "").strip()
    missing = [
        key for key, value in
        (("manufacturer", manufacturer), ("name", name), ("film_type", film_type))
        if not value
    ]
    if missing:
        message = f"Missing required metadata: {', '.join(missing)}"
        _record_import(app_state, "film", False, message)
        ui.notify(message, type="warning")
        return _IMPORT_FAILED
    if film_type not in set(FILM_TYPES):
        message = f"Invalid film type '{film_type}' (use color_negative, color_slide, or bw)"
        _record_import(app_state, "film", False, message)
        ui.notify(message, type="warning")
        return _IMPORT_FAILED

    def reload(_state: AppState, d: AppData) -> None:
        d.sensor_profiles, d.default_key = load_sensor_profiles()

    return _run_csv_import(
        app_state, data, "film", payload.get("content") or b"",
        lambda fh: import_film_from_csv(
            fh, manufacturer, name, film_type,
            iso=str(payload.get("iso") or ""),
            process=str(payload.get("process") or ""),
            notes=str(payload.get("notes") or ""),
            log_sensitivity=bool(payload.get("log_sensitivity", False)),
        ),
        reload,
    )


def _import_spectrum_csv(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    name = str(payload.get("name") or "").strip()
    if not name:
        _record_import(app_state, "spectrum", False, "Missing required metadata: name")
        ui.notify("Surface name is required", type="warning")
        return _IMPORT_FAILED
    meta = {
        "name": name,
        "description": str(payload.get("description") or "").strip(),
        "category": str(payload.get("category") or "other"),
        "data_type": str(payload.get("data_type") or "Reflectance"),
    }
    extrap_lower = bool(payload.get("extrap_lower"))
    extrap_upper = bool(payload.get("extrap_upper"))

    return _run_csv_import(
        app_state, data, "spectrum", payload.get("content") or b"",
        lambda fh: import_reflectance_absorption_from_csv(
            fh, meta, extrap_lower, extrap_upper
        ),
        _reload_reflectors,
    )


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
    "set_advanced_manufacturers": _set_advanced_manufacturers,
    "set_advanced_sort": _set_advanced_sort,
    "add_advanced_criterion": _add_advanced_criterion,
    "remove_advanced_criterion": _remove_advanced_criterion,
    "update_advanced_criterion": _update_advanced_criterion,
    "set_wb_enabled": _set_wb_enabled,
    "add_surface": _add_surface,
    "remove_surface": _remove_surface,
    "wb_from_surface": _wb_from_surface,
    "reset_surface_wb": _reset_surface_wb,
    "import_filter_csv": _import_filter_csv,
    "import_illuminant_csv": _import_illuminant_csv,
    "import_qe_csv": _import_qe_csv,
    "import_film_csv": _import_film_csv,
    "import_spectrum_csv": _import_spectrum_csv,
    "select_film_target": _select_film_target,
    "set_film_wb": _set_film_wb,
    "set_film_target_filters": _set_film_target_filters,
    "set_mixer_enabled": _set_mixer_enabled,
    "set_mixer_gain": _set_mixer_gain,
    "reset_mixer": _reset_mixer,
    "run_report": _run_report,
    "reset_all": _reset_all,
}

