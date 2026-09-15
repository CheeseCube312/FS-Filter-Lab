"""
FS FilterLab — Entry Point
===================================
Launch with:  python program/app.py
   or from project root:  .venv/Scripts/python program/app.py

Thin shell per UX_SPEC.md §6: data init, AppState init, page scaffolding.
All feature logic lives in views/ + services/.
"""
import logging
import sys
from pathlib import Path

# Ensure the program/ directory is on the import path so that
# `from services.xxx import ...` works regardless of working directory.
_program_dir = Path(__file__).resolve().parent
if str(_program_dir) not in sys.path:
    sys.path.insert(0, str(_program_dir))

from nicegui import ui

from services.data import initialize_application_data
from services.state_manager import AppState
from views.shell import build_page

logging.basicConfig(level=logging.INFO)


def _apply_startup_defaults(app_state: AppState, data) -> None:
    """Preselect a sensor and illuminant so the first charts render (Mode A)."""
    if app_state.active_profile is None:
        profiles = data.sensor_profiles
        if data.default_key in profiles:
            app_state.active_profile = profiles[data.default_key]
        else:
            sensors = [p for p in profiles.values() if not p.is_film]
            app_state.active_profile = sensors[0] if sensors else None
    if app_state.illuminant is None and data.illuminants:
        name = "D65" if "D65" in data.illuminants else sorted(data.illuminants)[0]
        app_state.select_illuminant(name, data.illuminants)


def main() -> None:
    """Build the NiceGUI application UI."""

    # 1. Load data
    data = initialize_application_data()
    if not data:
        @ui.page("/")
        def error_page():
            ui.label(
                "Failed to load application data. Check data files and the terminal for errors."
            ).classes("text-red-600 text-xl m-8")
        ui.run(title="FS FilterLab — Error", host="127.0.0.1", port=8080, reload=False, show=True)
        return

    # 2. State manager + startup defaults
    app_state = AppState()
    app_state.seed_isdefault_reflectors(data.reflector_collection)
    _apply_startup_defaults(app_state, data)

    # 3. Page
    @ui.page("/")
    def index():
        build_page(app_state, data)

    ui.run(
        title="FS FilterLab",
        host="127.0.0.1",
        port=8080,
        reload=False,
        show=True,
    )


if __name__ == "__main__":
    main()
