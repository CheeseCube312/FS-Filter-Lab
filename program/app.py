"""
FS FilterLab — Entry Point
===================================
Launch with:  python program/app.py

UI rebuild in progress (ui-rebuild branch). The previous views/ layer was
removed in slice U1; slices U2–U6 rebuild the interface per UX_SPEC.md.
This stub keeps the boot path (data load + state init) alive and renders a
placeholder page.
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

from services.app_operations import initialize_application_data
from services.state_manager import AppState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    # 1. Load data — keeps the boot path exercised during the rebuild.
    data = initialize_application_data()
    if data:
        app_state = AppState()
        app_state.seed_isdefault_reflectors(data.reflector_collection)
    else:
        logger.error("Failed to load application data; check the terminal output.")

    # 2. Placeholder page until slice U2 lands the new shell.
    @ui.page("/")
    def index():
        ui.add_head_html(
            "<style>body { font-family: Inter, system-ui, sans-serif; }</style>"
        )
        with ui.header().classes("items-center px-4 py-2 bg-blue-800"):
            ui.label("FS FilterLab").classes("text-white text-lg font-bold")
        with ui.column().classes("w-full items-center pt-24 gap-2"):
            ui.label("UI rebuild in progress").classes("text-2xl font-bold")
            ui.label(
                "The views layer was removed (slice U1) and is being rebuilt per "
                "UX_SPEC.md (slices U2–U6). Data loading still works; run the "
                "test suite from the program/ directory with: "
                "python -m unittest discover -s tests -t ."
            ).classes("text-sm text-gray-500 max-w-xl text-center")

    ui.run(
        title="FS FilterLab",
        host="127.0.0.1",
        port=8080,
        reload=False,
        show=True,
    )


if __name__ == "__main__":
    main()
