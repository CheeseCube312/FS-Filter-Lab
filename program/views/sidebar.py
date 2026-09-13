"""
Sidebar for FS FilterLab.

Left drawer with filter selection, analysis setup, display options,
export controls, and data management.
"""
import logging
from typing import Callable

from nicegui import ui

from models.constants import (
    DEFAULT_ILLUMINANT,
    UI_BUTTONS, UI_SECTIONS, UI_LABELS,
    UI_INFO_MESSAGES, UI_WARNING_MESSAGES, UI_HELP_TEXT, ACTION_TYPES,
    SENSOR_TYPE_LABELS,
)
from models.core import AppData, FilterCollection
from views.ui_utils import handle_error, stateful_expansion, inline_warning

logger = logging.getLogger(__name__)


# ============================================================================
# FILTER SELECTION
# ============================================================================

def _filter_selection_panel(
    filter_collection: FilterCollection,
    app_state,
    on_change: Callable,
) -> ui.select:
    """Multi-select dropdown for filter selection.

    Returns the ui.select element so callers can read .value.
    """
    display_names = sorted(filter_collection.get_display_names())
    
    # Filter to owned only when toggle is active
    if app_state.my_filters_only:
        owned = app_state.get_my_filters()
        display_names = [n for n in display_names if n in owned]

    sel = ui.select(
        options=display_names,
        multiple=True,
        label=UI_LABELS["select_filters"],
        value=app_state.selected_filters,
        on_change=lambda e: _on_filter_change(e, app_state, on_change),
    ).props("use-chips dense outlined use-input input-debounce=300").classes("w-full")
    return sel


def _on_filter_change(e, app_state, on_change: Callable):
    app_state.selected_filters = list(e.value) if e.value else []
    on_change()


# ============================================================================
# MY FILTERS ONLY TOGGLE
# ============================================================================

def _my_filters_toggle(app_state, filter_collection, on_change: Callable) -> None:
    """Toggle to restrict filter selection to owned filters only."""
    owned_count = app_state.get_my_filters_count()
    total_count = len(filter_collection.filters)
    
    def _on_toggle(e):
        if e.value and owned_count == 0:
            ui.notify(UI_WARNING_MESSAGES["no_owned_filters"], type="warning")
            # Keep toggle off
            return
        app_state.my_filters_only = e.value
        on_change()
    
    with ui.row().classes("w-full items-center gap-2"):
        ui.switch(
            f"Owned Filters Only ({owned_count}/{total_count})",
            value=app_state.my_filters_only,
            on_change=_on_toggle,
        ).props("dense")
    
    # Show warning for selected filters not in owned list
    if app_state.my_filters_only and app_state.selected_filters:
        owned = app_state.get_my_filters()
        unowned = [f for f in app_state.selected_filters if f not in owned]
        if unowned:
            names = ", ".join(unowned[:5])
            if len(unowned) > 5:
                names += f" (+{len(unowned) - 5} more)"
            inline_warning(f"{UI_WARNING_MESSAGES['unowned_selected']} {names}")


# ============================================================================
# FILTER MULTIPLIERS
# ============================================================================

def _filter_multipliers_panel(app_state, on_change: Callable) -> None:
    """Collapsible section with number inputs for stacking counts."""
    selected = app_state.selected_filters
    if not selected:
        return

    with stateful_expansion(UI_LABELS["set_filter_counts"], "filter_multipliers", app_state):
        for name in selected:
            current_val = app_state.filter_multipliers.get(name, 1)

            def _make_cb(n):
                def _cb(e):
                    mults = dict(app_state.filter_multipliers)
                    mults[n] = int(e.value)
                    app_state.filter_multipliers = mults
                    on_change()
                return _cb

            ui.number(
                label=name, value=current_val, min=1, max=5, step=1,
                on_change=_make_cb(name),
            ).props("dense outlined").classes("w-full")


# ============================================================================
# ANALYSIS SETUP (illuminant, QE, target)
# ============================================================================

def _analysis_setup_panel(
    data: AppData,
    app_state,
    on_change: Callable,
) -> None:
    """Illuminant, QE profile, and target reference selectors."""
    illuminants = data.illuminants
    illuminant_metadata = data.illuminant_metadata
    camera_keys = data.camera_keys
    qe_data = data.qe_data
    default_camera_key = data.default_key
    filter_collection = data.filter_collection

    # --- Illuminant ---
    if illuminants:
        illum_names = list(illuminants.keys())
        default_illum = app_state.illuminant_name or (
            DEFAULT_ILLUMINANT if DEFAULT_ILLUMINANT in illum_names else illum_names[0]
        )

        def _on_illum(e):
            app_state.select_illuminant(e.value, illuminants)
            on_change()

        ui.select(
            options=illum_names,
            label=UI_LABELS["scene_illuminant"],
            value=default_illum,
            on_change=_on_illum,
        ).props("dense outlined").classes("w-full")

        # Set initial values
        if app_state.illuminant is None:
            app_state.select_illuminant(default_illum, illuminants)
    else:
        handle_error(UI_WARNING_MESSAGES["no_illuminants"], "warning")

    # --- Sensor / Film profile ---
    sensor_profiles = data.sensor_profiles
    # Build ordered option list: digital sensors first, then films by type
    _TYPE_ORDER = ['rgb_sensor', 'color_slide', 'color_negative', 'bw']
    grouped: dict = {t: [] for t in _TYPE_ORDER}
    for key, profile in sensor_profiles.items():
        grouped.setdefault(profile.sensor_type, []).append(key)
    for grp in grouped.values():
        grp.sort()
    sensor_options = ["None"] + [
        key
        for t in _TYPE_ORDER
        for key in grouped.get(t, [])
    ]

    # Determine default: prefer current selection, then the data's default_key
    current_sel = app_state.selected_camera
    default_sensor = (
        current_sel if current_sel in sensor_options
        else (default_camera_key if default_camera_key in sensor_options else "None")
    )

    def _on_sensor(e):
        app_state.select_sensor(e.value, sensor_profiles)
        on_change()

    ui.select(
        options=sensor_options,
        label=UI_LABELS["sensor_qe_profile"],
        value=default_sensor,
        on_change=_on_sensor,
    ).props("dense outlined use-input input-debounce=300").classes("w-full")

    # Show a small type badge for the active selection
    active_profile = app_state.active_profile
    if active_profile is not None:
        type_label = SENSOR_TYPE_LABELS.get(active_profile.sensor_type, "")
        badge_color = (
            "blue-grey" if active_profile.sensor_type == "rgb_sensor"
            else "teal" if active_profile.sensor_type == "color_slide"
            else "deep-orange" if active_profile.sensor_type == "color_negative"
            else "brown"
        )
        with ui.row().classes("items-center gap-1 mt-1"):
            ui.badge(type_label, color=badge_color).props("outline")
            if active_profile.is_film:
                extras = []
                if active_profile.iso:
                    extras.append(f"ISO {active_profile.iso}")
                if active_profile.process:
                    extras.append(active_profile.process)
                if extras:
                    ui.label(" · ".join(extras)).classes("text-xs text-grey-7")

    # Set initial sensor state if not already set (current_qe derives from
    # active_profile, so checking the profile alone is sufficient)
    if app_state.active_profile is None and default_sensor != "None":
        app_state.select_sensor(default_sensor, sensor_profiles)

    # --- Target profile ---
    display_names = filter_collection.get_display_names()
    target_options = ["None"] + list(display_names)

    def _on_target(e):
        app_state.select_target(e.value, filter_collection)
        on_change()

    ui.select(
        options=target_options,
        label=UI_LABELS["reference_target"],
        value=app_state.selected_target_name or "None",
        on_change=_on_target,
    ).props("dense outlined use-input input-debounce=300").classes("w-full")

    # --- Film Emulation Target ---
    ui.separator().classes("my-2")
    ui.label(UI_SECTIONS["film_emulation"]).classes("text-xs font-semibold text-grey-7 uppercase tracking-wide mt-1")

    # Only offer film profiles as emulation targets
    film_options = ["None"] + [
        key for key, profile in sensor_profiles.items()
        if getattr(profile, 'is_film', False)
    ]
    current_emul = app_state.film_emulation_profile_name
    default_emul = current_emul if current_emul in film_options else "None"

    def _on_emulation_target(e):
        app_state.select_film_emulation_target(e.value, sensor_profiles)
        on_change()

    ui.select(
        options=film_options,
        label=UI_LABELS["film_emulation_target"],
        value=default_emul,
        on_change=_on_emulation_target,
    ).props("dense outlined use-input input-debounce=300").classes("w-full")

    with ui.row().classes("items-center gap-1 mt-1"):
        ui.icon("info_outline").classes("text-grey-6 text-sm cursor-help flex-shrink-0") \
            .tooltip(UI_HELP_TEXT["film_emulation_filters"])
        ui.label(UI_LABELS["film_emulation_filters"]).classes("text-xs text-grey-7")

    all_filter_names = sorted(filter_collection.get_display_names())
    ui.select(
        options=all_filter_names,
        multiple=True,
        label=UI_LABELS["film_emulation_filters"],
        value=app_state.film_emulation_filters,
        on_change=lambda e: (
            setattr(app_state, "film_emulation_filters", list(e.value) if e.value else []),
            on_change(),
        ),
    ).props("use-chips dense outlined use-input input-debounce=300").classes("w-full")

    with ui.row().classes("items-center gap-1 mt-1"):
        ui.switch(
            UI_LABELS["film_emulation_wb"],
            value=app_state.film_emulation_wb,
            on_change=lambda e: (
                setattr(app_state, "film_emulation_wb", e.value),
                on_change(),
            ),
        ).props("dense")
        ui.icon("info_outline").classes("text-grey-6 text-sm cursor-help") \
            .tooltip(UI_HELP_TEXT["film_emulation_wb"])


# ============================================================================
# DISPLAY & VISUALISATION TOGGLES
# ============================================================================

def _display_toggles_panel(app_state, on_change: Callable) -> None:
    """Checkboxes / switches for display options."""

    _PANEL_TOGGLES = [
        ("show_advanced_search", None),
        ("show_reflector_search", None),
        ("show_channel_mixer", UI_HELP_TEXT["channel_mixer"]),
        ("show_my_filters_manager", None),
    ]
    for attr, tooltip in _PANEL_TOGGLES:
        sw = ui.switch(
            UI_SECTIONS[attr],
            value=getattr(app_state, attr),
            on_change=lambda e, a=attr: (
                setattr(app_state, a, e.value),
                on_change(),
            ),
        ).props("dense")
        if tooltip:
            sw.tooltip(tooltip)

    # RGB channel visibilities
    ui.label(UI_SECTIONS["sensor_response_channels"]).classes("font-bold text-sm mt-2")
    for ch in ("R", "G", "B"):
        def _make_rgb_cb(channel=ch):
            def _cb(e):
                vis = app_state.rgb_channels_visibility
                vis[channel] = e.value
                app_state.rgb_channels_visibility = vis
                on_change()
            return _cb

        ui.switch(
            f"{ch} Channel",
            value=app_state.rgb_channels_visibility.get(ch, True),
            on_change=_make_rgb_cb(ch),
        ).props("dense")

    # Log / stop view
    ui.switch(
        UI_LABELS["stop_view_toggle"],
        value=app_state.log_view,
        on_change=lambda e: (
            setattr(app_state, "log_view", e.value),
            on_change(),
        ),
    ).props("dense").tooltip(UI_HELP_TEXT["stop_view"])


# ============================================================================
# EXPORT & REPORTS
# ============================================================================

def _export_panel(app_state, data: AppData, on_action: Callable) -> None:
    """Buttons for report generation and download."""
    selected = app_state.selected_filters
    report_disabled = (
        not selected
        or app_state.current_qe is None
        or app_state.illuminant is None
    )

    ui.button(
        UI_BUTTONS["generate_full_report"],
        on_click=lambda: on_action(ACTION_TYPES["generate_full_report"], app_state.selected_camera),
    ).props("dense outlined" + (" disable" if report_disabled else "")).classes("w-full")

    # Download button — only shown when a report has been generated
    export = app_state.last_export
    if export and export.get("bytes"):
        ui.button(
            UI_BUTTONS["download_report"],
            on_click=lambda: ui.download(export["bytes"], export["name"]),
        ).props("dense outlined").classes("w-full")


# ============================================================================
# DATA MANAGEMENT
# ============================================================================

def _data_management_panel(app_state, on_change: Callable) -> None:
    open_importers = not app_state.show_import_data
    ui.button(
        UI_BUTTONS["csv_importers" if open_importers else "close_importers"],
        on_click=lambda: (setattr(app_state, "show_import_data", open_importers), on_change()),
    ).props("dense outlined").classes("w-full")


# ============================================================================
# TOP-LEVEL SIDEBAR RENDERER
# ============================================================================

def render_sidebar(
    container: ui.column,
    app_state,
    data: AppData,
    on_change: Callable,
    on_action: Callable,
) -> None:
    """Populate the sidebar container with all sidebar controls.

    The container is cleared and rebuilt each time state changes,
    keeping all dynamic content (export enable/disable, multipliers, etc.)
    in sync with current state.

    Args:
        container: column element to fill (cleared first).
        app_state: AppState instance.
        data: Application data dict from initialize_application_data().
        on_change: Callback invoked whenever a setting changes (triggers full re-render).
        on_action: Callback(action_type, payload) for imperative actions (reports, cache).
    """
    filter_collection = data.filter_collection

    container.clear()

    with container:
        ui.label(UI_SECTIONS["filter_plotter"]).classes("text-xl font-bold mb-2")

        # 1. Filter selection
        _filter_selection_panel(filter_collection, app_state, on_change)
        
        # My Filters Only toggle
        _my_filters_toggle(app_state, filter_collection, on_change)

        # 2. Filter multipliers
        _filter_multipliers_panel(app_state, on_change)

        # 3. Analysis setup
        with stateful_expansion(UI_SECTIONS["analysis_setup"], "analysis_setup", app_state):
            _analysis_setup_panel(data, app_state, on_change)

        # 4. Display & vis
        with stateful_expansion(UI_SECTIONS["display_visualization"], "display_visualization", app_state):
            _display_toggles_panel(app_state, on_change)

        # 5. Export
        with stateful_expansion(UI_SECTIONS["export_reports"], "export_reports", app_state):
            _export_panel(app_state, data, on_action)

        # 6. Data management
        with stateful_expansion(UI_SECTIONS["data_management"], "data_management", app_state):
            _data_management_panel(app_state, on_change)
