"""
UI utilities for FS FilterLab.

Provides toast notifications, inline messages, color utilities,
and reusable rendering helpers.
"""
import re
from typing import Dict, Optional

import numpy as np
from nicegui import ui


# ============================================================================
# MESSAGING — TOAST NOTIFICATIONS (for action feedback)
# ============================================================================

def show_error_message(message: str) -> None:
    """Display an error notification (toast)."""
    ui.notify(message, type="negative", position="top", close_button=True, timeout=8000)


def show_warning_message(message: str) -> None:
    """Display a warning notification (toast)."""
    ui.notify(message, type="warning", position="top", close_button=True, timeout=5000)


def show_info_message(message: str) -> None:
    """Display an info notification (toast)."""
    ui.notify(message, type="info", position="top", close_button=True, timeout=4000)


def show_success_message(message: str) -> None:
    """Display a success notification (toast)."""
    ui.notify(message, type="positive", position="top", close_button=True, timeout=3000)


# ============================================================================
# MESSAGING — INLINE (for page-content messages that should persist)
# ============================================================================

def inline_warning(message: str) -> None:
    """Render an inline warning message that persists on the page."""
    with ui.row().classes("w-full items-center gap-2 p-2 bg-yellow-50 rounded border border-yellow-300"):
        ui.icon("warning").classes("text-yellow-600")
        ui.label(message).classes("text-sm text-yellow-800")


def inline_info(message: str) -> None:
    """Render an inline info message that persists on the page."""
    with ui.row().classes("w-full items-center gap-2 p-2 bg-blue-50 rounded border border-blue-200"):
        ui.icon("info").classes("text-blue-600")
        ui.label(message).classes("text-sm text-blue-800")


def handle_error(message: str, severity: str = "error") -> None:
    """Unified error display with severity routing."""
    if severity == "error":
        show_error_message(message)
    elif severity == "warning":
        show_warning_message(message)
    else:
        show_info_message(message)


def try_operation(operation, error_message: str, default_value=None, severity: str = "error"):
    """Execute an operation with error handling."""
    try:
        return operation()
    except Exception as e:
        handle_error(f"{error_message}: {e}", severity)
        return default_value


# ============================================================================
# COLOR UTILITIES
# ============================================================================

def is_dark_color(hex_color: str) -> bool:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b < 128


def is_valid_hex_color(hex_code: str) -> bool:
    return isinstance(hex_code, str) and bool(re.fullmatch(r"#([0-9a-fA-F]{6})", hex_code))


def rgb_to_hex(rgb: np.ndarray) -> str:
    """Convert 0-1 scale RGB array to hex string."""
    if rgb is None:
        return "#808080"
    return "#{:02x}{:02x}{:02x}".format(
        int(np.clip(rgb[0], 0, 1) * 255),
        int(np.clip(rgb[1], 0, 1) * 255),
        int(np.clip(rgb[2], 0, 1) * 255),
    )


# ============================================================================
# REFLECTOR METADATA FORMATTING
# ============================================================================

def format_reflector_metadata(
    metadata: Dict[str, str],
    api_fields: list,
    fallback_fields: list,
    relevant_meta_key: str,
    max_parts: int = 3,
) -> Optional[str]:
    """Build a concise metadata summary string for a reflector.

    Collects field values from *metadata* in priority order:
    api_fields first, then columns listed in the relevant-metadata key
    (or *fallback_fields* if none).  Returns at most *max_parts* items
    joined by `` | ``, or ``None`` if nothing is available.
    """
    display_fields = list(api_fields)
    rel_meta = metadata.get(relevant_meta_key, "")
    if rel_meta:
        display_fields.extend(c.strip() for c in rel_meta.split("|") if c.strip())
    else:
        display_fields.extend(fallback_fields)

    seen: set = set()
    parts: list = []
    for field in display_fields:
        if field in seen:
            continue
        seen.add(field)
        v = metadata.get(field, "").strip()
        if v:
            parts.append(f"{field}: {v}")
    return " | ".join(parts[:max_parts]) if parts else None


# ============================================================================
# COLOR SWATCH / CARD RENDERING
# ============================================================================

def render_filter_card(hex_color: str, label: str, text_color: Optional[str] = None) -> None:
    text_color = text_color or ("#FFF" if is_dark_color(hex_color) else "#000")
    ui.html(
        f'<div style="background-color:{hex_color};color:{text_color};'
        f'padding:8px 12px;border-radius:6px;font-weight:600;font-size:1rem;'
        f'margin-bottom:2px;">{label}</div>'
    )


# ============================================================================
# STATEFUL EXPANSION
# ============================================================================

def stateful_expansion(label: str, key: str, app_state, default_open: bool = False) -> ui.expansion:
    """Create a ui.expansion that remembers its open/closed state across rebuilds.

    Saves state directly to app_state.sidebar_expansions without triggering
    a full UI refresh — toggling an expander causes no flicker or re-render.

    Args:
        label: Header text for the expansion panel.
        key: Unique string key used to persist state in app_state.
        app_state: AppState instance.
        default_open: Initial open state if never saved (default False).
    """
    is_open = app_state.sidebar_expansions.get(key, default_open)

    def _save(e):
        exps = dict(app_state.sidebar_expansions)
        exps[key] = e.value
        app_state.sidebar_expansions = exps
        # Intentionally no on_change call — toggle doesn't need a full rebuild

    return ui.expansion(label, value=is_open, on_value_change=_save).classes("w-full")
