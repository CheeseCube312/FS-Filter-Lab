"""
UI utilities for FS FilterLab.

Provides error handling, messaging, color utilities, and safe operation execution.
"""
# Standard library imports
import colorsys
import re
from typing import Dict, Any, Optional, List, Tuple, Union, TypeVar, Callable

# Third-party imports
import streamlit as st
import numpy as np
import pandas as pd

# Local imports
from models.constants import UI_SECTIONS, ERROR_MESSAGE_TEMPLATES
from models.core import TargetProfile
from services.visualization import prepare_rgb_for_display

T = TypeVar('T')

# ============================================================================
# ERROR HANDLING AND MESSAGING
# ============================================================================

def show_error_message(message: str, stop_execution: bool = False) -> None:
    """Display an error message."""
    st.error(message)
    if stop_execution:
        st.stop()


def show_warning_message(message: str) -> None:
    """Display a warning message."""
    st.warning(message)


def show_info_message(message: str) -> None:
    """Display an info message."""
    st.info(message)


def show_success_message(message: str) -> None:
    """Display a success message."""
    st.success(message)


def handle_error(message: str, severity: str = "error", stop_execution: bool = False) -> None:
    """Display an error message with specified severity."""
    if severity == "error":
        show_error_message(message, stop_execution)
    elif severity == "warning":
        show_warning_message(message)
    else:
        show_info_message(message)


def try_operation(
    operation: Callable[[], T], 
    error_message: str, 
    default_value: Optional[T] = None, 
    severity: str = "error",
    stop_on_error: bool = False
) -> T:
    """Execute an operation with error handling."""
    try:
        return operation()
    except Exception as e:
        handle_error(f"{error_message}: {str(e)}", severity, stop_on_error)
        return default_value


def format_error_message(template_key: str, **kwargs) -> str:
    """
    Format an error message using predefined templates.
    
    Args:
        template_key: Key from ERROR_MESSAGE_TEMPLATES
        **kwargs: Template parameters
        
    Returns:
        Formatted error message string
    """
    if template_key not in ERROR_MESSAGE_TEMPLATES:
        return f"Unknown error template: {template_key}"
    
    try:
        return ERROR_MESSAGE_TEMPLATES[template_key].format(**kwargs)
    except KeyError as e:
        return f"Error formatting message template '{template_key}': missing parameter {e}"


def show_template_error(template_key: str, severity: str = "error", **kwargs) -> None:
    """
    Display an error message using a predefined template.
    
    Args:
        template_key: Key from ERROR_MESSAGE_TEMPLATES
        severity: Message severity ('error', 'warning', 'info')
        **kwargs: Template parameters
    """
    message = format_error_message(template_key, **kwargs)
    handle_error(message, severity)


# ============================================================================
# COLOR UTILITIES
# ============================================================================

def is_dark_color(hex_color: str) -> bool:
    """
    Check if a color is dark (for text contrast).
    
    Args:
        hex_color: Hex color code
        
    Returns:
        True if the color is dark
    """
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    luminance = 0.2126*r + 0.7152*g + 0.0722*b
    return luminance < 128


# ============================================================================
# COLOR PREVIEW UTILITIES
# ============================================================================

def reflector_preview(pixels: np.ndarray, reflector_names: Optional[List[str]] = None) -> None:
    """
    Display reflector color preview.
    
    Args:
        pixels: Array of RGB pixel values, shape (n, m, 3)
        reflector_names: List of reflector names (optional)
    """
    # Ensure pixels is a valid RGB array
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        message = format_error_message('invalid_format', item='pixel array')
        handle_error(message)
        return
    
    # Display in the sidebar
    st.sidebar.subheader(UI_SECTIONS['vegetation_preview'])
    
    # Use camera-realistic normalization with independent channel saturation
    pixels_normalized = prepare_rgb_for_display(pixels, auto_exposure=True)
    
    # Display the image in the sidebar
    st.sidebar.image(pixels_normalized, width=300, channels="RGB", output_format="PNG")


def single_reflector_preview(
    pixel_color: np.ndarray, 
    reflector_name: str,
    global_max: float = None
) -> None:
    """
    Display single reflector color preview.
    
    Args:
        pixel_color: Single RGB color as 1x1x3 array
        reflector_name: Name of the reflector
        global_max: Global maximum value for consistent scaling (optional)
    """
    # Ensure pixel_color is a valid RGB array
    if pixel_color.ndim != 3 or pixel_color.shape[2] != 3:
        message = format_error_message('invalid_format', item='pixel color')
        handle_error(message)
        return
    
    # Display in the sidebar
    st.sidebar.subheader(UI_SECTIONS['surface_preview'])
    
    # Use camera-realistic normalization
    
    if global_max is not None and global_max > 0:
        # Use consistent scaling with vegetation preview
        # Apply the same exposure scaling, then camera-realistic saturation
        pixel_normalized = prepare_rgb_for_display(
            pixel_color, 
            saturation_level=global_max, 
            auto_exposure=False
        )
    else:
        # Independent normalization when no global reference
        pixel_normalized = prepare_rgb_for_display(pixel_color, auto_exposure=True)
    
    # Display the single color as a larger image
    st.sidebar.image(pixel_normalized, width=200, channels="RGB", output_format="PNG")
    st.sidebar.caption(f"Selected: {reflector_name}")


def is_valid_hex_color(hex_code: str) -> bool:
    """
    Check if a string is a valid hex color code.
    
    Args:
        hex_code: String to check
        
    Returns:
        True if the string is a valid hex color code
    """
    return isinstance(hex_code, str) and bool(re.fullmatch(r"#([0-9a-fA-F]{6})", hex_code))


# ============================================================================
# COLOR SWATCH RENDERING
# ============================================================================

def render_color_swatch(
    hex_color: str, 
    size: int = 40, 
    border: bool = True
) -> None:
    """
    Render a simple color swatch square.
    
    Args:
        hex_color: Hex color code (e.g., "#FF5500")
        size: Size of the swatch in pixels (default 40)
        border: Whether to add a border (default True)
    """
    border_style = "border: 1px solid #ccc;" if border else ""
    st.markdown(f"""
        <div style="
            background-color: {hex_color};
            width: {size}px;
            height: {size}px;
            border-radius: 4px;
            {border_style}
        "></div>
    """, unsafe_allow_html=True)


def render_color_swatch_from_rgb(
    rgb_color: np.ndarray, 
    size: int = 40, 
    border: bool = True
) -> None:
    """
    Render a color swatch from RGB values (0-1 scale).
    
    Args:
        rgb_color: RGB array with values in 0-1 range
        size: Size of the swatch in pixels (default 40)
        border: Whether to add a border (default True)
    """
    if rgb_color is None:
        st.markdown("—")
        return
    
    hex_color = "#{:02x}{:02x}{:02x}".format(
        int(np.clip(rgb_color[0], 0, 1) * 255),
        int(np.clip(rgb_color[1], 0, 1) * 255),
        int(np.clip(rgb_color[2], 0, 1) * 255)
    )
    render_color_swatch(hex_color, size, border)


def render_filter_card(
    hex_color: str,
    label: str,
    text_color: Optional[str] = None
) -> None:
    """
    Render a filter card with colored background and label.
    
    Args:
        hex_color: Background hex color code
        label: Text to display on the card
        text_color: Text color (auto-detected if None)
    """
    if text_color is None:
        text_color = "#FFF" if is_dark_color(hex_color) else "#000"
    
    st.markdown(f"""
        <div style="
            background-color: {hex_color};
            color: {text_color};
            padding: 8px 12px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 1rem;
            margin-bottom: 0;
        ">
            {label}
        </div>
    """, unsafe_allow_html=True)



