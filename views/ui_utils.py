"""
UI utilities and components for FS FilterLab.

This module provides all UI-related functionality including:
- Reusable UI components and styling
- Error handling and messaging utilities
- Color utilities and validation
- Data display formatting
- Safe operation execution
"""
# Standard library imports
import colorsys
import logging
import re
from typing import Dict, Any, Optional, List, Tuple, Union, TypeVar, Callable

# Third-party imports
import streamlit as st
import numpy as np
import pandas as pd

# Local imports
from models.constants import DEFAULT_WB_GAINS
from models.core import TargetProfile

# Configure logging for error tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar('T')

# ============================================================================
# ERROR HANDLING AND MESSAGING
# ============================================================================

def show_error_message(message: str, stop_execution: bool = False) -> None:
    """Display an error message with consistent styling."""
    logger.error(f"Error displayed to user: {message}")
    st.error(message)
    if stop_execution:
        st.stop()


def show_warning_message(message: str) -> None:
    """Display a warning message with consistent styling."""
    st.warning(message)


def show_info_message(message: str) -> None:
    """Display an info message with consistent styling."""
    st.info(message)


def show_success_message(message: str) -> None:
    """Display a success message with consistent styling."""
    st.success(message)


def handle_error(message: str, severity: str = "error", stop_execution: bool = False) -> None:
    """Display an error message with consistent styling based on severity."""
    if severity == "error":
        show_error_message(message, stop_execution)
    elif severity == "warning":
        logger.warning(f"Warning displayed to user: {message}")
        show_warning_message(message)
    else:
        logger.info(f"Info displayed to user: {message}")
        show_info_message(message)


def try_operation(
    operation: Callable[[], T], 
    error_message: str, 
    default_value: Optional[T] = None, 
    severity: str = "error",
    stop_on_error: bool = False
) -> T:
    """Try to execute an operation and handle errors gracefully."""
    try:
        return operation()
    except Exception as e:
        logger.exception(f"Operation failed: {error_message}")
        handle_error(f"{error_message}: {str(e)}", severity, stop_on_error)
        return default_value


def handle_safe_operation(operation, error_message="Operation failed", default_value=None):
    """
    Safely execute an operation and handle any errors.
    
    Args:
        operation: Function to execute
        error_message: Error message to display
        default_value: Value to return if operation fails
        
    Returns:
        Result of operation or default_value if operation fails
    """
    try:
        return operation()
    except Exception as e:
        st.error(f"{error_message}: {str(e)}")
        return default_value


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


def is_valid_hex_color(hex_code: str) -> bool:
    """
    Check if a string is a valid hex color code.
    
    Args:
        hex_code: String to check
        
    Returns:
        True if the string is a valid hex color code
    """
    return isinstance(hex_code, str) and bool(re.fullmatch(r"#([0-9a-fA-F]{6})", hex_code))


def color_swatch(hex_color: str, width: int = 22, height: int = 16) -> str:
    """
    Generate HTML for a color swatch.
    
    Args:
        hex_color: Hex color code
        width: Width in pixels
        height: Height in pixels
        
    Returns:
        HTML string for the color swatch
    """
    return f"""
    <div style='
        display:inline-block;
        width:{width}px;
        height:{height}px;
        background-color:{hex_color};
        border:1px solid #aaa;
        border-radius:4px;
        margin:4px 0;
    '></div>
    """


# ============================================================================
# UI COMPONENTS
# ============================================================================

def styled_header(title: str, level: int = 3, icon: Optional[str] = None) -> None:
    """
    Display a styled header with optional icon.
    
    Args:
        title: Header text
        level: Header level (1-6)
        icon: Optional icon emoji
    """
    prefix = f"{icon} " if icon else ""
    st.markdown(f"{'#' * level} {prefix}{title}")


def colored_box(text: str, color: str, text_color: Optional[str] = None) -> None:
    """
    Display text in a colored box.
    
    Args:
        text: Text to display
        color: Background color (hex code)
        text_color: Text color (hex code)
    """
    if text_color is None:
        text_color = "#FFF" if is_dark_color(color) else "#000"
        
    st.markdown(
        f"""
        <div style="
            background-color: {color};
            color: {text_color};
            padding: 8px 12px;
            border-radius: 6px;
            font-weight: 600;
            margin: 8px 0;
        ">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )


def expandable_section(title: str, expanded: bool = False, icon: Optional[str] = None) -> st._DeltaGenerator:
    """
    Create an expandable section with a stylized title.
    
    Args:
        title: Section title
        expanded: Whether the section is expanded by default
        icon: Optional icon emoji
        
    Returns:
        Streamlit expander object
    """
    prefix = f"{icon} " if icon else ""
    return st.expander(f"{prefix}{title}", expanded=expanded)


def status_indicator(
    value: float, 
    label: str, 
    thresholds: Dict[str, float] = None, 
    is_percentage: bool = False
) -> None:
    """
    Display a status indicator with color based on thresholds.
    
    Args:
        value: Numeric value to display
        label: Label for the value
        thresholds: Dictionary mapping status to threshold values (ascending)
        is_percentage: Whether to format the value as a percentage
    """
    if thresholds is None:
        thresholds = {
            "success": 90,
            "warning": 70,
            "error": 0
        }
        
    format_str = f"{value:.1f}%" if is_percentage else f"{value:.2f}"
    
    # Determine status based on thresholds
    status = "error"
    for s, t in sorted(thresholds.items(), key=lambda x: x[1], reverse=True):
        if value >= t:
            status = s
            break
            
    if status == "success":
        st.success(f"{label}: {format_str}")
    elif status == "warning":
        st.warning(f"{label}: {format_str}")
    else:
        st.error(f"{label}: {format_str}")


def section_separator(margin_top: int = 20, margin_bottom: int = 20) -> None:
    """
    Display a horizontal separator with margins.
    
    Args:
        margin_top: Top margin in pixels
        margin_bottom: Bottom margin in pixels
    """
    st.markdown(
        f"""
        <div style="
            margin-top: {margin_top}px;
            margin-bottom: {margin_bottom}px;
            border-bottom: 1px solid rgba(49, 51, 63, 0.2);
        "></div>
        """,
        unsafe_allow_html=True
    )


# ============================================================================
# DATA DISPLAY UTILITIES
# ============================================================================

def format_data_table(df: pd.DataFrame, highlight_cols: List[str] = None) -> None:
    """
    Display a formatted data table with optional column highlighting.
    
    Args:
        df: DataFrame to display
        highlight_cols: List of column names to highlight
    """
    if highlight_cols is None:
        highlight_cols = []
        
    # Apply styling to the DataFrame
    def highlight(x):
        styles = []
        for col in x.index:
            if col in highlight_cols:
                styles.append('background-color: rgba(255, 215, 0, 0.2)')
            else:
                styles.append('')
        return styles
        
    styled_df = df.style.apply(highlight, axis=1)
    st.dataframe(styled_df)


def format_value_with_unit(value: float, unit: str, precision: int = 2) -> str:
    """
    Format a value with a unit.
    
    Args:
        value: Numeric value to format
        unit: Unit string
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    return f"{value:.{precision}f} {unit}"


def display_metric_card(
    title: str, 
    value: float, 
    unit: str = "", 
    delta: Optional[float] = None,
    precision: int = 2
) -> None:
    """
    Display a metric card with title, value, unit, and optional delta.
    
    Args:
        title: Card title
        value: Numeric value
        unit: Unit string
        delta: Optional delta value for comparison
        precision: Number of decimal places
    """
    formatted = format_value_with_unit(value, unit, precision)
    
    if delta is not None:
        st.metric(title, formatted, delta=delta)
    else:
        st.metric(title, formatted)
