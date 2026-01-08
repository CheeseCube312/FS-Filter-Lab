"""
Channel Mixer UI Components for FS FilterLab.

This module provides the user interface for channel mixing functionality,
including the main control panel with sliders for manual channel mixing control.
The UI provides a 3x3 matrix of sliders for intuitive channel mixing control.
"""
import streamlit as st
import numpy as np
from typing import Dict, Any, Optional

from models.core import ChannelMixerSettings
from models.constants import CHANNEL_MIXER_RANGE, CHANNEL_MIXER_STEP, DEFAULT_CHANNEL_MIXER
from services.channel_mixer import is_identity_matrix


def _reset_mixer_to_identity(mixer_settings: ChannelMixerSettings) -> None:
    """
    Reset channel mixer settings to identity matrix (no mixing).
    
    Args:
        mixer_settings: ChannelMixerSettings object to reset
    """
    # Reset object values to identity matrix
    mixer_settings.red_r = DEFAULT_CHANNEL_MIXER['red_r']
    mixer_settings.red_g = DEFAULT_CHANNEL_MIXER['red_g']
    mixer_settings.red_b = DEFAULT_CHANNEL_MIXER['red_b']
    mixer_settings.green_r = DEFAULT_CHANNEL_MIXER['green_r']
    mixer_settings.green_g = DEFAULT_CHANNEL_MIXER['green_g']
    mixer_settings.green_b = DEFAULT_CHANNEL_MIXER['green_b']
    mixer_settings.blue_r = DEFAULT_CHANNEL_MIXER['blue_r']
    mixer_settings.blue_g = DEFAULT_CHANNEL_MIXER['blue_g']
    mixer_settings.blue_b = DEFAULT_CHANNEL_MIXER['blue_b']
    
    # Update session state to match (this is safe because we're setting the values, not conflicts)
    st.session_state.red_r = DEFAULT_CHANNEL_MIXER['red_r']
    st.session_state.red_g = DEFAULT_CHANNEL_MIXER['red_g']
    st.session_state.red_b = DEFAULT_CHANNEL_MIXER['red_b']
    st.session_state.green_r = DEFAULT_CHANNEL_MIXER['green_r']
    st.session_state.green_g = DEFAULT_CHANNEL_MIXER['green_g']
    st.session_state.green_b = DEFAULT_CHANNEL_MIXER['green_b']
    st.session_state.blue_r = DEFAULT_CHANNEL_MIXER['blue_r']
    st.session_state.blue_g = DEFAULT_CHANNEL_MIXER['blue_g']
    st.session_state.blue_b = DEFAULT_CHANNEL_MIXER['blue_b']


def render_channel_mixer_panel(mixer_settings: ChannelMixerSettings) -> ChannelMixerSettings:
    """
    Render the main channel mixer control panel.
    
    Creates a clean, compact interface with:
    - 3x3 matrix of channel mixing sliders
    - Reset button
    - Real-time preview of changes
    
    Args:
        mixer_settings: Current channel mixer settings
        
    Returns:
        Updated ChannelMixerSettings object with user modifications
    """
    # Header with reset button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Channel Mixer")
    with col2:
        if st.button("Reset", help="Reset to no mixing"):
            _reset_mixer_to_identity(mixer_settings)
            st.rerun()
    
    # Enable by default when panel is shown
    mixer_settings.enabled = True
    
    # Compact slider grid
    _render_compact_sliders(mixer_settings)
    
    return mixer_settings


def _render_compact_sliders(mixer_settings: ChannelMixerSettings) -> None:
    """
    Render a clean, compact 3x3 grid of channel mixing sliders.
    
    Note: The StateManager now automatically builds channel mixer objects from
    session state, so we no longer need to manually sync values. This eliminates
    timing issues and ensures sliders are always immediately reflected in calculations.
    
    Args:
        mixer_settings: Channel mixer settings (used for initialization only)
    """
    # Initialize session state if not present (using mixer_settings for defaults)
    if "red_r" not in st.session_state:
        st.session_state.red_r = mixer_settings.red_r
    if "red_g" not in st.session_state:
        st.session_state.red_g = mixer_settings.red_g
    if "red_b" not in st.session_state:
        st.session_state.red_b = mixer_settings.red_b
    if "green_r" not in st.session_state:
        st.session_state.green_r = mixer_settings.green_r
    if "green_g" not in st.session_state:
        st.session_state.green_g = mixer_settings.green_g
    if "green_b" not in st.session_state:
        st.session_state.green_b = mixer_settings.green_b
    if "blue_r" not in st.session_state:
        st.session_state.blue_r = mixer_settings.blue_r
    if "blue_g" not in st.session_state:
        st.session_state.blue_g = mixer_settings.blue_g
    if "blue_b" not in st.session_state:
        st.session_state.blue_b = mixer_settings.blue_b
    
    # Column headers first for better clarity
    col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
    with col1:
        st.markdown("**Output**")
    with col2:
        st.markdown("*From Red*")
    with col3:
        st.markdown("*From Green*") 
    with col4:
        st.markdown("*From Blue*")
    
    # Red output row
    col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
    with col1:
        st.markdown("**Red**")
    with col2:
        st.slider(
            "Red→Red", *CHANNEL_MIXER_RANGE, step=CHANNEL_MIXER_STEP,
            key="red_r", label_visibility="collapsed"
        )
    with col3:
        st.slider(
            "Green→Red", *CHANNEL_MIXER_RANGE, step=CHANNEL_MIXER_STEP,
            key="red_g", label_visibility="collapsed"
        )
    with col4:
        st.slider(
            "Blue→Red", *CHANNEL_MIXER_RANGE, step=CHANNEL_MIXER_STEP,
            key="red_b", label_visibility="collapsed"
        )
    
    # Green output row
    col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
    with col1:
        st.markdown("**Green**")
    with col2:
        st.slider(
            "Red→Green", *CHANNEL_MIXER_RANGE, step=CHANNEL_MIXER_STEP,
            key="green_r", label_visibility="collapsed"
        )
    with col3:
        st.slider(
            "Green→Green", *CHANNEL_MIXER_RANGE, step=CHANNEL_MIXER_STEP,
            key="green_g", label_visibility="collapsed"
        )
    with col4:
        st.slider(
            "Blue→Green", *CHANNEL_MIXER_RANGE, step=CHANNEL_MIXER_STEP,
            key="green_b", label_visibility="collapsed"
        )
    
    # Blue output row
    col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
    with col1:
        st.markdown("**Blue**")
    with col2:
        st.slider(
            "Red→Blue", *CHANNEL_MIXER_RANGE, step=CHANNEL_MIXER_STEP,
            key="blue_r", label_visibility="collapsed"
        )
    with col3:
        st.slider(
            "Green→Blue", *CHANNEL_MIXER_RANGE, step=CHANNEL_MIXER_STEP,
            key="blue_g", label_visibility="collapsed"
        )
    with col4:
        st.slider(
            "Blue→Blue", *CHANNEL_MIXER_RANGE, step=CHANNEL_MIXER_STEP,
            key="blue_b", label_visibility="collapsed"
        )
    
    # Note: No manual sync needed - StateManager automatically builds mixer from session state


def render_channel_mixer_toggle() -> bool:
    """
    Render channel mixer toggle button for sidebar.
    
    Returns:
        True if channel mixer should be shown, False otherwise
    """
    return st.checkbox(
        "Show Channel Mixer",
        key="show_channel_mixer",
        help="Open channel mixer panel for RGB manipulation"
    )


def render_compact_channel_mixer_status(mixer_settings: ChannelMixerSettings) -> None:
    """
    Render compact status display for when channel mixer is not expanded.
    
    Args:
        mixer_settings: Current mixer settings
    """
    if not mixer_settings.enabled:
        st.markdown("*Channel Mixer: Disabled*")
        return
        
    if is_identity_matrix(mixer_settings):
        st.markdown("*Channel Mixer: Identity (no mixing)*")
    else:
        st.markdown("*Channel Mixer: Custom*")


def validate_and_warn_mixer_settings(mixer_settings: ChannelMixerSettings) -> None:
    """
    Validate mixer settings and show warnings if needed.
    Only shows critical warnings to avoid UI clutter.
    
    Args:
        mixer_settings: Settings to validate
    """
    if not mixer_settings.enabled:
        return
        
    matrix = mixer_settings.to_matrix()
    
    # Only show critical warnings
    det = np.linalg.det(matrix)
    if abs(det) < 1e-6:
        st.error("Settings may cause color loss - try adjusting values")
    elif np.any(np.abs(matrix) > 5.0):
        st.warning("Extreme values may produce unusual results")