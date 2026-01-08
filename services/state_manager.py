"""
Unified State Management for FS FilterLab.

This module provides centralized application state management using Streamlit's
session_state as the single source of truth, with a clean object-oriented
interface for type-safe state access and modification.
"""
import streamlit as st
from typing import Any, Dict, List, Optional, Union

from models.constants import DEFAULT_RGB_VISIBILITY, DEFAULT_WB_GAINS, DEFAULT_CHANNEL_MIXER
from models.core import TargetProfile, ChannelMixerSettings


class StateManager:
    """
    Unified state management with dynamic attribute access.
    
    This class provides a clean, object-oriented interface to Streamlit's session_state
    while maintaining session_state as the single source of truth. It handles:
    
    - Automatic initialization of required state keys with sensible defaults
    - Type-safe attribute access with fallback values
    - Protection against widget-controlled key modification conflicts
    - Dynamic attribute access for clean, Pythonic state management
    
    State Organization:
        Filter State: selected_filters, filter_multipliers
        Sensor Data: current_qe, selected_camera, illuminant, illuminant_name
        Analysis: target_profile, combined_transmission, white_balance_gains
        UI Settings: Various display preferences and toggles
        Export: last_export metadata for download functionality
        
    Widget Handling:
        Some state keys are controlled by Streamlit widgets and cannot be
        modified programmatically. The StateManager automatically detects
        and handles these cases gracefully.
        
    Thread Safety:
        Safe for use in Streamlit's single-threaded execution model.
        All state modifications go through session_state.
    """
    
    def __init__(self):
        """
        Initialize the state manager and ensure all required keys exist.
        
        Sets up default values for all non-widget state keys and prepares
        the manager for dynamic attribute access.
        """
        self._ensure_initialized()
    
    def _ensure_initialized(self) -> None:
        """
        Initialize all required state keys in session_state with defaults.
        
        This method is idempotent - calling it multiple times has no adverse
        effects and ensures state consistency across application restarts.
        
        Widget-controlled keys are explicitly excluded to avoid conflicts
        with Streamlit's widget management system.
        """
        defaults = {
            # Filter data
            'filter_multipliers': {},
            
            # QE and illuminant data  
            'current_qe': None,
            'selected_camera': None,
            'illuminant': None,
            'illuminant_name': None,
            
            # Target profile
            'target_profile': None,
            
            # Computed results
            'combined_transmission': None,
            'white_balance_gains': DEFAULT_WB_GAINS.copy(),
            
            # Note: Channel mixer is now handled dynamically via _build_live_channel_mixer()
            # to ensure immediate UI responsiveness
            
            # Export/report state
            'last_export': {},
            'last_tsv_export': {},
            
            # UI state (non-widget controlled)
            'import_status': None,
            'import_error_message': None,
        }
        
        # Widget-controlled keys (managed by Streamlit widgets, don't initialize manually)
        widget_keys = {
            'selected_filters', 'show_advanced_search', 'show_import_data', 'sidebar_log_view_toggle',
            'apply_white_balance_toggle', 'show_R', 'show_G', 'show_B', 'show_channel_mixer'
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state and key not in widget_keys:
                st.session_state[key] = default_value


    
    # ========================================================================
    # DYNAMIC ATTRIBUTE ACCESS
    # ========================================================================
    
    def __getattr__(self, name: str) -> Any:
        """
        Get state attribute with appropriate default values.
        
        Provides dynamic access to session_state keys as object attributes,
        with intelligent defaults based on the attribute name and type.
        
        Args:
            name: Attribute name to retrieve from session_state
            
        Returns:
            Value from session_state or appropriate default if not set
            
        Default Value Logic:
            - Lists: Empty list []
            - Dicts: Empty dict {} (with special cases for gains/visibility)
            - Booleans: False
            - Optional types: None
            - Numbers: 0 or 1.0 as appropriate
            
        Example:
            state.selected_filters  # Returns [] if not set
            state.current_qe        # Returns None if not set
            state.log_view         # Returns False if not set
        """
        defaults = {
            'selected_filters': [],
            'filter_multipliers': {},
            'current_qe': None,
            'selected_camera': None,
            'illuminant': None,
            'illuminant_name': None,
            'target_profile': None,
            'combined_transmission': None,
            'white_balance_gains': DEFAULT_WB_GAINS.copy(),
            # Note: channel_mixer is now handled dynamically via _build_live_channel_mixer()
            'last_export': {},
            'last_tsv_export': {},
        }
        
        # Handle special cases for widget-controlled state
        if name == 'log_view':
            return st.session_state.get('sidebar_log_view_toggle', False)
        elif name == 'show_advanced_search':
            return st.session_state.get('show_advanced_search', False)
        elif name == 'show_import_data':
            return st.session_state.get('show_import_data', False)
        elif name == 'apply_white_balance':
            return st.session_state.get('apply_white_balance_toggle', False)
        elif name == 'rgb_channels_visibility':
            # Read RGB channel states from individual widget keys
            return {
                'R': st.session_state.get('show_R', True),
                'G': st.session_state.get('show_G', True),
                'B': st.session_state.get('show_B', True)
            }
        elif name == 'show_channel_mixer':
            return st.session_state.get('show_channel_mixer', False)
        elif name == 'channel_mixer':
            # Always return fresh mixer with current session state values
            # This ensures sliders are always immediately reflected in calculations
            return self._build_live_channel_mixer()
        
        if name.startswith('_'):
            # Don't interfere with private attributes
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        return st.session_state.get(name, defaults.get(name))
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute in session state."""
        if name.startswith('_') or name in ['_ensure_initialized']:
            # Allow private attributes and methods to be set normally
            super().__setattr__(name, value)
        else:
            # Widget-controlled keys should not be manually set, they're managed by Streamlit
            widget_keys = {
                'log_view': 'sidebar_log_view_toggle', 
                'show_advanced_search': 'show_advanced_search',
                'show_import_data': 'show_import_data',
                'apply_white_balance': 'apply_white_balance_toggle',
                'rgb_channels_visibility': None  # Special case - managed by individual keys
            }
            
            if name in widget_keys:
                # These are managed by widgets, don't try to set them manually
                # Just log that we're ignoring the attempt
                pass
            else:
                st.session_state[name] = value
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _build_live_channel_mixer(self) -> ChannelMixerSettings:
        """
        Build a ChannelMixerSettings object with current session state values.
        
        This ensures that the channel mixer always reflects the current UI state,
        eliminating timing issues where slider changes aren't immediately reflected
        in calculations and visualizations.
        
        Returns:
            ChannelMixerSettings object with current slider values from session state
        """
        mixer = ChannelMixerSettings()
        
        # Get enabled state from the show_channel_mixer widget
        mixer.enabled = st.session_state.get('show_channel_mixer', False)
        
        # Get all 9 channel mixing values from session state with fallbacks to defaults
        mixer.red_r = st.session_state.get('red_r', DEFAULT_CHANNEL_MIXER['red_r'])
        mixer.red_g = st.session_state.get('red_g', DEFAULT_CHANNEL_MIXER['red_g'])
        mixer.red_b = st.session_state.get('red_b', DEFAULT_CHANNEL_MIXER['red_b'])
        
        mixer.green_r = st.session_state.get('green_r', DEFAULT_CHANNEL_MIXER['green_r'])
        mixer.green_g = st.session_state.get('green_g', DEFAULT_CHANNEL_MIXER['green_g'])
        mixer.green_b = st.session_state.get('green_b', DEFAULT_CHANNEL_MIXER['green_b'])
        
        mixer.blue_r = st.session_state.get('blue_r', DEFAULT_CHANNEL_MIXER['blue_r'])
        mixer.blue_g = st.session_state.get('blue_g', DEFAULT_CHANNEL_MIXER['blue_g'])
        mixer.blue_b = st.session_state.get('blue_b', DEFAULT_CHANNEL_MIXER['blue_b'])
        
        return mixer
    

    

    



# ============================================================================
# GLOBAL INSTANCE MANAGEMENT
# ============================================================================

# Singleton instance for application-wide state management
_state_manager: Optional[StateManager] = None

def get_state_manager() -> StateManager:
    """
    Get or create the global StateManager instance.
    
    Implements singleton pattern to ensure consistent state access across
    all application modules. The StateManager is created on first access
    and reused for the entire session.
    
    Returns:
        The global StateManager instance for the current session
        
    Thread Safety:
        Safe in Streamlit's single-threaded execution model.
        Each browser session gets its own StateManager instance.
    """
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager


