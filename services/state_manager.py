"""
Unified State Management for FS FilterLab.

This module provides centralized application state management using Streamlit's
session_state as the single source of truth, with a clean object-oriented
interface for type-safe state access and modification.
"""
import streamlit as st
from typing import Any, Dict, List, Optional, Union, TypeVar

from models.constants import DEFAULT_RGB_VISIBILITY, DEFAULT_WB_GAINS, DEFAULT_CHANNEL_MIXER
from models.core import TargetProfile, ChannelMixerSettings

T = TypeVar('T')


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
            
            # UI state (non-widget controlled)
            'show_import_data': False,
            'import_status': None,
            'import_error_message': None,
        }
        
        # Widget-controlled keys (managed by Streamlit widgets, don't initialize manually)
        widget_keys = {
            'selected_filters', 'show_advanced_search', 'sidebar_log_view_toggle',
            'apply_white_balance_toggle', 'show_R', 'show_G', 'show_B', 'show_channel_mixer'
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state and key not in widget_keys:
                st.session_state[key] = default_value

    def _safe_set_session_state(self, key: str, value: Any) -> None:
        """
        Safely set a session state value with widget conflict protection.
        
        Attempts to set the session state key to the given value, but gracefully
        handles the case where the key is controlled by a Streamlit widget.
        
        Args:
            key: Session state key to set
            value: Value to assign to the key
            
        Note:
            Widget-controlled keys cannot be modified programmatically and will
            raise StreamlitAPIException. This method catches and ignores these
            exceptions to prevent application crashes.
        """
        try:
            st.session_state[key] = value
        except st.errors.StreamlitAPIException:
            # Key is managed by a widget, ignore the set operation
            pass
    
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
            'show_import_data': False,
        }
        
        # Handle special cases for widget-controlled state
        if name == 'log_view':
            return st.session_state.get('sidebar_log_view_toggle', False)
        elif name == 'show_advanced_search':
            return st.session_state.get('show_advanced_search', False)
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
        if name.startswith('_') or name in ['_ensure_initialized', '_safe_set_session_state']:
            # Allow private attributes and methods to be set normally
            super().__setattr__(name, value)
        else:
            # Widget-controlled keys should not be manually set, they're managed by Streamlit
            widget_keys = {
                'log_view': 'sidebar_log_view_toggle', 
                'show_advanced_search': 'show_advanced_search',
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
    
    def reset(self) -> None:
        """
        Reset the application state to default values.
        
        Clears all user selections and computed results while preserving
        the underlying data. Useful for starting fresh analysis or clearing
        problematic state.
        
        Reset Operations:
            - Clear filter selections and multipliers
            - Reset display preferences to defaults
            - Clear computed results and cached calculations
            - Reset UI toggles and expanded sections
            - Preserve loaded data (filters, QE, illuminants)
            
        Note:
            Widget-controlled state is reset by directly modifying session_state
            since widgets cannot be programmatically controlled through the
            StateManager interface.
        """
        # Reset non-widget managed state
        self.selected_filters = []
        self.filter_multipliers = {}
        self.combined_transmission = None
        self.white_balance_gains = DEFAULT_WB_GAINS.copy()
        self.last_export = {}
        self.show_import_data = False
        self.import_status = None
        self.import_error_message = None
        
        # Reset widget-controlled state directly in session_state
        st.session_state['sidebar_log_view_toggle'] = False
        st.session_state['show_advanced_search'] = False
        st.session_state['apply_white_balance_toggle'] = False
        st.session_state['show_R'] = True
        st.session_state['show_G'] = True  
        st.session_state['show_B'] = True
        st.session_state['show_channel_mixer'] = False
    
    def update_multiple(self, **kwargs) -> None:
        """
        Update multiple state values atomically.
        
        Provides efficient bulk updates while maintaining the same error
        handling and widget conflict protection as individual updates.
        
        Args:
            **kwargs: Key-value pairs of state attributes to update
            
        Example:
            state.update_multiple(
                selected_filters=['Filter1', 'Filter2'],
                log_view=True,
                target_profile=new_profile
            )
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                st.session_state[key] = value
    
    def get_raw_session_state(self) -> Dict[str, Any]:
        """
        Get direct access to session_state for debugging and inspection.
        
        Returns a copy of the current session_state contents, useful for
        debugging state issues or understanding the complete application state.
        
        Returns:
            Dictionary containing all current session_state key-value pairs
            
        Note:
            This is primarily for debugging purposes. Normal application code
            should use the dynamic attribute access provided by StateManager.
        """
        return dict(st.session_state)


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

def reset_state_manager() -> None:
    """
    Reset the global StateManager instance.
    
    Primarily used for testing scenarios where a clean state is required
    between test runs. Normal application use should not need this function.
    
    Warning:
        This will destroy the current StateManager and all associated state.
        Use with caution in production code.
    """
    global _state_manager
    _state_manager = None
