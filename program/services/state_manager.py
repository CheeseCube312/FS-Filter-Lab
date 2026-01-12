"""
Unified State Management for FS FilterLab.

This module provides centralized application state management using Streamlit's
session_state as the single source of truth, with a clean object-oriented
interface for type-safe state access and modification.
"""
import streamlit as st
import numpy as np
from typing import Any, Dict, List, Optional, Union

from models.constants import DEFAULT_WB_GAINS, DEFAULT_CHANNEL_MIXER
from models.core import TargetProfile, ChannelMixerSettings


# =============================================================================
# STATE CONFIGURATION CONSTANTS
# =============================================================================

# Default values for all state keys - single source of truth
# Note: white_balance_gains uses .copy() when accessed to prevent mutation
STATE_DEFAULTS = {
    # Filter data
    'selected_filters': [],
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
    'white_balance_gains': DEFAULT_WB_GAINS,  # .copy() applied on access
    'wb_reference_surface': None,  # Source file of surface used for WB reference
    
    # Export/report state
    'last_export': {},
    'last_tsv_export': {},
    
    # UI state (non-widget controlled)
    'import_status': None,
    'import_error_message': None,
}

# Keys that are managed by Streamlit widgets - don't initialize manually
WIDGET_CONTROLLED_KEYS = {
    'selected_filters', 'show_advanced_search', 'show_import_data', 
    'sidebar_log_view_toggle', 'apply_white_balance_toggle', 
    'show_R', 'show_G', 'show_B', 'show_channel_mixer'
}


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
        for key, default_value in STATE_DEFAULTS.items():
            if key not in st.session_state and key not in WIDGET_CONTROLLED_KEYS:
                # Use .copy() for mutable defaults to prevent shared state
                if isinstance(default_value, dict):
                    st.session_state[key] = default_value.copy()
                elif isinstance(default_value, list):
                    st.session_state[key] = default_value.copy()
                else:
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
        
        # Use STATE_DEFAULTS for fallback values, with .copy() for mutable types
        default = STATE_DEFAULTS.get(name)
        if isinstance(default, (dict, list)):
            default = default.copy() if default else default
        return st.session_state.get(name, default)
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute in session state."""
        if name.startswith('_'):
            # Allow private attributes to be set normally
            super().__setattr__(name, value)
        elif name in WIDGET_CONTROLLED_KEYS or name in ('log_view', 'apply_white_balance', 'rgb_channels_visibility'):
            # Widget-controlled keys should not be manually set, they're managed by Streamlit
            pass
        else:
            st.session_state[name] = value
    
        
    def get_selected_reflector_idx(self) -> Optional[Union[int, str]]:
        """Get currently selected reflector index."""
        return st.session_state.get("selected_reflector_idx", None)
    
    def set_selected_reflector_idx(self, idx: Optional[Union[int, str]]) -> None:
        """Set selected reflector index."""
        st.session_state["selected_reflector_idx"] = idx
    
    def set_white_balance_from_surface(
        self, 
        reflector: np.ndarray, 
        transmission: np.ndarray,
        source_file: Optional[str] = None
    ) -> None:
        """
        Update white balance gains using a selected surface as reference.
        
        Args:
            reflector: Reflectance spectrum of the reference surface
            transmission: Combined filter transmission values
            source_file: Source file path of the reference surface (for tracking)
        """
        from services.calculations import compute_white_balance_gains_from_surface
        
        if self.current_qe and self.illuminant is not None:
            new_gains = compute_white_balance_gains_from_surface(
                reflector, transmission, self.current_qe, self.illuminant
            )
            self.white_balance_gains = new_gains
            # Store reference surface so WB can be recalculated when filters change
            self.wb_reference_surface = source_file
    
    def reset_white_balance(self) -> None:
        """
        Reset white balance to default behavior (standard computation from transmission/QE/illuminant).
        """
        from models.constants import DEFAULT_WB_GAINS
        
        # Set back to default gains, which will trigger standard WB computation
        self.white_balance_gains = DEFAULT_WB_GAINS.copy()
        # Clear reference surface so standard WB computation is used
        self.wb_reference_surface = None

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
    
    # ========================================================================
    # DEFAULT REFLECTOR LIST MANAGEMENT
    # ========================================================================
    
    _DEFAULT_REFLECTORS_FILE = "program/data/reflectors/default_reflectors.json"
    
    def _load_default_reflectors_from_file(self) -> List[str]:
        """Load default reflector list from JSON file."""
        import json
        from pathlib import Path
        
        file_path = Path(self._DEFAULT_REFLECTORS_FILE)
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('default_reflectors', [])
            except (json.JSONDecodeError, IOError):
                return []
        return []
    
    def _save_default_reflectors_to_file(self, reflector_files: List[str]) -> None:
        """Save default reflector list to JSON file."""
        import json
        from pathlib import Path
        
        file_path = Path(self._DEFAULT_REFLECTORS_FILE)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({'default_reflectors': reflector_files}, f, indent=2)
        except IOError:
            pass  # Silently fail on write errors
    
    def get_default_reflector_files(self) -> List[str]:
        """
        Get list of default reflector source file paths.
        
        Returns:
            List of source file paths for default reflectors
        """
        # First check session state, then fall back to file
        if 'default_reflector_files' not in st.session_state:
            st.session_state['default_reflector_files'] = self._load_default_reflectors_from_file()
        return st.session_state.get('default_reflector_files', [])
    
    def is_default_reflector(self, source_file: str) -> bool:
        """
        Check if a reflector is in the default list.
        
        Args:
            source_file: Source file path of the reflector
            
        Returns:
            True if the reflector is in the default list
        """
        return source_file in self.get_default_reflector_files()
    
    def add_to_default_reflectors(self, source_file: str) -> None:
        """
        Add a reflector to the default list.
        
        Args:
            source_file: Source file path of the reflector to add
        """
        current = self.get_default_reflector_files()
        if source_file not in current:
            current.append(source_file)
            st.session_state['default_reflector_files'] = current
            self._save_default_reflectors_to_file(current)
    
    def remove_from_default_reflectors(self, source_file: str) -> None:
        """
        Remove a reflector from the default list.
        
        Args:
            source_file: Source file path of the reflector to remove
        """
        current = self.get_default_reflector_files()
        if source_file in current:
            current.remove(source_file)
            st.session_state['default_reflector_files'] = current
            self._save_default_reflectors_to_file(current)



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


