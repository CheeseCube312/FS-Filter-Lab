"""
Application state management and user action handling.

This module provides centralized coordination between the UI and application state,
managing the flow of user interactions and ensuring consistent state updates.

Key Components:

State Initialization:
- initialize_session_state(): Sets up the unified StateManager for the session
- Integrates with Streamlit's session_state for persistence across interactions
- Ensures all required state keys are properly initialized

Action Processing:
- handle_app_actions(): Processes user-triggered actions from UI components
- Coordinates between UI events and business logic services
- Provides error handling and user feedback for all operations

Supported Actions:
- Report Generation: Creates and prepares PNG analysis reports for download
- Cache Management: Rebuilds data caches when source files change
- State Resets: Clears user selections and computed results
- Data Refreshes: Triggers reload of external data sources

Integration Points:
This module serves as the main coordination point between:
- UI components (sidebar, main_content, forms)
- Business services (app_operations, calculations, visualization)
- State management (StateManager, Streamlit session_state)
- Error handling (ui_utils messaging functions)

The design ensures that complex user workflows are handled consistently
while maintaining clean separation between UI presentation and business logic.
"""

# Standard library imports
from pathlib import Path
from typing import Dict, Any

# Third-party imports
import streamlit as st

# Local imports
from models.constants import CACHE_DIR, UI_SUCCESS_MESSAGES
from services.app_operations import generate_application_report, rebuild_application_cache
from services.state_manager import get_state_manager, StateManager
from views.ui_utils import handle_error, show_info_message, show_success_message


def initialize_session_state() -> StateManager:
    """
    Initialize and return the unified application state manager.
    
    Creates or retrieves the StateManager instance that provides centralized
    access to all application state using Streamlit's session_state as the
    underlying storage mechanism.
    
    The StateManager handles:
    - Default value initialization for all state keys
    - Type-safe access to state variables
    - Integration with Streamlit widgets
    - State persistence across page interactions
    
    Returns:
        StateManager instance configured for the current session
        
    Note:
        This function is idempotent - calling it multiple times returns
        the same StateManager instance for the session.
    """
    # Initialize and return the unified StateManager
    return get_state_manager()


def handle_app_actions(actions: Dict[str, Any], state_manager: StateManager, data: Dict) -> None:
    """
    Process and execute actions triggered by user interface interactions.
    
    This function serves as the central dispatcher for user actions, coordinating
    between UI events and the appropriate business logic services. It provides
    consistent error handling and user feedback across all operations.
    
    Args:
        actions: Dictionary of action types and parameters from UI components
                Keys represent action types, values contain action parameters
        state_manager: Unified state manager for accessing and updating application state
        data: Loaded application data including filters, QE, illuminants, etc.
    
    Supported Actions:
        'generate_report': Create PNG analysis report
            - Parameter: selected_camera (str) - name of camera for QE analysis
            - Triggers comprehensive report generation workflow
            - Updates state with export metadata for download functionality
            
        'rebuild_cache': Clear and rebuild data caches
            - Parameter: boolean flag indicating rebuild request
            - Clears all cached data files and triggers application reload
            - Useful when source data files have been updated externally
    
    Error Handling:
        - All operations wrapped in try_operation for consistent error recovery
        - User-friendly error messages displayed via UI messaging system
        - Failed operations don't crash the application or corrupt state
        - Success messages provide clear feedback on completed operations
        
    State Management:
        - Updates state_manager with operation results
        - Triggers Streamlit rerun when state changes require UI refresh
        - Maintains operation history for user reference
    """
    if not actions:
        return
    
    # Handle report generation
    if 'generate_report' in actions:
        selected_camera = actions['generate_report']
        def _generate_report():
            success = generate_application_report(
                app_state=state_manager,
                filter_collection=data['filter_collection'],
                selected_camera=selected_camera
            )
            if success:
                show_success_message(UI_SUCCESS_MESSAGES['report_generated'])
                st.rerun()
            else:
                handle_error("Failed to generate report. Check console for details.")
        
        from views.ui_utils import try_operation
        try_operation(_generate_report, "Report generation failed")
    
    # Handle full report generation (PNG + TSV)
    if 'generate_full_report' in actions:
        selected_camera = actions['generate_full_report']
        def _generate_full_report():
            from services.app_operations import generate_full_report
            success = generate_full_report(
                app_state=state_manager,
                filter_collection=data['filter_collection'],
                selected_camera=selected_camera
            )
            if success:
                show_success_message("✅ Full report generated! Files saved to output folder.")
                st.rerun()
            else:
                handle_error("Failed to generate full report. Check console for details.")
        
        from views.ui_utils import try_operation
        try_operation(_generate_full_report, "Full report generation failed")

    # Handle TSV generation for download
    if 'export_tsv' in actions and actions['export_tsv']:
        def _generate_tsv():
            from services.app_operations import generate_tsv_for_download
            success = generate_tsv_for_download(
                app_state=state_manager,
                filter_collection=data['filter_collection']
            )
            if success:
                show_success_message("✅ TSV generated and ready for download!")
                st.rerun()
            else:
                handle_error("Failed to generate TSV. Make sure you have filters selected.")
        
        from views.ui_utils import try_operation
        try_operation(_generate_tsv, "TSV generation failed")

    # Handle cache rebuild
    if 'rebuild_cache' in actions and actions['rebuild_cache']:
        def _rebuild_cache():
            cache_path = Path(CACHE_DIR)
            success = rebuild_application_cache(cache_path)
            if success:
                show_success_message(UI_SUCCESS_MESSAGES['cache_rebuilt'])
                st.rerun()
            else:
                handle_error("Failed to rebuild cache.")
        
        from views.ui_utils import try_operation
        try_operation(_rebuild_cache, "Cache rebuild failed")
