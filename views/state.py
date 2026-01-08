"""
UI action coordination for FS FilterLab.

Handles user-triggered actions from UI components by coordinating
between UI events and business logic services.
"""

# Standard library imports
from pathlib import Path
from typing import Dict, Any

# Third-party imports
import streamlit as st

# Local imports
from models.constants import CACHE_DIR, UI_SUCCESS_MESSAGES, UI_WARNING_MESSAGES, UI_OPERATION_ERRORS, ACTION_TYPES
from services.app_operations import (
    generate_application_report, rebuild_application_cache,
    generate_full_report, generate_tsv_for_download
)
from services.state_manager import get_state_manager, StateManager
from views.ui_utils import handle_error, show_info_message, show_success_message, try_operation


def handle_app_actions(actions: Dict[str, Any], state_manager: StateManager, data: Dict) -> None:
    """
    Process user actions from UI components.
    
    Args:
        actions: Dictionary of action types and parameters from UI
        state_manager: Application state manager
        data: Application data (filters, QE, illuminants, etc.)
    
    Supported Actions:
        ACTION_TYPES['generate_report']: Create PNG analysis report
        ACTION_TYPES['generate_full_report']: Create PNG + TSV reports to output folder  
        ACTION_TYPES['export_tsv']: Generate TSV for download
        ACTION_TYPES['rebuild_cache']: Clear and rebuild data caches
    """
    if not actions:
        return
    
    # Handle report generation
    if ACTION_TYPES['generate_report'] in actions:
        selected_camera = actions[ACTION_TYPES['generate_report']]
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
                handle_error(UI_WARNING_MESSAGES['report_generation_failed'])
        
        try_operation(_generate_report, UI_OPERATION_ERRORS['report_generation'])
    
    # Handle full report generation (PNG + TSV)
    if ACTION_TYPES['generate_full_report'] in actions:
        selected_camera = actions[ACTION_TYPES['generate_full_report']]
        def _generate_full_report():
            success = generate_full_report(
                app_state=state_manager,
                filter_collection=data['filter_collection'],
                selected_camera=selected_camera
            )
            if success:
                show_success_message(UI_SUCCESS_MESSAGES['full_report_generated'])
                st.rerun()
            else:
                handle_error(UI_WARNING_MESSAGES['full_report_generation_failed'])
        
        try_operation(_generate_full_report, UI_OPERATION_ERRORS['full_report_generation'])

    # Handle TSV generation for download
    if ACTION_TYPES['export_tsv'] in actions and actions[ACTION_TYPES['export_tsv']]:
        def _generate_tsv():
            success = generate_tsv_for_download(
                app_state=state_manager,
                filter_collection=data['filter_collection']
            )
            if success:
                show_success_message(UI_SUCCESS_MESSAGES['tsv_generated'])
                st.rerun()
            else:
                handle_error(UI_WARNING_MESSAGES['tsv_generation_failed'])
        
        try_operation(_generate_tsv, UI_OPERATION_ERRORS['tsv_generation'])

    # Handle cache rebuild
    if ACTION_TYPES['rebuild_cache'] in actions and actions[ACTION_TYPES['rebuild_cache']]:
        def _rebuild_cache():
            cache_path = Path(CACHE_DIR)
            success = rebuild_application_cache(cache_path)
            if success:
                show_success_message(UI_SUCCESS_MESSAGES['cache_rebuilt'])
                st.rerun()
            else:
                handle_error(UI_WARNING_MESSAGES['cache_rebuild_failed'])
        
        try_operation(_rebuild_cache, UI_OPERATION_ERRORS['cache_rebuild'])
