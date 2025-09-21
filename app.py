"""
FS FilterLab - Optical Filter Analysis Tool

A comprehensive Streamlit application for analyzing optical filter stacks, quantum efficiency
curves, and illuminant spectra. This tool enables optical engineers and photographers to:

- Design and analyze multi-filter optical systems
- Calculate combined transmission characteristics
- Evaluate filter effects on camera sensors
- Generate detailed technical reports
- Import custom spectral data

Main Features:
- Interactive filter selection and stacking
- Real-time transmission calculations
- RGB channel analysis with white balance
- Deviation metrics from target profiles
- Advanced search and filtering capabilities
- PNG report generation

Usage:
    Run with: streamlit run app.py
    Or use the provided batch/shell scripts

License: Open Source (see LICENSE file)
"""

import streamlit as st
from pathlib import Path
from models.constants import CACHE_DIR

# Configure Streamlit
st.set_page_config(page_title="FS FilterLab", layout="wide")
Path(CACHE_DIR).mkdir(exist_ok=True)

# Import main application components
from services.app_operations import initialize_application_data
from views.main_content import render_main_content
from views.sidebar import render_sidebar
from views.state import initialize_session_state, handle_app_actions
from views.ui_utils import handle_error


def main():
    """
    Main application entry point.
    
    Initializes the Streamlit application by:
    1. Loading and validating application data (filters, QE, illuminants)
    2. Setting up the unified state management system
    3. Rendering the sidebar with controls and filter selection
    4. Displaying the main content area with charts and analysis
    5. Handling user actions and interactions
    
    The application uses a modular architecture with separate services for
    data loading, calculations, and visualization.
    """
    # 1. Initialize data and state (using new unified state management)
    app_state = initialize_session_state()  # Returns StateManager directly
    data = initialize_application_data()
    
    if not data:
        handle_error("❌ Failed to load application data. Check data files.", stop_execution=True)
        return
    
    # 2. Render sidebar
    sidebar_actions = render_sidebar(app_state, data)
    
    # 3. Render main content  
    render_main_content(app_state, data)
    
    # 4. Handle actions
    handle_app_actions(sidebar_actions, app_state, data)


if __name__ == "__main__":
    main()
