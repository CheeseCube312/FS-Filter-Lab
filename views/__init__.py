"""
User Interface components and views for FS FilterLab.

This package contains all Streamlit UI components organized by functionality:

Component Categories:

UI Utilities (ui_utils):
- Color processing and validation functions
- Reusable styled components (headers, boxes, separators)
- Error handling and user messaging
- Data formatting and display utilities
- Safe operation wrappers with error recovery

Sidebar Components (sidebar):
- Filter selection and search interfaces
- Filter stack multiplier controls  
- Camera QE and illuminant selection
- Settings panels and preferences
- Reflector preview functionality

Main Content (main_content):
- Transmission analysis charts and metrics
- RGB sensor response visualization
- White balance calculations and display
- Chart rendering with interactive controls
- Expandable sections and data tables

Forms (forms):
- Advanced filter search with multiple criteria
- Data import interfaces for custom spectra
- File upload and validation workflows
- Search result filtering and sorting

State Management (state):
- Session state initialization and management
- User action handling and coordination
- State persistence across page interactions

Architecture Principles:
- Modular component design for reusability
- Consistent error handling across all components
- Separation of UI logic from business logic
- Type hints for better IDE support and documentation
- Streamlit best practices for performance and UX
"""

# ============== Components =================
# Color utilities
from views.ui_utils import (
    is_dark_color,
    is_valid_hex_color
)

# UI components - removed unused functions

# Data display utilities - removed unused functions

# ============== Sidebar Components =================
from views.sidebar import (
    filter_selection,
    filter_multipliers,
    analysis_setup
)

# ============== Main Content Components =================
from views.main_content import (
    transmission_metrics,
    deviation_metrics,
    white_balance_display,
    raw_qe_and_illuminant,
    filter_response_display,
    sensor_response_display,
    render_chart
)

# ============== Form Components =================
from views.forms import (
    advanced_filter_search,
    import_data_form
)

# ============== Utilities =================
from views.ui_utils import (
    # Error handling
    show_error_message,
    show_warning_message,
    show_info_message,
    show_success_message,
    handle_error,
    try_operation,
    format_error_message,
    show_template_error
)
