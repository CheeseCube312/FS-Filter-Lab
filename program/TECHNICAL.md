# FS FilterLab - Technical Overview

This is a quick technical reference for developers working on or extending the project.

## Project Structure

```
models/      # Data structures and constants
  - constants.py   # Centralized configuration, error messages, UI text
  - core.py        # Filter, QE, TargetProfile, StateManager classes
services/    # Data processing and logic
  - calculations.py    # Mathematical operations, transmission metrics
  - visualization.py   # Chart generation (Plotly + Matplotlib)
  - data.py           # TSV loading, caching, data normalization
  - importing.py      # Data import functionality
  - app_operations.py # Export operations
  - channel_mixer.py  # RGB channel manipulation
  - state_manager.py  # Application state management
views/       # UI components
  - main_content.py   # Primary display area
  - sidebar.py        # Control panels and inputs
  - forms.py          # Advanced search, import forms
  - ui_utils.py       # Reusable UI components (color previews, etc.)
data/        # Spectral data files (TSV format)
cache/       # Auto-generated cache files
```

## Main Concepts

### Data Scale Architecture
**CRITICAL**: All spectral data uses **fractional scale (0-1) internally** for mathematical correctness:
- **Transmission**: 0.0 = no transmission, 1.0 = full transmission
- **QE values**: 0.0 = no response, 1.0 = full quantum efficiency  
- **Reflectance**: 0.0 = no reflection, 1.0 = full reflection
- **UI Display**: Internal values are converted to percentages (* 100) only for user display
- **Filter Combinations**: Natural multiplication works correctly (0.5 * 0.8 = 0.4)

### Core Components
- **Filters, QE, Illuminants, Reflectors**: Loaded from TSV files, interpolated to standard grid (300-1100nm, 1nm steps), normalized to 0-1 scale internally
- **FilterCollection**: Matrix-based storage for fast vectorized calculations
- **StateManager**: Centralized application state built on Streamlit's session_state
- **Constants System**: All configuration, UI text, and error messages centralized in `models/constants.py`
- **Template System**: Reusable error message and UI component templates for consistency

## Data Flow

1. **Data Loading**: TSV files loaded and cached, **normalized to 0-1 scale** during import
2. **State Management**: User selections update centralized StateManager
3. **Calculations**: Performed using NumPy vectorization on 0-1 scale data
4. **UI Display**: Internal 0-1 values converted to percentages (* 100) for user display  
5. **Export**: Matplotlib reports maintain percentage scale for user readability
6. **Caching**: Processed data cached for performance, maintains internal scale consistency

## Code Quality & Architecture

### Recent Improvements (2026)
- **Comprehensive cleanup**: Eliminated deprecated/unused code across entire codebase
- **Scale consistency**: Established uniform 0-1 internal / 0-100% display architecture  
- **Constants consolidation**: Moved all configuration to `models/constants.py`
- **Import organization**: Consolidated and organized all import statements
- **Function organization**: Moved functions to appropriate modules (e.g., UI utilities to `views/ui_utils.py`)
- **Error handling**: Centralized error messages with reusable templates
- **Documentation**: Added clear scale expectations throughout codebase

### Development Patterns
- **Internal calculations**: Always use 0-1 fractional scale
- **UI conversions**: Apply `* 100` only at display layer
- **Input processing**: Convert user percentages with `/ 100` 
- **Function organization**: Keep related functionality in appropriate service modules
- **Error messages**: Use centralized templates from constants
- **Import structure**: Group by standard library, third-party, then local imports

## Extending the Project

### Adding New Data Types
- Define dataclass in `models/core.py` with 0-1 scale expectations documented
- Add loader in `services/data.py` with proper normalization (`/ 100` for percentages)
- Update UI in `views/` with percentage display (`* 100`)
- Add constants to `models/constants.py` for configuration
- Follow established import organization patterns

### Adding New Calculations  
- Implement in `services/calculations.py` using 0-1 scale internally
- Document scale expectations in function docstrings
- Use vectorized NumPy operations for performance
- Apply percentage conversion only at UI display layer

### Adding New UI Components
- Place in appropriate `views/` module or create in `views/ui_utils.py` for reusables
- Use constants from `models/constants.py` for text and configuration
- Follow established error handling patterns with centralized templates
- Convert internal 0-1 scale to percentages for user display

### Code Quality Guidelines
- **Scale consistency**: Keep 0-1 internal, 0-100% display
- **Function placement**: Group related functionality in appropriate modules
- **Import organization**: Follow established patterns (stdlib, third-party, local)
- **Error handling**: Use templates from `models/constants.py`
- **Documentation**: Always specify scale expectations in docstrings

## Testing

- Use NumPy's testing tools for calculations
- Test data loading with sample TSV files
- UI can be tested by running the app and trying different workflows

## Deployment

- Standard Python/Streamlit app, no special requirements
- Data files go in the `data/` folder
- Cache is auto-managed in `cache/`

## Troubleshooting

### Common Issues
- **Data doesn't show**: Check TSV file format and try "Rebuild Cache"
- **Wrong scale display**: Verify 0-1 internal / 0-100% display conversion is applied correctly  
- **Calculation errors**: Check that all inputs use consistent 0-1 fractional scale
- **UI inconsistencies**: Ensure percentage conversion (`* 100`) is applied at display layer only
- **Import errors**: Verify import organization follows established patterns (stdlib, third-party, local)

### Scale Debugging
- **Internal data**: Should always be 0-1 fractional values
- **UI displays**: Should show 0-100% with proper labels  
- **Filter combinations**: Should use natural multiplication (works correctly in 0-1 scale)
- **Export reports**: Should match UI scale (0-100% for user readability)

### Performance Issues  
- **Data loading**: Check cache files in `cache/` directory
- **Calculations**: Verify vectorized NumPy operations are used
- **UI responsiveness**: Large filter selections may impact performance

---

For more details, read the code and docstrings. The structure is meant to be straightforward and easy to follow.
