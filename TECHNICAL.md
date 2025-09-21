# FS FilterLab - Technical Overview

This is a quick technical reference for developers working on or extending the project.

## Project Structure

```
models/      # Data structures and constants
services/    # Data processing and logic
views/       # UI components
data/        # Spectral data files (TSV)
cache/       # Auto-generated cache
```

## Main Concepts

- **Filters, QE, Illuminants, Reflectors**: All are loaded from TSV files, interpolated to a standard grid (300-1100nm, 1nm steps), and stored as NumPy arrays.
- **FilterCollection**: Holds all filters and their transmission data as a matrix for fast calculations.
- **StateManager**: Central place for all app state, built on Streamlit's session_state.
- **Services**: Handle calculations (transmission, RGB response, metrics), data loading/caching, and report generation.
- **Views**: Streamlit UI components for sidebar, main content, and forms.

## Data Flow

1. Data is loaded and cached on startup (filters, QE, illuminants, reflectors)
2. User selections update the StateManager
3. Calculations are performed using NumPy (vectorized)
4. Results are shown in the UI (charts, tables, metrics)
5. Reports can be generated as PNGs

## Extending the Project

- **Add a new data type**: Define a dataclass in `models/core.py`, add a loader in `services/data.py`, and update the UI in `views/`.
- **Add a new calculation**: Put the function in `services/calculations.py` and call it from the UI as needed.
- **Add a new chart or UI element**: Add a function in `views/` and use Streamlit components.

## Testing

- Use NumPy's testing tools for calculations
- Test data loading with sample TSV files
- UI can be tested by running the app and trying different workflows

## Deployment

- Standard Python/Streamlit app, no special requirements
- Data files go in the `data/` folder
- Cache is auto-managed in `cache/`

## Troubleshooting

- If data doesn't show up, check your TSV files and try "Rebuild Cache"
- For errors, check the logs in the terminal and Streamlit's error messages

---

For more details, read the code and docstrings. The structure is meant to be straightforward and easy to follow.
