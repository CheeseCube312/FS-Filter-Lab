# FS FilterLab - Technical Overview

As-is reference plus the core design principles that keep the codebase consistent — deliberately lean so it fits in an agent's context.

## Project Structure

```
app.py       # NiceGUI application entry point
models/      # Data structures and constants
  - constants.py   # Centralized configuration, error messages, UI text
  - core.py        # Filter, FilterCollection, Reflector*, SensorProfile hierarchy, AppData
services/    # Framework-agnostic data processing and logic
  - calculations.py    # Mathematical operations, transmission metrics
  - visualization.py   # Plotly charts for UI views + matplotlib PNG report rendering
  - presenter.py       # View-model builder; advanced search compute
  - data.py            # TSV loading, caching, data normalization
  - importing.py       # Data import (filters, illuminants, camera QE, film, reflectance)
  - spectra.py         # Pure spectral-physics core (combine, integrate, WB gains)
  - state_manager.py   # Application state management
views/       # NiceGUI UI components (zero computation — render view models only)
  - shell.py            # App shell/layout; action dispatch wiring
  - setup_rail.py       # Setup rail controls
  - shelf_data.py       # Data shelf + per-card CSV import expansions
  - shelf_filters.py    # Filters shelf: quick/advanced search, sparkline cards, My Filters
  - shelf_reflectors.py # Reflectors shelf: library search, spectrum sparklines, add-as-surface actions
  - actions.py          # Imperative action handlers (imports, report generation)
  - film_match.py       # Film Match feature view
  - sensor_response.py  # Sensor response view
  - transmission.py     # Transmission metrics view
  - scene_preview.py    # Scene Preview card: render swatches, camera-look toggle
  - reference.py        # Raw QE / illuminant lookup view (collapsed fold-out)
tests/       # unittest suite (calculations, data, importing, presenter, spectra, state, report)
data/        # Spectral data files (TSV format)
  - filters_data/     # Filter transmissions (+ my_filters.json)
  - QE_data/          # Camera sensor QE curves
  - film_data/        # Sensor/film profiles (digital, slide, color negative, B&W)
  - illuminants/      # Light source spectra
  - reflectors/       # Surface reflectance library
cache/       # Auto-generated cache files (one .npz per dataset)
output/      # Generated PNG reports
```

## Main Concepts

### Data Scale Architecture
**CRITICAL**: Two-tier scale system for mathematical correctness and user clarity:

**File Storage (0-100 percentage scale)**:
- All TSV data files store values as percentages: 0 = 0%, 100 = 100%
- This matches user expectations and common spectral data formats
- Examples: transmission = 85.3, QE = 42.7, reflectance = 12.5

**Internal Processing (0-1 fractional scale)**:
- Data is automatically converted from 0-100 → 0-1 during loading (`/ 100`)
- All calculations use fractional scale for mathematical correctness:
  - **Transmission**: 0.0 = no transmission, 1.0 = full transmission
  - **QE values**: 0.0 = no response, 1.0 = full quantum efficiency  
  - **Reflectance**: 0.0 = no reflection, 1.0 = full reflection
  - **Filter Combinations**: Natural multiplication (0.5 × 0.8 = 0.4)
- UI Display: Values converted back to percentages (`* 100`) only for display

### Core Components
- **Filters, QE, Illuminants, Reflectors, Film profiles**: Loaded from TSV files (stored as 0-100%), interpolated to standard grid (300-1100nm, 1nm steps), converted to 0-1 scale for internal use
- **FilterCollection**: Matrix-based storage for fast vectorized calculations (all values in 0-1 scale)
- **AppData**: Typed container for all loaded data (filters, QE curves, illuminants, reflectors, sensor profiles)
- **AppState**: plain dataclass state containers in `services/state_manager.py` — no listener machinery; views trigger their own refresh via the `on_change` callback they pass around
- **Constants System**: All configuration, UI text, and error messages centralized in `models/constants.py`
- **Template System**: Reusable error message and UI component templates for consistency

### Sensor/Film Profile System
Unified `SensorProfile` hierarchy in `models/core.py` models both digital sensors and film stocks. Every profile exposes `display_name`, `sensor_type`, `get_channels()` (→ `{'R','G','B'}` arrays) and `is_film`:

- **RGBSensorProfile** (`rgb_sensor`): standard digital camera with independent R/G/B QE curves
- **ColorSlideFilmProfile** (`color_slide`): reversal film (Ektachrome, Velvia); dye-layer sensitivities, positive image — no display inversion
- **ColorNegativeFilmProfile** (`color_negative`): C-41/ECN-2 negative film; `invert_for_display = True`, so visualization applies `1 - response` before rendering
- **BWFilmProfile** (`bw`): single sensitivity curve mapped to all three RGB keys; visualization collapses the identical traces into one grey "Luminance" trace

Profiles load from `data/film_data/` TSV files with `# Key: Value` headers (`Manufacturer`, `Name`, `Type` required; `ISO`, `Process`, `Notes` optional — see `FILM_TSV_FIELDS`) and are stored in `AppData.sensor_profiles` keyed by display name. Human labels come from `SENSOR_TYPE_LABELS`.

### Presenter / View-Model Pattern
`services/presenter.py` is framework-agnostic and bridges business logic and UI:

- Computes everything a view needs (charts, metrics, messages) and bundles it in typed view-model dataclasses: `FilterAnalysisVM`, `SensorAnalysisVM`, `TransmissionMetricsVM`, `WhiteBalanceVM`, `ScenePreviewVM`/`SceneSwatchVM`
- Views in `views/` contain **zero computation** — they render view models and dispatch user input
- Imperative actions (report generation) are handled by `views/actions.py`, which calls services directly (`services.visualization` for PNG reports, `services.importing` for staged imports)

### Feature Map
- **Advanced Filter Search**: manufacturer/hex-color/wavelength-transmission criteria with live sparkline previews (`views/shelf_filters.py`; compute in `presenter.search_filters`, previews via `presenter._create_filter_sparkline`)
- **Advanced Reflector Search**: filter by organization or target type — only bites on reflector metadata, which the stock plant TSVs don't carry (`views/shelf_reflectors.py`; compute in `presenter.search_reflectors`, previews via `presenter._create_reflector_sparkline`)
- **Scene Preview**: card below the Sensor response card with one render swatch per surface, following the camera-look toggle (filmic tone curve + chroma boost, default on; negative film is inverted). Each swatch carries a reflectance sparkline and a partial-data flag when >10% of the chain's spectral weight (`SCENE_COVERAGE_LIMIT`) lands on wavelengths without reflectance data; the popover shows the active render's linear R/G/B. WB + remove live in the swatch popover (`views/scene_preview.py`; compute in `presenter.prepare_scene_preview`, look via `calculations.apply_camera_look`)
- **My Filters Manager**: curate owned filters, persisted to `data/filters_data/my_filters.json`, with a "My Filters only" sidebar toggle
- **White Balance from Surface Reference**: pick any loaded reflector as WB source (popover on a Scene Preview swatch); gains recompute against it (`AppState.recompute_white_balance`, `wb_reference_surface` state)
- **Film Emulation Target**: overlay a film profile's response as a target with its own independent filter stack and white balance
- **Imports**: per-card CSV upload expansion cards on the Data shelf (`views/shelf_data.py`); dispatched via `views/actions.py` → `services/importing.py`

## Data Flow

1. **Data Loading**: TSV files (0-100% format) loaded from disk, automatically converted to 0-1 scale (`/ 100`), then cached
2. **State Management**: User selections mutate the plain `AppState` dataclass; the owning view calls its `on_change` callback to re-render
3. **Calculations**: Performed using NumPy vectorization on 0-1 scale data for mathematical correctness
4. **UI Display**: Internal 0-1 values converted to percentages (`* 100`) at display layer only
5. **Export**: PNG reports show percentage scale (0-100%) for user readability
6. **Caching**: Processed data cached as npz (one file per dataset) in 0-1 format for performance, invalidated when source files change

## Conventions

- **Scale discipline**: internal calculations use 0-1 fractional values; user/file percentages enter via `/ 100`; `* 100` only at the display layer
- **Function placement**: related functionality lives in the appropriate service module; views contain no computation
- **Error messages**: centralized templates from `models/constants.py`
- **Import organization**: stdlib, third-party, then local imports
- **Docstrings**: always state scale expectations

## Extending the Project

### Adding New Data Types
- Define dataclass in `models/core.py` with 0-1 internal scale documented
- Add loader in `services/data.py` that:
  - Reads TSV files expecting 0-100% values
  - Converts to 0-1 scale (`/ 100`) immediately after loading
  - Stores in 0-1 format internally
- Add importer in `services/importing.py` that saves as 0-100% to TSV files
- Update UI in `views/` with percentage display (`* 100`) at render time
- Add constants to `models/constants.py` for configuration
- Follow established import organization patterns (stdlib, third-party, local)

### Adding New Calculations  
- Implement in `services/calculations.py` using 0-1 scale internally
- Document scale expectations in function docstrings
- Use vectorized NumPy operations for performance
- Apply percentage conversion only at UI display layer

### Adding New UI Components
- Place in the appropriate `views/` module; there is no shared `ui_utils.py` — create a new module if a component is reused across shelves
- Use constants from `models/constants.py` for text and configuration
- Follow established error handling patterns with centralized templates
- Convert internal 0-1 scale to percentages for user display

## Testing

Standard-library `unittest` suite in `tests/` — run `python -m unittest discover` from `program/` (183 tests covering calculations, data loading, importing, presenter, spectra, state, and report rendering). Report tests compare against golden outputs via `numpy.testing`. Harness strategy beyond this suite is an open decision.

## Deployment

- Standard Python/NiceGUI web application
- Launch: `python program/app.py` (runs on http://127.0.0.1:8080 by default)
- Requirements: Python 3.10+, see `requirements.txt`
- Data files go in `program/data/` subfolders (TSV format, 0-100% scale): `filters_data/`, `QE_data/`, `film_data/`, `illuminants/`, `reflectors/`
- Cache is auto-managed in `cache/` (one `.npz` per dataset, stores processed 0-1 scale data; silently rebuilt when any source `.tsv` or the loading code changes)
- Stop the app with Ctrl+C in the console (or close the console window) — closing the browser tab alone leaves the server running

## Troubleshooting

### Common Issues
- **Data doesn't show**: Check TSV file format; if issues persist, delete `program/cache/` — it rebuilds automatically on the next start
- **Wrong scale display**: Verify 0-1 internal / 0-100% display conversion is applied correctly  
- **Calculation errors**: Check that all inputs use consistent 0-1 fractional scale
- **UI inconsistencies**: Ensure percentage conversion (`* 100`) is applied at display layer only
- **Import errors**: Verify import organization follows established patterns (stdlib, third-party, local)

### Scale Debugging
- **TSV files on disk**: 0-100 percentage values (e.g., 85.3, not 0.853)
- **After loading (internal)**: 0-1 fractional values everywhere
- **UI displays**: 0-100% (conversion at display layer only)
- **Filter combinations**: natural multiplication in 0-1 scale (0.5 × 0.8 = 0.4)
- **Data loaders**: divide by 100 unconditionally — sole exception: the film-profile loader keeps a `> 1.5` dual-scale guard for legacy 0-1 files (see the scale-convention comment in `models/constants.py`)
- **Importers**: save as 0-100% to files, clamping negatives to 0

### Performance Issues
- **Data loading**: cache lives in `cache/` as one `.npz` per dataset; delete the folder to force a cold reload

## Design Principles

The refactor arc (S0–S6) landed the six principles below as standing rules.
Endpoint decision: **Lean** — keep every feature; the maximal "7,000-cut"
endpoint (feature removal) was rejected and stays rejected unless revisited
explicitly (S8 unused). Optional remainder: S7, pandas removal — high effort,
a real rewrite of data + search, not planned.

### Ideal-shape principles
1. **Single-source physics** — every spectral multiplication chain
   (T × QE × illuminant [× reflector]) lives in `services/spectra.py`;
   `calculations.py` is thin public wrappers; presenter and visualization never
   multiply spectra inline.
2. **Plain state** — `state_manager.py` is a typed dataclass plus one
   `refresh()` path; no listener machinery; one key per concept.
3. **One importer pipeline** — generic parse → validate → interpolate → write in
   `services/importing.py`, plus thin per-type metadata adapters (filters,
   illuminants, camera QE, film, reflectance).
4. **Lean dependencies** — numpy, plotly, nicegui stay; pandas is structural,
   keep it; matplotlib stays only while PNG reports stay (Plotly PNG export
   would need kaleido ≈ Chromium, so matplotlib is the light option).
5. **Views render only** — zero computation; search/filter logic lives behind
   the presenter.
6. **Feature cuts are explicit decisions**, never side effects of refactoring.

### Open items & known debts

- **Test harness**: `unittest` suite covers the core services (see §Testing); broader strategy (CI, coverage gaps) undecided
- **Reporting stays a service**: PNG report generation lives in `services/visualization.py` (driven from `views/actions.py`); do not fold it into the presenter
- **Presenter VMs carry `go.Figure` objects** — couples view models to Plotly; revisit only on a UI-framework swap
- **Per-shelf manufacturer option lists and reflector target-type lists are built in views** — should come from presenter/constants
- **`my_filters_only` toggle applies only in the quick-search branch**, not advanced search (product decision pending)
