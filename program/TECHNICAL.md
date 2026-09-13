# FS FilterLab - Technical Overview

This is a quick technical reference for developers working on or extending the project.

## Project Structure

```
app.py       # NiceGUI application entry point
models/      # Data structures and constants
  - constants.py   # Centralized configuration, error messages, UI text
  - core.py        # Filter, FilterCollection, Reflector*, SensorProfile hierarchy, AppData
services/    # Framework-agnostic data processing and logic
  - calculations.py    # Mathematical operations, transmission metrics
  - visualization.py   # Chart generation (Plotly + Matplotlib)
  - presenter.py       # View-model builder; advanced search compute
  - data.py            # TSV loading, caching, data normalization
  - importing.py       # Data import (filters, film profiles, QE, ECOSIS)
  - app_operations.py  # Report generation and utility operations
  - channel_mixer.py   # RGB channel matrix manipulation
  - state_manager.py   # Application state management
views/       # NiceGUI UI components (zero computation — render view models only)
  - main_content.py   # Primary display area
  - sidebar.py        # Control panels and inputs
  - forms.py          # Advanced search, import dialog, My Filters manager
  - actions.py        # Imperative action handlers (report generation)
  - channel_mixer.py  # Channel mixer panel
  - ui_utils.py       # Reusable UI components (color previews, error handling)
data/        # Spectral data files (TSV format)
  - filters_data/     # Filter transmissions (+ my_filters.json)
  - QE_data/          # Camera sensor QE curves
  - film_data/        # Sensor/film profiles (digital, slide, color negative, B&W)
  - illuminants/      # Light source spectra
  - reflectors/       # Surface reflectance library (incl. ECOSIS imports)
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

- Computes everything a view needs (charts, metrics, messages) and bundles it in typed view-model dataclasses: `FilterAnalysisVM`, `SensorAnalysisVM`, `TransmissionMetricsVM`, `WhiteBalanceVM`
- Views in `views/` contain **zero computation** — they render view models and dispatch user input
- Imperative actions (report generation) are handled by `views/actions.py`, which maps action types to operations in `services/app_operations.py`

### Feature Map
- **Advanced Filter Search**: manufacturer/hex-color/wavelength-transmission criteria with live sparkline previews (`views/forms.py`)
- **Advanced Reflector Search**: filter by organization, package, or target type (`views/forms.py`)
- **My Filters Manager**: curate owned filters, persisted to `data/filters_data/my_filters.json`, with a "My Filters only" sidebar toggle
- **White Balance from Surface Reference**: pick any loaded reflector as WB source; gains recompute against it (`AppState.recompute_white_balance`, `wb_reference_surface` state)
- **Film Emulation Target**: overlay a film profile's response as a target with its own independent filter stack and white balance
- **Import Dialog**: five tabs — Filters, Illuminants, Camera QE, Film, Reflectance/ECOSIS (`IMPORT_TABS` in constants, panels in `views/forms.py`)

## Data Flow

1. **Data Loading**: TSV files (0-100% format) loaded from disk, automatically converted to 0-1 scale (`/ 100`), then cached
2. **State Management**: User selections mutate the plain `AppState` dataclass; the owning view calls its `on_change` callback to re-render
3. **Calculations**: Performed using NumPy vectorization on 0-1 scale data for mathematical correctness
4. **UI Display**: Internal 0-1 values converted to percentages (`* 100`) at display layer only
5. **Export**: PNG reports show percentage scale (0-100%) for user readability
6. **Caching**: Processed data cached as npz (one file per dataset) in 0-1 format for performance, invalidated when source files change

## Code Quality & Architecture

### Recent Improvements (August-September 2026)
- **Sensor/Film Profile System**: Unified `SensorProfile` hierarchy (digital sensor, slide film, color negative with display inversion, B&W film) replacing the flat QE dict
- **Film Emulation Target**: Film response overlay with its own filter stack and independent white balance
- **Presenter/View-Model Refactor**: All computation moved to `services/presenter.py`; views render typed view models with zero computation
- **My Filters Manager**: Curate and persist owned filters (`my_filters.json`) with a "My Filters only" toggle
- **White Balance from Surface Reference**: Use any loaded reflector as the white balance source
- **Advanced Reflector Search**: Filter reflectors by organization, package, or target type
- **Film Import Tab**: Import dialog now covers Filters, Illuminants, Camera QE, Film, Reflectance/ECOSIS
- **Feature Removal**: Dropped the multispectral band-composite image preview (UI, services, sample data)

### Earlier Improvements (April-May 2026)
- **UI Framework Migration**: Migrated from Streamlit to NiceGUI for better layout control and async support
- **File Upload Fix**: Updated to NiceGUI's async file upload API (e.g., `await e.file.read()`)
- **Scale Consistency Enforcement**: 
  - Removed legacy 0-1 auto-detection (`> 1.5` checks) from data loaders
  - Standardized all file storage to 0-100%, all internal processing to 0-1
  - Updated all import functions to handle WebPlotDigitizer artifacts (clamp negatives to 0)
- **Comprehensive Cleanup**: Eliminated deprecated/unused code across entire codebase
- **Constants Consolidation**: Moved all configuration, UI text, and error messages to `models/constants.py`
- **Import Organization**: Standardized import structure (stdlib, third-party, local)
- **Function Organization**: Moved functions to appropriate modules (e.g., UI utilities to `views/ui_utils.py`)
- **Documentation**: Added clear scale expectations throughout codebase and updated technical docs

### Development Patterns
- **Internal calculations**: Always use 0-1 fractional scale
- **UI conversions**: Apply `* 100` only at display layer
- **Input processing**: Convert user percentages with `/ 100` 
- **Function organization**: Keep related functionality in appropriate service modules
- **Error messages**: Use centralized templates from constants
- **Import structure**: Group by standard library, third-party, then local imports

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
- Test data loading with sample TSV files (ensure 0-100% format in files)
- UI can be tested by running the app and trying different workflows
- Verify scale conversions: files (0-100%) → internal (0-1) → display (0-100%)

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
- **TSV files on disk**: Should contain 0-100 percentage values (e.g., 85.3, not 0.853)
- **After loading (internal)**: Should be 0-1 fractional values in all data structures
- **UI displays**: Should show 0-100% with proper labels (conversion at display layer)
- **Filter combinations**: Should use natural multiplication in 0-1 scale (0.5 × 0.8 = 0.4)
- **Export reports**: Should show 0-100% scale for user readability
- **Data loaders**: Should divide by 100 unconditionally (no conditional normalization)
- **Importers**: Should save as 0-100% to files, clamping negatives to 0

### Performance Issues  
- **Data loading**: Cache lives in `cache/` as one `.npz` per dataset; delete the folder to force a cold reload
- **Calculations**: Verify vectorized NumPy operations are used
- **UI responsiveness**: Large filter selections may impact performance

## North-Star Architecture (Refactoring Spec — S0)

> Added 2026-09-11. This section is the S0 deliverable: the target the refactoring
> roadmap codes against. Status legend: ⬜ not started · ◐ partial · ✅ done.

### Goal
Shrink the codebase from the verified 8,115-line baseline (2026-09-11) toward the
ideal thin shape. Two candidate endpoints — **decided 2026-09-11: Lean**:
- **Lean (roadmap default, ≈ 4,000–4,500 lines): ✅ CHOSEN 2026-09-11.** Ship every
  phase below, keep all features.
- **Maximal (≈ 1,100 lines, the "7,000 cut"):** rejected — would additionally remove
  features (channel mixer, PNG report, possibly film emulation) and collapse the
  views layer to a minimal renderer. S8 stays unused unless this is revisited.

### Ideal-shape principles
1. **Single-source physics** — every spectral multiplication chain
   (T × QE × illuminant [× reflector]) lives in `services/spectra.py`;
   `calculations.py` is thin public wrappers; presenter and visualization never
   multiply spectra inline.
2. **Plain state** — `state_manager.py` becomes a typed dataclass plus one
   `refresh()` path; no listener machinery; one key per concept (collapse the
   `active_profile` / `current_qe` / `selected_camera` triple).
3. **One importer pipeline** — generic parse → validate → interpolate → write in
   `services/importing.py`, plus thin per-type metadata adapters (filters,
   illuminants, camera QE, film, reflectance/ECOSIS).
4. **Lean dependencies** — keep numpy, plotly, nicegui; scipy and requests
   eliminated in S1 (→ `np.interp` helper and `urllib`); pandas is structural,
   keep it (honest tradeoff); matplotlib stays only if PNG reports stay (Plotly
   PNG export would need kaleido ≈ Chromium, so matplotlib is the light option).
5. **Views render only** — zero computation; search/filter logic moves behind
   the presenter.
6. **Feature cuts are explicit decisions**, never side effects of refactoring.

### Verified baseline (2026-09-11)
| File | Lines | Roadmap touchpoint |
|---|---|---|
| services/importing.py | 1,153 | S1 (deps), S5 done (unify) |
| views/forms.py | 888 | S6 done (search behind presenter) |
| services/visualization.py | 860 | S2a ✅, S6 done (docstring compaction) |
| services/data.py | 938 | S4 done (npz cache), S7 (pandas) |
| models/constants.py | 474 | shared text, no phase |
| views/sidebar.py | 436 | S6 done (importer toggle dedupe) |
| services/calculations.py | 425 | S2a ✅ (thin wrappers now) |
| services/state_manager.py | 366 | S3 done (dataclass + refresh) |
| models/core.py | 342 | S3 (plain containers) |
| services/presenter.py | 457 | S2b ✅, S6 done (search compute landed here) |
| views/main_content.py | 300 | S6 verified (presenter-driven, no change) |
| services/spectra.py | 275 | domain layer (S2 home) |
| services/app_operations.py | 267 | S4 done |
| views/ui_utils.py | 174 | S6 verified (no change) |
| app.py | 107 | S4 done (rebuild/shutdown dance removed) |
| channel_mixer (services + views) | 183 | S8 cut candidate |
| views/actions.py | 71 | S4 done |
| `__init__` files | 41 | — |
| **Total** | **7,757** | |

Estimated savings: S1 ~15 (+2 deps gone) · S2b ~30 · S3 ~450–550 · S4 ~300–400 ·
S5 ~600–700 · S6 ~800–900 (incl. visualization cleanup) · S7 ~800–1,200.
Roadmap total ≈ 3,300–4,200 → lands ≈ 4,000–4,800 lines. The gap to ≈ 1,100 is
feature cuts plus views consolidation (principle 6) — not reachable by
refactoring alone.

### Roadmap status (live)
| Phase | Scope | Status |
|---|---|---|
| S0 | this spec | ✅ 2026-09-11 |
| S1 | scipy → np.interp, requests → urllib | ✅ 2026-09-11 — golden-verified (bit-exact / ≤ 7.3e-12); requirements 7 → 5; suite 62/62 (see log) |
| S2 | domain layer: spectra.py; route calculations + presenter film block + visualization; unit tests | ✅ 2026-09-11 — calculations ✅ · visualization ✅ (S2a) · presenter film block ✅ · committed tests ✅ `tests/test_spectra.py` (S2b) |
| S3 | state simplification (dataclass, single refresh, collapse sensor triple-key, delete listeners) | ✅ 2026-09-11 — plain `AppState` dataclass (706 → 366 lines); triple-key collapsed to derived properties; listeners, generic get/set, legacy `select_camera`, singleton deleted; 28 new tests (`tests/test_state.py`) |
| S4 | measure cold load; pickle → npz + silent mtime invalidation; remove "Rebuild Cache" UX + reload/shutdown dance | ✅ 2026-09-11 — cold 2.0–2.7 s → 1.75 s, warm 170–305 ms → 168–212 ms; one `.npz` per key, skew-free kernel-mtime invalidation; rebuild button/action/operation/constants + auto-shutdown machinery deleted; 18 new tests (`tests/test_cache.py`), suite 80/80 (see log) |
| S5 | unify importer: generic pipeline + five thin adapters | ✅ 2026-09-11 — two shared pipelines (`_extrapolated_series`, `_resample_channel`) + `_import_errors`/`_parse_upload`; adapters thin; ECOSIS format-locked, untouched; golden-verified byte-exact incl. loader round-trips; 3 deliberate fixes (Reflectance column contract, illuminant sort, clamp-log count); importing.py 1,175 → 1,153 (see log) |
| S6 | search behind presenter; forms.py slims; visualization cleanup | ✅ 2026-09-13 — search compute moved to presenter (`search_filters`/`search_reflectors`); forms delegates + dedupes re-render via `_set_criteria`; visualization docstring compaction, logic untouched; sidebar importer toggle single button; forms 1,006 → 888, visualization 985 → 860, sidebar 441 → 436, presenter 338 → 457 (+119) — net −129, far short of the −800–900 estimate; 15 new tests (`tests/test_search.py`), suite 95/95 (see log) |
| S7 (opt) | pandas removal | ⬜ high effort, real rewrite of data + search |
| S8 (opt) | feature cuts toward the maximal endpoint (channel mixer first — smallest feature, widest blast radius) | ➖ n/a — Lean endpoint chosen 2026-09-11 |

### Progress log
- **2026-09-11 — S2a shipped.** `services/calculations.py` now delegates
  combine/validity/WB/integration to `services/spectra.py` (private helpers
  `_compute_combined_transmission` and `_is_valid_transmission` deleted; public
  API and signatures unchanged, all consumers untouched). `services/visualization.py`
  delegates display inversion via `invert_for_display`. Verified by a 32/32
  parity regression against embedded pre-refactor originals: plain WB bitwise
  identical, surface WB ≤ 1 ulp (2.2e-16), reflector color ≤ 8e-15, full
  consumer import chain loads. Bookkeeping notes: this work was labeled "S1" in
  session logs — per this roadmap it is S2 part 1 (S2a); S2b = presenter film
  block + committed tests. Line delta so far: net +145 (spectra.py added,
  duplication deleted) — line cuts arrive with S3+.
- **2026-09-11 — S2b shipped; S2 complete.** The film-emulation block in
  `presenter.prepare_sensor_analysis` moved into a module-level helper
  `_compute_film_emulation_target` (presenter.py:200) that delegates the
  response chain to `spectra.sensor_channel_curves` — the multiply order
  (`t·s → ·ill → ·scale`) is left-associative identical to the old inline
  `film_filter * sens * illuminant * 100`, so the swap is bitwise-safe and
  the block is now directly unit-testable without chart machinery.
  Committed the regression suite as `tests/test_spectra.py` — stdlib
  `unittest`, zero new deps (pytest deliberately not added; lean-deps
  principle from S0). 34 tests / 48 checks: combine+clip per-path, plain
  and surface WB bitwise vs embedded references, RGB response semantics
  (gain flooring, `nan_to_num`, visibility, mixer off), per-wavelength
  chain + NaN-aware integral, film-target helper (with-WB oracle
  reproducing the original inline block, non-positive-gain guard),
  signature lock on all 14 public functions, consumer import chain.
  Run: `python -m unittest discover -s tests` from `program/` — 34/34 OK
  in 0.05 s. Line delta: presenter.py 312 → 341 (+29, helper + docstring);
  tests add 506 lines of guardrails against future drift in spectra.py.
  Next: S1 (dep removal), then S3 — blocked on the Lean vs Maximal
  endpoint decision.
- **2026-09-11 — Endpoint decided: Lean.** User picked the roadmap-default Lean
  endpoint (~4,000–4,500 lines, every feature kept). S8 (feature cuts) is n/a.
  This unblocks S3: state simplification proceeds with all feature state
  (film emulation, channel mixer, PNG report) retained — S3 is structural only.
- **2026-09-11 — S3 shipped.** `services/state_manager.py` rewritten from the
  706-line dict-backed `StateManager` into a 366-line plain `@dataclass
  AppState`: per-property pairs became real fields (fresh-collection
  defaults); `_store`, the listener machinery (it had zero registrations),
  generic `.get()`/`.set()`, legacy `select_camera`, and the dead
  `import_status`/`import_error_message` keys are deleted. Sensor triple
  collapsed: `current_qe`/`selected_camera` are read-only derived properties
  of `active_profile`; `invert_output` also deleted (no callers since S2a
  routed display inversion through `spectra.invert_for_display`). The
  `get_state_manager()` singleton is gone — `app.py` constructs `AppState()`
  directly; 5 `StateManager` type-hint sites renamed across
  `presenter.py`/`app_operations.py`. Refresh stays a view concern via the
  existing `on_change` callback — that already was the single refresh path;
  state-side listeners were never registered by anyone. WB / selection /
  My Filters / default-reflector behaviour preserved verbatim (My Filters
  JSON format unchanged, `sorted()` on write kept). New
  `tests/test_state.py` (416 lines, 28 stdlib tests): defaults freshness,
  the derived triple, selection helpers, every WB branch with monkeypatched
  calculators, JSON persistence round-trips in temp dirs, and guards that
  the deleted machinery stays deleted. Full suite 62/62 OK; consumer import
  chain (`app` → presenter → app_operations → state_manager) loads.
  Line delta: state_manager.py −340 (below the ~450–550 estimate because the
  ~200 lines of behaviour-bearing WB/persistence/selection methods were kept
  intact rather than reworked); tests add 416 guardrail lines. Next: S1
  (dep removal), then S4.
- **2026-09-11 — S1 shipped.** scipy and requests eliminated from
  `services/importing.py`: `_interpolate_spectrum` now uses `np.interp` with
  NaN-blanking outside the data range (plus an ascending-order guard matching
  scipy's `assume_sorted=False`), and a new `_interp_with_extrapolation` helper
  reproduces `interp1d(fill_value='extrapolate')` via slope extension for the
  ECOSIS grid fill. `_fetch_ecosis_api_metadata` switched to
  `urllib.request.urlopen` (timeout kept at 30 s; HTTPError/URLError/timeout
  mapped onto the same four user-facing error strings; the dead
  `if requests is None` guard deleted — the import was unconditional anyway).
  Verified with a one-shot golden harness: 124 spectra × 4 extrapolation-flag
  combos + 120 ECOSIS grids recorded against the scipy build —
  `_interpolate_spectrum` bit-exact (max diff 0, NaN positions identical),
  ECOSIS max |diff| 7.3e-12 (ulp-level; vanishes in the 3-decimal rounding
  applied before TSV write). Full suite 62/62 OK; grep confirms no
  scipy/requests imports remain. `requirements.txt` 7 → 5 deps (nicegui,
  pandas, numpy, plotly, matplotlib) — drops ~100 MB of installed wheels.
  Line delta: importing.py 1,141 → 1,175 (+34: extrapolation helper + sort
  guards vs removed dep code). Dependency-audit verdicts:
  pandas/matplotlib/plotly/nicegui stay per principle 4. Two recorded
  UI-swap-readiness debts for the "NiceGUI is swappable" goal: (1) presenter
  VMs carry `go.Figure` objects (`filter_response_fig`, `sensor_response_fig`)
  — a future UI swap should carry raw (x, y, labels) instead; (2) DataFrame
  search/filter logic lives in `views/forms.py` — already roadmap-tracked as
  S6. nicegui imports are otherwise confined to `app.py` + `views/`. Next: S4.
- **2026-09-11 — S4 shipped.** Data cache moved from pickle to npz:
  `_cached_loader` now writes one `{key}.npz` per dataset — numeric arrays as
  npz entries, string/scalar metadata in a JSON blob inside the same file,
  `allow_pickle=False` end to end — via per-loader typed `pack`/`unpack`
  serializer pairs. FilterCollection, the QE tuple, illuminants,
  ReflectorCollection and the film-profile hierarchy all round-trip exactly
  (df dtypes preserved incl. the bool `is_lee` column; film subclasses
  restored via their `sensor_type` tag). Invalidation is a skew-free kernel-
  mtime comparison: the cache file's own mtime vs the newest source `.tsv`
  (data.py included, so loader-code changes invalidate too); ties count as
  valid (make semantics), and missing/stale/corrupt caches rebuild silently.
  Measured (1,578 filters): cold 2.0–2.7 s → 1.75 s, warm 170–305 ms →
  168–212 ms — npz parity, no warm-path regression; `cache/` holds 5 files
  instead of 10. Removed the "Rebuild Cache" UX end to end (sidebar button,
  `actions.py` branch with its `location.reload()` dance,
  `rebuild_application_cache()`, four constants) and the `app.py`
  auto-shutdown machinery (30 s grace, reconnect tracking); `main()` is now
  data init + UI render + `ui.run(...)`. Behaviour note: the server keeps
  running when the browser tab closes — stop it with Ctrl+C in the console
  (documented in README + Deployment). New `tests/test_cache.py` (18 stdlib
  tests): per-payload npz round-trips, format/mtime/corruption rejection,
  loader hit/invalidate/recover flows, pickle-stays-deleted guard. Full
  suite 80/80 OK (62 existing + 18 new). Line delta: data.py 770 → 937 (the
  typed serializers are the price of dropping pickle), app.py 148 → 111,
  actions.py 89 → 71, app_operations.py 292 → 267, sidebar.py −6,
  constants.py −8 — net +73; S5/S6 carry the remaining cuts. No migration
  needed for legacy `.pkl` files: `cache/` is gitignored and stale files
  are simply ignored (delete the folder for a clean start). Next: S5.

- **2026-09-11 — post-S4 debt sweep.** Removed the redundant cache-dir
  bootstrap from `app.py` (`_cached_loader` self-creates `program/cache/`,
  so `main()`'s `mkdir` + `CACHE_DIR` import were dead) and the dead
  `SPECTRAL_CONFIG['normalization_threshold']` entry ("legacy, no longer
  used"). Stale-comment fixes: sidebar's "legacy default_key" is a live
  `AppData` field; the redundant `current_qe is None` check folded into
  `active_profile is None` (S3 derived property); data.py's "legacy format"
  wording; test_cache's meta description (format tag only — write time is
  the file mtime); two blank-line seams in data.py. File table re-synced to
  measured counts — state_manager 706 → 366 and app_operations 292 → 267
  had gone stale after S3/S4: total 8,115 → 7,914. Suite 80/80 OK,
  py_compile clean. Next: S5.

- **2026-09-11 — debt sweep tranche 2 (S2/S3 files).** Audited
  state_manager, core, presenter, spectra, calculations; only presenter
  and core held remnants. Removed presenter's dead `logging`/`logger`
  (zero log calls in the module) and unused `field`/`Any` imports, and
  fixed its "Application data dict" docstring (S3 made it a typed
  `AppData` container); dropped core.py's dead `INTERP_GRID` import
  (only a docstring mention — the prose stays). state_manager, spectra
  and calculations verified fully live, including `get_display_names()`
  (5 call sites). Line delta: presenter 341 → 338, core 345 → 342:
  total 7,914 → 7,908. Suite 80/80 OK, py_compile clean. Next: S5.

- **2026-09-11 — S5 shipped: importer unification.** The five
  single-spectrum importers in `services/importing.py` now run through two
  shared pipelines: `_extrapolated_series` (validate → optional 0-100 cap /
  negative clamp → sort → 5nm-rounded range with optional edge
  extrapolation → interpolate; filters, reflectance) and
  `_resample_channel` (trim → validate → optional cap → clamp → sort →
  INTERP_GRID resample with per-type edge-fill and optional peak
  normalisation; illuminants, camera QE, film channels). The three
  serialization styles stay per-type — they are distinct loader contracts
  (tall TSV with broadcast metadata columns / first-row metadata /
  comment-header TSV). A `_import_errors` decorator centralises the
  (success, message) UI contract and `_parse_upload` the CSV error prefix,
  leaving the five adapters thin. ECOSIS (~45% of the file: wide-CSV
  geometry, per-sample naming, custom float precision, API attribution) is
  format-locked and untouched — asserted source-identical by the golden
  harness. Verification: old vs new module run side-by-side on sandboxed
  output dirs — 13 happy paths byte-identical (extrap-flag combos, log10
  sensitivity, ragged/reversed channels, comma separator, plant/other
  routing), 11 error paths message-identical, and every new output
  re-parsed by the real `data.py` loaders. Three deliberate fixes:
  reflectance/absorption imports now always write the value column as
  'Reflectance' — the loader requires it, so 'Absorption' files were
  silently skipped before (the chosen type stays in the status message);
  the illuminant importer sorts by wavelength before `np.interp`
  (descending CSVs previously produced a wrong curve); the filter
  negative-clamp log counted after clipping (always logged 0). Line delta:
  importing.py 1,175 → 1,153 (−22 — the −600 roadmap estimate ignored that
  the ECOSIS block is format-locked). Suite 80/80 OK, py_compile clean.
  Next: S6.

- **2026-09-13 — S6 shipped: search behind the presenter.** Advanced
  search compute moved from `views/forms.py` into a new presenter
  section: `search_filters()` (manufacturer, trans-@-λ, multi-criterion
  AND wavelength matching, My-Filters-only via AppState's owned set, all
  five sort modes incl. Hex-Rainbow and trans-@-λ) and
  `search_reflectors()` (text match across orgs/packages/targets).
  forms.py now renders only: `_apply_filter_search` /
  `_apply_reflector_search` delegate to the presenter; criterion-row
  callbacks take a `ctx` bundle instead of loose args; add/remove
  re-render deduped into `_set_criteria()`. visualization.py: 14
  surgical docstring/comment compactions, zero logic changes.
  sidebar.py: `_data_management_panel` if/else collapsed to one button
  with a computed key. Honest delta vs the −800–900 estimate: forms
  1,006 → 888, visualization 985 → 860, sidebar 441 → 436, presenter
  338 → 457 (+119 — the compute has to live somewhere), main_content
  and ui_utils unchanged → net −129 across the six S6 files, total
  7,886 → 7,757. The estimate ignored that a layering move is mostly
  displacement, not deletion; the real win is that search is now
  unit-testable without a UI. 15 new tests (`tests/test_search.py`:
  helpers, owned-only, every sort mode, reflector search), suite 95/95
  OK, py_compile clean on all five touched files. Next: S7 (optional
  pandas removal) — or stop here; S8 already ruled out.

---

For more details, read the code and docstrings. The structure is meant to be straightforward and easy to follow.
