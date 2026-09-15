# ECOSIS Extraction Record — FS Filter Lab

Extracted 2026-09-15 from branch `ui-rebuild` @ `9f588a4` (line numbers are
pre-deletion). FS Filter Lab no longer contains any of this code; this file is
the single source for rebuilding ECOSIS import as a standalone tool.

## 1. Inventory (pre-deletion locations)

| #  | Location | Unit |
|----|----------|------|
| 1  | `program/services/importing.py:285-442` | `import_ecosis_csv` — wide CSV → per-sample TSV importer |
| 2  | `program/services/importing.py:246-271` | `_interp_with_extrapolation` — linear interp + edge extrapolation (sole caller: #1, line 394) |
| 3  | `program/services/presenter.py:787-817` | `EcosisPreviewVM` + `prepare_ecosis_preview` — in-memory preview parse |
| 4  | `program/views/shelf_reflectors.py:106-159` | `render_ecosis_card` + `_handle_upload` — preview-before-commit card |
| 5  | `program/views/actions.py:593-625` | `_import_ecosis` action handler |
| 6  | `program/views/actions.py:651` | `"import_ecosis"` entry in `_HANDLERS` |
| 7  | `program/views/shelf_reflectors.py:56-57` | Reflectors-shelf expansion wrapper |
| 8  | `program/views/shelf_data.py:51-52` | Data-shelf expansion (second render site) |
| 9  | `program/models/constants.py:61` | `OUTPUT_FOLDERS['ecosis']` |
| 10 | `program/models/constants.py:86-87, 90` | `METADATA_FIELDS`: `name_for_search`, `species`, `package_title` |
| 11 | `program/services/data.py:634-652` | Loader-side name resolution for ECOSIS-written headers |
| 12 | `program/tests/test_importing.py:308-360` | `ecosis_csv` fixture + `EcosisImport` (3 tests) |
| 13 | `program/tests/test_data.py:254-266` | `test_name_for_search_resolves_through_the_column_indirection` |
| 14 | `program/tests/test_presenter.py:17, 101` | preview import + call in `PresenterReadOnly` |
| 15 | `program/tests/helpers.py:168` | `"ecosis"` output folder in `import_sandbox` |
| 16 | `TECHNICAL.md:17, 25, 37, 92, 164, 186, 198` · `USAGE.md:71` · `README.md:60` | doc mentions |

`program/data/reflectors/Ecosis/` never existed on disk — the folder was only
created on first import, and no crawled ECOSIS data was ever retained. The
`plant/*.tsv` reflector files came from the single-spectrum importer and stay.

## 2. File-format contract

### Input: wide ECOSIS CSV (one row per sample)
- Metadata columns: any names (`Sample`, `Organization`, `Species`,
  `Package Title`, ...).
- Wavelength columns: headers that parse as floats, accepted range 200–3000 nm;
  fewer than 10 wavelength columns raises `ValueError`.
- Separator auto-detected (pandas `read_csv(sep=None, engine="python")`).
- Values may be 0–1 fractions or 0–100 percents: if `max(values) <= 1.5` the
  spectrum is scaled ×100 before interpolation, otherwise passed through.

### Processing
1. Sample name: first 3 non-empty metadata cell values joined with `_`,
   sanitized to `[\w\-\.]`, underscore runs collapsed, ≤ 50 chars;
   fallback `Sample_{seq}` (1-based row number).
2. Spectral points: skip empty/NaN cells; ≥ 5 valid points overall and
   ≥ 5 within 300–1100 nm, else the row is skipped (logged warning).
3. Trim to the grid range, then interpolate onto
   `INTERP_GRID = np.arange(300, 1101, 1)` with edge extrapolation (unit 3.2).

### Output: one comment-header TSV per sample
- `# CSV Metadata:` block — `# {col}\t{value}` per metadata column, then a bare `#` line.
- Optional `# name_for_search\t{column_name}` — stores the COLUMN NAME, not the
  value; the loader resolves the indirection (unit 3.7).
- Optional `# relevant_metadata\t{col1|col2}` — **write-only dead weight**: no
  reader anywhere ever consumed it. Do not port to the standalone tool.
- Table: header `Wavelength\tReflectance`, then `{wl}\t{refl:.10f}` for all 801
  grid points, values in 0–100 scale (loaders divide by 100).
- Filename `{sample_name}_{seq}.tsv`; returns list of created paths; raises
  `ValueError` on failure. (All other FS importers return `(bool, str)`.)

### Latent bug worth knowing
The UI handler passed `OUTPUT_FOLDERS["ecosis"]` but never created the folder
(the test does `out_dir.mkdir(...)` with the comment "UI creates it first" —
it didn't). With the folder absent, `open(filepath, "w")` raises
`FileNotFoundError`, surfaced as a generic "Error importing ECOSIS CSV".
Create the output dir in the standalone tool before writing.

## 3. Code units (verbatim, pre-deletion)

### 3.1 `import_ecosis_csv` — program/services/importing.py:285-442
```python
def import_ecosis_csv(file_path: str, output_dir: str, name_column: Optional[str] = None, relevant_metadata: Optional[List[str]] = None) -> List[str]:
    """
    Import ECOSIS CSV file directly. User explicitly chooses this format.
    
    Args:
        file_path: Path to ECOSIS CSV file
        output_dir: Directory to save processed files
        name_column: Optional column name to use for search names (stores column name, not value)
        relevant_metadata: Optional list of column names for Surface Color Preview display
        
    Returns:
        List of created file paths
    """
    try:
        df = pd.read_csv(file_path, sep=None, engine="python")
        
        if df.empty:
            raise ValueError("CSV file is empty")
            
        # Find wavelength columns (numeric headers)
        wavelength_cols = []
        metadata_cols = []
        
        for col in df.columns:
            try:
                wavelength = float(str(col).strip())
                # Accept wavelengths in reasonable spectral range
                if 200 <= wavelength <= 3000:
                    wavelength_cols.append((col, wavelength))
                else:
                    metadata_cols.append(col)
            except (ValueError, TypeError):
                metadata_cols.append(col)
        
        if len(wavelength_cols) < 10:
            raise ValueError(f"Not enough wavelength columns found: {len(wavelength_cols)}. Expected at least 10.")
            
        # Sort wavelength columns by wavelength value
        wavelength_cols.sort(key=lambda x: x[1])
        
        created_files = []
        
        for seq, (index, row) in enumerate(df.iterrows(), start=1):
            # Create sample name from available metadata - use any non-empty metadata columns
            sample_parts = []
            
            # Go through all metadata columns and use the first few non-empty ones
            for col_name in metadata_cols:
                if pd.notna(row[col_name]) and str(row[col_name]).strip():
                    value = str(row[col_name]).strip()
                    if value.lower() != 'nan' and value != '':
                        sample_parts.append(value)
                        if len(sample_parts) >= 3:  # Limit to first 3 meaningful metadata values
                            break
            
            # Fallback to row index if no metadata found
            if not sample_parts:
                sample_parts = [f"Sample_{seq}"]
            
            # Clean sample name for filename
            sample_name = "_".join(sample_parts).replace(" ", "_")
            sample_name = re.sub(r'[^\w\-_\.]', '_', sample_name)
            sample_name = re.sub(r'_{2,}', '_', sample_name).strip('_')
            
            # Limit filename length
            if len(sample_name) > 50:
                sample_name = sample_name[:50]
            
            # Extract spectral data and handle NaN properly
            spectral_wavelengths = []
            spectral_values = []
            
            for col_name, wavelength in wavelength_cols:
                value = row[col_name]
                # Only include data points with valid values
                if pd.notna(value) and value != '':
                    try:
                        float_val = _safe_float(value)
                        if not np.isnan(float_val):
                            spectral_wavelengths.append(wavelength)
                            spectral_values.append(float_val)
                    except (ValueError, TypeError):
                        continue  # Skip invalid values
            
            if len(spectral_wavelengths) < 5:
                logger.warning("Skipping sample %s - insufficient valid data points (%d)", sample_name, len(spectral_wavelengths))
                continue
            
            spectral_wavelengths = np.array(spectral_wavelengths)
            spectral_values = np.array(spectral_values)
            
            # Filter to INTERP_GRID range (300-1100nm) before interpolation
            wl_min, wl_max = INTERP_GRID[0], INTERP_GRID[-1]
            mask = (spectral_wavelengths >= wl_min) & (spectral_wavelengths <= wl_max)
            
            if np.sum(mask) < 5:
                logger.warning("Skipping sample %s - insufficient data in %s-%snm range", sample_name, wl_min, wl_max)
                continue
                
            filtered_wavelengths = spectral_wavelengths[mask]
            filtered_values = spectral_values[mask]
            # Normalize to the 0-100 file convention (services/data.py divides
            # reflectance by 100 on load). ECOSIS CSVs ship 0-1 fractions;
            # 0-100-scaled sources pass through untouched.
            if filtered_values.size and filtered_values.max() <= 1.5:
                filtered_values = filtered_values * 100.0

            
            # Interpolate to standard 1nm grid (300-1100nm)
            interp_values = _interp_with_extrapolation(filtered_wavelengths, filtered_values, INTERP_GRID)
            
            # Prepare TSV content with metadata as header comments
            rows = []
            
            # Add CSV metadata as comments
            if metadata_cols:
                rows.append("# CSV Metadata:")
                for col in metadata_cols:
                    value = row[col] if pd.notna(row[col]) and row[col] != '' else ''
                    if value:
                        rows.append(f"# {col}\t{value}")
                rows.append("#")  # Blank comment line
            
            # Add name_for_search field if user selected a column (store column name, not value)
            if name_column and name_column in metadata_cols:
                rows.append(f"# name_for_search\t{name_column}")
                rows.append("#")  # Blank comment line
            
            # Add relevant_metadata field if user selected columns (store column names, not values)
            if relevant_metadata:
                # Only include columns that exist in this CSV file
                valid_relevant = [col for col in relevant_metadata if col in metadata_cols]
                if valid_relevant:
                    relevant_str = "|".join(valid_relevant)  # Join multiple column names with |
                    rows.append(f"# relevant_metadata\t{relevant_str}")
                    rows.append("#")  # Blank comment line
            
            # Simple data table: just Wavelength and Reflectance
            rows.append("Wavelength\tReflectance")
            
            # Data rows - clean and simple
            for wl, refl in zip(INTERP_GRID, interp_values):
                rows.append(f"{wl}\t{refl:.10f}")
            
            # Save as TSV file
            filename = f"{sample_name}_{seq}.tsv"  # Sequence number avoids duplicates
            filepath = Path(output_dir) / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(rows))
            
            created_files.append(filepath)
            
        logger.info("Successfully processed %d ECOSIS samples", len(created_files))
        return created_files
        
    except Exception as e:
        raise ValueError(f"Error importing ECOSIS CSV: {str(e)}")
```

### 3.2 `_interp_with_extrapolation` — program/services/importing.py:246-271
```python
def _interp_with_extrapolation(wavelengths, values, target_wavelengths):
    """Linear interpolation onto ``target_wavelengths``, extending the first and
    last segments linearly beyond the data range (replaces scipy's
    ``interp1d(..., fill_value='extrapolate')``; call sites guarantee >= 2 points)."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    values = np.asarray(values, dtype=float)
    target = np.asarray(target_wavelengths, dtype=float)
    
    # Match the previous scipy behaviour (assume_sorted=False): sort by
    # wavelength first if the input is not already ascending.
    if np.any(np.diff(wavelengths) < 0):
        order = np.argsort(wavelengths)
        wavelengths, values = wavelengths[order], values[order]
    
    # Interior points; np.interp clamps outside the data range, fixed up below.
    interpolated = np.interp(target, wavelengths, values)
    
    if target[0] < wavelengths[0]:
        slope = (values[1] - values[0]) / (wavelengths[1] - wavelengths[0])
        below = target < wavelengths[0]
        interpolated[below] = values[0] + slope * (target[below] - wavelengths[0])
    if target[-1] > wavelengths[-1]:
        slope = (values[-1] - values[-2]) / (wavelengths[-1] - wavelengths[-2])
        above = target > wavelengths[-1]
        interpolated[above] = values[-1] + slope * (target[above] - wavelengths[-1])
    return interpolated
```

SciPy-free replacement for `interp1d(..., fill_value='extrapolate')`;
call sites guarantee >= 2 points. Port it — plain `np.interp` clamps at
the edges instead of extrapolating.

### 3.3 Preview VM + parse — program/services/presenter.py:787-817
```python
# =============================================================================
# ECOSIS IMPORT PREVIEW (preview-before-commit; commit runs via actions)
# =============================================================================

@dataclass
class EcosisPreviewVM:
    """Parsed summary of an ECOSIS CSV, shown before anything is written."""
    filename: str
    n_rows: int
    n_cols: int
    sample_names: List[str]
    organizations: List[str]
    error: Optional[str] = None


def prepare_ecosis_preview(content: bytes, filename: str = "") -> EcosisPreviewVM:
    """Parse an ECOSIS CSV in memory for display — no files are written."""
    try:
        df = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
    except Exception as exc:
        return EcosisPreviewVM(
            filename=filename, n_rows=0, n_cols=0,
            sample_names=[], organizations=[], error=str(exc),
        )
    orgs = sorted(set(df["Organization"].astype(str))) if "Organization" in df.columns else []
    name_col = "Sample" if "Sample" in df.columns else df.columns[0]
    samples = [str(v) for v in df[name_col].head(8).tolist()]
    return EcosisPreviewVM(
        filename=filename, n_rows=len(df), n_cols=int(df.shape[1]),
        sample_names=samples, organizations=orgs,
    )
```

Standalone needs: `io`, `pandas`, `dataclasses.dataclass`, `typing`.
The card showed "N rows × M columns", up to 6 organizations, the first 8
sample names (preferring a `Sample` column), and the confirm button.

### 3.4 UI card — program/views/shelf_reflectors.py:106-159
```python
def render_ecosis_card(ctx: Dict[str, Any], page_state: Dict[str, Any]) -> None:
    ui.label(
        "Upload an ECOSIS multi-spectrum CSV. Nothing is written to disk until "
        "you confirm the import."
    ).classes("text-caption text-grey-7")

    @ui.refreshable
    def preview_section():
        pvm = page_state.get("ecosis_preview")
        if pvm is None:
            return
        if pvm.error:
            ui.label(f"Could not parse file: {pvm.error}").classes("text-body2 text-negative")
            return
        with ui.card().classes("w-full q-pa-sm"):
            ui.label(f"{pvm.filename}: {pvm.n_rows} rows × {pvm.n_cols} columns").classes(
                "text-body2"
            )
            if pvm.organizations:
                shown = ", ".join(pvm.organizations[:6]) + (
                    "…" if len(pvm.organizations) > 6 else ""
                )
                ui.label(f"Organizations: {shown}").classes("text-caption text-grey-6")
            tail = "…" if pvm.n_rows > 8 else ""
            ui.label("Samples: " + ", ".join(pvm.sample_names) + tail).classes(
                "text-caption text-grey-6"
            )
            ui.button(
                f"Import {pvm.n_rows} spectra", icon="download_done",
                on_click=lambda: ctx["dispatch"]("import_ecosis", {
                    "content": page_state.get("ecosis_content"),
                    "filename": pvm.filename,
                }),
            ).props("unelevated dense color=positive")

    preview_section()

    ui.upload(
        auto_upload=True,
        on_upload=lambda e: _handle_upload(ctx, e, preview_section),
    ).classes("w-full max-w-xs").props("accept=.csv label='Choose ECOSIS CSV'")

    # Persistent per-card import status, shared with the Data shelf cards.
    # Function-local import: shelf_data imports this module at module level.
    from views.shelf_data import _status_line

    _status_line(ctx["app_state"], "ecosis")


def _handle_upload(ctx: Dict[str, Any], event, preview_section) -> None:
    content = event.content.read()
    ctx["page_state"]["ecosis_content"] = content
    ctx["page_state"]["ecosis_preview"] = prepare_ecosis_preview(content, event.name)
    preview_section.refresh()
```

Wiring that surrounded the card:
- Reflectors shelf: `with ui.expansion("ECOSIS import (preview before commit)", icon="upload").classes("w-full"): render_ecosis_card(ctx, page_state)`
- Data shelf (shelf_data.py:51-52, second render site): `with ui.expansion("ECOSIS CSV (multi-surface)", icon="cloud_download").classes("w-full"): render_ecosis_card(ctx, ctx["page_state"])`, import `from views.shelf_reflectors import render_ecosis_card`
- `page_state` keys: `ecosis_content` (raw uploaded bytes), `ecosis_preview` (EcosisPreviewVM)
- dispatch `"import_ecosis"` payload: `{"content": <bytes>, "filename": <str>}`; per-card status read `AppState.last_import["ecosis"]`

### 3.5 Action handler — program/views/actions.py:593-625
```python
def _import_ecosis(app_state: AppState, data: AppData, payload: Dict) -> Set[str]:
    content: bytes = payload.get("content") or b""
    if not content:
        _record_import(app_state, "ecosis", False, "No ECOSIS file staged.")
        ui.notify("No ECOSIS file staged", type="warning")
        return _IMPORT_FAILED
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".csv", prefix="ecosis_", delete=False
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        created = import_ecosis_csv(tmp_path, OUTPUT_FOLDERS["ecosis"])
    except ValueError as exc:
        _record_import(app_state, "ecosis", False, f"ECOSIS import failed: {exc}")
        ui.notify(f"ECOSIS import failed: {exc}", type="negative")
        return _IMPORT_FAILED
    except Exception:
        logger.exception("ECOSIS import failed unexpectedly")
        _record_import(
            app_state, "ecosis", False, "ECOSIS import failed — see terminal log"
        )
        ui.notify("ECOSIS import failed — see terminal log", type="negative")
        return _IMPORT_FAILED
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
    _reload_reflectors(app_state, data)
    message = f"Imported {len(created)} ECOSIS spectra"
    _record_import(app_state, "ecosis", True, message)
    ui.notify(message, type="positive")
    return _IMPORT_OK
```

Registered as `"import_ecosis": _import_ecosis` in `_HANDLERS`
(actions.py:651). Handler imports it needed: `tempfile`,
`OUTPUT_FOLDERS` (was imported alongside `FILM_TYPES`), and
`import_ecosis_csv` from `services.importing`.

### 3.6 Constants — program/models/constants.py
```python
OUTPUT_FOLDERS = {
    'reports': "program/output",  # Main output directory for generated reports
    'ecosis': "program/data/reflectors/Ecosis",  # ECOSIS import destination
    'filter_import': "program/data/filters_data"  # Filter import destination
}
```

`METADATA_FIELDS` pre-removal (FS keeps `name` / `sample_type` /
`collector`; the other three were ECOSIS-only):
```python
METADATA_FIELDS = {
    'name': 'Name',                    # Display name for the spectrum
    'name_for_search': 'name_for_search',  # Column name to use for naming (not the value)
    'species': 'species',              # Species name (ECOSIS data)
    'sample_type': 'sample_type',      # Sample type classification
    'collector': 'collector',          # Data collector information
    'package_title': 'Package Title'   # ECOSIS package title
}
```

### 3.7 Loader-side name resolution — program/services/data.py:634-656
```python
    # Extract name from metadata, prioritizing name_for_search field
    name = None
    
    # First priority: use the user-selected name_for_search field (now contains column name, not value)
    if METADATA_FIELDS['name_for_search'] in metadata and metadata[METADATA_FIELDS['name_for_search']].strip():
        column_name = metadata[METADATA_FIELDS['name_for_search']].strip()
        # Look up the actual value using the column name
        if column_name in metadata and metadata[column_name].strip():
            name = metadata[column_name].strip()
    
    if not name:
        # Fallback: try common name fields from metadata
        name_fields = [METADATA_FIELDS['species'], METADATA_FIELDS['name'], 
                      METADATA_FIELDS['sample_type'], METADATA_FIELDS['collector'], 
                      METADATA_FIELDS['package_title']]
        for field in name_fields:
            if field in metadata and metadata[field].strip():
                name = metadata[field].strip()
                break
    
    # Final fallback to filename if no name found in metadata
    if not name:
        name = path.stem
```

The generic comment-header parser (reads `# key\tvalue` lines into a
dict) is unchanged and worth copying to the standalone tool for reading
ECOSIS-written TSVs back. FS now resolves the display name as
`Name` → `sample_type` → `collector` → filename stem.

### 3.8 Tests (the behavioral spec)
program/tests/test_importing.py:308-360 — fixture + `EcosisImport` class:
```python
def ecosis_csv(scales):
    """Wide ECOSIS-style CSV: metadata cols + 11 wavelength columns."""
    wl_cols = [str(wl) for wl in range(300, 850, 50)]
    rows = []
    for i, scale in enumerate(scales):
        row = {"Sample": f"Plant{chr(65 + i)}", "Organization": "Uni",
               "Species": "Ficus"}
        for j, wl in enumerate(wl_cols):
            base = 0.8 - j * 0.05 if scale == "fraction" else 80.0 - j * 5.0
            row[wl] = "" if (i == 0 and j == 3) else base  # one interior blank
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


class EcosisImport(TestCase):
    def test_fractions_are_scaled_to_the_0_100_file_convention(self):
        with import_sandbox() as (root, folders, output):
            src = root / "ecosis_source.csv"
            src.write_text(ecosis_csv(["fraction"]), encoding="utf-8")
            out_dir = root / "ecosis_out"
            out_dir.mkdir(parents=True, exist_ok=True)  # UI creates it first
            created = imp.import_ecosis_csv(str(src), str(out_dir),
                                            name_column="Sample",
                                            relevant_metadata=["Species"])
            self.assertEqual(len(created), 1)
            self.assertEqual(created[0].name, "PlantA_Uni_Ficus_1.tsv")
            text = created[0].read_text(encoding="utf-8")
            self.assertIn("# name_for_search\tSample", text)
            self.assertIn("# relevant_metadata\tSpecies", text)
            df = read_tsv(created[0])
            self.assertEqual(list(df.columns), ["Wavelength", "Reflectance"])
            self.assertEqual(len(df), 801)  # 300..1100 on the standard grid
            self.assertClose(float(df["Reflectance"].max()), 80.0)  # 0.8 * 100

    def test_percent_scaled_sources_pass_through_unscaled(self):
        with import_sandbox() as (root, folders, output):
            src = root / "ecosis_source.csv"
            src.write_text(ecosis_csv(["percent"]), encoding="utf-8")
            out_dir = root / "ecosis_out"
            out_dir.mkdir(parents=True, exist_ok=True)  # UI creates it first
            created = imp.import_ecosis_csv(str(src), str(out_dir))
            df = read_tsv(created[0])
            self.assertClose(float(df["Reflectance"].max()), 80.0)
            # No double-scaling: 80 stayed 80 (only <=1.5 data is scaled up).

    def test_fewer_than_ten_wavelength_columns_raises(self):
        with import_sandbox() as (root, folders, output):
            src = root / "tiny.csv"
            src.write_text("Sample,300,400\nPlantA,0.5,0.4\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                imp.import_ecosis_csv(str(src), str(root / "out"))
            self.assertIn("Not enough wavelength columns", str(ctx.exception))
```

program/tests/test_data.py:254-266 — loader-side indirection spec:
```python
    def test_name_for_search_resolves_through_the_column_indirection(self):
        with data_sandbox() as (root, folders):
            wl = [400 + 10 * i for i in range(20)]
            write_reflector_tsv(
                Path(folders["reflectors"]) / "leaf.tsv", wl, [50.0] * 20,
                header={"Organization": "Uni", "name_for_search": "Common Name",
                        "Common Name": "Ficus Leaf", "IsDefault": "yes"})
            c = data_mod.load_reflector_collection()
            self.assertEqual(len(c.reflectors), 1)
            self.assertEqual(c.reflectors[0].name, "Ficus Leaf")
            self.assertClose(float(c.reflector_matrix[0][GRID == 400][0]), 0.5)
            self.assertEqual(c.reflectors[0].metadata["source_file"],
                             str(Path(folders["reflectors"]) / "leaf.tsv"))
```

Also removed: test_presenter.py:17 (import) and :101 (smoke call of
`prepare_ecosis_preview`); helpers.py:168 (`"ecosis": str(root / "ecosis"),`
in `import_sandbox`'s `output` dict).

## 4. Standalone rebuild notes
- **Dependencies**: numpy + pandas only. No scipy (unit 3.2 replaces
  `interp1d` extrapolation).
- **Port alongside units 3.1/3.2**: `_safe_float` (importing.py:51-56;
  still used by FS's generic pipeline, so copy, don't move):
  ```python
  def _safe_float(val):
      """Convert a value to float safely, handling various formats."""
      try:
          return float(str(val).replace(',', '.').strip())
      except (ValueError, TypeError):
          return np.nan
  ```
  plus `INTERP_GRID = np.arange(300, 1101, 1)`.
- **Create the output directory** before writing (latent bug, §2).
- **Drop `relevant_metadata`** — write-only, never read anywhere (§2).
- Suggested CLI: `ecosis_import input.csv -o out_dir [--name-column Sample]`,
  keeping the §2 contract fixed; keep writing `# key\tvalue` comment-header
  TSVs so any output stays loadable by FS's reflector loader via the
  `Name` / `sample_type` / `collector` metadata keys.
