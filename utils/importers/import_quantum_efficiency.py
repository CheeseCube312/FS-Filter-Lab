"""
quantum_efficiency.py

takes digitized quantum efficiency curves in form of WebPlotDigitizer .csv files,
    turns them into proper .tsv format for use in this tool

--- Key Functions ---

process_qe_csv(csv_path: Path, brand: str, model: str, out_dir: Path) -> Path
    - Parses CSVs where the first row specifies color channels (e.g., Red, Green, Blue),
      and the second row specifies axis types (X/Y for wavelength/response).
    - Interpolates QE data onto a 1 nm grid (300–1100 nm).
    - Saves interpolated and metadata-annotated TSV file per camera model.
    - Returns the full path to the saved file.

import_qe_from_csv(uploaded_file, brand: str, model: str) -> tuple[bool, str]
    - Web UI wrapper that handles file upload, temp saving, processing, and cleanup.
    - Returns (success flag, status message).

--- Input Format (Expected from WebPlotDigitizer) ---
Row 0: Color labels (e.g., Red, Green, Blue) — repeated across X and Y columns.  
Row 1: Axis types (X or Y) — alternating with color labels.  
Row 2+: Data rows.

--- Output Format ---
- Rows: Channels (Red, Green, Blue)
- Columns: Interpolated QE values from 300–1100 nm
- Metadata columns: "Camera Brand", "Camera Model"
- File saved as: data/QE_data/QE_<Brand>_<Model>.tsv

--- Intended Use ---
- Provides clean, structured QE files for use in sensor simulation, visualization, or spectral modeling.
- Designed to work with UI and integrate with broader data ingestion pipeline
"""




# /utils/importers/import_quantum_efficiency.py

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path

def process_qe_csv(csv_path: Path, brand: str, model: str, out_dir: Path) -> Path:
    """
    Processes a CSV containing WebPlotDigitizer QE data.
    Saves interpolated data as TSV and returns the output path.
    """
    df_raw = pd.read_csv(csv_path, header=None)

    if df_raw.shape[0] < 3:
        raise ValueError("CSV must contain at least 3 rows: color headers, axis headers, and data.")

    # Row 0 = color names, Row 1 = X/Y axis
    color_row = df_raw.iloc[0].fillna(method='ffill').astype(str).str.strip()
    axis_row = df_raw.iloc[1].astype(str).str.strip().str.upper()

    # Build multi-index column headers
    columns = []
    for color, axis in zip(color_row, axis_row):
        if axis in {"X", "Y"}:
            columns.append((color, axis))
        else:
            columns.append(None)
    
    df = df_raw.iloc[2:].copy()
    df.columns = columns
    df = df.loc[:, df.columns.notnull()]  # Drop bad columns

    channels = sorted(set(color for color, axis in df.columns if axis in {"X", "Y"}))
    required = {"Red", "Green", "Blue"}
    if not required.issubset(channels):
        raise ValueError(f"Missing required channels. Found: {channels}")

    # Target wavelength grid
    wl_target = np.arange(300, 1100, 1)
    results = {}

    for channel in channels:
        try:
            x = df[(channel, "X")].astype(float).values
            y = df[(channel, "Y")].astype(float).values
            interp = interp1d(x, y, kind='linear', bounds_error=False, fill_value=0.0)
            y_interp = np.clip(np.round(interp(wl_target), 3), 0.0, None)
            results[channel] = y_interp
        except Exception as e:
            print(f"⚠️ Skipping channel {channel}: {e}")

    if not results:
        raise ValueError("No valid QE curves were interpolated.")

    # Assemble DataFrame: Channels x Wavelength
    out_df = pd.DataFrame(results, index=wl_target).T
    out_df.columns.name = "Wavelength (nm)"
    out_df.index.name = "Channel"

    # Remove columns where all values are 0
    out_df = out_df.loc[:, ~(out_df == 0).all(axis=0)]

    # Add metadata columns
    out_df.insert(0, "Camera Brand", brand)
    out_df.insert(1, "Camera Model", model)

    # Build filename
    filename = f"QE_{brand}_{model}.tsv".replace(" ", "_")
    out_path = out_dir / filename
    out_df.to_csv(out_path, sep="\t")

    return out_path


def import_qe_from_csv(uploaded_file, brand, model):
    try:
        out_dir = Path("data/QE_data")
        out_dir.mkdir(parents=True, exist_ok=True)

        tmp_path = out_dir / "tmp_upload.csv"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        out_path = process_qe_csv(tmp_path, brand, model, out_dir)
        tmp_path.unlink()

        return True, f"QE data saved to {out_path}"
    except Exception as e:
        return False, f"Error: {str(e)}"
