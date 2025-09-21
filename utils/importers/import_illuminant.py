"""
import_illuminants.py

takes digitized illuminant curves in form of WebPlotDigitizer .csv files,
    turns them into proper .tsv format for use in this tool

--- Key Functions ---

parse_webplot_csv(path: Path) -> tuple[np.ndarray, np.ndarray]
    - Loads a CSV file using ';' as separator and ',' as decimal.
    - Returns a tuple of (wavelengths, intensities).

convert_to_illuminant_tsv(csv_path: Path, description: str, out_dir: Path) -> Path
    - Interpolates raw illuminant data to a standard 300–1100 nm wavelength grid.
    - Normalizes intensity values to a 0–100% relative scale.
    - Inserts a user-provided description at the start of the file.
    - Saves result as a `.tsv` file to the specified output directory.
    - Returns the output file path.

import_illuminant_from_csv(uploaded_file, description) -> tuple[bool, str]
    - Wrapper for UI uploads: accepts a file-like object and metadata description.
    - Handles temporary saving, conversion, and cleanup.
    - Returns (success flag, status message).

--- Output Format ---
- Columns: Wavelength (nm), Relative Power, Description
- File saved as: `data/illuminants/<name>.tsv`

--- Intended Use ---
- Backend utility for frontend UI (e.g., Streamlit) to allow user import of real-world or synthetic illuminant data.
- Enables consistent formatting and normalization across spectral analysis modules.
"""




# /utils/importers/import_illuminants.py
import pandas as pd
import numpy as np
from pathlib import Path


def parse_webplot_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and parse a CSV from WebPlotDigitizer with ; separator and , as decimal.
    Returns a tuple: (wavelengths, intensities)
    """
    raw_df = pd.read_csv(path, sep=";", header=None, engine="python")
    raw_df = raw_df.applymap(lambda x: str(x).strip().replace(",", "."))
    raw_df = raw_df.astype(float)

    wavelengths = raw_df.iloc[:, 0].values
    intensity = raw_df.iloc[:, 1].values
    return wavelengths, intensity


def convert_to_illuminant_tsv(
    csv_path: Path,
    description: str,
    out_dir: Path
) -> Path:
    """
    Converts a WebPlotDigitizer CSV into a formatted .tsv file for illuminants.
    Saves to `out_dir`. Returns the output path.
    """
    wl, intensity = parse_webplot_csv(csv_path)

    # Target wavelength range: 300–1100 nm
    full_range = np.arange(300, 1101, 1)

    # Interpolate intensity over this range
    intensity_interp = np.interp(full_range, wl, intensity, left=0, right=0)

    # Normalize to 0–100 scale
    max_val = np.max(intensity_interp)
    if max_val > 0:
        intensity_rel = np.round((intensity_interp / max_val) * 100, 3)
    else:
        intensity_rel = intensity_interp

    # Construct output DataFrame
    df_out = pd.DataFrame({
        "Wavelength (nm)": full_range,
        "Relative Power": intensity_rel,
        "Description": [description] + [""] * (len(full_range) - 1)
    })

    # Filename
    out_name = csv_path.stem.replace(" ", "_") + ".tsv"
    out_path = out_dir / out_name

    df_out.to_csv(out_path, sep="\t", index=False)
    return out_path

def import_illuminant_from_csv(uploaded_file, description):
    try:
        out_dir = Path("data/illuminants")
        out_dir.mkdir(parents=True, exist_ok=True)

        tmp_path = out_dir / "tmp_upload.csv"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        out_path = convert_to_illuminant_tsv(tmp_path, description, out_dir)
        tmp_path.unlink()  # Clean up temp

        return True, f"Illuminant saved to {out_path}"
    except Exception as e:
        return False, f"Error: {str(e)}"
