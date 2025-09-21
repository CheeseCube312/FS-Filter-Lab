# /utils/importers/import_reflectance_absorption.py

import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d

def safe_float(val):
    try:
        return float(str(val).replace(',', '.').strip())
    except Exception:
        return np.nan

def import_reflectance_absorption_from_csv(uploaded_file, meta, extrap_lower, extrap_upper):
    """
    Converts a digitized reflectance or absorption spectrum CSV into a wide TSV.

    Parameters:
    - uploaded_file: file-like object, CSV with two columns: wavelength, value
    - meta: dict with keys:
        - spectrum_name: str
        - description: str (optional)
        - spectrum_type: str, either 'reflectance' or 'absorption' (case-insensitive)
        - hex_color: str (optional, e.g. '#00FF00')
    - extrap_lower: bool, extend data down to 300 nm if True
    - extrap_upper: bool, extend data up to 1100 nm if True

    Returns:
    - (success: bool, message: str)
    """
    try:
        raw_data = pd.read_csv(uploaded_file, sep=';', header=None, engine='python')
        if raw_data.shape[1] < 2:
            return False, "Could not read two columns from the CSV."

        raw_data = raw_data.applymap(safe_float)
        wavelengths = raw_data.iloc[:, 0].dropna().values
        values = raw_data.iloc[:, 1].dropna().values

        # Clip to [300, 1100] nm
        valid_mask = (wavelengths >= 300) & (wavelengths <= 1100)
        wavelengths = wavelengths[valid_mask]
        values = values[valid_mask]

        if wavelengths.size == 0:
            return False, "No data points remain after clipping wavelengths to 300-1100 nm."

        sort_idx = np.argsort(wavelengths)
        wavelengths = wavelengths[sort_idx]
        values = values[sort_idx]

        min_wl = 300 if extrap_lower else int(np.ceil(wavelengths.min() / 5.0)) * 5
        max_wl = 1100 if extrap_upper else int(np.floor(wavelengths.max() / 5.0)) * 5

        if min_wl > max_wl:
            return False, "Adjusted wavelength range is invalid after cropping."

        new_wavelengths = np.arange(min_wl, max_wl + 1, 1)
        interpolator = interp1d(wavelengths, values, kind='linear', bounds_error=False, fill_value=np.nan)
        interpolated = interpolator(new_wavelengths)

        if extrap_lower:
            below_mask = new_wavelengths < wavelengths.min()
            interpolated[below_mask] = values[0]
        if extrap_upper:
            above_mask = new_wavelengths > wavelengths.max()
            interpolated[above_mask] = values[-1]

        interpolated = np.clip(np.round(interpolated, 3), 0.0, 1.0)

        spectrum_type = meta.get("spectrum_type", "").strip().lower()
        if spectrum_type == "absorption":
            interpolated = 1.0 - interpolated
        elif spectrum_type != "reflectance":
            return False, "spectrum_type must be 'reflectance' or 'absorption'."

        output_df = pd.DataFrame([interpolated], columns=new_wavelengths)
        output_df.insert(0, 'Hex Color', meta.get("hex_color", "#000000"))  # new hex color column
        output_df.insert(0, 'Spectrum Type', spectrum_type)
        output_df.insert(0, 'Description', meta.get("description", ""))
        output_df.insert(0, 'Spectrum Name', meta.get("spectrum_name", "Unnamed_Spectrum"))

        base = f"{meta.get('spectrum_name', 'Unnamed')}_{spectrum_type}"
        sanitized = ''.join(c for c in base if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')

        suffix_parts = []
        if extrap_lower: suffix_parts.append("300")
        if extrap_upper: suffix_parts.append("1100")
        suffix = f"_extrapolated_{'_'.join(suffix_parts)}" if suffix_parts else ""

        filename = f"{sanitized}{suffix}.tsv"
        out_dir = os.path.join("data", "reflectance_absorption")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        output_df.to_csv(out_path, sep='\t', index=False)

        return True, f"Reflectance/Absorption data saved to {out_path}"

    except Exception as e:
        return False, f"Error: {str(e)}"
