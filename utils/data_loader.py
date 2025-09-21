"""
data_loaders.py

Handles loading, preprocessing, interpolation, and caching of:
- Filter transmission curves
- Camera sensor QE (quantum efficiency) data
- Illuminant spectral power distributions

All data is loaded from .tsv files, interpolated onto a common wavelength grid, and cached for performance.

Utilities:
- _is_float: Safely checks if a value can be interpreted as a float.
- _save_cache: Saves an object to the 'cache/' folder using "pickle".
- _load_cache: Loads an object from the 'cache/' folder if it exists.
- _get_data_files_hash: Generates a hash of filenames + modification times to detect changes.

Loaders:
- load_filter_data: Loads and caches filter transmission curves and metadata from `data/filters_data/`.
- load_qe_data: Loads and caches RGB quantum efficiency curves from `data/QE_data/`.
- load_illuminants: Loads and caches illuminant power spectra from `data/illuminants/`.
"""



import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
import pickle
import hashlib
from .constants import INTERP_GRID

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def _is_float(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False

def _save_cache(obj, filename):
    with open(CACHE_DIR / filename, "wb") as f:
        pickle.dump(obj, f)

def _load_cache(filename):
    path = CACHE_DIR / filename
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def _get_data_files_hash(folder: str, pattern: str = "**/*.*") -> str:
    """
    Generate a hash string based on file names and modification times in the folder.
    """
    file_info = []
    folder_path = Path(folder)
    for filepath in sorted(folder_path.glob(pattern)):
        if filepath.is_file():
            stat = filepath.stat()
            file_info.append(f"{filepath.name}-{stat.st_mtime}")
    file_info_str = "|".join(file_info)
    return hashlib.md5(file_info_str.encode()).hexdigest()

# --- FILTER DATA ---
def load_filter_data():
    folder = os.path.join("data", "filters_data")
    os.makedirs(folder, exist_ok=True)

    version_hash = _get_data_files_hash(folder, pattern="**/*.tsv")
    cached = _load_cache("filter_data.pkl")
    cached_version = _load_cache("filter_data_version.pkl")

    if cached is not None and cached_version == version_hash:
        return cached

    # Rebuild cache
    files = glob.glob(os.path.join(folder, "**", "*.tsv"), recursive=True)
    meta_list, matrix, masks = [], [], []

    for path in files:
        try:
            df = pd.read_csv(path, sep="\t")
            df.columns = [str(c).strip() for c in df.columns]

            wl_cols = sorted([float(c) for c in df.columns if _is_float(c)])
            str_wl_cols = [str(int(w)) for w in wl_cols]
            if not wl_cols or 'Filter Number' not in df.columns:
                continue

            for _, row in df.iterrows():
                fn = str(row['Filter Number'])

                name_raw = row.get('Filter Name')
                name = str(name_raw).strip() if pd.notnull(name_raw) and str(name_raw).strip() else "Unnamed"
                manufacturer = row.get('Manufacturer', 'Unknown')
                hex_color_raw = row.get('Hex Color')
                hex_color = str(hex_color_raw).strip() if pd.notnull(hex_color_raw) and str(hex_color_raw).strip().startswith("#") else "#838383"
                is_lee = 'LeeFilters' in os.path.basename(path)

                raw = np.array([row.get(w, np.nan) for w in str_wl_cols], dtype=float)
                if raw.max() > 1.5:
                    raw /= 100.0

                interp_vals = np.interp(INTERP_GRID, wl_cols, raw, left=np.nan, right=np.nan)
                extrap_mask = (INTERP_GRID > 700) if is_lee else np.zeros_like(INTERP_GRID, dtype=bool)

                meta_list.append({
                    'Filter Number': fn,
                    'Filter Name': name,
                    'Manufacturer': manufacturer,
                    'Hex Color': hex_color,
                    'is_lee': is_lee
                })
                matrix.append(interp_vals)
                masks.append(extrap_mask)

        except Exception as e:
            st.warning(f"⚠️ Failed to load filter file '{os.path.basename(path)}': {e}")

    if not matrix:
        result = (pd.DataFrame(), np.empty((0, len(INTERP_GRID))), np.empty((0, len(INTERP_GRID)), dtype=bool))
    else:
        result = (pd.DataFrame(meta_list), np.vstack(matrix), np.vstack(masks))

    _save_cache(result, "filter_data.pkl")
    _save_cache(version_hash, "filter_data_version.pkl")
    return result


# --- QE DATA ---
def load_qe_data():
    folder = os.path.join('data', 'QE_data')
    os.makedirs(folder, exist_ok=True)

    version_hash = _get_data_files_hash(folder, pattern="*.tsv")
    cached = _load_cache("qe_data.pkl")
    cached_version = _load_cache("qe_data_version.pkl")

    if cached is not None and cached_version == version_hash:
        return cached

    files = glob.glob(os.path.join(folder, '*.tsv'))
    qe_dict = {}
    default_key = None

    for path in files:
        try:
            df = pd.read_csv(path, sep='\t')
            df.columns = [str(c).strip() for c in df.columns]

            wl_cols = sorted([float(c) for c in df.columns if _is_float(c)])
            str_wl_cols = [str(int(w)) for w in wl_cols]

            for _, row in df.iterrows():
                brand = row['Camera Brand'].strip()
                model = row['Camera Model'].strip()
                channel = row['Channel'].strip().upper()[0]
                key = f"{brand} {model}"

                raw = row[str_wl_cols].astype(float).values
                interp = np.interp(INTERP_GRID, wl_cols, raw, left=np.nan, right=np.nan)

                qe_dict.setdefault(key, {})[channel] = interp

                if os.path.basename(path) == 'Default_QE.tsv' and default_key is None:
                    default_key = key

        except Exception as e:
            st.warning(f"⚠️ Failed to load QE file '{os.path.basename(path)}': {e}")

    result = (sorted(qe_dict.keys()), qe_dict, default_key)
    _save_cache(result, "qe_data.pkl")
    _save_cache(version_hash, "qe_data_version.pkl")
    return result


# --- ILLUMINANTS ---
def load_illuminants():
    folder = os.path.join('data', 'illuminants')
    os.makedirs(folder, exist_ok=True)

    version_hash = _get_data_files_hash(folder, pattern="*.tsv")
    cached = _load_cache("illuminants.pkl")
    cached_version = _load_cache("illuminants_version.pkl")

    if cached is not None and cached_version == version_hash:
        return cached

    illum, meta = {}, {}

    for path in glob.glob(os.path.join(folder, '*.tsv')):
        try:
            df = pd.read_csv(path, sep='\t')
            if df.shape[1] < 2:
                raise ValueError("File must have at least two columns (wavelength and power)")

            wl = df.iloc[:, 0].astype(float).values
            power = df.iloc[:, 1].astype(float).values

            interp = np.interp(INTERP_GRID, wl, power, left=np.nan, right=np.nan)
            name = os.path.splitext(os.path.basename(path))[0]

            illum[name] = interp
            if 'Description' in df.columns:
                meta[name] = df['Description'].dropna().iloc[0]

        except Exception as e:
            st.warning(f"⚠️ Failed to load illuminant '{os.path.basename(path)}': {e}")

    result = (illum, meta)
    _save_cache(result, "illuminants.pkl")
    _save_cache(version_hash, "illuminants_version.pkl")
    return result
