"""
Consolidated data importing utilities for FS FilterLab.

This module contains all data import functionality for filters, illuminants,
quantum efficiencies, and reflectance data with comprehensive error handling
and user-friendly validation.

Key Features:
- Robust CSV parsing with automatic separator detection
- Comprehensive data validation with descriptive error messages
- Consistent error handling across all import types
- File format validation and range checking
- Safe filename generation and file saving

Generic pipeline shared by the five single-spectrum importers:
- _extrapolated_series: validate -> optional 0-100 cap / negative clamp ->
  sort -> 5nm-rounded output range with optional edge extrapolation ->
  interpolate (filters, reflectance/absorption spectra)
- _resample_channel: trim -> validate -> optional 0-100 cap -> clamp ->
  sort -> resample onto INTERP_GRID with edge-fill policy and optional
  peak normalisation (illuminants, camera QE, film channels)

Import adapters (one per on-disk format):
- import_filter_from_csv: filter transmission -> tall TSV (filters_data)
- import_illuminant_from_csv: illuminant SPD -> TSV (illuminants/)
- import_qe_from_csv: camera QE, 3 X-Y pairs -> TSV (QE_data/)
- import_film_from_csv: film sensitivity -> comment-header TSV (film_data/)
- import_reflectance_absorption_from_csv: surface spectrum -> TSV
  (reflectors/plant|other; value column is always 'Reflectance' - the
  loader contract in services/data.py)
- import_ecosis_csv: wide ECOSIS CSV -> one comment-header TSV per sample
  (reflectors/Ecosis/...); sample values are normalized to the 0-100 file
  scale (the loaders divide by 100 on load)

All import_* functions except import_ecosis_csv return
(success: bool, message: str) tuples for consistent error handling in the
UI layer; import_ecosis_csv returns the list of created files and raises
ValueError on failure (views/actions.py wraps the call in try/except).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import functools
import logging
import re
from models.constants import INTERP_GRID, DATA_FOLDERS, OUTPUT_FOLDERS, FILM_TSV_FIELDS

# Configure logging for debugging import issues
logger = logging.getLogger(__name__)

# ============================================================================
# COMMON UTILITIES
# ============================================================================

def _safe_float(val):
    """Convert a value to float safely, handling various formats."""
    try:
        return float(str(val).replace(',', '.').strip())
    except (ValueError, TypeError):
        return np.nan


def _validate_wavelength_data(wavelengths, values, value_type="spectral values"):
    """
    Common validation for wavelength and spectral data.
    
    Args:
        wavelengths: Array of wavelength values  
        values: Array of spectral values
        value_type: Description of value type for error messages
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for empty data
    if wavelengths.size == 0:
        return False, "Wavelength column (first column) is empty or contains no valid numbers."
    
    if values.size == 0:
        return False, f"{value_type.title()} column is empty or contains no valid numbers."
    
    # Check size match
    if wavelengths.size != values.size:
        return False, f"Wavelength and {value_type} columns have different lengths ({wavelengths.size} vs {values.size})."
    
    # Check wavelength range validity
    min_wl, max_wl = wavelengths.min(), wavelengths.max()
    if min_wl < 200 or max_wl > 2000:
        return False, f"Wavelength range ({min_wl:.1f}-{max_wl:.1f} nm) seems invalid. Expected 200-2000 nm."
        
    if max_wl - min_wl < 50:
        return False, f"Wavelength range too narrow ({min_wl:.1f}-{max_wl:.1f} nm). Need at least 50nm range."
    
    return True, ""



def _safe_file_save(df: pd.DataFrame, file_path: Path, data_type: str,
                    header_lines: Optional[List[str]] = None) -> Tuple[bool, str]:
    """
    Safely save a DataFrame to a TSV file with error handling.

    Args:
        df: DataFrame to save
        file_path: Path to save the file
        data_type: Type of data being saved (for error messages)
        header_lines: Optional comment-header lines written above the data
            table (film profiles use "# Key\tvalue" metadata headers)

    Returns:
        Tuple of (success, message)
    """
    try:
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the file
        if header_lines:
            content = '\n'.join(header_lines) + '\n' + df.to_csv(sep='\t', index=False)
            file_path.write_text(content, encoding='utf-8')
        else:
            df.to_csv(file_path, sep='\t', index=False)
        return True, f"{data_type} data saved successfully to {file_path}"

    except PermissionError:
        return False, f"Permission denied: Cannot write to {file_path}. File may be open in another application."
    except OSError as e:
        return False, f"File system error: {str(e)}"
    except Exception as e:
        return False, f"Failed to save file to {file_path}: {str(e)}"


def _import_errors(label: str):
    """
    Decorator for the single-spectrum import adapters: run the adapter and
    map errors onto the (success, message) UI contract. Validation errors
    (ValueError) are shown verbatim; anything else is logged and reported
    generically.
    """
    def decorator(handler):
        @functools.wraps(handler)
        def wrapper(*args, **kwargs):
            try:
                return handler(*args, **kwargs)
            except ValueError as e:
                return False, str(e)
            except Exception as e:
                logger.exception("Unexpected error in %s import: %s", label, e)
                return False, f"Unexpected error during import: {str(e)}. Please check the file format and try again."
        return wrapper
    return decorator


def _parse_upload(uploaded_file):
    """Parse an uploaded CSV; parse failures become user-facing errors."""
    try:
        return _parse_csv(uploaded_file)
    except ValueError as e:
        raise ValueError(f"CSV parsing failed: {str(e)}") from None


def _parse_csv(file, separator=';', fallback_separator=','):
    """Parse a CSV file with auto-detection of separator and better error handling."""
    try:
        # Try primary separator first
        raw_data = pd.read_csv(file, sep=separator, header=None, engine='python')
        
        # If we don't have enough columns, try fallback separator
        if raw_data.shape[1] < 2:
            file.seek(0)  # Reset file position for re-reading
            raw_data = pd.read_csv(file, sep=fallback_separator, header=None, engine='python')
        
        # Final validation
        if raw_data.shape[1] < 2:
            raise ValueError(f"CSV file must have at least 2 columns. Found {raw_data.shape[1]} columns.")
            
        if raw_data.shape[0] == 0:
            raise ValueError("CSV file is empty or contains no data rows.")
        
        # Convert to float with better error reporting
        try:
            raw_data = raw_data.map(_safe_float)
        except AttributeError:
            # Fallback for older pandas versions
            raw_data = raw_data.applymap(_safe_float)
        
        # Check for too many NaN values
        nan_count = raw_data.isna().sum().sum()
        total_count = raw_data.size
        if nan_count > total_count * 0.5:
            raise ValueError(f"Too many invalid values in CSV ({nan_count}/{total_count}). Check data format.")
            
        return raw_data
        
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty.")
    except pd.errors.ParserError as e:
        raise ValueError(f"CSV parsing error: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {str(e)}")


def _get_wavelength_range(wavelengths, extrap_lower=False, extrap_upper=False):
    """Determine the wavelength range based on data and extrapolation settings."""
    base_min = int(np.ceil(wavelengths.min() / 5.0)) * 5
    base_max = int(np.floor(wavelengths.max() / 5.0)) * 5
    
    # Use constants for consistent wavelength range
    target_min, target_max = INTERP_GRID.min(), INTERP_GRID.max()
    
    min_wl = target_min if extrap_lower else base_min
    max_wl = target_max if extrap_upper else min(target_max, base_max)
    
    if min_wl > max_wl:
        raise ValueError("Data range is outside allowable bounds.")
        
    return min_wl, max_wl


def _interpolate_spectrum(wavelengths, values, target_wavelengths, extrap_lower=False, extrap_upper=False):
    """Interpolate spectral data to a target wavelength range."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    values = np.asarray(values, dtype=float)
    
    # Match the previous scipy behaviour (assume_sorted=False): sort by
    # wavelength first if the input is not already ascending.
    if np.any(np.diff(wavelengths) < 0):
        order = np.argsort(wavelengths)
        wavelengths, values = wavelengths[order], values[order]
    
    # np.interp clamps outside the data range to the first/last value.
    interpolated = np.interp(target_wavelengths, wavelengths, values)
    
    # Blank out targets beyond the data range unless edge-value extrapolation
    # was requested (the clamp already yields exactly values[0] / values[-1]).
    if not extrap_lower:
        interpolated[target_wavelengths < wavelengths.min()] = np.nan
    if not extrap_upper:
        interpolated[target_wavelengths > wavelengths.max()] = np.nan
    
    return np.clip(np.round(interpolated, 3), 0.0, None)


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


def _sanitize_filename(name):
    """Convert a string to a valid import filename (spaces -> underscores).

    Deliberately different from reporting.sanitize_filename_component
    (report paths; invalid chars -> '-'): unifying the two would change the
    on-disk names future imports produce and orphan existing files.
    """
    return ''.join(c for c in name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')


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
                rows.append(f"# name_for_search	{name_column}")
                rows.append("#")  # Blank comment line
            
            # Add relevant_metadata field if user selected columns (store column names, not values)
            if relevant_metadata:
                # Only include columns that exist in this CSV file
                valid_relevant = [col for col in relevant_metadata if col in metadata_cols]
                if valid_relevant:
                    relevant_str = "|".join(valid_relevant)  # Join multiple column names with |
                    rows.append(f"# relevant_metadata	{relevant_str}")
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


def _get_extrapolation_suffix(extrap_lower, extrap_upper):
    """Get a standardized suffix for extrapolated files."""
    suffix_parts = []
    if extrap_lower: suffix_parts.append("300")
    if extrap_upper: suffix_parts.append("1100")
    return f"_extrapolated_{'_'.join(suffix_parts)}" if suffix_parts else ""

def _extrapolated_series(wavelengths, values, value_label, extrap_lower, extrap_upper,
                         clamp_negatives=False, cap_message_fn=None, on_negative=None):
    """
    Shared filter/reflectance pipeline: validate the raw 2-column series,
    optionally cap it at 100 and clamp negatives, sort by wavelength, pick
    the 5nm-rounded output range (extended to 300/1100nm when extrapolation
    is requested) and interpolate onto it.

    Raises ValueError with the user-facing message on invalid input.
    Returns (new_wavelengths, interpolated_values).
    """
    is_valid, error_msg = _validate_wavelength_data(wavelengths, values, value_label)
    if not is_valid:
        raise ValueError(error_msg)

    if cap_message_fn is not None:
        max_val = values.max()
        if max_val > 100:
            raise ValueError(cap_message_fn(max_val))

    if clamp_negatives and values.min() < 0:
        if on_negative is not None:
            on_negative(values.min())
        values = np.clip(values, 0, None)

    sort_idx = np.argsort(wavelengths)
    wavelengths = wavelengths[sort_idx]
    values = values[sort_idx]

    min_wl, max_wl = _get_wavelength_range(wavelengths, extrap_lower, extrap_upper)
    new_wavelengths = np.arange(min_wl, max_wl + 1, 1)

    interpolated = _interpolate_spectrum(
        wavelengths, values, new_wavelengths, extrap_lower, extrap_upper
    )
    return new_wavelengths, interpolated


def _resample_channel(wavelengths, values, value_label, cap_message_fn=None,
                      zero_edges=True, normalize_peak=False,
                      clamp_negatives=True, on_negative=None):
    """
    Shared illuminant/QE/film pipeline for one X-Y channel: trim to the
    shorter column, validate, optionally cap-check at 100, clamp negatives,
    sort by wavelength and resample onto INTERP_GRID (rounded to 3 decimals).

    zero_edges=True fills beyond the data range with 0 (film/illuminant
    behaviour); False clamps to the edge values (camera QE behaviour).
    normalize_peak=True rescales the resampled curve so its peak is 100
    (illuminants).

    Raises ValueError with the user-facing message on invalid input.
    """
    min_size = min(wavelengths.size, values.size)
    wavelengths = wavelengths[:min_size]
    values = values[:min_size]

    is_valid, error_msg = _validate_wavelength_data(wavelengths, values, value_label)
    if not is_valid:
        raise ValueError(error_msg)

    if cap_message_fn is not None:
        max_val = values.max()
        if max_val > 100:
            raise ValueError(cap_message_fn(max_val))

    if clamp_negatives and values.min() < 0:
        if on_negative is not None:
            on_negative(values.min())
        values = np.clip(values, 0, None)

    sort_idx = np.argsort(wavelengths)
    resampled = np.interp(
        INTERP_GRID, wavelengths[sort_idx], values[sort_idx],
        left=0.0 if zero_edges else None, right=0.0 if zero_edges else None,
    )

    if normalize_peak:
        peak = resampled.max()
        if peak > 0:
            return np.round((resampled / peak) * 100.0, 3)
        return resampled  # all-zero curve stays all-zero
    return np.round(resampled, 3)


# ============================================================================
# FILTER IMPORT
# ============================================================================

def _paired_columns(df: pd.DataFrame, left: int, right: int):
    """Row-aligned extraction of a (wavelength, value) column pair.

    Rows where EITHER column is NaN are dropped from both series, so
    wavelength/value rows stay paired. The previous per-column
    dropna() desynced the two series whenever a CSV had an interior
    blank in just one column of the pair.
    """
    pair = df.iloc[:, [left, right]].dropna()
    return pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()


@_import_errors("filter")
def import_filter_from_csv(uploaded_file, meta, extrap_lower, extrap_upper):
    """
    Import filter data from a CSV file and save it to the data directory.

    Args:
        uploaded_file: The uploaded CSV file
        meta: Dictionary containing filter metadata
        extrap_lower: Whether to extrapolate to 300nm
        extrap_upper: Whether to extrapolate to 1100nm

    Returns:
        Tuple of (success, message)
    """
    # Validate inputs
    if uploaded_file is None:
        return False, "No file provided for import."

    if not meta or not isinstance(meta, dict):
        return False, "Invalid metadata provided."

    required_keys = ['manufacturer', 'filter_name', 'filter_number', 'hex_color']
    missing_keys = [key for key in required_keys if key not in meta or not meta[key]]
    if missing_keys:
        return False, f"Missing required metadata: {', '.join(missing_keys)}"

    raw_data = _parse_upload(uploaded_file)

    wavelengths, transmissions = _paired_columns(raw_data, 0, 1)

    # Validate, cap at 100, clamp negatives, sort, pick the output range and
    # interpolate via the shared pipeline
    new_wavelengths, interpolated = _extrapolated_series(
        wavelengths, transmissions, "transmission", extrap_lower, extrap_upper,
        clamp_negatives=True,
        cap_message_fn=lambda m: f"Transmission values seem too high (max: {m:.2f}). Expected 0-100%.",
        on_negative=lambda m: logger.info(f"Clamped negative transmission values to 0 (min was {m:.3f})"),
    )

    # Create DataFrame in tall format
    output_df = pd.DataFrame({
        'Wavelength': new_wavelengths,
        'Transmittance': interpolated,
        'hex_color': meta["hex_color"],
        'Manufacturer': meta["manufacturer"],
        'Name': meta["filter_name"],
        'Filter Number': meta["filter_number"]
    })

    # Generate filename and save
    base = f"{meta['manufacturer']}_{meta['filter_number']}_{meta['filter_name']}"
    sanitized = _sanitize_filename(base)
    suffix = _get_extrapolation_suffix(extrap_lower, extrap_upper)
    filename = f"{sanitized}{suffix}.tsv"

    out_dir = Path(OUTPUT_FOLDERS['filter_import']) / meta["manufacturer"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    return _safe_file_save(output_df, out_path, "Filter")

# ============================================================================
# ILLUMINANT IMPORT
# ============================================================================

@_import_errors("illuminant")
def import_illuminant_from_csv(uploaded_file, description):
    """
    Import illuminant data from a CSV file and save it to the data directory.

    Args:
        uploaded_file: The uploaded CSV file
        description: Illuminant description

    Returns:
        Tuple of (success, message)
    """
    # Validate inputs
    if uploaded_file is None:
        return False, "No file provided for import."

    if not description or not description.strip():
        return False, "Description cannot be empty."

    out_dir = Path(DATA_FOLDERS['illuminants'])
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_data = _parse_upload(uploaded_file)

    wavelengths, intensity = _paired_columns(raw_data, 0, 1)

    # Validate, clamp negatives, sort and resample onto the standard grid
    # normalised so the peak is 100 (shared pipeline)
    intensity_rel = _resample_channel(
        wavelengths, intensity, "intensity",
        normalize_peak=True,
        on_negative=lambda m: logger.info(f"Clamped negative intensity values to 0 (min was {m:.3f})"),
    )

    # Create output DataFrame
    df_out = pd.DataFrame({
        "Wavelength (nm)": INTERP_GRID,
        "Relative Power": intensity_rel,
        "Description": description
    })

    # Save file
    filename = _sanitize_filename(f"illuminant_{description}") + ".tsv"
    out_path = out_dir / filename

    return _safe_file_save(df_out, out_path, "Illuminant")

# ============================================================================
# QUANTUM EFFICIENCY IMPORT
# ============================================================================

@_import_errors("QE")
def import_qe_from_csv(uploaded_file, brand, model):
    """
    Import quantum efficiency data from a CSV file.

    Expected format: X-Y pairs for each channel (6 columns)
    R_wavelength, R_qe, G_wavelength, G_qe, B_wavelength, B_qe

    Args:
        uploaded_file: The uploaded CSV file
        brand: Camera brand
        model: Camera model

    Returns:
        Tuple of (success, message)
    """
    # Validate inputs
    if uploaded_file is None:
        return False, "No file provided for import."

    if not brand or not brand.strip():
        return False, "Camera brand cannot be empty."

    if not model or not model.strip():
        return False, "Camera model cannot be empty."

    raw_data = _parse_upload(uploaded_file)

    if raw_data.shape[1] < 6:
        return False, f"Expected 6 columns (R_wl, R_qe, G_wl, G_qe, B_wl, B_qe), found {raw_data.shape[1]} columns."

    # X-Y pairs format: R_wl, R_qe, G_wl, G_qe, B_wl, B_qe
    channels = {}
    for channel, wl_col, val_col in [('R', 0, 1), ('G', 2, 3), ('B', 4, 5)]:
        wl, qe = _paired_columns(raw_data, wl_col, val_col)
        if wl.size == 0 or qe.size == 0:
            return False, f"{channel} channel has empty wavelength or QE data."

        # Note: QE clamps to the edge values beyond the data range
        # (zero_edges=False), unlike film/illuminant which zero-fill.
        channels[channel] = _resample_channel(
            wl, qe, f"{channel} QE",
            cap_message_fn=lambda m, ch=channel: f"{ch} channel QE values seem too high (max: {m:.3f}). Expected 0-100%.",
            zero_edges=False,
            on_negative=lambda m, ch=channel: logger.info(f"{ch} channel: Clamped negative QE values to 0 (min was {m:.3f})"),
        )

    # Create output DataFrame with rounded values (0.001 precision)
    output_df = pd.DataFrame({
        'Wavelength': INTERP_GRID,
        'R': channels['R'],
        'G': channels['G'],
        'B': channels['B'],
        'Manufacturer': brand,
        'Name': model
    })

    # Save file
    filename = f"{_sanitize_filename(brand)}_{_sanitize_filename(model)}_QE.tsv"
    out_dir = Path(DATA_FOLDERS['qe'])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    return _safe_file_save(output_df, out_path, "QE")


# ============================================================================
# FILM SENSITIVITY IMPORT
# ============================================================================

@_import_errors("film")
def import_film_from_csv(
    uploaded_file,
    manufacturer: str,
    name: str,
    film_type: str,
    iso: str = "",
    process: str = "",
    notes: str = "",
    log_sensitivity: bool = False,
) -> Tuple[bool, str]:
    """
    Import analog film spectral sensitivity data from a CSV file.

    Colour film format (columns): R_wl, R_val, G_wl, G_val, B_wl, B_val
    B&W film format (columns):    Wavelength, V   (single sensitivity curve)

    Values may be linear 0-100 scale OR log10 sensitivity (log_sensitivity=True).
    Log sensitivity is converted via 10^x then normalised so peak = 100.
    """
    VALID_TYPES = {'color_negative', 'color_slide', 'bw'}

    # --- Input validation ---
    if uploaded_file is None:
        return False, "No file provided for import."
    if not manufacturer or not manufacturer.strip():
        return False, "Manufacturer cannot be empty."
    if not name or not name.strip():
        return False, "Film name cannot be empty."
    if film_type not in VALID_TYPES:
        return False, f"Invalid film type '{film_type}'. Must be one of: {', '.join(sorted(VALID_TYPES))}."

    manufacturer = manufacturer.strip()
    name = name.strip()

    # --- Parse CSV ---
    raw_data = _parse_upload(uploaded_file)

    is_bw = film_type == 'bw'
    min_cols = 2 if is_bw else 6
    if raw_data.shape[1] < min_cols:
        if is_bw:
            return False, f"B&W film CSV requires 2 columns (Wavelength, Sensitivity). Found {raw_data.shape[1]}."
        else:
            return False, f"Colour film CSV requires 6 columns (R_wl, R_val, G_wl, G_val, B_wl, B_val). Found {raw_data.shape[1]}."

    # --- Resample the channel(s) via the shared pipeline ---
    if is_bw:
        wavelengths, sensitivity = _paired_columns(raw_data, 0, 1)

        if log_sensitivity:
            # Log10 input -> linear, normalised so the peak is 100
            sensitivity = np.power(10.0, sensitivity)
            peak = sensitivity.max()
            if peak > 0:
                sensitivity = sensitivity / peak * 100.0

        v_interp = _resample_channel(
            wavelengths, sensitivity, "sensitivity",
            cap_message_fn=lambda m: f"Sensitivity values seem too high (max: {m:.1f}). Expected 0-100.",
        )
        channel_rows = {'Wavelength': INTERP_GRID, 'V': v_interp}
    else:
        # 6-column X-Y pair format: R_wl, R_val, G_wl, G_val, B_wl, B_val
        col_pairs = [('R', 0, 1), ('G', 2, 3), ('B', 4, 5)]

        # For log data, find the global peak across all channels first so
        # relative channel sensitivity is preserved after normalisation.
        global_peak = None
        if log_sensitivity:
            all_linear = []
            for _, wl_col, val_col in col_pairs:
                v = raw_data.iloc[:, val_col].dropna().values
                all_linear.append(np.power(10.0, v))
            global_peak = max(a.max() for a in all_linear if len(a) > 0)
            global_peak = global_peak if global_peak > 0 else 1.0

        channels = {}
        for ch, wl_col, val_col in col_pairs:
            wl_ch, vals = _paired_columns(raw_data, wl_col, val_col)

            if log_sensitivity:
                vals = np.power(10.0, vals) / global_peak * 100.0

            channels[ch] = _resample_channel(
                wl_ch, vals, f"{ch} sensitivity",
                cap_message_fn=lambda m, c=ch: f"{c} channel sensitivity values too high (max: {m:.1f}). Expected 0-100.",
            )

        channel_rows = {'Wavelength': INTERP_GRID, **channels}

    # --- Build TSV with comment headers ---
    header_lines = [
        f"# {FILM_TSV_FIELDS['manufacturer']}\t{manufacturer}",
        f"# {FILM_TSV_FIELDS['name']}\t{name}",
        f"# {FILM_TSV_FIELDS['type']}\t{film_type}",
    ]
    if iso and iso.strip():
        header_lines.append(f"# {FILM_TSV_FIELDS['iso']}\t{iso.strip()}")
    if process and process.strip():
        header_lines.append(f"# {FILM_TSV_FIELDS['process']}\t{process.strip()}")
    if notes and notes.strip():
        header_lines.append(f"# {FILM_TSV_FIELDS['notes']}\t{notes.strip()}")

    # --- Save file ---
    data_df = pd.DataFrame(channel_rows)
    filename = f"{_sanitize_filename(manufacturer)}_{_sanitize_filename(name)}_film.tsv"
    out_dir = Path(DATA_FOLDERS['film'])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    success, message = _safe_file_save(data_df, out_path, "Film profile", header_lines=header_lines)
    if success:
        return True, f"Film profile '{manufacturer} {name}' saved to {out_path}"
    return False, message

# ============================================================================
# REFLECTANCE/ABSORPTION IMPORT
# ============================================================================

@_import_errors("reflectance")
def import_reflectance_absorption_from_csv(uploaded_file, meta, extrap_lower, extrap_upper):
    """
    Import reflectance or absorption data from a CSV file.

    Args:
        uploaded_file: The uploaded CSV file
        meta: Dictionary containing metadata
        extrap_lower: Whether to extrapolate to target minimum
        extrap_upper: Whether to extrapolate to target maximum

    Returns:
        Tuple of (success, message)
    """
    # Validate inputs
    if uploaded_file is None:
        return False, "No file provided for import."

    if not meta or not isinstance(meta, dict):
        return False, "Invalid metadata provided."

    # Process as standard reflectance/absorption file
    raw_data = _parse_upload(uploaded_file)
    return _import_standard_reflectance(raw_data, meta, extrap_lower, extrap_upper)


def _import_standard_reflectance(raw_data, meta, extrap_lower, extrap_upper):
    """Import standard 2-column reflectance data."""
    wavelengths, values = _paired_columns(raw_data, 0, 1)

    # Validate, sort, pick the output range and interpolate via the shared
    # filter pipeline
    new_wavelengths, interpolated = _extrapolated_series(
        wavelengths, values, "reflectance", extrap_lower, extrap_upper
    )

    # The value column is always 'Reflectance': that is the column name the
    # reflector loader (services/data.py) requires, so 'Absorption' imports
    # stay loadable too (the chosen data_type is kept for the status message).
    data_type = meta.get("data_type", "Reflectance")
    name = meta.get("name", "Unknown")
    description = meta.get("description", "")

    # Create DataFrame - metadata only in first row (efficient: no wasted arrays)
    output_df = pd.DataFrame({
        'Wavelength': new_wavelengths,
        'Reflectance': interpolated,
        'Name': [name] + [""] * (len(new_wavelengths) - 1),
        'Description': [description] + [""] * (len(new_wavelengths) - 1)
    })

    # Save file
    return _save_reflectance_file(output_df, meta, extrap_lower, extrap_upper, data_type)


def _save_reflectance_file(df, meta, extrap_lower, extrap_upper, data_type):
    """Save reflectance DataFrame to file with proper naming."""
    base_name = meta.get("name", "spectrum")
    sanitized = _sanitize_filename(base_name)
    suffix = _get_extrapolation_suffix(extrap_lower, extrap_upper)
    filename = f"{sanitized}{suffix}.tsv"

    folder = "plant" if "plant" in meta.get("category", "").lower() else "other"
    out_dir = Path(DATA_FOLDERS['reflectors']) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    # Save the file
    success, message = _safe_file_save(df, out_path, data_type)
    return success, message

# ============================================================================
# All import functions are defined above and can be imported individually
# The UI import flows live in views/ (shelf_data.py, shelf_reflectors.py, actions.py).
# ============================================================================
