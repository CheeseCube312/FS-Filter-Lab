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

Import Functions:
- import_filter_from_csv: Import optical filter transmission data
- import_illuminant_from_csv: Import illuminant spectral power distributions
- import_qe_from_csv: Import camera sensor quantum efficiency data
- import_reflectance_absorption_from_csv: Import surface reflectance/absorption spectra

All functions return (success: bool, message: str) tuples for consistent
error handling in the UI layer.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy.interpolate import interp1d
from typing import Tuple, Dict, Any
import logging

# Configure logging for debugging import issues
logger = logging.getLogger(__name__)

# ============================================================================
# COMMON UTILITIES
# ============================================================================

def safe_float(val):
    """Convert a value to float safely, handling various formats."""
    try:
        return float(str(val).replace(',', '.').strip())
    except Exception:
        return np.nan


def validate_wavelength_range(wavelengths: np.ndarray, data_type: str = "data") -> Tuple[bool, str]:
    """
    Validate wavelength range for spectral data.
    
    Args:
        wavelengths: Array of wavelength values
        data_type: Type of data being validated (for error messages)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if wavelengths.size == 0:
        return False, f"Wavelength column is empty for {data_type}."
        
    min_wl, max_wl = wavelengths.min(), wavelengths.max()
    
    if min_wl < 200 or max_wl > 2000:
        return False, f"Wavelength range ({min_wl:.1f}-{max_wl:.1f} nm) seems invalid. Expected 200-2000 nm."
        
    if max_wl - min_wl < 50:
        return False, f"Wavelength range too narrow ({min_wl:.1f}-{max_wl:.1f} nm). Need at least 50nm range."
        
    return True, ""


def validate_spectral_values(values: np.ndarray, value_type: str, min_val: float = 0, max_val: float = None) -> Tuple[bool, str]:
    """
    Validate spectral values (transmission, reflectance, QE, etc.).
    
    Args:
        values: Array of spectral values
        value_type: Type of values being validated
        min_val: Minimum allowed value
        max_val: Maximum allowed value (None for no limit)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if values.size == 0:
        return False, f"{value_type} column is empty or contains no valid numbers."
        
    if values.min() < min_val:
        return False, f"{value_type} values cannot be below {min_val} (found: {values.min():.3f})."
        
    if max_val is not None and values.max() > max_val:
        return False, f"{value_type} values seem too high (max: {values.max():.3f}). Expected 0-{max_val}."
        
    return True, ""


def safe_file_save(df: pd.DataFrame, file_path: Path, data_type: str) -> Tuple[bool, str]:
    """
    Safely save a DataFrame to a file with error handling.
    
    Args:
        df: DataFrame to save
        file_path: Path to save the file
        data_type: Type of data being saved (for error messages)
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the file
        df.to_csv(file_path, sep='\\t', index=False)
        return True, f"{data_type} data saved successfully to {file_path}"
        
    except PermissionError:
        return False, f"Permission denied: Cannot write to {file_path}. File may be open in another application."
    except OSError as e:
        return False, f"File system error: {str(e)}"
    except Exception as e:
        return False, f"Failed to save file to {file_path}: {str(e)}"


def parse_csv(file, separator=';', fallback_separator=','):
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
            raw_data = raw_data.map(safe_float)  # Use map instead of deprecated applymap
        except AttributeError:
            # Fallback for older pandas versions
            raw_data = raw_data.applymap(safe_float)
        
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


def get_wavelength_range(wavelengths, extrap_lower=False, extrap_upper=False):
    """Determine the wavelength range based on data and extrapolation settings."""
    base_min = int(np.ceil(wavelengths.min() / 5.0)) * 5
    base_max = int(np.floor(wavelengths.max() / 5.0)) * 5
    
    min_wl = 300 if extrap_lower else base_min
    max_wl = 1100 if extrap_upper else min(1100, base_max)
    
    if min_wl > max_wl:
        raise ValueError("Data range is outside allowable bounds.")
        
    return min_wl, max_wl


def interpolate_spectrum(wavelengths, values, target_wavelengths, extrap_lower=False, extrap_upper=False):
    """Interpolate spectral data to a target wavelength range."""
    interpolator = interp1d(wavelengths, values, kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated = interpolator(target_wavelengths)
    
    # Handle extrapolation
    if extrap_lower:
        below_mask = target_wavelengths < wavelengths.min()
        interpolated[below_mask] = values[0]
    if extrap_upper:
        above_mask = target_wavelengths > wavelengths.max()
        interpolated[above_mask] = values[-1]
    
    return np.clip(np.round(interpolated, 3), 0.0, None)


def sanitize_filename(name):
    """Convert a string to a valid filename."""
    return ''.join(c for c in name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')


def get_extrapolation_suffix(extrap_lower, extrap_upper):
    """Get a standardized suffix for extrapolated files."""
    suffix_parts = []
    if extrap_lower: suffix_parts.append("300")
    if extrap_upper: suffix_parts.append("1100")
    return f"_extrapolated_{'_'.join(suffix_parts)}" if suffix_parts else ""

# ============================================================================
# FILTER IMPORT
# ============================================================================

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
    try:   
        # Validate inputs
        if uploaded_file is None:
            return False, "No file provided for import."
            
        if not meta or not isinstance(meta, dict):
            return False, "Invalid metadata provided."
            
        required_keys = ['manufacturer', 'filter_name', 'filter_number', 'hex_color']
        missing_keys = [key for key in required_keys if key not in meta or not meta[key]]
        if missing_keys:
            return False, f"Missing required metadata: {', '.join(missing_keys)}"
        
        # Parse the CSV with detailed error handling
        try:
            raw_data = parse_csv(uploaded_file)
        except ValueError as e:
            return False, f"CSV parsing failed: {str(e)}"
            
        wavelengths = raw_data.iloc[:, 0].dropna().values
        transmissions = raw_data.iloc[:, 1].dropna().values

        # Validate data
        if wavelengths.size == 0:
            return False, "Wavelength column (first column) is empty or contains no valid numbers."
            
        if transmissions.size == 0:
            return False, "Transmission column (second column) is empty or contains no valid numbers."
            
        if wavelengths.size != transmissions.size:
            return False, f"Wavelength and transmission columns have different lengths ({wavelengths.size} vs {transmissions.size})."
            
        # Check wavelength range validity
        min_wl, max_wl = wavelengths.min(), wavelengths.max()
        if min_wl < 200 or max_wl > 2000:
            return False, f"Wavelength range ({min_wl:.1f}-{max_wl:.1f} nm) seems invalid. Expected 200-2000 nm."
            
        if max_wl - min_wl < 50:
            return False, f"Wavelength range too narrow ({min_wl:.1f}-{max_wl:.1f} nm). Need at least 50nm range."
        
        # Validate transmission values (should be 0-100 or 0-1)
        trans_min, trans_max = transmissions.min(), transmissions.max()
        if trans_max > 100:
            return False, f"Transmission values seem too high (max: {trans_max:.2f}). Expected 0-100% or 0-1."
        if trans_min < 0:
            return False, f"Transmission values cannot be negative (min: {trans_min:.2f})."

        # Sort by wavelength
        sort_idx = np.argsort(wavelengths)
        wavelengths = wavelengths[sort_idx]
        transmissions = transmissions[sort_idx]

        # Determine wavelength range
        min_wl, max_wl = get_wavelength_range(wavelengths, extrap_lower, extrap_upper)
        new_wavelengths = np.arange(min_wl, max_wl + 1, 1)
        
        # Interpolate to the new wavelength grid
        interpolated = interpolate_spectrum(
            wavelengths, transmissions, new_wavelengths, 
            extrap_lower, extrap_upper
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
        sanitized = sanitize_filename(base)
        suffix = get_extrapolation_suffix(extrap_lower, extrap_upper)
        filename = f"{sanitized}{suffix}.tsv"
        
        out_dir = os.path.join("data", "filters_data", meta["manufacturer"])
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        
        # Save the file
        try:
            output_df.to_csv(out_path, sep='\t', index=False)
        except Exception as e:
            return False, f"Failed to save file to {out_path}: {str(e)}"

        return True, f"Filter data saved successfully to {out_path}"
        
    except ValueError as e:
        # These are validation errors we want to show to the user
        return False, str(e)
    except Exception as e:
        # Unexpected errors - log them for debugging
        import logging
        logging.exception(f"Unexpected error in filter import: {str(e)}")
        return False, f"Unexpected error during import: {str(e)}. Please check the file format and try again."

# ============================================================================
# ILLUMINANT IMPORT
# ============================================================================

def import_illuminant_from_csv(uploaded_file, description):
    """
    Import illuminant data from a CSV file and save it to the data directory.
    
    Args:
        uploaded_file: The uploaded CSV file
        description: Illuminant description
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Validate inputs
        if uploaded_file is None:
            return False, "No file provided for import."
            
        if not description or not description.strip():
            return False, "Description cannot be empty."
        
        out_dir = Path("data/illuminants")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Parse CSV data with error handling
        try:
            raw_data = parse_csv(uploaded_file)
        except ValueError as e:
            return False, f"CSV parsing failed: {str(e)}"
            
        wavelengths = raw_data.iloc[:, 0].dropna().values
        intensity = raw_data.iloc[:, 1].dropna().values
        
        # Validate data
        if wavelengths.size == 0:
            return False, "Wavelength column (first column) is empty or contains no valid numbers."
            
        if intensity.size == 0:
            return False, "Intensity column (second column) is empty or contains no valid numbers."
            
        if wavelengths.size != intensity.size:
            return False, f"Wavelength and intensity columns have different lengths ({wavelengths.size} vs {intensity.size})."
            
        # Check wavelength range validity
        min_wl, max_wl = wavelengths.min(), wavelengths.max()
        if min_wl < 200 or max_wl > 2000:
            return False, f"Wavelength range ({min_wl:.1f}-{max_wl:.1f} nm) seems invalid. Expected 200-2000 nm."
            
        if max_wl - min_wl < 50:
            return False, f"Wavelength range too narrow ({min_wl:.1f}-{max_wl:.1f} nm). Need at least 50nm range."
        
        # Validate intensity values
        if intensity.min() < 0:
            return False, f"Intensity values cannot be negative (min: {intensity.min():.2f})."

        # Target wavelength range: 300–1100 nm
        full_range = np.arange(300, 1101, 1)
        intensity_interp = np.interp(full_range, wavelengths, intensity, left=0, right=0)

        # Normalize to 0–100 scale
        max_val = np.max(intensity_interp)
        if max_val > 0:
            intensity_rel = np.round((intensity_interp / max_val) * 100, 3)
        else:
            intensity_rel = intensity_interp

        # Create output DataFrame
        df_out = pd.DataFrame({
            "Wavelength (nm)": full_range,
            "Relative Power": intensity_rel,
            "Description": description
        })

        # Save file
        filename = sanitize_filename(f"illuminant_{description}") + ".tsv"
        out_path = out_dir / filename
        
        # Save the file
        try:
            df_out.to_csv(out_path, sep="\t", index=False)
        except Exception as e:
            return False, f"Failed to save file to {out_path}: {str(e)}"

        return True, f"Illuminant data saved successfully to {out_path}"
        
    except ValueError as e:
        # These are validation errors we want to show to the user
        return False, str(e)
    except Exception as e:
        # Unexpected errors - log them for debugging
        import logging
        logging.exception(f"Unexpected error in illuminant import: {str(e)}")
        return False, f"Unexpected error during import: {str(e)}. Please check the file format and try again."

# ============================================================================
# QUANTUM EFFICIENCY IMPORT
# ============================================================================

def import_qe_from_csv(uploaded_file, brand, model):
    """
    Import quantum efficiency data from a CSV file.
    
    Args:
        uploaded_file: The uploaded CSV file
        brand: Camera brand
        model: Camera model
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Validate inputs
        if uploaded_file is None:
            return False, "No file provided for import."
            
        if not brand or not brand.strip():
            return False, "Camera brand cannot be empty."
            
        if not model or not model.strip():
            return False, "Camera model cannot be empty."
        
        # Parse the CSV with error handling
        try:
            raw_data = parse_csv(uploaded_file)
        except ValueError as e:
            return False, f"CSV parsing failed: {str(e)}"
        
        if raw_data.shape[1] < 4:
            return False, f"Expected at least 4 columns (Wavelength, R, G, B), found {raw_data.shape[1]} columns."

        wavelengths = raw_data.iloc[:, 0].dropna().values
        r_qe = raw_data.iloc[:, 1].dropna().values
        g_qe = raw_data.iloc[:, 2].dropna().values  
        b_qe = raw_data.iloc[:, 3].dropna().values
        
        # Validate data
        if wavelengths.size == 0:
            return False, "Wavelength column (first column) is empty or contains no valid numbers."
            
        min_size = min(wavelengths.size, r_qe.size, g_qe.size, b_qe.size)
        if min_size == 0:
            return False, "One or more QE columns (R, G, B) are empty or contain no valid numbers."
            
        # Trim arrays to same size
        wavelengths = wavelengths[:min_size]
        r_qe = r_qe[:min_size]
        g_qe = g_qe[:min_size]
        b_qe = b_qe[:min_size]
        
        # Check wavelength range validity
        min_wl, max_wl = wavelengths.min(), wavelengths.max()
        if min_wl < 200 or max_wl > 2000:
            return False, f"Wavelength range ({min_wl:.1f}-{max_wl:.1f} nm) seems invalid. Expected 200-2000 nm."
            
        if max_wl - min_wl < 50:
            return False, f"Wavelength range too narrow ({min_wl:.1f}-{max_wl:.1f} nm). Need at least 50nm range."
        
        # Validate QE values
        for channel, values in [("R", r_qe), ("G", g_qe), ("B", b_qe)]:
            if values.min() < 0:
                return False, f"{channel} channel QE values cannot be negative (min: {values.min():.3f})."
            if values.max() > 100:
                return False, f"{channel} channel QE values seem too high (max: {values.max():.3f}). Expected 0-1 or 0-100."

        # Interpolate to standard grid (300-1100nm)
        target_wl = np.arange(300, 1101, 1)
        r_interp = np.interp(target_wl, wavelengths, r_qe)
        g_interp = np.interp(target_wl, wavelengths, g_qe)
        b_interp = np.interp(target_wl, wavelengths, b_qe)

        # Create output DataFrame
        output_df = pd.DataFrame({
            'Wavelength': target_wl,
            'R': r_interp,
            'G': g_interp,
            'B': b_interp,
            'Manufacturer': brand,
            'Name': model
        })

        # Save file
        filename = f"{sanitize_filename(brand)}_{sanitize_filename(model)}_QE.tsv"
        out_dir = Path("data/QE_data")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        
        # Save the file
        try:
            output_df.to_csv(out_path, sep='\t', index=False)
        except Exception as e:
            return False, f"Failed to save file to {out_path}: {str(e)}"
            
        return True, f"QE data saved successfully to {out_path}"
        
    except ValueError as e:
        # These are validation errors we want to show to the user
        return False, str(e)
    except Exception as e:
        # Unexpected errors - log them for debugging
        import logging
        logging.exception(f"Unexpected error in QE import: {str(e)}")
        return False, f"Unexpected error during import: {str(e)}. Please check the file format and try again."

# ============================================================================
# REFLECTANCE/ABSORPTION IMPORT
# ============================================================================

def import_reflectance_absorption_from_csv(uploaded_file, meta, extrap_lower, extrap_upper):
    """
    Import reflectance or absorption data from a CSV file.
    
    Args:
        uploaded_file: The uploaded CSV file
        meta: Dictionary containing metadata
        extrap_lower: Whether to extrapolate to 300nm
        extrap_upper: Whether to extrapolate to 1100nm
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Validate inputs
        if uploaded_file is None:
            return False, "No file provided for import."
            
        if not meta or not isinstance(meta, dict):
            return False, "Invalid metadata provided."
            
        # Parse the CSV with detailed error handling
        try:
            raw_data = parse_csv(uploaded_file)
        except ValueError as e:
            return False, f"CSV parsing failed: {str(e)}"
            
        wavelengths = raw_data.iloc[:, 0].dropna().values
        values = raw_data.iloc[:, 1].dropna().values

        # Validate data
        if wavelengths.size == 0:
            return False, "Wavelength column (first column) is empty or contains no valid numbers."
            
        if values.size == 0:
            return False, "Value column (second column) is empty or contains no valid numbers."
            
        if wavelengths.size != values.size:
            return False, f"Wavelength and value columns have different lengths ({wavelengths.size} vs {values.size})."
            
        # Check wavelength range validity
        min_wl, max_wl = wavelengths.min(), wavelengths.max()
        if min_wl < 200 or max_wl > 2000:
            return False, f"Wavelength range ({min_wl:.1f}-{max_wl:.1f} nm) seems invalid. Expected 200-2000 nm."
            
        if max_wl - min_wl < 50:
            return False, f"Wavelength range too narrow ({min_wl:.1f}-{max_wl:.1f} nm). Need at least 50nm range."

        # Sort by wavelength
        sort_idx = np.argsort(wavelengths)
        wavelengths = wavelengths[sort_idx] 
        values = values[sort_idx]

        # Determine wavelength range
        min_wl, max_wl = get_wavelength_range(wavelengths, extrap_lower, extrap_upper)
        new_wavelengths = np.arange(min_wl, max_wl + 1, 1)
        
        # Interpolate
        interpolated = interpolate_spectrum(
            wavelengths, values, new_wavelengths,
            extrap_lower, extrap_upper
        )

        # Create DataFrame
        data_type = meta.get("data_type", "Reflectance")
        output_df = pd.DataFrame({
            'Wavelength': new_wavelengths,
            data_type: interpolated,
            'Name': meta.get("name", "Unknown"),
            'Description': meta.get("description", "")
        })

        # Save file
        base_name = meta.get("name", "spectrum")
        sanitized = sanitize_filename(base_name)
        suffix = get_extrapolation_suffix(extrap_lower, extrap_upper)
        filename = f"{sanitized}{suffix}.tsv"
        
        folder = "plant" if "plant" in meta.get("category", "").lower() else "other"
        out_dir = Path("data/reflectors") / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        
        # Save the file
        try:
            output_df.to_csv(out_path, sep='\t', index=False)
        except Exception as e:
            return False, f"Failed to save file to {out_path}: {str(e)}"
            
        return True, f"{data_type} data saved successfully to {out_path}"
        
    except ValueError as e:
        # These are validation errors we want to show to the user
        return False, str(e)
    except Exception as e:
        # Unexpected errors - log them for debugging
        import logging
        logging.exception(f"Unexpected error in reflectance import: {str(e)}")
        return False, f"Unexpected error during import: {str(e)}. Please check the file format and try again."

# ============================================================================
# All import functions are defined above and can be imported individually
# The UI components have been moved to views/forms.py
# ============================================================================
