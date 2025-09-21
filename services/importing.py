"""
Consolidated data importing utilities for FS FilterLab.

This module contains all data import functionality for filters, illuminants,
quantum efficiencies, and reflectance data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy.interpolate import interp1d
from typing import Tuple, Dict, Any

# ============================================================================
# COMMON UTILITIES
# ============================================================================

def safe_float(val):
    """Convert a value to float safely, handling various formats."""
    try:
        return float(str(val).replace(',', '.').strip())
    except Exception:
        return np.nan


def parse_csv(file, separator=';', fallback_separator=','):
    """Parse a CSV file with auto-detection of separator."""
    raw_data = pd.read_csv(file, sep=separator, header=None, engine='python')
    if raw_data.shape[1] < 2:
        raw_data = pd.read_csv(file, sep=fallback_separator, header=None, engine='python')
    if raw_data.shape[1] < 2:
        raise ValueError("Could not read two columns from the CSV.")
    
    raw_data = raw_data.applymap(safe_float)
    return raw_data


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
        # Parse the CSV
        raw_data = parse_csv(uploaded_file)
        wavelengths = raw_data.iloc[:, 0].dropna().values
        transmissions = raw_data.iloc[:, 1].dropna().values

        if wavelengths.size == 0 or transmissions.size == 0:
            return False, "Wavelength or transmission columns are empty."

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
        output_df.to_csv(out_path, sep='\t', index=False)

        return True, f"Filter data saved to {out_path}"
    except Exception as e:
        return False, f"Error: {str(e)}"

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
        out_dir = Path("data/illuminants")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Parse CSV data
        raw_data = parse_csv(uploaded_file)
        wavelengths = raw_data.iloc[:, 0].dropna().values
        intensity = raw_data.iloc[:, 1].dropna().values

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
        df_out.to_csv(out_path, sep="\t", index=False)

        return True, f"Illuminant saved to {out_path}"
    except Exception as e:
        return False, f"Error: {str(e)}"

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
        # Parse the CSV
        raw_data = parse_csv(uploaded_file)
        
        if raw_data.shape[1] < 4:
            return False, "Expected at least 4 columns (Wavelength, R, G, B)"

        wavelengths = raw_data.iloc[:, 0].dropna().values
        r_qe = raw_data.iloc[:, 1].dropna().values
        g_qe = raw_data.iloc[:, 2].dropna().values  
        b_qe = raw_data.iloc[:, 3].dropna().values

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
        
        output_df.to_csv(out_path, sep='\t', index=False)
        return True, f"QE data saved to {out_path}"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

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
        # Parse the CSV
        raw_data = parse_csv(uploaded_file)
        wavelengths = raw_data.iloc[:, 0].dropna().values
        values = raw_data.iloc[:, 1].dropna().values

        if wavelengths.size == 0 or values.size == 0:
            return False, "Wavelength or value columns are empty."

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
        
        output_df.to_csv(out_path, sep='\t', index=False)
        return True, f"{data_type} data saved to {out_path}"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

# ============================================================================
# All import functions are defined above and can be imported individually
# The UI components have been moved to views/forms.py
# ============================================================================
