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
from typing import Tuple, Dict, Any, List, Optional
import requests
import json
import logging
import re
from models.constants import INTERP_GRID, DATA_FOLDERS, METADATA_FIELDS, OUTPUT_FOLDERS, TSV_ATTRIBUTION_FIELDS

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


def validate_wavelength_data(wavelengths, values, value_type="spectral values"):
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
            raw_data = raw_data.map(safe_float)
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
    
    # Use constants for consistent wavelength range
    target_min, target_max = INTERP_GRID.min(), INTERP_GRID.max()
    
    min_wl = target_min if extrap_lower else base_min
    max_wl = target_max if extrap_upper else min(target_max, base_max)
    
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


def _get_api_field(data: Dict, key: str, join_list: bool = True) -> str:
    """Extract a field from API metadata, handling list/string variations."""
    value = data.get(key, [])
    if isinstance(value, list):
        return ', '.join(value) if join_list else (value[0] if value else '')
    return str(value) if value else ''


def fetch_ecosis_api_metadata(api_url: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Fetch metadata from ECOSIS API endpoint.
    
    Args:
        api_url: ECOSIS API URL (e.g., https://ecosis.org/api/package/package-name)
        
    Returns:
        Tuple of (success, metadata_dict, error_message)
    """
    try:
        logger.info(f"Fetching ECOSIS API metadata from: {api_url}")
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        
        metadata = response.json()
        
        # Validate that this looks like ECOSIS API data
        if not isinstance(metadata, dict):
            return False, None, "API response is not a valid JSON object"
            
        # Check for expected ECOSIS API structure
        if 'ecosis' not in metadata:
            return False, None, "API response does not contain ECOSIS metadata structure"
            
        logger.info("Successfully fetched ECOSIS API metadata")
        return True, metadata, ""
        
    except requests.exceptions.Timeout:
        return False, None, "Request timeout - API server took too long to respond"
    except requests.exceptions.ConnectionError:
        return False, None, "Connection error - check internet connection and API URL"
    except requests.exceptions.HTTPError as e:
        return False, None, f"HTTP error {e.response.status_code}: {e.response.reason}"
    except requests.exceptions.RequestException as e:
        return False, None, f"Request error: {str(e)}"
    except json.JSONDecodeError:
        return False, None, "Invalid JSON response from API"
    except Exception as e:
        logger.error(f"Unexpected error fetching ECOSIS metadata: {e}")
        return False, None, f"Unexpected error: {str(e)}"


def extract_attribution_info(api_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key attribution and metadata information from ECOSIS API response.
    
    Args:
        api_metadata: Full API metadata dictionary
        
    Returns:
        Dictionary with cleaned attribution information
    """
    ecosis_data = api_metadata.get('ecosis', {})
    
    # Build attribution from ecosis nested data and top-level metadata
    attribution = {
        # From ecosis object
        'package_id': ecosis_data.get('package_id', ''),
        'package_name': ecosis_data.get('package_name', ''),
        'package_title': ecosis_data.get('package_title', ''),
        'organization': ecosis_data.get('organization', ''),
        'description': ecosis_data.get('description', ''),
        'license': ecosis_data.get('license', ''),
        'created': ecosis_data.get('created', ''),
        'modified': ecosis_data.get('modified', ''),
        'spectra_count': ecosis_data.get('spectra_count', 0),
        
        # From top-level (use helper for list/string handling)
        'author': _get_api_field(api_metadata, 'Author', join_list=False),
        'citation': _get_api_field(api_metadata, 'Citation', join_list=False),
        'funding_source': _get_api_field(api_metadata, 'Funding Source', join_list=False),
        'year': _get_api_field(api_metadata, 'Year', join_list=False),
        'keywords': _get_api_field(api_metadata, 'Keywords'),
        'theme': _get_api_field(api_metadata, 'Theme'),
        'target_type': _get_api_field(api_metadata, 'Target Type'),
        'measurement_quantity': _get_api_field(api_metadata, 'Measurement Quantity'),
        'measurement_units': _get_api_field(api_metadata, 'Measurement Units'),
        'instrument_manufacturer': _get_api_field(api_metadata, 'Instrument Manufacturer'),
        'instrument_model': _get_api_field(api_metadata, 'Instrument Model'),
        'nasa_gcmd_keywords': _get_api_field(api_metadata, 'NASA GCMD Keywords'),
        'funding_grant_numbers': _get_api_field(api_metadata, 'Funding Source Grant Number'),
        
        # Single-value fields
        'acquisition_method': _get_api_field(api_metadata, 'Acquisition Method', join_list=False),
        'foreoptic_type': _get_api_field(api_metadata, 'Foreoptic Type', join_list=False),
        'light_source': _get_api_field(api_metadata, 'Light Source', join_list=False),
        'target_status': _get_api_field(api_metadata, 'Target Status', join_list=False),
        'ecosystem_type': _get_api_field(api_metadata, 'Ecosystem Type', join_list=False),
        'measurement_venue': _get_api_field(api_metadata, 'Measurement Venue', join_list=False),
    }
    
    # DOI handling (with fallback to linked_data)
    if 'doi' in ecosis_data:
        doi = ecosis_data['doi']
        attribution['doi'] = doi
        if doi.startswith('doi:'):
            attribution['doi_url'] = f"https://doi.org/{doi[4:]}"
    
    # Linked publications (and DOI fallback)
    if 'linked_data' in ecosis_data:
        linked_data = ecosis_data['linked_data']
        if isinstance(linked_data, list) and linked_data:
            links = []
            for link in linked_data:
                if isinstance(link, dict) and 'url' in link and 'label' in link:
                    links.append(f"{link['label']}: {link['url']}")
                    # Fallback: extract DOI URL if none exists
                    if 'doi_url' not in attribution and 'doi.org/' in link.get('url', ''):
                        attribution['doi_url'] = link['url']
            if links:
                attribution['related_publications'] = ' | '.join(links)
    
    # Processing info
    proc_flags = []
    for field in ['Processing Averaged', 'Processing Interpolated', 'Processing Resampled']:
        val = _get_api_field(api_metadata, field, join_list=False)
        if val:
            proc_flags.append(f"{field.replace('Processing ', '')}: {val}")
    if proc_flags:
        attribution['processing_info'] = ' | '.join(proc_flags)
    
    # Measurement date (handle range)
    dates = api_metadata.get('Measurement Date', [])
    if isinstance(dates, list) and dates:
        dates = sorted([d for d in dates if d])
        attribution['measurement_date'] = dates[0] if len(dates) == 1 else f"{dates[0]} to {dates[-1]}"
    elif isinstance(dates, str) and dates:
        attribution['measurement_date'] = dates
    
    # Remove empty values
    return {k: v for k, v in attribution.items() if v and str(v).strip()}


# ============================================================================
# ECOSIS UTILITIES
# ============================================================================

def get_ecosis_csv_metadata_columns(file_path: str) -> List[str]:
    """
    Get available metadata column names from an ECOSIS CSV file.
    
    Args:
        file_path: Path to ECOSIS CSV file
        
    Returns:
        List of non-wavelength column names
    """
    try:
        df = pd.read_csv(file_path, nrows=1)  # Only read header row for efficiency
        
        metadata_cols = []
        for col in df.columns:
            try:
                wavelength = float(str(col).strip())
                # Skip if this looks like a wavelength column
                if 200 <= wavelength <= 3000:
                    continue
            except (ValueError, TypeError):
                # This is a metadata column
                metadata_cols.append(str(col))
        
        return metadata_cols
        
    except Exception:
        return []


def import_ecosis_csv(file_path: str, output_dir: str, api_url: Optional[str] = None, name_column: Optional[str] = None, relevant_metadata: Optional[List[str]] = None) -> List[str]:
    """
    Import ECOSIS CSV file directly. User explicitly chooses this format.
    
    Args:
        file_path: Path to ECOSIS CSV file
        output_dir: Directory to save processed files
        api_url: Optional ECOSIS API URL for metadata enhancement
        name_column: Optional column name to use for search names (stores column name, not value)
        relevant_metadata: Optional list of column names for Surface Color Preview display
        
    Returns:
        List of created file paths
    """
    try:
        # Fetch API metadata if provided
        api_attribution = {}
        
        if api_url:
            success, api_metadata, error_msg = fetch_ecosis_api_metadata(api_url)
            if success and api_metadata:
                api_attribution = extract_attribution_info(api_metadata)
                logger.info(f"Fetched API metadata with {len(api_attribution)} attribution fields")
            else:
                logger.warning(f"Failed to fetch API metadata: {error_msg}")
                # Continue without API metadata
        
        # Always add source file and API URL to attribution (even if API fetch failed)
        source_filename = os.path.basename(file_path)
        api_attribution['source_csv_file'] = source_filename
        if api_url:
            api_attribution['api_url'] = api_url
        
        df = pd.read_csv(file_path)
        
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
        
        for index, row in df.iterrows():
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
                sample_parts = [f"Sample_{index + 1}"]
            
            # Add dataset identifier if available from API
            if api_attribution and 'organization' in api_attribution:
                org_short = api_attribution['organization'][:10] if api_attribution['organization'] else ''
                if org_short:
                    org_short = re.sub(r'[^\w]', '', org_short)  # Clean organization name
                    sample_parts.insert(0, org_short)
            
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
                        float_val = safe_float(value)
                        if not np.isnan(float_val):
                            spectral_wavelengths.append(wavelength)
                            spectral_values.append(float_val)
                    except:
                        continue  # Skip invalid values
            
            if len(spectral_wavelengths) < 5:
                logger.warning(f"Skipping sample {sample_name} - insufficient valid data points ({len(spectral_wavelengths)})")
                continue
            
            spectral_wavelengths = np.array(spectral_wavelengths)
            spectral_values = np.array(spectral_values)
            
            # Filter to INTERP_GRID range (300-1100nm) before interpolation
            wl_min, wl_max = INTERP_GRID[0], INTERP_GRID[-1]
            mask = (spectral_wavelengths >= wl_min) & (spectral_wavelengths <= wl_max)
            
            if np.sum(mask) < 5:
                logger.warning(f"Skipping sample {sample_name} - insufficient data in {wl_min}-{wl_max}nm range")
                continue
                
            filtered_wavelengths = spectral_wavelengths[mask]
            filtered_values = spectral_values[mask]
            
            # Interpolate to standard 1nm grid (300-1100nm)
            interp_func = interp1d(filtered_wavelengths, filtered_values, kind='linear', 
                                 bounds_error=False, fill_value='extrapolate')
            interp_values = interp_func(INTERP_GRID)
            
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
            
            # Add API attribution as comments if available
            if api_attribution:
                rows.append("# Dataset Attribution (from ECOSIS API):")
                for display_name, key in TSV_ATTRIBUTION_FIELDS:
                    if key in api_attribution and api_attribution[key]:
                        value = str(api_attribution[key])
                        if len(value) > 100:
                            value = value[:100] + "..."
                        rows.append(f"# {display_name}\t{value}")
                rows.append("#")  # Blank comment line
            
            # Simple data table: just Wavelength and Reflectance
            rows.append("Wavelength\tReflectance")
            
            # Data rows - clean and simple
            for wl, refl in zip(INTERP_GRID, interp_values):
                rows.append(f"{wl}\t{refl:.10f}")
            
            # Save as TSV file  
            filename = f"{sample_name}_{index+1}.tsv"  # Add index to avoid duplicates
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write('\n'.join(rows))
            
            created_files.append(filepath)
            
        # Save complete API attribution information to a separate JSON file
        if api_attribution:
            attribution_file = os.path.join(output_dir, "dataset_attribution.json")
            with open(attribution_file, 'w', encoding='utf-8') as f:
                json.dump(api_attribution, f, indent=2, ensure_ascii=False)
            logger.info(f"Complete attribution information saved to {attribution_file}")
            
        # Log summary of what was processed
        attribution_info = ""
        if api_attribution:
            org = api_attribution.get('organization', 'Unknown')
            title = api_attribution.get('package_title', 'Unknown')
            attribution_info = f" with attribution to {org} - {title}"
        
        logger.info(f"Successfully processed {len(created_files)} ECOSIS samples{attribution_info}")
        return created_files
        
    except Exception as e:
        raise ValueError(f"Error importing ECOSIS CSV: {str(e)}")


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

        # Validate wavelength data using common utility
        is_valid, error_msg = validate_wavelength_data(wavelengths, transmissions, "transmission")
        if not is_valid:
            return False, error_msg
        
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
        
        out_dir = os.path.join(OUTPUT_FOLDERS['filter_import'], meta["manufacturer"])
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
        logger.exception(f"Unexpected error in filter import: {str(e)}")
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
        
        out_dir = Path(DATA_FOLDERS['illuminants'])
        out_dir.mkdir(parents=True, exist_ok=True)

        # Parse CSV data with error handling
        try:
            raw_data = parse_csv(uploaded_file)
        except ValueError as e:
            return False, f"CSV parsing failed: {str(e)}"
            
        wavelengths = raw_data.iloc[:, 0].dropna().values
        intensity = raw_data.iloc[:, 1].dropna().values
        
        # Validate wavelength data using common utility
        is_valid, error_msg = validate_wavelength_data(wavelengths, intensity, "intensity")
        if not is_valid:
            return False, error_msg
        
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
        logger.exception(f"Unexpected error in illuminant import: {str(e)}")
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
        
        # Basic validation - trim to minimum size
        if wavelengths.size == 0:
            return False, "Wavelength column is empty or contains no valid numbers."
            
        min_size = min(wavelengths.size, r_qe.size, g_qe.size, b_qe.size)
        if min_size == 0:
            return False, "One or more QE columns (R, G, B) are empty or contain no valid numbers."
            
        wavelengths, r_qe, g_qe, b_qe = wavelengths[:min_size], r_qe[:min_size], g_qe[:min_size], b_qe[:min_size]
        
        # Validate wavelength range
        is_valid, error_msg = validate_wavelength_data(wavelengths, r_qe, "QE")
        if not is_valid:
            return False, error_msg
        
        # Validate QE values for each channel
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
        out_dir = Path(DATA_FOLDERS['qe'])
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
        logger.exception(f"Unexpected error in QE import: {str(e)}")
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
        extrap_lower: Whether to extrapolate to target minimum
        extrap_upper: Whether to extrapolate to target maximum
        
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
            
        # Process as standard reflectance/absorption file
        return _import_standard_reflectance(raw_data, meta, extrap_lower, extrap_upper)
        
    except ValueError as e:
        # These are validation errors we want to show to the user
        return False, str(e)
    except Exception as e:
        # Unexpected errors - log them for debugging
        logger.exception(f"Unexpected error in reflectance import: {str(e)}")
        return False, f"Unexpected error during import: {str(e)}. Please check the file format and try again."


def _import_standard_reflectance(raw_data, meta, extrap_lower, extrap_upper):
    """Import standard 2-column reflectance data."""
    wavelengths = raw_data.iloc[:, 0].dropna().values
    values = raw_data.iloc[:, 1].dropna().values

    # Validate wavelength data using common utility
    is_valid, error_msg = validate_wavelength_data(wavelengths, values, "reflectance")
    if not is_valid:
        return False, error_msg

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

    # Normalize reflectance units: if values look like percents (>1.5), convert to fraction [0..1]
    if np.nanmax(interpolated) > 1.5:
        interpolated = interpolated / 100.0

    interpolated = np.round(interpolated, 3)

    # Create DataFrame - metadata only in first row (efficient: no wasted arrays)
    data_type = meta.get("data_type", "Reflectance")
    name = meta.get("name", "Unknown")
    description = meta.get("description", "")
    
    output_df = pd.DataFrame({
        'Wavelength': new_wavelengths,
        data_type: interpolated,
        'Name': [name] + [""] * (len(new_wavelengths) - 1),
        'Description': [description] + [""] * (len(new_wavelengths) - 1)
    })

    # Save file
    return _save_reflectance_file(output_df, meta, extrap_lower, extrap_upper, data_type)


def _save_reflectance_file(df, meta, extrap_lower, extrap_upper, data_type):
    """Save reflectance DataFrame to file with proper naming."""
    base_name = meta.get("name", "spectrum")
    sanitized = sanitize_filename(base_name)
    suffix = get_extrapolation_suffix(extrap_lower, extrap_upper)
    filename = f"{sanitized}{suffix}.tsv"
    
    folder = "plant" if "plant" in meta.get("category", "").lower() else "other"
    out_dir = Path(DATA_FOLDERS['reflectors']) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    
    # Save the file
    success, message = safe_file_save(df, out_path, data_type)
    return success, message

# ============================================================================
# All import functions are defined above and can be imported individually
# The UI components have been moved to views/forms.py
# ============================================================================
