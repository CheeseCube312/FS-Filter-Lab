"""
Data loading services for FS FilterLab.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Any, Optional, TypeVar, Callable

from models import (
    Filter, FilterCollection, TargetProfile,
    ReflectorSpectrum, ReflectorCollection
)
from models.constants import CACHE_DIR, DEFAULT_HEX_COLOR, DEFAULT_ILLUMINANT, INTERP_GRID

# Ensure cache directory exists
Path(CACHE_DIR).mkdir(exist_ok=True)

# Generic type for cached data
T = TypeVar('T')

def parse_tsv_file(file_path: str | Path) -> pd.DataFrame:
    """
    Parse a TSV file with standardized error handling.
    
    Args:
        file_path: Path to the TSV file
        
    Returns:
        DataFrame with parsed data, columns stripped of whitespace
    """
    path = Path(file_path) if isinstance(file_path, str) else file_path
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def cached_loader(cache_key: str, data_folder: str | Path, 
                  load_function: Callable[[], T]) -> T:
    """
    Simple caching mechanism for data loading.
    
    Args:
        cache_key: Base name for the cache file (without extension)
        data_folder: Path to the folder containing source data files
        load_function: Function to call when cache is invalid or missing
        
    Returns:
        Data from cache or freshly loaded
    """
    data_dir = Path(data_folder)
    cache_file = Path(CACHE_DIR) / f"{cache_key}.pkl"
    cache_time_file = Path(CACHE_DIR) / f"{cache_key}_time.pkl"
    
    # Check if cache exists and is newer than source files
    cache_valid = False
    if cache_file.exists() and cache_time_file.exists():
        try:
            # Load the last modification time we saved
            with open(cache_time_file, 'rb') as f:
                cached_timestamp = pickle.load(f)
            
            # Find the newest file in the data folder
            newest_time = 0
            for filepath in data_dir.glob("**/*.tsv"):
                if filepath.is_file():
                    newest_time = max(newest_time, filepath.stat().st_mtime)
            
            # If our cached time is newer than any data file, cache is valid
            cache_valid = cached_timestamp > newest_time
            
            if cache_valid:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception:
            pass  # Silent failure for cache operations
    
    # Cache miss or invalid - load fresh data
    data = load_function()
    
    # Save to cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
            
        # Save current timestamp
        import time
        with open(cache_time_file, 'wb') as f:
            pickle.dump(time.time(), f)
    except Exception:
        pass  # Silent failure for cache operations
        
    return data


# Helper functions for empty collection creation
def create_empty_filter_collection() -> FilterCollection:
    """Create an empty filter collection."""
    return FilterCollection(
        filters=[],
        df=pd.DataFrame(),
        filter_matrix=np.empty((0, len(INTERP_GRID))),
        extrapolated_masks=np.empty((0, len(INTERP_GRID)), dtype=bool)
    )

def create_empty_reflector_collection() -> ReflectorCollection:
    """Create an empty reflector collection."""
    return ReflectorCollection(
        reflectors=[],
        reflector_matrix=np.empty((0, len(INTERP_GRID)))
    )

def safely_load_file(path: Path, processor_func: Callable) -> Optional[Any]:
    """
    Load and process a file with standardized error handling.
    
    Args:
        path: Path to the file to load
        processor_func: Function to process the file contents
        
    Returns:
        Processed file contents or None if loading failed
    """
    try:
        return processor_func(path)
    except Exception:
        return None


def _process_filter_file(path: Path) -> Optional[Tuple[dict, np.ndarray, np.ndarray, Filter]]:
    """
    Process a single filter file and return its data components.
    
    Args:
        path: Path to the filter file
        
    Returns:
        Tuple of (metadata, transmission, mask, filter) or None if invalid
    """
    df = parse_tsv_file(path)
    
    # Check if file has required columns
    if "Wavelength" not in df.columns or "Transmittance" not in df.columns:
        return None
    
    filename = path.name
    is_lee = 'LeeFilters' in filename
    
    # Extract metadata from first row
    first_row = df.iloc[0]
    
    fn = str(first_row.get('Filter Number', path.stem))
    name_raw = first_row.get('Name')
    name = str(name_raw).strip() if pd.notnull(name_raw) and str(name_raw).strip() else path.stem
    manufacturer = first_row.get('Manufacturer', 'Unknown')
    hex_color_raw = first_row.get('hex_color', DEFAULT_HEX_COLOR)
    hex_color = str(hex_color_raw).strip() if pd.notnull(hex_color_raw) and str(hex_color_raw).strip().startswith("#") else DEFAULT_HEX_COLOR
    
    # Extract wavelength and transmittance values
    wavelengths = df["Wavelength"].astype(float).values
    transmittance = df["Transmittance"].astype(float).values
    
    # Normalize if needed
    if transmittance.max() > 1.5:
        transmittance /= 100.0
    
    # Interpolate to standard grid
    interp_vals = np.interp(INTERP_GRID, wavelengths, transmittance, left=np.nan, right=np.nan)
    extrap_mask = (INTERP_GRID > 700) if is_lee else np.zeros_like(INTERP_GRID, dtype=bool)
    
    metadata = {
        'Filter Number': fn,
        'Filter Name': name,
        'Manufacturer': manufacturer,
        'Hex Color': hex_color,
        'is_lee': is_lee
    }
    
    filter_obj = Filter(
        name=name,
        number=fn,
        manufacturer=manufacturer,
        hex_color=hex_color,
        transmission=interp_vals,
        extrapolated_mask=extrap_mask
    )
    
    return metadata, interp_vals, extrap_mask, filter_obj

def _load_filter_collection_from_files() -> FilterCollection:
    """
    Load filter data from files without using cache.
    
    Returns:
        FilterCollection object
    """
    data_folder = Path("data") / "filters_data"
    data_folder.mkdir(exist_ok=True, parents=True)
    
    files = list(data_folder.glob("**/*.tsv"))
    meta_list, matrix, masks = [], [], []
    filters = []
    
    for path in files:
        result = safely_load_file(path, _process_filter_file)
        if result:
            metadata, transmission, mask, filter_obj = result
            meta_list.append(metadata)
            matrix.append(transmission)
            masks.append(mask)
            filters.append(filter_obj)

    # Return empty collection if no valid data found
    if not matrix:
        return create_empty_filter_collection()
    
    try:
        df_result = pd.DataFrame(meta_list)
        matrix_result = np.vstack(matrix)
        masks_result = np.vstack(masks)
        
        return FilterCollection(
            filters=filters,
            df=df_result,
            filter_matrix=matrix_result,
            extrapolated_masks=masks_result
        )
    except Exception:
        return create_empty_filter_collection()


def load_filter_collection() -> FilterCollection:
    """Load filter data and return a FilterCollection object."""
    def _create_cached_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[Filter]]:
        """Load fresh filter data for caching"""
        collection = _load_filter_collection_from_files()
        return (collection.df, collection.filter_matrix, collection.extrapolated_masks, collection.filters)
    
    # Load data using cache wrapper
    try:
        cached_data = cached_loader(
            cache_key="filter_data",
            data_folder=str(Path("data") / "filters_data"),
            load_function=_create_cached_data
        )
        
        # Unpack cached data
        df, matrix, masks, filters = cached_data
        
        return FilterCollection(
            filters=filters,
            df=df,
            filter_matrix=matrix,
            extrapolated_masks=masks
        )
    except Exception:
        return _load_filter_collection_from_files()


def _process_qe_file(path: Path) -> Optional[Tuple[str, Dict[str, np.ndarray], bool]]:
    """
    Process a quantum efficiency file.
    
    Args:
        path: Path to the QE file
        
    Returns:
        Tuple of (sensor_key, channel_data, is_default) or None if invalid
    """
    df = parse_tsv_file(path)
    
    # Check if file has required columns
    if 'Wavelength' not in df.columns or not any(col in df.columns for col in ['R', 'G', 'B']):
        return None
    
    # Extract sensor info
    brand = df['Manufacturer'].iloc[0].strip() if 'Manufacturer' in df.columns else "Generic"
    model = df['Name'].iloc[0].strip() if 'Name' in df.columns else path.stem
    key = f"{brand} {model}"
    
    # Process channels
    channel_data = {}
    wavelength = df['Wavelength'].astype(float).values
    
    for channel in ['R', 'G', 'B']:
        if channel not in df.columns:
            continue
            
        valid_mask = ~pd.isna(df[channel])
        if not valid_mask.any():
            continue
            
        channel_values = df[channel].values
        interp = np.interp(
            INTERP_GRID, 
            wavelength, 
            channel_values,
            left=np.nan, 
            right=np.nan
        )
        channel_data[channel[0]] = interp
    
    # Check if this is the default QE file
    is_default = (path.name == 'Default_QE.tsv')
    
    return key, channel_data, is_default


def _load_quantum_efficiencies_from_files() -> Tuple[List[str], Dict[str, Dict[str, np.ndarray]], Optional[str]]:
    """
    Load quantum efficiency data from files without using cache.
    
    Returns:
        Tuple of (qe_keys, qe_data, default_key)
    """
    folder = Path('data') / 'QE_data'
    folder.mkdir(exist_ok=True, parents=True)
    
    files = list(folder.glob('*.tsv'))
    qe_dict = {}
    default_key = None

    for path in files:
        result = safely_load_file(path, _process_qe_file)
        if result:
            key, channel_data, is_default = result
            qe_dict[key] = channel_data
            if is_default and default_key is None:
                default_key = key

    return (sorted(qe_dict.keys()), qe_dict, default_key)


def load_quantum_efficiencies() -> Tuple[List[str], Dict[str, Dict[str, np.ndarray]], Optional[str]]:
    """Load quantum efficiency data for camera sensors."""
    return cached_loader(
        cache_key="qe_data",
        data_folder=Path('data') / 'QE_data',
        load_function=_load_quantum_efficiencies_from_files
    )


def _process_illuminant_file(path: Path) -> Optional[Tuple[str, np.ndarray, Optional[str]]]:
    """
    Process an illuminant file.
    
    Args:
        path: Path to the illuminant file
        
    Returns:
        Tuple of (name, interpolated_data, description) or None if invalid
    """
    df = parse_tsv_file(path)
    
    if df.shape[1] < 2:
        return None
    
    # Extract wavelength and power data from the first two columns
    wl = df.iloc[:, 0].astype(float).values
    power = df.iloc[:, 1].astype(float).values
    
    # Interpolate to standard grid
    interp = np.interp(INTERP_GRID, wl, power, left=np.nan, right=np.nan)
    name = path.stem
    
    # Extract description if available
    description = None
    if 'Description' in df.columns and not df['Description'].dropna().empty:
        description = df['Description'].dropna().iloc[0]
    
    return name, interp, description


def _load_illuminant_collection_from_files() -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """
    Load illuminant data from files without using cache.
    
    Returns:
        Tuple of (illuminants, metadata)
    """
    folder = Path('data') / 'illuminants'
    folder.mkdir(exist_ok=True, parents=True)

    illum, meta = {}, {}
    
    for path in folder.glob('*.tsv'):
        result = safely_load_file(path, _process_illuminant_file)
        if result:
            name, interp, description = result
            illum[name] = interp
            if description:
                meta[name] = description

    return (illum, meta)


def load_illuminant_collection() -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """Load illuminant collection."""
    return cached_loader(
        cache_key="illuminants",
        data_folder=Path('data') / 'illuminants',
        load_function=_load_illuminant_collection_from_files
    )


def _process_reflector_file(path: Path) -> Optional[Tuple[str, np.ndarray]]:
    """
    Process a reflector file.
    
    Args:
        path: Path to the reflector file
        
    Returns:
        Tuple of (name, interpolated_data) or None if invalid
    """
    df = parse_tsv_file(path)
    
    # Check for required columns
    if "Wavelength" not in df.columns or "Reflectance" not in df.columns:
        return None
    
    # Extract name from the first row if present, else use filename
    name = None
    if "Name" in df.columns:
        name_values = df["Name"].dropna()
        if len(name_values) > 0:
            name = name_values.iloc[0]
    
    if not name:
        name = path.stem
    
    # Process wavelength and reflectance data
    wl = df["Wavelength"].astype(float).values
    refl = df["Reflectance"].astype(float).values
    
    # Check for sufficient valid data points
    valid_mask = ~np.isnan(refl)
    if np.sum(valid_mask) < 2:
        return None
        
    wl = wl[valid_mask]
    refl = refl[valid_mask]
    
    # Interpolate to standard grid
    interp_vals = np.interp(INTERP_GRID, wl, refl, left=np.nan, right=np.nan)
    
    # Normalize reflectance units: if values look like percents (>1.5), convert to fraction [0..1]
    # This handles existing files that may have been imported before normalization was added
    if np.nanmax(interp_vals) > 1.5:
        interp_vals = interp_vals / 100.0
    
    # Round to 3 decimal places to avoid floating point precision issues
    interp_vals = np.round(interp_vals, 3)
    
    return name, interp_vals


def _load_reflector_collection_from_files() -> ReflectorCollection:
    """
    Load reflector data from files without using cache.
    
    Returns:
        ReflectorCollection object
    """
    folder = Path('data') / 'reflectors'
    folder.mkdir(exist_ok=True, parents=True)

    files = list(folder.glob("**/*.tsv"))
    reflectors = []
    matrix = []

    for path in files:
        result = safely_load_file(path, _process_reflector_file)
        if result:
            name, interp_vals = result
            reflectors.append(ReflectorSpectrum(name=name, values=interp_vals))
            matrix.append(interp_vals)

    if not matrix:
        return create_empty_reflector_collection()
    
    reflector_matrix = np.vstack(matrix)
    return ReflectorCollection(reflectors=reflectors, reflector_matrix=reflector_matrix)


def load_reflector_collection() -> ReflectorCollection:
    """Load reflector collection."""
    return cached_loader(
        cache_key="reflectors",
        data_folder=Path('data') / 'reflectors',
        load_function=_load_reflector_collection_from_files
    )
