"""
Data loading services for FS FilterLab.
"""
# Standard library imports
import json
from io import StringIO
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, TypeVar, Callable

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from models import (
    Filter, FilterCollection,
    ReflectorSpectrum, ReflectorCollection
)
from models.core import (
    RGBSensorProfile, ColorSlideFilmProfile,
    ColorNegativeFilmProfile, BWFilmProfile, SensorProfile
)
from models.constants import (
    CACHE_DIR, DEFAULT_HEX_COLOR, INTERP_GRID,
    DATA_FOLDERS, TSV_COLUMNS, METADATA_FIELDS, SPECTRAL_CONFIG, FILM_TSV_FIELDS
)


def _interpolate_to_standard_grid(wavelengths: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Interpolate spectral data to the standard wavelength grid.
    
    Args:
        wavelengths: Input wavelength array
        values: Corresponding spectral values (transmission, power, reflectance, etc.)
        
    Returns:
        Interpolated values on the standard INTERP_GRID
    """
    interp_vals = np.interp(INTERP_GRID, wavelengths, values, left=np.nan, right=np.nan)
    # Replace NaN with 0 (represents no transmission/response outside measured range)
    return np.nan_to_num(interp_vals, nan=0.0)

# Generic type for cached data
T = TypeVar('T')

def _parse_comment_headers(file_path: str | Path) -> Tuple[Dict[str, str], List[str]]:
    """
    Parse comment headers from a TSV file and return metadata and data lines.
    
    Args:
        file_path: Path to the TSV file with comment headers
        
    Returns:
        Tuple of (metadata_dict, data_lines) where data_lines excludes comment headers
    """
    path = Path(file_path) if isinstance(file_path, str) else file_path
    
    metadata = {}
    data_lines = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n\r')
            
            # Skip empty lines
            if not line.strip():
                continue
                
            # Check if this is a comment line with metadata
            if line.startswith('# ') and '\t' in line:
                # Extract key-value pair from comment
                content = line[2:]  # Remove "# "
                if '\t' in content:
                    key, value = content.split('\t', 1)
                    metadata[key.strip()] = value.strip()
            elif line.startswith('#'):
                # Skip section headers and other comment-only lines
                continue
            else:
                # This is a data line
                data_lines.append(line)
    
    return metadata, data_lines


def _parse_tsv_with_comments(file_path: str | Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Parse a TSV file with comment headers, returning both data and metadata.
    
    Args:
        file_path: Path to the TSV file
        
    Returns:
        Tuple of (dataframe, metadata_dict)
    """
    metadata, data_lines = _parse_comment_headers(file_path)
    
    if not data_lines:
        return pd.DataFrame(), metadata
    
    # Create a temporary file-like object from data lines
    data_content = '\n'.join(data_lines)
    df = pd.read_csv(StringIO(data_content), sep='\t')
    df.columns = [str(c).strip() for c in df.columns]
    
    return df, metadata


def _parse_tsv_file(file_path: str | Path) -> pd.DataFrame:
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


CACHE_FORMAT = "fs-filterlab-npz-1"  # bump to invalidate all cache files


def _newest_source_mtime(data_folder: str | Path) -> float:
    """
    Newest mtime among a data folder's TSV files and this module.

    Including this module means edits to the loading code silently
    invalidate every cache on the next start.
    """
    newest = Path(__file__).stat().st_mtime
    for path in Path(data_folder).glob("**/*.tsv"):
        if path.is_file():
            newest = max(newest, path.stat().st_mtime)
    return newest


def _read_cache(cache_file: Path, data_folder: str | Path
                ) -> Optional[Tuple[Dict[str, np.ndarray], dict]]:
    """
    Load an npz cache file if present and newer than every source file.

    Returns:
        Tuple of (arrays, meta) for a valid cache, otherwise None.
    """
    if not cache_file.exists():
        return None
    try:
        with np.load(cache_file, allow_pickle=False) as npz:
            arrays = {name: npz[name] for name in npz.files}
        meta = json.loads(arrays.pop("__meta__").item())
        if meta.get("format") != CACHE_FORMAT:
            return None
        # The cache file's own kernel mtime is the write stamp: same clock as
        # the source mtimes, so the comparison is skew-free (no time.time()).
        # A tie counts as valid: the build read the sources within that tick.
        if cache_file.stat().st_mtime < _newest_source_mtime(data_folder):
            return None
        return arrays, meta
    except Exception:
        return None  # Silent failure for cache operations


def _write_cache(cache_file: Path, arrays: Dict[str, np.ndarray], meta: dict) -> None:
    """Write numeric arrays plus a JSON meta blob into a single npz file."""
    meta = {"format": CACHE_FORMAT, **meta}
    payload = {name: np.asarray(arr) for name, arr in arrays.items()}
    payload["__meta__"] = np.array(json.dumps(meta))
    with open(cache_file, "wb") as f:
        np.savez(f, **payload)


def _cached_loader(cache_key: str, data_folder: str | Path,
                   load_function: Callable[[], T],
                   pack: Callable[[T], Tuple[Dict[str, np.ndarray], dict]],
                   unpack: Callable[[Dict[str, np.ndarray], dict], T]) -> T:
    """
    npz-backed caching: one ``{key}.npz`` file per dataset in program/cache/.

    Numeric arrays are stored as npz entries; strings and scalars travel in a
    JSON blob inside the same file, so no pickle is involved. A cache stays
    valid while it is newer than every source ``.tsv`` (and this module);
    anything else -- missing, stale, corrupt -- silently triggers a rebuild.

    Args:
        cache_key: Cache file base name (without extension)
        data_folder: Folder containing the source .tsv files
        load_function: Builds the data when the cache is missing or stale
        pack: data -> (arrays, meta) for storage
        unpack: (arrays, meta) -> data on a cache hit

    Returns:
        Data from cache or freshly loaded
    """
    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.npz"

    hit = _read_cache(cache_file, data_folder)
    if hit is not None:
        try:
            return unpack(*hit)
        except Exception:
            pass  # Undecodable payload -> rebuild silently

    data = load_function()
    try:
        arrays, meta = pack(data)
        _write_cache(cache_file, arrays, meta)
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
        df=pd.DataFrame(),
        reflector_matrix=np.empty((0, len(INTERP_GRID)))
    )

def _safely_load_file(path: Path, processor_func: Callable) -> Optional[Any]:
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
    df = _parse_tsv_file(path)
    
    # Check if file has required columns
    if TSV_COLUMNS['wavelength'] not in df.columns or TSV_COLUMNS['transmittance'] not in df.columns:
        return None
    
    filename = path.name
    is_lee = 'LeeFilters' in filename
    
    # Extract metadata from first row
    first_row = df.iloc[0]
    
    fn = str(first_row.get(TSV_COLUMNS['filter_number'], path.stem))
    name_raw = first_row.get(TSV_COLUMNS['filter_name'])
    name = str(name_raw).strip() if pd.notnull(name_raw) and str(name_raw).strip() else path.stem
    manufacturer = first_row.get(TSV_COLUMNS['manufacturer'], 'Unknown')
    hex_color_raw = first_row.get(TSV_COLUMNS['hex_color'], DEFAULT_HEX_COLOR)
    hex_color = str(hex_color_raw).strip() if pd.notnull(hex_color_raw) and str(hex_color_raw).strip().startswith("#") else DEFAULT_HEX_COLOR
    
    # Extract wavelength and transmittance values
    wavelengths = df[TSV_COLUMNS['wavelength']].astype(float).values
    transmittance = df[TSV_COLUMNS['transmittance']].astype(float).values
    
    # Convert from percentage (0-100) to fractional (0-1) scale
    # All data files store values in 0-100 scale; internal calculations use 0-1
    transmittance = transmittance / 100.0
    
    # Interpolate to standard grid
    interp_vals = _interpolate_to_standard_grid(wavelengths, transmittance)
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
    data_folder = Path(DATA_FOLDERS['filters'])
    data_folder.mkdir(exist_ok=True, parents=True)
    
    files = list(data_folder.glob("**/*.tsv"))
    meta_list, matrix, masks = [], [], []
    filters = []
    
    for path in files:
        result = _safely_load_file(path, _process_filter_file)
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


def _pack_filter_collection(collection: FilterCollection) -> Tuple[Dict[str, np.ndarray], dict]:
    """FilterCollection -> npz arrays (matrices) + JSON meta (df, filter fields)."""
    arrays = {
        "filter_matrix": collection.filter_matrix,
        "extrapolated_masks": collection.extrapolated_masks,
    }
    meta = {
        "df": {col: collection.df[col].tolist() for col in collection.df.columns},
        "filters": [
            {"name": f.name, "number": f.number,
             "manufacturer": f.manufacturer, "hex_color": f.hex_color}
            for f in collection.filters
        ],
    }
    return arrays, meta


def _unpack_filter_collection(arrays: Dict[str, np.ndarray], meta: dict) -> FilterCollection:
    """Rebuild a FilterCollection; per-filter arrays are matrix rows."""
    filters = [
        Filter(
            name=spec["name"], number=spec["number"],
            manufacturer=spec["manufacturer"], hex_color=spec["hex_color"],
            transmission=arrays["filter_matrix"][i].copy(),
            extrapolated_mask=arrays["extrapolated_masks"][i].copy(),
        )
        for i, spec in enumerate(meta["filters"])
    ]
    return FilterCollection(
        filters=filters,
        df=pd.DataFrame(meta["df"]),
        filter_matrix=arrays["filter_matrix"],
        extrapolated_masks=arrays["extrapolated_masks"],
    )


def load_filter_collection() -> FilterCollection:
    """Load filter data and return a FilterCollection object."""
    return _cached_loader(
        cache_key="filter_data",
        data_folder=DATA_FOLDERS['filters'],
        load_function=_load_filter_collection_from_files,
        pack=_pack_filter_collection,
        unpack=_unpack_filter_collection,
    )


def _process_qe_file(path: Path) -> Optional[Tuple[str, Dict[str, np.ndarray], bool]]:
    """
    Process a quantum efficiency file.
    
    Args:
        path: Path to the QE file
        
    Returns:
        Tuple of (sensor_key, channel_data, is_default) or None if invalid
    """
    df = _parse_tsv_file(path)
    
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
        # Replace NaN with 0 (no response outside measured range)
        interp = np.nan_to_num(interp, nan=0.0)
        # Convert QE percentages (0-100) to fractional (0-1) scale for internal calculations
        interp = interp / 100.0
        channel_data[channel[0]] = interp
    
    # Check if this is the default QE file
    is_default = (path.name == 'Default_QE.tsv')
    
    return key, channel_data, is_default
def _process_film_file(path: Path) -> Optional[Tuple[str, 'SensorProfile']]:
    """
    Process a film sensitivity TSV file and return the appropriate SensorProfile subclass.

    File format (comment headers, then a tab-separated data table):
        # Manufacturer: <value>
        # Name:         <value>
        # Type:         color_negative | color_slide | bw
        # ISO:          <optional>
        # Process:      <optional>
        # Notes:        <optional>
        Wavelength  R   G   B      (or single column V for B&W)

    Returns:
        Tuple of (profile_key, SensorProfile subclass instance), or None if invalid.
    """
    try:
        df, metadata = _parse_tsv_with_comments(path)
    except Exception:
        return None

    if 'Wavelength' not in df.columns:
        return None

    # --- Required metadata ---
    manufacturer = metadata.get(FILM_TSV_FIELDS['manufacturer'], '').strip()
    name = metadata.get(FILM_TSV_FIELDS['name'], '').strip()
    film_type = metadata.get(FILM_TSV_FIELDS['type'], '').strip().lower()

    if not manufacturer or not name or not film_type:
        return None

    valid_types = {'rgb_sensor', 'color_slide', 'color_negative', 'bw'}
    if film_type not in valid_types:
        return None

    # --- Optional metadata ---
    iso = metadata.get(FILM_TSV_FIELDS['iso'], '').strip() or None
    process = metadata.get(FILM_TSV_FIELDS['process'], '').strip() or None
    notes = metadata.get(FILM_TSV_FIELDS['notes'], '').strip() or None

    wavelength = df['Wavelength'].astype(float).values

    def _interp_channel(col: str) -> Optional[np.ndarray]:
        if col not in df.columns:
            return None
        valid = ~pd.isna(df[col])
        if not valid.any():
            return None
        vals = np.interp(INTERP_GRID, wavelength, df[col].values, left=np.nan, right=np.nan)
        vals = np.nan_to_num(vals, nan=0.0)
        # Normalise to 0-1 if stored as 0-100
        if vals.max() > 1.5:
            vals = vals / 100.0
        return vals

    key = f"{manufacturer} {name}"
    common = dict(manufacturer=manufacturer, name=name, iso=iso, process=process, notes=notes)

    if film_type == 'bw':
        sensitivity = _interp_channel('V')
        if sensitivity is None:
            # Fall back to first non-Wavelength column
            data_cols = [c for c in df.columns if c != 'Wavelength']
            if not data_cols:
                return None
            sensitivity = _interp_channel(data_cols[0])
        if sensitivity is None:
            return None
        profile = BWFilmProfile(**common, sensitivity=sensitivity)

    else:
        channel_data = {}
        for ch in ['R', 'G', 'B']:
            arr = _interp_channel(ch)
            if arr is not None:
                channel_data[ch] = arr
        if not channel_data:
            return None

        if film_type == 'color_slide':
            profile = ColorSlideFilmProfile(**common, channel_data=channel_data)
        elif film_type == 'color_negative':
            profile = ColorNegativeFilmProfile(**common, channel_data=channel_data)
        else:  # rgb_sensor handled via film folder (unusual but valid)
            profile = RGBSensorProfile(**common, channel_data=channel_data)

    return key, profile


def _load_film_profiles_from_files() -> Dict[str, 'SensorProfile']:
    """Load all film sensitivity profiles from the film_data folder."""
    folder = Path(DATA_FOLDERS['film'])
    folder.mkdir(exist_ok=True, parents=True)

    profiles: Dict[str, SensorProfile] = {}
    for path in sorted(folder.glob('*.tsv')):
        result = _safely_load_file(path, _process_film_file)
        if result:
            key, profile = result
            profiles[key] = profile
    return profiles


# sensor_type tag -> profile class, for film-profile cache round-trips
_FILM_PROFILE_CLASSES = {
    'rgb_sensor': RGBSensorProfile,
    'color_slide': ColorSlideFilmProfile,
    'color_negative': ColorNegativeFilmProfile,
    'bw': BWFilmProfile,
}


def _pack_film_profiles(profiles: Dict[str, SensorProfile]) -> Tuple[Dict[str, np.ndarray], dict]:
    """Profile dict -> npz arrays (sensitivity curves) + JSON meta (fields)."""
    arrays: Dict[str, np.ndarray] = {}
    entries = []
    for i, (key, profile) in enumerate(profiles.items()):
        channels = []
        if profile.sensor_type == 'bw':
            arrays[f"film_{i}_V"] = profile.sensitivity
        else:
            for ch, arr in profile.channel_data.items():
                arrays[f"film_{i}_{ch}"] = arr
                channels.append(ch)
        entries.append({
            "key": key, "type": profile.sensor_type,
            "manufacturer": profile.manufacturer, "name": profile.name,
            "iso": profile.iso, "process": profile.process, "notes": profile.notes,
            "channels": channels,
        })
    return arrays, {"profiles": entries}


def _unpack_film_profiles(arrays: Dict[str, np.ndarray], meta: dict) -> Dict[str, SensorProfile]:
    profiles: Dict[str, SensorProfile] = {}
    for i, entry in enumerate(meta["profiles"]):
        cls = _FILM_PROFILE_CLASSES[entry["type"]]
        common = dict(
            manufacturer=entry["manufacturer"], name=entry["name"],
            iso=entry["iso"], process=entry["process"], notes=entry["notes"],
        )
        if entry["type"] == 'bw':
            profiles[entry["key"]] = cls(**common, sensitivity=arrays[f"film_{i}_V"].copy())
        else:
            channel_data = {ch: arrays[f"film_{i}_{ch}"].copy() for ch in entry["channels"]}
            profiles[entry["key"]] = cls(**common, channel_data=channel_data)
    return profiles


def load_film_profiles() -> Dict[str, 'SensorProfile']:
    """Load film sensitivity profiles, with caching."""
    return _cached_loader(
        cache_key="film_profiles",
        data_folder=DATA_FOLDERS['film'],
        load_function=_load_film_profiles_from_files,
        pack=_pack_film_profiles,
        unpack=_unpack_film_profiles,
    )


def build_sensor_profiles(
    qe_data: Dict[str, Dict[str, np.ndarray]],
    film_profiles: Dict[str, 'SensorProfile'],
) -> Dict[str, 'SensorProfile']:
    """
    Merge digital-sensor QE data and film profiles into a single dict of
    SensorProfile objects keyed by display name.

    Digital sensor QE dicts are wrapped in RGBSensorProfile; film profiles
    are included as-is.
    """
    profiles: Dict[str, SensorProfile] = {}

    for key, channel_data in qe_data.items():
        # Key is already "Brand Model" from _process_qe_file
        parts = key.split(' ', 1)
        manufacturer = parts[0] if len(parts) > 1 else 'Generic'
        name = parts[1] if len(parts) > 1 else key
        profiles[key] = RGBSensorProfile(
            manufacturer=manufacturer,
            name=name,
            channel_data=channel_data,
        )

    # Film profiles override any QE entry with the same key (unlikely but safe)
    profiles.update(film_profiles)
    return profiles


def _load_quantum_efficiencies_from_files() -> Tuple[List[str], Dict[str, Dict[str, np.ndarray]], Optional[str]]:
    """
    Load quantum efficiency data from files without using cache.
    
    Returns:
        Tuple of (qe_keys, qe_data, default_key)
    """
    folder = Path(DATA_FOLDERS['qe'])
    folder.mkdir(exist_ok=True, parents=True)
    
    files = list(folder.glob('*.tsv'))
    qe_dict = {}
    default_key = None

    for path in files:
        result = _safely_load_file(path, _process_qe_file)
        if result:
            key, channel_data, is_default = result
            qe_dict[key] = channel_data
            if is_default and default_key is None:
                default_key = key

    return (sorted(qe_dict.keys()), qe_dict, default_key)


def _pack_qe_payload(payload: Tuple[List[str], Dict[str, Dict[str, np.ndarray]], Optional[str]]
                     ) -> Tuple[Dict[str, np.ndarray], dict]:
    """QE tuple -> npz arrays (channel curves) + JSON meta (keys, default)."""
    keys, qe_data, default_key = payload
    arrays: Dict[str, np.ndarray] = {}
    channels_per_key = []
    for i, key in enumerate(keys):
        channels = qe_data[key]
        channels_per_key.append(list(channels))
        for ch, arr in channels.items():
            arrays[f"qe_{i}_{ch}"] = arr
    return arrays, {"keys": keys, "channels": channels_per_key, "default_key": default_key}


def _unpack_qe_payload(arrays: Dict[str, np.ndarray], meta: dict
                       ) -> Tuple[List[str], Dict[str, Dict[str, np.ndarray]], Optional[str]]:
    qe_data = {
        key: {ch: arrays[f"qe_{i}_{ch}"].copy() for ch in channels}
        for i, (key, channels) in enumerate(zip(meta["keys"], meta["channels"]))
    }
    return meta["keys"], qe_data, meta["default_key"]


def load_quantum_efficiencies() -> Tuple[List[str], Dict[str, Dict[str, np.ndarray]], Optional[str]]:
    """Load quantum efficiency data for camera sensors."""
    return _cached_loader(
        cache_key="qe_data",
        data_folder=DATA_FOLDERS['qe'],
        load_function=_load_quantum_efficiencies_from_files,
        pack=_pack_qe_payload,
        unpack=_unpack_qe_payload,
    )


def _process_illuminant_file(path: Path) -> Optional[Tuple[str, np.ndarray, Optional[str]]]:
    """
    Process an illuminant file.
    
    Args:
        path: Path to the illuminant file
        
    Returns:
        Tuple of (name, interpolated_data, description) or None if invalid
    """
    df = _parse_tsv_file(path)
    
    if df.shape[1] < 2:
        return None
    
    # Extract wavelength and power data from the first two columns
    wl = df.iloc[:, 0].astype(float).values
    power = df.iloc[:, 1].astype(float).values
    
    # Interpolate to standard grid
    interp = _interpolate_to_standard_grid(wl, power)
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
    folder = Path(DATA_FOLDERS['illuminants'])
    folder.mkdir(exist_ok=True, parents=True)

    illum, meta = {}, {}
    
    for path in folder.glob('*.tsv'):
        result = _safely_load_file(path, _process_illuminant_file)
        if result:
            name, interp, description = result
            illum[name] = interp
            if description:
                meta[name] = description

    return (illum, meta)


def _pack_illuminants(payload: Tuple[Dict[str, np.ndarray], Dict[str, str]]
                      ) -> Tuple[Dict[str, np.ndarray], dict]:
    """Illuminant tuple -> npz arrays (spectra) + JSON meta (names, descriptions)."""
    illum, descriptions = payload
    arrays = {f"illum_{i}": arr for i, arr in enumerate(illum.values())}
    return arrays, {"names": list(illum), "descriptions": descriptions}


def _unpack_illuminants(arrays: Dict[str, np.ndarray], meta: dict
                        ) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    illum = {name: arrays[f"illum_{i}"].copy() for i, name in enumerate(meta["names"])}
    return illum, meta["descriptions"]


def load_illuminant_collection() -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """Load illuminant collection."""
    return _cached_loader(
        cache_key="illuminants",
        data_folder=DATA_FOLDERS['illuminants'],
        load_function=_load_illuminant_collection_from_files,
        pack=_pack_illuminants,
        unpack=_unpack_illuminants,
    )


def _process_reflector_file(path: Path) -> Optional[Tuple[str, np.ndarray, Dict[str, str]]]:
    """
    Process a reflector file with comment-based metadata.
    
    Args:
        path: Path to the reflector file
        
    Returns:
        Tuple of (name, interpolated_data, metadata_dict) or None if invalid
    """
    try:
        df, metadata = _parse_tsv_with_comments(path)
    except Exception:
        return None
    
    # Check for required columns
    if TSV_COLUMNS['wavelength'] not in df.columns or TSV_COLUMNS['reflectance'] not in df.columns:
        return None
    
    # Add source file info to metadata
    metadata['source_file'] = str(path)
    metadata['source_folder'] = path.parent.name
    
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
    
    # Process wavelength and reflectance data
    wl = df[TSV_COLUMNS['wavelength']].astype(float).values
    refl = df[TSV_COLUMNS['reflectance']].astype(float).values
    
    # Check for sufficient valid data points
    valid_mask = ~np.isnan(refl)
    if np.sum(valid_mask) < SPECTRAL_CONFIG['min_data_points']:
        return None
        
    wl = wl[valid_mask]
    refl = refl[valid_mask]
    
    # Interpolate to standard grid
    interp_vals = _interpolate_to_standard_grid(wl, refl)
    
    # Convert from percentage (0-100) to fractional (0-1) scale
    # All data files store values in 0-100 scale; internal calculations use 0-1
    interp_vals = interp_vals / 100.0
    
    # Round to specified decimal places to avoid floating point precision issues
    interp_vals = np.round(interp_vals, SPECTRAL_CONFIG['precision_decimals'])
    
    return name, interp_vals, metadata


def _load_reflector_collection_from_files() -> ReflectorCollection:
    """
    Load reflector data from files without using cache.
    
    Returns:
        ReflectorCollection object
    """
    folder = Path(DATA_FOLDERS['reflectors'])
    folder.mkdir(exist_ok=True, parents=True)

    files = list(folder.glob("**/*.tsv"))
    reflectors = []
    matrix = []
    meta_list = []

    for path in files:
        result = _safely_load_file(path, _process_reflector_file)
        if result:
            name, interp_vals, metadata = result
            reflectors.append(ReflectorSpectrum(name=name, values=interp_vals, metadata=metadata))
            matrix.append(interp_vals)
            
            # Build metadata row for DataFrame
            meta_row = {
                'Name': name,
                'Organization': metadata.get('Organization', ''),
                'Package Title': metadata.get('Package Title', ''),
                'Target Type': metadata.get('Target Type', ''),
                'Common Name': metadata.get('Common Name', ''),
                'Latin Genus': metadata.get('Latin Genus', ''),
                'Latin Species': metadata.get('Latin Species', ''),
                'Source File': metadata.get('source_file', ''),
                'Source Folder': metadata.get('source_folder', ''),
                'IsDefault': metadata.get('IsDefault', '')
            }
            meta_list.append(meta_row)

    if not matrix:
        return create_empty_reflector_collection()
    
    reflector_matrix = np.vstack(matrix)
    df_result = pd.DataFrame(meta_list)
    
    return ReflectorCollection(reflectors=reflectors, df=df_result, reflector_matrix=reflector_matrix)


def _pack_reflector_collection(collection: ReflectorCollection) -> Tuple[Dict[str, np.ndarray], dict]:
    """ReflectorCollection -> npz arrays (matrix) + JSON meta (df, spectrum fields)."""
    arrays = {"reflector_matrix": collection.reflector_matrix}
    meta = {
        "df": {col: collection.df[col].tolist() for col in collection.df.columns},
        "reflectors": [
            {"name": r.name, "metadata": r.metadata} for r in collection.reflectors
        ],
    }
    return arrays, meta


def _unpack_reflector_collection(arrays: Dict[str, np.ndarray], meta: dict) -> ReflectorCollection:
    """Rebuild a ReflectorCollection; spectrum values are matrix rows."""
    reflectors = [
        ReflectorSpectrum(
            name=spec["name"],
            values=arrays["reflector_matrix"][i].copy(),
            metadata=spec["metadata"],
        )
        for i, spec in enumerate(meta["reflectors"])
    ]
    return ReflectorCollection(
        reflectors=reflectors,
        df=pd.DataFrame(meta["df"]),
        reflector_matrix=arrays["reflector_matrix"],
    )


def load_reflector_collection() -> ReflectorCollection:
    """Load reflector collection."""
    return _cached_loader(
        cache_key="reflectors",
        data_folder=DATA_FOLDERS['reflectors'],
        load_function=_load_reflector_collection_from_files,
        pack=_pack_reflector_collection,
        unpack=_unpack_reflector_collection,
    )
