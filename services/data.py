"""
Data loading services for FS FilterLab.

This module handles loading and processing spectral data:
- Filter transmission curves
- Camera sensor quantum efficiency (QE) curves
- Illuminant spectral power distributions
- Surface reflectance spectra

Features include NaN handling, normalization, interpolation,
file caching, and consistent data processing.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Any, Optional, TypeVar, Callable, NamedTuple, Union

from models import (
    Filter, FilterCollection, 
    ReflectorSpectrum, ReflectorCollection
)
from models.constants import (
    CACHE_DIR, DEFAULT_HEX_COLOR, DEFAULT_ILLUMINANT, INTERP_GRID,
    TRANSMISSION_NORMALIZATION_THRESHOLD, MIN_VALID_DATAPOINTS, DECIMAL_PRECISION
)

# Configure logging for this module
import logging
logger = logging.getLogger(__name__)

# Ensure cache directory exists
Path(CACHE_DIR).mkdir(exist_ok=True)

# Generic type for cached data
T = TypeVar('T')

class SpectralConfig(NamedTuple):
    """Configuration for spectral file processing."""
    wavelength_col: str = "Wavelength"        # Column name for wavelength values
    value_col: str = "Transmittance"          # Column name for spectral values
    normalize: bool = True                    # Whether to normalize values from 0-100 to 0-1
    round_precision: Optional[int] = None     # Optional rounding precision
    name_cols: List[str] = ["Name"]           # Columns to check for name (in priority order)
    metadata_cols: List[str] = []             # Additional metadata columns to extract
    default_name: Optional[str] = None        # Default name if none found in file


class SpectralFileProcessor:
    """
    Processes spectral data files with configurable column handling.
    
    This class handles common operations for different spectral file types:
    - TSV file parsing with column validation
    - NaN filtering and normalization
    - Interpolation to standard wavelength grids
    - Metadata extraction with fallbacks
    """
    
    def __init__(self, config: SpectralConfig):
        """Initialize processor with configuration."""
        self.config = config
    
    def parse_file(self, path: Path) -> Optional[pd.DataFrame]:
        """Parse a TSV file with error handling and cleaning."""
        try:
            df = pd.read_csv(path, sep="\t")
            
            # Clean column names
            df.columns = [str(c).strip() for c in df.columns]
            
            # Drop rows where all numeric columns are NaN
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                df = df.dropna(subset=numeric_cols, how='all')
                
            # Verify required columns
            if self.config.wavelength_col not in df.columns:
                return None
                
            if self.config.value_col and self.config.value_col not in df.columns:
                # For illuminants, we allow using the second column regardless of name
                if not (len(df.columns) >= 2 and not self.config.value_col):
                    return None
                    
            return df
            
        except Exception as e:
            logger.error(f"Failed to load {path}: {str(e)}")
            return None
    
    def extract_name(self, df: pd.DataFrame, path: Path) -> str:
        """Extract a name from dataframe with fallbacks."""
        # Try each name column in priority order
        for col in self.config.name_cols:
            if col in df.columns:
                name_values = df[col].dropna()
                if len(name_values) > 0:
                    name = str(name_values.iloc[0]).strip()
                    if name:
                        return name
        
        # Use default name from config if provided
        if self.config.default_name:
            return self.config.default_name
            
        # Fall back to filename stem
        return path.stem
    
    def extract_metadata(self, df: pd.DataFrame, path: Path) -> Dict[str, Any]:
        """Extract metadata from a dataframe."""
        metadata = {}
        
        # Extract name first
        name = self.extract_name(df, path)
        if name:
            metadata["name"] = name
        
        # Extract additional metadata
        first_row = df.iloc[0] if not df.empty else pd.Series()
        
        for col in self.config.metadata_cols:
            if col in df.columns and pd.notnull(first_row.get(col, None)):
                metadata[col] = first_row.get(col)
                
        return metadata
    
    def get_spectral_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract spectral data arrays from a dataframe."""
        # Handle wavelength column
        wavelengths = df[self.config.wavelength_col].astype(float).values
        
        # Handle value column (or use second column for illuminants)
        if self.config.value_col and self.config.value_col in df.columns:
            values = df[self.config.value_col].astype(float).values
        else:
            # For illuminants, use the second column regardless of name
            values = df.iloc[:, 1].astype(float).values
        
        # Filter out NaN values
        valid_mask = ~np.isnan(wavelengths) & ~np.isnan(values)
        if not np.any(valid_mask):
            return np.array([]), np.array([])
            
        return wavelengths[valid_mask], values[valid_mask]
    
    def normalize_values(self, values: np.ndarray) -> np.ndarray:
        """Normalize values from 0-100 to 0-1 scale if needed."""
        if not self.config.normalize:
            return values
            
        valid_values = values[~np.isnan(values)]
        if len(valid_values) == 0:
            return values
            
        if np.max(valid_values) > TRANSMISSION_NORMALIZATION_THRESHOLD:
            return values / 100.0
            
        return values
    
    def interpolate(self, wavelengths: np.ndarray, values: np.ndarray, 
                   target_grid: np.ndarray = INTERP_GRID) -> np.ndarray:
        """Interpolate spectral data to a standard wavelength grid."""
        # Check if we have enough data points to interpolate
        if len(wavelengths) < MIN_VALID_DATAPOINTS:
            return np.zeros_like(target_grid, dtype=float)
            
        # Perform interpolation
        result = np.interp(
            target_grid,
            wavelengths,
            values,
            left=np.nan,
            right=np.nan
        )
        
        # Apply rounding if configured
        if self.config.round_precision is not None:
            result = np.round(result, self.config.round_precision)
            
        return result
    
    def process(self, path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
        """
        Process a spectral file and return metadata and interpolated values.
        
        Args:
            path: Path to the spectral file
            
        Returns:
            Tuple of (metadata dict, interpolated values) or (None, None) if invalid
        """
        # Parse the file
        df = self.parse_file(path)
        if df is None:
            return None, None
            
        # Extract metadata
        metadata = self.extract_metadata(df, path)
        
        # Add file path to metadata for special processing (like Lee filters)
        metadata["__file_path__"] = str(path)
        
        # Get spectral data
        wavelengths, values = self.get_spectral_data(df)
        if len(wavelengths) == 0:
            return None, None
            
        # Normalize if needed
        values = self.normalize_values(values)
        
        # Interpolate to standard grid
        interp_values = self.interpolate(wavelengths, values)
        
        # Check if interpolation succeeded
        if np.all(interp_values == 0):
            return None, None
            
        return metadata, interp_values


class DirectoryLoader:
    """
    Utility for loading and processing spectral files from a directory.
    
    This class handles common directory operations:
    - Recursively finding files matching a pattern
    - Processing each file with a configured processor
    - Collecting results into structured collections
    """
    
    def __init__(self, processor: SpectralFileProcessor):
        """Initialize with a processor for individual files."""
        self.processor = processor
    
    def find_files(self, directory: Union[str, Path], 
                  pattern: str = "**/*.tsv", 
                  recursive: bool = True) -> List[Path]:
        """Find files in a directory matching a pattern."""
        data_dir = Path(directory)
        data_dir.mkdir(exist_ok=True, parents=True)
        
        if recursive:
            return list(data_dir.glob(pattern))
        else:
            return list(data_dir.glob(f"*.tsv"))
    
    def load_directory(self, directory: Union[str, Path], 
                      recursive: bool = True) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
        """
        Load and process all spectral files in a directory.
        
        Args:
            directory: Directory to scan for spectral files
            recursive: Whether to search subdirectories
            
        Returns:
            Tuple of (metadata_list, values_list) where each list contains
            the metadata and interpolated values for each valid file
        """
        files = self.find_files(directory, recursive=recursive)
        metadata_list = []
        values_list = []
        
        for path in files:
            metadata, values = self.processor.process(path)
            if metadata is not None and values is not None:
                metadata_list.append(metadata)
                values_list.append(values)
                
        return metadata_list, values_list
        
    def process_directory(self, directory: Union[str, Path], 
                         process_results_fn: Callable[[List[Dict[str, Any]], List[np.ndarray]], T],
                         recursive: bool = True) -> T:
        """
        Process a directory and transform the results with a custom function.
        
        Args:
            directory: Directory to scan for spectral files
            process_results_fn: Function to process the collected metadata and values
            recursive: Whether to search subdirectories
            
        Returns:
            Result of applying process_results_fn to the collected data
        """
        metadata_list, values_list = self.load_directory(directory, recursive)
        return process_results_fn(metadata_list, values_list)

# Create processor configurations for each data type
FILTER_PROCESSOR_CONFIG = SpectralConfig(
    wavelength_col="Wavelength",
    value_col="Transmittance",
    normalize=True,
    name_cols=["Name", "Filter Name"],
    metadata_cols=["Filter Number", "Manufacturer", "hex_color"]
)

QE_PROCESSOR_CONFIG = SpectralConfig(
    wavelength_col="Wavelength", 
    value_col="",  # RGB channels are handled separately
    normalize=False,
    name_cols=["Name"],
    metadata_cols=["Manufacturer"]
)

ILLUMINANT_PROCESSOR_CONFIG = SpectralConfig(
    wavelength_col="Wavelength (nm)",
    value_col="SPD",  # Default column name, but the processor will fall back to second column
    normalize=False,
    name_cols=["Name"],
    metadata_cols=["Description"]
)

REFLECTOR_PROCESSOR_CONFIG = SpectralConfig(
    wavelength_col="Wavelength",
    value_col="Reflectance",
    normalize=True,
    round_precision=DECIMAL_PRECISION,
    name_cols=["Name"]
)

# Create processors for each data type
filter_processor = SpectralFileProcessor(FILTER_PROCESSOR_CONFIG)
qe_processor = SpectralFileProcessor(QE_PROCESSOR_CONFIG)
illuminant_processor = SpectralFileProcessor(ILLUMINANT_PROCESSOR_CONFIG)
reflector_processor = SpectralFileProcessor(REFLECTOR_PROCESSOR_CONFIG)

# Helper functions for filter collection and reflector collection


def cached_loader(cache_key: str, data_folder: Union[str, Path], 
                  load_function: Callable[[], T]) -> T:
    """Simple caching mechanism for data loading."""
    import time
    
    data_dir = Path(data_folder)
    cache_file = Path(CACHE_DIR) / f"{cache_key}.pkl"
    cache_time_file = Path(CACHE_DIR) / f"{cache_key}_time.pkl"
    
    # Check if cache exists and is newer than source files
    if cache_file.exists() and cache_time_file.exists():
        try:
            with open(cache_time_file, 'rb') as f:
                cached_timestamp = pickle.load(f)
            
            # Find the newest file in the data folder
            newest_time = max(
                (p.stat().st_mtime for p in data_dir.glob("**/*.tsv") if p.is_file()), 
                default=0
            )
            
            # If cache is valid, use it
            if cached_timestamp > newest_time and newest_time > 0:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception:
            pass
    
    # Cache miss or invalid - load fresh data
    data = load_function()
    
    # Save to cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        with open(cache_time_file, 'wb') as f:
            pickle.dump(time.time(), f)
    except Exception:
        pass
        
    return data


# Helper functions
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
    """Process a file with error handling."""
    try:
        return processor_func(path)
    except Exception as e:
        logger.warning(f"Failed to process {path.name}: {type(e).__name__}: {str(e)}")
        return None


def is_lee_filter(path_or_name: Union[Path, str]) -> bool:
    """
    Check if a file represents a Lee filter based on its path or name.
    Lee filters need special handling for extrapolation above 700nm.
    
    Args:
        path_or_name: Path object or string filename/path
        
    Returns:
        True if the file is a Lee filter
    """
    name = path_or_name.name if isinstance(path_or_name, Path) else str(path_or_name)
    return 'LeeFilters' in name

def _load_filter_collection_from_files() -> FilterCollection:
    """Load filter data from files using the DirectoryLoader."""
    data_folder = Path("data") / "filters_data"
    
    # Create a DirectoryLoader for filters
    loader = DirectoryLoader(filter_processor)
    
    # Define a custom results processor function
    def process_results(metadata_list: List[Dict[str, Any]], values_list: List[np.ndarray]) -> FilterCollection:
        if not metadata_list:
            return create_empty_filter_collection()
            
        meta_list = []
        matrix = []
        masks = []
        filters = []
        
        # Process each result
        for i, (metadata, values) in enumerate(zip(metadata_list, values_list)):
            # Handle Lee filters specially - check file path if available
            path_name = str(metadata.get("__file_path__", ""))
            filter_is_lee = is_lee_filter(path_name)
            extrap_mask = (INTERP_GRID > 700) if filter_is_lee else np.zeros_like(INTERP_GRID, dtype=bool)
            
            # Set defaults for metadata
            name = metadata.get("name", f"Filter {i}")
            fn = metadata.get("Filter Number", f"F{i}")
            manufacturer = metadata.get("Manufacturer", "Unknown")
            
            # Ensure hex_color is valid
            hex_color = metadata.get("hex_color", DEFAULT_HEX_COLOR)
            if not (isinstance(hex_color, str) and hex_color.startswith("#")):
                hex_color = DEFAULT_HEX_COLOR
                
            # Create metadata dictionary
            filter_metadata = {
                'Filter Number': fn,
                'Filter Name': name,
                'Manufacturer': manufacturer,
                'Hex Color': hex_color,
                'is_lee': filter_is_lee
            }
            
            # Create filter object
            filter_obj = Filter(
                name=name,
                number=fn,
                manufacturer=manufacturer,
                hex_color=hex_color,
                transmission=values,
                extrapolated_mask=extrap_mask
            )
            
            # Add to result collections
            meta_list.append(filter_metadata)
            matrix.append(values)
            masks.append(extrap_mask)
            filters.append(filter_obj)
            
        try:
            return FilterCollection(
                filters=filters,
                df=pd.DataFrame(meta_list),
                filter_matrix=np.vstack(matrix),
                extrapolated_masks=np.vstack(masks)
            )
        except Exception as e:
            logger.error(f"Failed to create filter collection: {e}")
            return create_empty_filter_collection()
    
    # Use DirectoryLoader to process the files
    return loader.process_directory(data_folder, process_results, recursive=True)


def load_filter_collection() -> FilterCollection:
    """Load filter data and return a FilterCollection object."""
    def _create_cached_data():
        collection = _load_filter_collection_from_files()
        return (collection.df, collection.filter_matrix, 
                collection.extrapolated_masks, collection.filters)
    
    try:
        df, matrix, masks, filters = cached_loader(
            cache_key="filter_data",
            data_folder=str(Path("data") / "filters_data"),
            load_function=_create_cached_data
        )
        return FilterCollection(
            filters=filters, df=df,
            filter_matrix=matrix, extrapolated_masks=masks
        )
    except Exception:
        return _load_filter_collection_from_files()


def _process_qe_file(path: Path) -> Optional[Tuple[str, Dict[str, np.ndarray], bool]]:
    """
    Process a quantum efficiency file with RGB channel handling.
    
    QE files have a special structure with RGB columns that need individual processing.
    
    Args:
        path: Path to the QE file
        
    Returns:
        Tuple of (sensor_name, channel_data_dict, is_default) or None if invalid
    """
    # Parse the file using the QE processor
    df = qe_processor.parse_file(path)
    if df is None:
        return None
    
    # Check if file has required columns: Wavelength and at least one RGB channel
    if 'Wavelength' not in df.columns or not any(col in df.columns for col in ['R', 'G', 'B']):
        return None
    
    # Extract metadata for the sensor
    metadata = qe_processor.extract_metadata(df, path)
    brand = metadata.get("Manufacturer", "Generic").strip()
    model = metadata.get("name", path.stem).strip()
    key = f"{brand} {model}"
    
    # Process each RGB channel individually
    channel_data = {}
    wavelength = df['Wavelength'].astype(float).values
    
    for channel in ['R', 'G', 'B']:
        if channel not in df.columns:
            continue
            
        channel_values = df[channel].values.astype(float)
        
        # Use the processor's interpolation for consistent handling
        interp_vals = qe_processor.interpolate(wavelength, channel_values)
        
        # Only add channel if we have valid data
        if not np.all(interp_vals == 0):
            channel_data[channel[0]] = interp_vals
    
    # Check if we have any valid channels
    if not channel_data:
        return None
    
    # Flag if this is the default QE file
    is_default = (path.name == 'Default_QE.tsv')
    return key, channel_data, is_default


def _load_quantum_efficiencies_from_files() -> Tuple[List[str], Dict[str, Dict[str, np.ndarray]], Optional[str]]:
    """
    Load quantum efficiency data from files.
    
    QE files require special handling for RGB channels, but we use the DirectoryLoader
    for consistent file finding and leverage the processor for metadata extraction.
    
    Returns:
        Tuple of (qe_keys, qe_data, default_key)
    """
    folder = Path('data') / 'QE_data'
    folder.mkdir(exist_ok=True, parents=True)
    
    # Create a DirectoryLoader for QE files
    loader = DirectoryLoader(qe_processor)
    
    # Define a custom processor for QE files that handles the RGB channels
    def process_qe_results(paths: List[Path]) -> Tuple[List[str], Dict[str, Dict[str, np.ndarray]], Optional[str]]:
        qe_dict = {}
        default_key = None
        
        for path in paths:
            result = safely_load_file(path, _process_qe_file)
            if result:
                key, channel_data, is_default = result
                qe_dict[key] = channel_data
                if is_default and default_key is None:
                    default_key = key
                    
        return sorted(qe_dict.keys()), qe_dict, default_key
    
    # Find files using DirectoryLoader's file finding capability
    files = loader.find_files(folder, recursive=False)
    
    return process_qe_results(files)


def load_quantum_efficiencies() -> Tuple[List[str], Dict[str, Dict[str, np.ndarray]], Optional[str]]:
    """Load quantum efficiency data for camera sensors."""
    return cached_loader(
        cache_key="qe_data",
        data_folder=Path('data') / 'QE_data',
        load_function=_load_quantum_efficiencies_from_files
    )


def _load_illuminant_collection_from_files() -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """
    Load illuminant data from files using DirectoryLoader.
    
    Returns:
        Tuple of (illuminants, metadata)
    """
    folder = Path('data') / 'illuminants'
    folder.mkdir(exist_ok=True, parents=True)
    
    # Create a directory loader for illuminants
    loader = DirectoryLoader(illuminant_processor)
    
    # Process the directory
    def process_results(metadata_list: List[Dict[str, Any]], values_list: List[np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        illum, meta = {}, {}
        
        # Process each result
        for metadata, values in zip(metadata_list, values_list):
            name = metadata.get("name", "Unnamed Illuminant")
            description = metadata.get("Description")
            
            illum[name] = values
            if description:
                meta[name] = description
                
        return illum, meta
    
    # Use DirectoryLoader to process the directory
    return loader.process_directory(folder, process_results, recursive=False)


def load_illuminant_collection() -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """Load illuminant collection."""
    return cached_loader(
        cache_key="illuminants",
        data_folder=Path('data') / 'illuminants',
        load_function=_load_illuminant_collection_from_files
    )


def _load_reflector_collection_from_files() -> ReflectorCollection:
    """Load reflector data from files using DirectoryLoader."""
    folder = Path('data') / 'reflectors'
    
    # Create a directory loader for reflectors
    loader = DirectoryLoader(reflector_processor)
    
    # Process the directory
    def process_results(metadata_list: List[Dict[str, Any]], values_list: List[np.ndarray]) -> ReflectorCollection:
        if not metadata_list:
            return create_empty_reflector_collection()
        
        reflectors = []
        
        # Create ReflectorSpectrum objects for each valid result
        for metadata, values in zip(metadata_list, values_list):
            name = metadata.get("name", "Unnamed Reflector")
            reflectors.append(ReflectorSpectrum(name=name, values=values))
        
        if not reflectors:
            return create_empty_reflector_collection()
        
        # Create the matrix
        reflector_matrix = np.vstack([r.values for r in reflectors])
        return ReflectorCollection(reflectors=reflectors, reflector_matrix=reflector_matrix)
    
    # Use DirectoryLoader to process the directory
    return loader.process_directory(folder, process_results, recursive=True)


def load_reflector_collection() -> ReflectorCollection:
    """Load reflector collection."""
    return cached_loader(
        cache_key="reflectors",
        data_folder=Path('data') / 'reflectors',
        load_function=_load_reflector_collection_from_files
    )
