"""
Core data models for FS FilterLab.

This module defines all fundamental data structures used throughout the application:

Data Models:
- Filter: Represents individual optical filters with transmission data
- FilterCollection: Manages collections of filters with efficient matrix operations
- ReflectorSpectrum: Individual reflector spectral data
- ReflectorCollection: Collection of reflector spectra
- ChannelMixerSettings: RGB channel mixing transformation settings

Design Principles:
- Uses dataclasses for clean, type-safe data structures
- Leverages NumPy for efficient numerical operations
- Maintains separation between data models and business logic
- Provides utility methods for common operations
"""
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional
import numpy as np
import pandas as pd

# Filter Models
@dataclass
class Filter:
    """
    Represents an individual optical filter with its properties and transmission data.
    
    Attributes:
        name: Human-readable filter name (e.g., "UV Cut", "IR Block")
        number: Manufacturer part number or model identifier
        manufacturer: Filter manufacturer name
        hex_color: Hexadecimal color code for UI visualization (#RRGGBB format)
        transmission: NumPy array of transmission values across wavelength spectrum
        extrapolated_mask: Boolean array indicating which wavelengths were extrapolated
    
    The transmission array should correspond to the wavelength grid defined in INTERP_GRID.
    Values should be in the range [0, 1] representing fractional transmission.
    """
    name: str
    number: str
    manufacturer: str
    hex_color: str
    transmission: np.ndarray
    extrapolated_mask: np.ndarray = None
    
    def __str__(self) -> str:
        """Return a formatted display name for the filter."""
        return f"{self.name} ({self.number}, {self.manufacturer})"


@dataclass
class FilterCollection:
    """
    Efficient collection of optical filters with matrix operations support.
    
    This class manages multiple filters and provides optimized access patterns
    for common operations like transmission calculations and filter selection.
    
    Attributes:
        filters: List of Filter objects containing metadata and individual transmissions
        df: Pandas DataFrame with filter metadata for efficient searching/filtering
        filter_matrix: 2D NumPy array where each row contains a filter's transmission data
        extrapolated_masks: 2D boolean array indicating extrapolated wavelengths per filter
    
    The matrix structure enables vectorized operations across multiple filters,
    significantly improving performance for large filter collections.
    """
    filters: List[Filter]
    df: pd.DataFrame  # Pandas DataFrame containing filter metadata
    filter_matrix: np.ndarray  # Shape: (n_filters, n_wavelengths)
    extrapolated_masks: np.ndarray  # Shape: (n_filters, n_wavelengths), dtype=bool
    
    def get_display_to_index_map(self) -> Dict[str, int]:
        """Map display names to matrix row indices."""
        return {str(f): i for i, f in enumerate(self.filters)}


@dataclass
class ChannelMixerSettings:
    """
    Channel mixer transformation settings for RGB channel manipulation.
    
    Implements Photoshop-style channel mixing where each output channel is a
    linear combination of the input RGB channels. Commonly used in IR photography
    for channel swapping and false color enhancement.
    
    Mathematical representation:
        R_out = R_in * red_r + G_in * red_g + B_in * red_b
        G_out = R_in * green_r + G_in * green_g + B_in * green_b  
        B_out = R_in * blue_r + G_in * blue_g + B_in * blue_b
    
    Attributes:
        red_r, red_g, red_b: Contributions to red output channel
        green_r, green_g, green_b: Contributions to green output channel
        blue_r, blue_g, blue_b: Contributions to blue output channel
        enabled: Whether channel mixing is currently active
    """
    # Red output channel = R*red_r + G*red_g + B*red_b
    red_r: float = 1.0    # Red contribution to red output
    red_g: float = 0.0    # Green contribution to red output  
    red_b: float = 0.0    # Blue contribution to red output
    
    # Green output channel = R*green_r + G*green_g + B*green_b
    green_r: float = 0.0  # Red contribution to green output
    green_g: float = 1.0  # Green contribution to green output
    green_b: float = 0.0  # Blue contribution to green output
    
    # Blue output channel = R*blue_r + G*blue_g + B*blue_b
    blue_r: float = 0.0   # Red contribution to blue output
    blue_g: float = 0.0   # Green contribution to blue output
    blue_b: float = 1.0   # Blue contribution to blue output
    
    enabled: bool = False # Whether channel mixing is active
    
    def to_matrix(self) -> np.ndarray:
        """
        Convert channel mixer settings to 3x3 transformation matrix.
        
        Returns:
            3x3 NumPy array where each row represents output channel weights
        """
        return np.array([
            [self.red_r, self.red_g, self.red_b],      # Red output weights
            [self.green_r, self.green_g, self.green_b], # Green output weights  
            [self.blue_r, self.blue_g, self.blue_b]    # Blue output weights
        ])
    

# Reflector Models
@dataclass
class ReflectorSpectrum:
    """
    Reflector spectrum data representing surface reflectance characteristics.
    
    Used for analyzing how filtered light interacts with different surfaces
    and materials in real-world scenarios.
    
    Attributes:
        name: Descriptive name for the reflector (e.g., "Green Vegetation", "Skin Tone")
        values: Reflectance values across wavelength spectrum (0-1 range typical)
        metadata: Dictionary containing all metadata from TSV comments (Organization, 
                  Target Type, Package Title, species, etc.)
    """
    name: str
    values: np.ndarray  # Reflectance values across the spectrum
    metadata: Dict[str, str] = field(default_factory=dict)  # All TSV comment metadata


@dataclass
class ReflectorCollection:
    """
    Collection of reflector spectra for batch analysis operations.
    
    Enables efficient computation of color responses across multiple
    surface types simultaneously.
    
    Attributes:
        reflectors: List of individual ReflectorSpectrum objects
        df: Pandas DataFrame with reflector metadata for searching/filtering
        reflector_matrix: 2D array with each row containing reflectance data
    """
    reflectors: List[ReflectorSpectrum]
    df: pd.DataFrame  # Searchable metadata (Name, Organization, Package Title, Target Type, etc.)
    reflector_matrix: np.ndarray  # Shape: (n_reflectors, n_wavelengths)


# =============================================================================
# SENSOR / FILM PROFILE MODELS
# =============================================================================

# Valid sensor type identifiers (matches SENSOR_TYPE_LABELS keys in constants.py)
SensorType = Literal['rgb_sensor', 'color_slide', 'color_negative', 'bw']


@dataclass
class SensorProfile:
    """
    Base class for all sensor and film sensitivity profiles.

    Subclasses represent different capture media and override ``sensor_type``,
    ``is_film``, and ``invert_for_display`` as appropriate.  All subclasses
    expose ``get_channels()`` which always returns a ``{'R', 'G', 'B'}`` dict
    so that the rest of the calculation pipeline needs no changes.

    Required fields (enforced at load time): manufacturer, name, sensor_type.
    Optional fields: iso, process, notes.
    """
    manufacturer: str
    name: str
    iso: Optional[str] = None       # Film speed, e.g. "400"
    process: Optional[str] = None   # Development process, e.g. "E-6" or "D-76 1:1"
    notes: Optional[str] = None     # Free-text notes shown in the UI

    @property
    def display_name(self) -> str:
        """Display string used in dropdowns and report headers."""
        return f"{self.manufacturer} {self.name}"

    @property
    def sensor_type(self) -> SensorType:
        raise NotImplementedError

    @property
    def is_film(self) -> bool:
        """True for all film subclasses; False for digital sensors."""
        return False

    @property
    def invert_for_display(self) -> bool:
        """True when response curves must be inverted (1 - x) before display."""
        return False

    def get_channels(self) -> Dict[str, np.ndarray]:
        """Return {'R': ..., 'G': ..., 'B': ...} sensitivity arrays."""
        raise NotImplementedError


@dataclass
class RGBSensorProfile(SensorProfile):
    """Standard digital camera sensor with independent R, G, B QE curves."""
    channel_data: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def sensor_type(self) -> SensorType:
        return 'rgb_sensor'

    def get_channels(self) -> Dict[str, np.ndarray]:
        return self.channel_data


@dataclass
class FilmProfile(SensorProfile):
    """Base for film stocks: three dye-layer curves, ``is_film`` always True."""
    channel_data: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def is_film(self) -> bool:
        return True

    def get_channels(self) -> Dict[str, np.ndarray]:
        return self.channel_data


@dataclass
class ColorSlideFilmProfile(FilmProfile):
    """
    Colour reversal (slide/transparency) film, e.g. Ektachrome, Velvia.

    The sensitivity curves represent the three dye layers directly.
    The final image is a positive, so no display inversion is needed.
    """

    @property
    def sensor_type(self) -> SensorType:
        return 'color_slide'


@dataclass
class ColorNegativeFilmProfile(FilmProfile):
    """
    Colour negative film (C-41, ECN-2, etc.), e.g. Kodak Gold, Portra.

    Sensitivity curves describe the CMY dye layers.  The latent image is a
    negative, so ``invert_for_display`` is True: the visualisation layer
    applies ``1 - response`` before rendering colour previews.
    """

    @property
    def sensor_type(self) -> SensorType:
        return 'color_negative'

    @property
    def invert_for_display(self) -> bool:
        return True


@dataclass
class BWFilmProfile(FilmProfile):
    """
    Black-and-white panchromatic (or orthochromatic) film.

    A single sensitivity curve is stored.  ``get_channels()`` maps it to all
    three RGB keys so the downstream calculation pipeline requires no changes.
    The visualisation layer collapses the three identical traces into a single
    grey "Luminance" trace.
    """
    sensitivity: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def sensor_type(self) -> SensorType:
        return 'bw'

    def get_channels(self) -> Dict[str, np.ndarray]:
        return {'R': self.sensitivity, 'G': self.sensitivity, 'B': self.sensitivity}


# =============================================================================
# APPLICATION DATA CONTAINER
# =============================================================================

@dataclass
class AppData:
    """Typed container for all loaded application data."""
    filter_collection: FilterCollection
    default_key: str
    illuminants: Dict[str, np.ndarray]
    illuminant_metadata: Dict[str, str]
    reflector_collection: ReflectorCollection
    # Unified sensor + film profiles (key = profile.display_name)
    sensor_profiles: Dict[str, SensorProfile] = field(default_factory=dict)



