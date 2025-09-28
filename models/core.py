"""
Core data models for FS FilterLab.

This module defines all fundamental data structures used throughout the application:

Data Models:
- Filter: Represents individual optical filters with transmission data
- FilterCollection: Manages collections of filters with efficient matrix operations
- TargetProfile: Stores target transmission profiles for comparison analysis
- ReflectorSpectrum: Individual reflector spectral data
- ReflectorCollection: Collection of reflector spectra
- ApplicationState: Global application state management

Design Principles:
- Uses dataclasses for clean, type-safe data structures
- Leverages NumPy for efficient numerical operations
- Maintains separation between data models and business logic
- Provides utility methods for common operations

Note: Some legacy classes are deprecated in favor of simpler Dict/np.ndarray
structures for better performance and maintainability.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

# Import constants from the constants module
from models.constants import INTERP_GRID, DEFAULT_RGB_VISIBILITY, DEFAULT_WB_GAINS

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
    df: Any  # Pandas DataFrame containing filter metadata
    filter_matrix: np.ndarray  # Shape: (n_filters, n_wavelengths)
    extrapolated_masks: np.ndarray  # Shape: (n_filters, n_wavelengths), dtype=bool
    
    def get_display_to_index_map(self) -> Dict[str, int]:
        """
        Create mapping from display names to filter indices.
        
        Returns:
            Dictionary mapping filter display names to their matrix row indices
        """
        return {str(f): i for i, f in enumerate(self.filters)}

    def get_display_names(self) -> List[str]:
        """
        Get list of display names for all filters in the collection.
        
        Returns:
            List of formatted display names suitable for UI selection widgets
        """
        return [str(f) for f in self.filters]


# Deprecated classes - kept for reference but not exported
# These have been replaced by simpler Dict/np.ndarray structures

# @dataclass
# class QuantumEfficiency:
#     """Camera quantum efficiency model - DEPRECATED: Use Dict[str, np.ndarray] directly."""
#     brand: str
#     model: str
#     channels: Dict[str, np.ndarray]  # Dict of channel name to QE values
#     
#     def __str__(self) -> str:
#         return f"{self.brand} {self.model}"


# @dataclass
# class Illuminant:
#     """Light source illuminant model - DEPRECATED: Use np.ndarray with metadata dict directly."""
#     name: str
#     values: np.ndarray  # Illuminant values across the spectrum
#     metadata: Dict[str, Any] = field(default_factory=dict)
#     
#     def __str__(self) -> str:
#         return self.name


@dataclass
class TargetProfile:
    """
    Target transmission profile for comparison and optimization analysis.
    
    Used to define desired transmission characteristics and calculate
    deviation metrics from actual filter performance.
    
    Attributes:
        name: Descriptive name for the target profile
        values: Target transmission values across the wavelength spectrum
        valid: Boolean mask indicating which wavelengths have valid target data
    
    The values array should align with INTERP_GRID wavelengths and contain
    transmission values in the range [0, 1].
    """
    name: str
    values: np.ndarray  # Target transmission values
    valid: np.ndarray   # Boolean mask for valid target values
    
    def __str__(self) -> str:
        """Return the target profile name."""
        return self.name


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
    
    def from_dict(self, settings_dict: Dict[str, Any]) -> None:
        """Update settings from dictionary (for preset loading)."""
        for key, value in settings_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary (for preset saving)."""
        return {
            'red_r': self.red_r, 'red_g': self.red_g, 'red_b': self.red_b,
            'green_r': self.green_r, 'green_g': self.green_g, 'green_b': self.green_b,
            'blue_r': self.blue_r, 'blue_g': self.blue_g, 'blue_b': self.blue_b,
            'enabled': self.enabled
        }


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
    """
    name: str
    values: np.ndarray  # Reflectance values across the spectrum


@dataclass
class ReflectorCollection:
    """
    Collection of reflector spectra for batch analysis operations.
    
    Enables efficient computation of color responses across multiple
    surface types simultaneously.
    
    Attributes:
        reflectors: List of individual ReflectorSpectrum objects
        reflector_matrix: 2D array with each row containing reflectance data
    """
    reflectors: List[ReflectorSpectrum]
    reflector_matrix: np.ndarray  # Shape: (n_reflectors, n_wavelengths)


@dataclass
class ApplicationState:
    """
    Global application state container for FS FilterLab.
    
    Manages all persistent application data including user selections,
    computed results, and UI state. This class serves as a data container
    while the StateManager handles session state integration.
    
    Filter State:
        filter_collection: Available filters loaded from data files
        selected_filters: Currently selected filter display names
        filter_multipliers: Stack count for each selected filter
    
    Sensor and Illumination:
        current_qe: Camera quantum efficiency data by RGB channel
        selected_illuminant: Current illuminant spectrum and metadata
    
    Analysis:
        target_profile: Reference transmission profile for comparison
        combined_transmission: Computed result from selected filter stack
    
    Display Options:
        log_view: Toggle for logarithmic transmission scale
        rgb_channels_visibility: Show/hide individual RGB channels
        white_balance_gains: Color correction multipliers
        apply_white_balance: Enable white balance correction
    
    UI State:
        show_advanced_search: Toggle advanced filter search interface
        show_import_data: Toggle data import interface
        last_export: Metadata from most recent report generation
    """
    # Filter data
    filter_collection: Optional[FilterCollection] = None
    selected_filters: List[str] = field(default_factory=list)
    filter_multipliers: Dict[str, int] = field(default_factory=dict)
    
    # QE and illuminant data
    current_qe: Optional[Dict[str, np.ndarray]] = None
    selected_illuminant: Optional[Dict[str, Any]] = None
    
    # Target profile
    target_profile: Optional[TargetProfile] = None
    
    # Display options
    log_view: bool = False
    rgb_channels_visibility: Dict[str, bool] = field(default_factory=lambda: DEFAULT_RGB_VISIBILITY.copy())
    
    # Computed results
    combined_transmission: Optional[np.ndarray] = None
    white_balance_gains: Dict[str, float] = field(default_factory=lambda: DEFAULT_WB_GAINS.copy())
    apply_white_balance: bool = False
    
    # Export/report state
    last_export: Dict[str, Any] = field(default_factory=dict)
    
    # Advanced search state
    show_advanced_search: bool = False
    
    # Import state
    show_import_data: bool = False
    
    def reset(self):
        """Reset the application state to defaults."""
        self.selected_filters = []
        self.filter_multipliers = {}
        self.combined_transmission = None
        self.white_balance_gains = DEFAULT_WB_GAINS.copy()
        self.apply_white_balance = False
        self.log_view = False
        self.rgb_channels_visibility = DEFAULT_RGB_VISIBILITY.copy()
        self.last_export = {}
        self.show_advanced_search = False
        self.show_import_data = False
