"""
Application state for FS FilterLab.

Plain dataclass containers (S3): no backing dict, no change listeners, no
generic get()/set(). One field per concept — the old
active_profile/current_qe/selected_camera triple is collapsed into
``active_profile`` plus the two derived properties below. UI refresh is owned
by views via the ``on_change`` callback they pass around, not by state.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from models.constants import DEFAULT_WB_GAINS, MY_FILTERS_FILE
from models.core import ChannelMixerSettings, Filter, SensorProfile

logger = logging.getLogger(__name__)

DEFAULT_REFLECTORS_FILE = "program/data/reflectors/default_reflectors.json"


def _default_wavelength_criteria() -> List[Dict[str, int]]:
    """Fresh default wavelength-criteria row (mutable default, so a factory)."""
    return [{"wavelength": 550, "trans_min": 0, "trans_max": 100}]


@dataclass
class AppState:
    """
    All mutable application state as plain typed fields.

    Writers assign fresh collections; computed values are derived properties
    or live in services/ (spectra, calculations). Persistence helpers (My
    Filters, default reflectors) read/write small JSON files and are loaded
    once at construction.
    """

    # --- Filter data ---
    selected_filters: List[str] = field(default_factory=list)
    filter_multipliers: Dict[str, int] = field(default_factory=dict)
    # --- Sensor / film (single source of truth) ---
    active_profile: Optional[SensorProfile] = None
    illuminant: Optional[np.ndarray] = None
    illuminant_name: Optional[str] = None
    # --- Target profile ---
    target_profile: Optional[Filter] = None
    selected_target_name: Optional[str] = None
    # --- Film emulation target ---
    film_emulation_profile: Optional[SensorProfile] = None
    film_emulation_profile_name: Optional[str] = None
    film_emulation_wb: bool = False
    film_emulation_wb_gains: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_WB_GAINS)
    )
    film_emulation_filters: List[str] = field(default_factory=list)
    # --- Computed results ---
    combined_transmission: Optional[np.ndarray] = None
    white_balance_gains: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_WB_GAINS)
    )
    wb_reference_surface: Optional[str] = None
    # --- Surfaces instrument: reflectors added as color-preview chips (§3.2.3) ---
    selected_surfaces: List[str] = field(default_factory=list)
    last_export: Dict[str, Any] = field(default_factory=dict)
    # --- Display toggles ---
    log_view: bool = False
    apply_white_balance: bool = False
    show_advanced_search: bool = False
    show_import_data: bool = False
    show_reflector_search: bool = False
    show_my_filters_manager: bool = False
    my_filters_only: bool = False
    # --- Advanced filter search (persists across UI rebuilds) ---
    advanced_search_manufacturers: List[str] = field(default_factory=list)
    advanced_search_wavelength_criteria: List[Dict[str, int]] = field(
        default_factory=_default_wavelength_criteria
    )
    advanced_search_sort: str = "Filter Number"
    # --- RGB channel visibility ---
    show_R: bool = True
    show_G: bool = True
    show_B: bool = True
    # --- Channel mixer ---
    channel_mixer: ChannelMixerSettings = field(default_factory=ChannelMixerSettings)
    # --- Sidebar expansion open/closed state (persists across rebuilds) ---
    sidebar_expansions: Dict[str, bool] = field(default_factory=dict)
    # --- Private: persisted lists/sets, loaded once at construction ---
    _default_reflector_files: List[str] = field(default_factory=list)
    _my_filters_set: set = field(default_factory=set)

    def __post_init__(self) -> None:
        self._default_reflector_files = self._load_default_reflectors_from_file()
        self._my_filters_set = set(self._load_my_filters_from_file())

    # ====================================================================
    # DERIVED STATE (the collapsed sensor triple-key)
    # ====================================================================

    @property
    def current_qe(self) -> Optional[Dict[str, np.ndarray]]:
        """Channel data of the active profile; None when no profile is set."""
        return self.active_profile.get_channels() if self.active_profile else None

    @property
    def selected_camera(self) -> Optional[str]:
        """Display name of the active profile (sensor or film)."""
        return self.active_profile.display_name if self.active_profile else None

    # ====================================================================
    # COMPOSED-VIEW PROPERTIES
    # ====================================================================

    @property
    def rgb_channels_visibility(self) -> Dict[str, bool]:
        return {"R": self.show_R, "G": self.show_G, "B": self.show_B}

    @rgb_channels_visibility.setter
    def rgb_channels_visibility(self, value: Dict[str, bool]) -> None:
        for ch in ("R", "G", "B"):
            setattr(self, f"show_{ch}", value.get(ch, True))

    @property
    def show_channel_mixer(self) -> bool:
        return self.channel_mixer.enabled

    @show_channel_mixer.setter
    def show_channel_mixer(self, value: bool) -> None:
        mixer = self.channel_mixer
        if mixer.enabled != value:
            mixer.enabled = value
            self.channel_mixer = mixer

    # ====================================================================
    # WHITE BALANCE
    # ====================================================================

    def set_white_balance_from_surface(
        self,
        reflector: np.ndarray,
        transmission: np.ndarray,
        source_file: Optional[str] = None,
    ) -> None:
        from services.calculations import compute_white_balance_gains_from_surface

        if self.current_qe and self.illuminant is not None:
            new_gains = compute_white_balance_gains_from_surface(
                reflector, transmission, self.current_qe, self.illuminant
            )
            self.white_balance_gains = new_gains
            self.wb_reference_surface = source_file

    def reset_white_balance(self) -> None:
        self.white_balance_gains = DEFAULT_WB_GAINS.copy()
        self.wb_reference_surface = None

    def recompute_white_balance(
        self,
        trans_interp: np.ndarray,
        reflector_collection: Any = None,
    ) -> Dict[str, float]:
        """Recompute and store white balance gains.

        If a reference surface is set, recalculates gains from that surface.
        Otherwise computes standard WB from the filter transmission, QE, and
        illuminant.  Returns the (possibly updated) gains dict.
        """
        from services.calculations import (
            compute_white_balance_gains,
            compute_white_balance_gains_from_surface,
        )

        wb_gains = self.white_balance_gains
        if not self.current_qe or self.illuminant is None:
            return wb_gains

        wb_ref = self.wb_reference_surface
        if wb_ref and reflector_collection:
            ref_reflector = None
            for r in reflector_collection.reflectors:
                if r.metadata.get("source_file", "") == wb_ref:
                    ref_reflector = r
                    break

            if ref_reflector is not None:
                wb_gains = compute_white_balance_gains_from_surface(
                    ref_reflector.values, trans_interp,
                    self.current_qe, self.illuminant,
                )
                self.white_balance_gains = wb_gains
            else:
                logger.warning("WB reference surface not found: %s", wb_ref)
                wb_gains = compute_white_balance_gains(
                    trans_interp, self.current_qe, self.illuminant,
                )
                self.white_balance_gains = wb_gains
                self.wb_reference_surface = None
        else:
            wb_gains = compute_white_balance_gains(
                trans_interp, self.current_qe, self.illuminant,
            )
            self.white_balance_gains = wb_gains

        return wb_gains

    # ====================================================================
    # MULTI-FIELD SELECTION HELPERS
    # ====================================================================

    def select_illuminant(self, name: str, illuminants: Dict[str, Any]) -> None:
        """Set the active illuminant by name."""
        self.illuminant_name = name
        self.illuminant = illuminants.get(name)

    def select_sensor(self, name: Optional[str], sensor_profiles: Dict[str, Any]) -> None:
        """Set the active sensor or film profile by display name."""
        if name and name != "None":
            self.active_profile = sensor_profiles.get(name)
        else:
            self.active_profile = None

    def select_target(self, name: Optional[str], filter_collection: Any) -> None:
        """Set the target profile from a named filter."""
        if not name or name == "None":
            self.target_profile = None
            self.selected_target_name = None
        else:
            display_to_index = filter_collection.get_display_to_index_map()
            idx = display_to_index[name]
            self.target_profile = filter_collection.filters[idx]
            self.selected_target_name = name

    def select_film_emulation_target(self, name: Optional[str], sensor_profiles: Dict[str, Any]) -> None:
        """Set the film emulation target from a named SensorProfile (film only)."""
        if not name or name == "None":
            self.film_emulation_profile = None
            self.film_emulation_profile_name = None
        else:
            self.film_emulation_profile = sensor_profiles.get(name)
            self.film_emulation_profile_name = name

    # ====================================================================
    # DEFAULT REFLECTOR LIST MANAGEMENT
    # ====================================================================

    def _load_default_reflectors_from_file(self) -> List[str]:
        fp = Path(DEFAULT_REFLECTORS_FILE)
        if fp.exists():
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                return data.get("default_reflectors", [])
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save_default_reflectors_to_file(self, reflector_files: List[str]) -> None:
        fp = Path(DEFAULT_REFLECTORS_FILE)
        fp.parent.mkdir(parents=True, exist_ok=True)
        try:
            fp.write_text(
                json.dumps({"default_reflectors": reflector_files}, indent=2),
                encoding="utf-8",
            )
        except IOError:
            pass

    def get_default_reflector_files(self) -> List[str]:
        return self._default_reflector_files

    def is_default_reflector(self, source_file: str) -> bool:
        return source_file in self.get_default_reflector_files()

    def add_to_default_reflectors(self, source_file: str) -> None:
        if source_file not in self._default_reflector_files:
            self._default_reflector_files.append(source_file)
            self._save_default_reflectors_to_file(self._default_reflector_files)

    def remove_from_default_reflectors(self, source_file: str) -> None:
        if source_file in self._default_reflector_files:
            self._default_reflector_files.remove(source_file)
            self._save_default_reflectors_to_file(self._default_reflector_files)

    def seed_isdefault_reflectors(self, reflector_collection: Any) -> None:
        if not reflector_collection or not hasattr(reflector_collection, "reflectors"):
            return
        for reflector in reflector_collection.reflectors:
            is_default = reflector.metadata.get("IsDefault", "").strip()
            if is_default:
                source_file = reflector.metadata.get("source_file", "")
                if source_file and not self.is_default_reflector(source_file):
                    self.add_to_default_reflectors(source_file)

    # ====================================================================
    # MY FILTERS (OWNED FILTERS) MANAGEMENT
    # ====================================================================

    def _load_my_filters_from_file(self) -> List[str]:
        fp = Path(MY_FILTERS_FILE)
        if fp.exists():
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                return data.get("my_filters", [])
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save_my_filters_to_file(self, filter_names: List[str]) -> None:
        fp = Path(MY_FILTERS_FILE)
        fp.parent.mkdir(parents=True, exist_ok=True)
        try:
            fp.write_text(
                json.dumps({"my_filters": sorted(filter_names)}, indent=2),
                encoding="utf-8",
            )
        except IOError:
            pass

    def get_my_filters(self) -> set:
        return self._my_filters_set

    def get_my_filters_count(self) -> int:
        return len(self._my_filters_set)

    def is_my_filter(self, display_name: str) -> bool:
        return display_name in self._my_filters_set

    def add_to_my_filters(self, display_name: str) -> None:
        if display_name not in self._my_filters_set:
            self._my_filters_set.add(display_name)
            self._save_my_filters_to_file(list(self._my_filters_set))

    def remove_from_my_filters(self, display_name: str) -> None:
        if display_name in self._my_filters_set:
            self._my_filters_set.discard(display_name)
            self._save_my_filters_to_file(list(self._my_filters_set))

    def add_manufacturer_to_my_filters(self, manufacturer: str, filter_collection) -> int:
        """Add all filters from a manufacturer. Returns count added."""
        added = 0
        for f in filter_collection.filters:
            name = str(f)
            if f.manufacturer == manufacturer and name not in self._my_filters_set:
                self._my_filters_set.add(name)
                added += 1
        if added:
            self._save_my_filters_to_file(list(self._my_filters_set))
        return added

    def remove_manufacturer_from_my_filters(self, manufacturer: str, filter_collection) -> int:
        """Remove all filters from a manufacturer. Returns count removed."""
        removed = 0
        for f in filter_collection.filters:
            name = str(f)
            if f.manufacturer == manufacturer and name in self._my_filters_set:
                self._my_filters_set.discard(name)
                removed += 1
        if removed:
            self._save_my_filters_to_file(list(self._my_filters_set))
        return removed

    def clear_all_my_filters(self) -> None:
        self._my_filters_set = set()
        self._save_my_filters_to_file([])
