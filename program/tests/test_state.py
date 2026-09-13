"""
Regression suite for the plain-dataclass application state (roadmap S3).

Locks the S3 rewrite of services/state_manager.py: StateManager's dict-backed
store, change listeners, generic get()/set(), legacy select_camera path, and
the active_profile/current_qe/selected_camera triple are gone. AppState is a
plain dataclass; current_qe and selected_camera are read-only derived
properties of active_profile. The white-balance, selection, and JSON
persistence behaviours are exercised end-to-end with monkeypatched
calculators and temp-dir files -- no real user data is touched.

Run from the program/ directory:

    python -m unittest discover -s tests -v

The suite is stdlib-only (unittest + numpy); no pytest dependency.
"""
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

# Make the program/ package root importable regardless of invocation cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from models.constants import DEFAULT_WB_GAINS
from models.core import ChannelMixerSettings
from services.state_manager import AppState


def _fake_profile(name="TestCam", gain=2.0):
    channels = {ch: np.full(5, gain) for ch in ("R", "G", "B")}
    return SimpleNamespace(display_name=name, get_channels=lambda: channels)


class SimpleNamespaceWithStr:
    """Filter stand-in whose str() is 'Manufacturer Name' like models.Filter."""

    def __init__(self, manufacturer, name):
        self.manufacturer = manufacturer
        self.name = name

    def __str__(self):
        return f"{self.manufacturer} {self.name}"


def _fake_filter_collection(names):
    """Fake FilterCollection: display name == str(filter), index == position."""
    filters = [SimpleNamespaceWithStr("M", n) for n in names]
    return SimpleNamespace(
        filters=filters,
        get_display_to_index_map=lambda: {str(f): i for i, f in enumerate(filters)},
    )


# ============================================================================
# Defaults & dataclass hygiene
# ============================================================================

class TestDefaults(unittest.TestCase):
    def test_initial_values(self):
        s = AppState()
        self.assertEqual(s.selected_filters, [])
        self.assertEqual(s.filter_multipliers, {})
        self.assertIsNone(s.active_profile)
        self.assertIsNone(s.illuminant)
        self.assertIsNone(s.target_profile)
        self.assertFalse(s.film_emulation_wb)
        self.assertEqual(s.white_balance_gains, dict(DEFAULT_WB_GAINS))
        self.assertEqual(s.film_emulation_wb_gains, dict(DEFAULT_WB_GAINS))
        self.assertEqual(s.advanced_search_sort, "Filter Number")
        self.assertEqual(
            s.advanced_search_wavelength_criteria,
            [{"wavelength": 550, "trans_min": 0, "trans_max": 100}],
        )
        self.assertTrue(s.show_R and s.show_G and s.show_B)
        self.assertEqual(s.channel_mixer, ChannelMixerSettings())
        self.assertEqual(s.sidebar_expansions, {})

    def test_mutable_defaults_are_per_instance(self):
        a, b = AppState(), AppState()
        self.assertIsNot(a.selected_filters, b.selected_filters)
        self.assertIsNot(a.filter_multipliers, b.filter_multipliers)
        self.assertIsNot(a.sidebar_expansions, b.sidebar_expansions)
        self.assertIsNot(a.white_balance_gains, b.white_balance_gains)
        self.assertIsNot(a.advanced_search_wavelength_criteria,
                         b.advanced_search_wavelength_criteria)
        a.selected_filters.append("X")
        a.sidebar_expansions["k"] = True
        self.assertEqual(b.selected_filters, [])
        self.assertEqual(b.sidebar_expansions, {})


# ============================================================================
# Derived state: the collapsed sensor triple-key
# ============================================================================

class TestDerivedSensorState(unittest.TestCase):
    def test_no_profile(self):
        s = AppState()
        self.assertIsNone(s.current_qe)
        self.assertIsNone(s.selected_camera)

    def test_profile_drives_both(self):
        s = AppState()
        p = _fake_profile("Nikon D850")
        s.active_profile = p
        self.assertEqual(s.selected_camera, "Nikon D850")
        self.assertEqual(set(s.current_qe), {"R", "G", "B"})
        self.assertEqual(s.current_qe["R"][0], 2.0)

    def test_select_sensor_set_and_clear(self):
        s = AppState()
        profiles = {"Nikon D850": _fake_profile("Nikon D850")}
        s.select_sensor("Nikon D850", profiles)
        self.assertIs(s.active_profile, profiles["Nikon D850"])
        self.assertEqual(s.selected_camera, "Nikon D850")
        s.select_sensor("None", profiles)
        self.assertIsNone(s.active_profile)
        self.assertIsNone(s.selected_camera)
        self.assertIsNone(s.current_qe)


# ============================================================================
# Composed-view properties
# ============================================================================

class TestComposedProperties(unittest.TestCase):
    def test_rgb_visibility_roundtrip(self):
        s = AppState()
        self.assertEqual(s.rgb_channels_visibility, {"R": True, "G": True, "B": True})
        s.rgb_channels_visibility = {"R": False, "G": True}
        self.assertFalse(s.show_R)
        self.assertTrue(s.show_G)
        self.assertTrue(s.show_B)  # missing key defaults to True

    def test_show_channel_mixer_bridges_enabled(self):
        s = AppState()
        self.assertFalse(s.show_channel_mixer)
        s.show_channel_mixer = True
        self.assertTrue(s.channel_mixer.enabled)
        s.show_channel_mixer = False
        self.assertFalse(s.channel_mixer.enabled)

    def test_sidebar_expansions_copy_then_assign(self):
        # Mirrors views/ui_utils.stateful_expansion usage
        s = AppState()
        exps = dict(s.sidebar_expansions)
        exps["filters"] = True
        s.sidebar_expansions = exps
        self.assertEqual(s.sidebar_expansions.get("filters", False), True)


# ============================================================================
# Selection helpers
# ============================================================================

class TestSelectionHelpers(unittest.TestCase):
    def test_select_illuminant(self):
        s = AppState()
        illuminants = {"D65": np.arange(3), "D50": np.arange(3) * 2}
        s.select_illuminant("D65", illuminants)
        self.assertEqual(s.illuminant_name, "D65")
        np.testing.assert_array_equal(s.illuminant, illuminants["D65"])

    def test_select_target(self):
        s = AppState()
        # Display names are str(filter) == "Manufacturer Name"
        fc = _fake_filter_collection(["ND8", "Polarizer"])
        s.select_target("M Polarizer", fc)
        self.assertEqual(s.selected_target_name, "M Polarizer")
        self.assertIs(s.target_profile, fc.filters[1])
        s.select_target(None, fc)
        self.assertIsNone(s.target_profile)
        self.assertIsNone(s.selected_target_name)

    def test_select_film_emulation_target(self):
        s = AppState()
        profiles = {"Portra 400": _fake_profile("Portra 400")}
        s.select_film_emulation_target("Portra 400", profiles)
        self.assertIs(s.film_emulation_profile, profiles["Portra 400"])
        self.assertEqual(s.film_emulation_profile_name, "Portra 400")
        s.select_film_emulation_target("None", profiles)
        self.assertIsNone(s.film_emulation_profile)
        self.assertIsNone(s.film_emulation_profile_name)


# ============================================================================
# White balance paths (calculators monkeypatched)
# ============================================================================

class TestWhiteBalance(unittest.TestCase):
    def _state_with_profile(self):
        s = AppState()
        s.active_profile = _fake_profile()
        s.illuminant = np.ones(5)
        return s

    def test_set_white_balance_from_surface(self):
        s = self._state_with_profile()
        gains = {"R": 1.2, "G": 1.0, "B": 0.9}
        with mock.patch(
            "services.calculations.compute_white_balance_gains_from_surface",
            return_value=gains,
        ) as m:
            s.set_white_balance_from_surface(np.ones(5), np.ones(5), "refl.tsv")
        m.assert_called_once()
        self.assertEqual(s.white_balance_gains, gains)
        self.assertEqual(s.wb_reference_surface, "refl.tsv")

    def test_set_white_balance_skipped_without_profile(self):
        s = AppState()  # no active profile -> current_qe None
        with mock.patch(
            "services.calculations.compute_white_balance_gains_from_surface"
        ) as m:
            s.set_white_balance_from_surface(np.ones(5), np.ones(5), "x")
        m.assert_not_called()
        self.assertEqual(s.white_balance_gains, dict(DEFAULT_WB_GAINS))

    def test_recompute_plain_path(self):
        s = self._state_with_profile()
        gains = {"R": 1.5, "G": 1.0, "B": 0.8}
        with mock.patch(
            "services.calculations.compute_white_balance_gains",
            return_value=gains,
        ) as m:
            out = s.recompute_white_balance(np.ones(5))
        m.assert_called_once()
        self.assertEqual(out, gains)
        self.assertEqual(s.white_balance_gains, gains)

    def test_recompute_with_known_reference(self):
        s = self._state_with_profile()
        s.wb_reference_surface = "refl.tsv"
        reflector = SimpleNamespace(
            values=np.ones(5), metadata={"source_file": "refl.tsv"}
        )
        collection = SimpleNamespace(reflectors=[reflector])
        gains = {"R": 1.1, "G": 1.0, "B": 0.95}
        with mock.patch(
            "services.calculations.compute_white_balance_gains_from_surface",
            return_value=gains,
        ) as m_surface, mock.patch(
            "services.calculations.compute_white_balance_gains"
        ) as m_plain:
            out = s.recompute_white_balance(np.ones(5), collection)
        m_surface.assert_called_once()
        m_plain.assert_not_called()
        self.assertEqual(out, gains)

    def test_recompute_unknown_reference_falls_back_and_clears(self):
        s = self._state_with_profile()
        s.wb_reference_surface = "missing.tsv"
        collection = SimpleNamespace(reflectors=[])
        gains = {"R": 1.3, "G": 1.0, "B": 0.9}
        with mock.patch(
            "services.calculations.compute_white_balance_gains",
            return_value=gains,
        ):
            out = s.recompute_white_balance(np.ones(5), collection)
        self.assertEqual(out, gains)
        self.assertIsNone(s.wb_reference_surface)

    def test_recompute_noop_without_profile(self):
        s = AppState()
        with mock.patch("services.calculations.compute_white_balance_gains") as m:
            out = s.recompute_white_balance(np.ones(5))
        m.assert_not_called()
        self.assertEqual(out, dict(DEFAULT_WB_GAINS))

    def test_reset_white_balance(self):
        s = self._state_with_profile()
        s.white_balance_gains = {"R": 2.0, "G": 1.0, "B": 0.5}
        s.wb_reference_surface = "refl.tsv"
        s.reset_white_balance()
        self.assertEqual(s.white_balance_gains, dict(DEFAULT_WB_GAINS))
        self.assertIsNone(s.wb_reference_surface)


# ============================================================================
# JSON persistence: my filters + default reflectors (temp dirs)
# ============================================================================

class TestMyFiltersPersistence(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch(
            "services.state_manager.MY_FILTERS_FILE",
            os.path.join(self._tmp.name, "my_filters.json"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_add_remove_roundtrip_and_file_sorted(self):
        s = AppState()  # constructor loads (missing file -> empty set)
        s.add_to_my_filters("Zeta")
        s.add_to_my_filters("Alpha")
        self.assertTrue(s.is_my_filter("Alpha"))
        self.assertEqual(s.get_my_filters_count(), 2)
        path = os.path.join(self._tmp.name, "my_filters.json")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["my_filters"], ["Alpha", "Zeta"])  # sorted on write
        s.remove_from_my_filters("Alpha")
        self.assertFalse(s.is_my_filter("Alpha"))
        self.assertEqual(s.get_my_filters_count(), 1)

    def test_add_is_idempotent(self):
        s = AppState()
        s.add_to_my_filters("Same")
        s.add_to_my_filters("Same")
        self.assertEqual(s.get_my_filters_count(), 1)

    def test_manufacturer_add_remove(self):
        s = AppState()
        fc = _fake_filter_collection(["A One", "A Two", "B Three"])
        added = s.add_manufacturer_to_my_filters("M", fc)
        self.assertEqual(added, 3)
        removed = s.remove_manufacturer_from_my_filters("M", fc)
        self.assertEqual(removed, 3)
        self.assertEqual(s.get_my_filters_count(), 0)

    def test_clear_all(self):
        s = AppState()
        s.add_to_my_filters("X")
        s.clear_all_my_filters()
        self.assertEqual(s.get_my_filters_count(), 0)
        path = os.path.join(self._tmp.name, "my_filters.json")
        with open(path, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["my_filters"], [])

    def test_state_reload_picks_up_file(self):
        s = AppState()
        s.add_to_my_filters("Persisted")
        fresh = AppState()
        self.assertTrue(fresh.is_my_filter("Persisted"))


class TestDefaultReflectorsPersistence(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch(
            "services.state_manager.DEFAULT_REFLECTORS_FILE",
            os.path.join(self._tmp.name, "default_reflectors.json"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_add_remove_roundtrip_and_reload(self):
        s = AppState()
        s.add_to_default_reflectors("a_reflector.tsv")
        s.add_to_default_reflectors("a_reflector.tsv")  # idempotent
        self.assertTrue(s.is_default_reflector("a_reflector.tsv"))
        path = os.path.join(self._tmp.name, "default_reflectors.json")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["default_reflectors"], ["a_reflector.tsv"])
        fresh = AppState()
        self.assertTrue(fresh.is_default_reflector("a_reflector.tsv"))
        fresh.remove_from_default_reflectors("a_reflector.tsv")
        self.assertFalse(fresh.is_default_reflector("a_reflector.tsv"))

    def test_seed_isdefault_reflectors(self):
        s = AppState()
        collection = SimpleNamespace(
            reflectors=[
                SimpleNamespace(
                    metadata={"IsDefault": "yes", "source_file": "auto.tsv"}
                ),
                SimpleNamespace(
                    metadata={"IsDefault": "", "source_file": "manual.tsv"}
                ),
            ]
        )
        s.seed_isdefault_reflectors(collection)
        self.assertTrue(s.is_default_reflector("auto.tsv"))
        self.assertFalse(s.is_default_reflector("manual.tsv"))

    def test_seed_tolerates_non_collection(self):
        s = AppState()
        s.seed_isdefault_reflectors(None)
        s.seed_isdefault_reflectors(SimpleNamespace())  # no .reflectors
        self.assertEqual(s.get_default_reflector_files(), [])


# ============================================================================
# S3 decision guards: the deleted machinery must stay deleted
# ============================================================================

class TestDeletedMachineryStaysDeleted(unittest.TestCase):
    def test_no_listener_or_dict_machinery(self):
        for name in ("on_change", "_notify", "_set", "_store", "_listeners",
                     "get", "set", "select_camera", "invert_output",
                     "get_state_manager"):
            self.assertFalse(hasattr(AppState, name), msg=name)

    def test_derived_keys_are_read_only(self):
        s = AppState()
        with self.assertRaises(AttributeError):
            s.current_qe = {"R": None}
        with self.assertRaises(AttributeError):
            s.selected_camera = "X"


if __name__ == "__main__":
    unittest.main()



