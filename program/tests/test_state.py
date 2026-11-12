"""services.state_manager — AppState contracts.

Behavior only: the derived sensor triple (current_qe / selected_camera),
white-balance recompute branches (standard / surface / dangling surface),
My-Filters and default-reflector persistence round-trips, and the mutable
-default factory rule.  Persistence files live in a throwaway dir; the patch
is applied BEFORE AppState() is constructed.
"""
import json
import unittest

import numpy as np

from models.constants import DEFAULT_WB_GAINS, REPO_ROOT
from services.state_manager import AppState
from tests.helpers import (
    GRID,
    TestCase,
    display_name,
    flat,
    make_bw,
    make_collection,
    make_reflector,
    make_sensor,
    state_sandbox,
)


def wb_scene():
    """Sensor R:2 G:1 B:1, illum 2, trans 0.5 -> gains {R:2, G:1, B:1}."""
    state = AppState()
    state.active_profile = make_sensor(r=2.0, g=1.0, b=1.0)
    state.illuminant = flat(2.0)
    trans = flat(0.5)
    return state, trans


class DerivedProperties(TestCase):
    def test_no_profile_gives_no_qe_and_no_camera(self):
        state = AppState()
        self.assertIsNone(state.current_qe)
        self.assertIsNone(state.selected_camera)

    def test_active_profile_drives_both_derived_values(self):
        state = AppState()
        state.active_profile = make_sensor(manufacturer="Cam", name="X")
        self.assertEqual(set(state.current_qe), {"R", "G", "B"})
        self.assertEqual(state.selected_camera, "Cam X")

    def test_bw_profile_maps_one_curve_to_three_keys(self):
        state = AppState()
        profile = make_bw(sensitivity=1.0)
        state.active_profile = profile
        self.assertIs(state.current_qe["R"], state.current_qe["B"])


class WhiteBalanceRecompute(TestCase):
    def test_standard_recompute_hand_computed(self):
        state, trans = wb_scene()
        gains = state.recompute_white_balance(trans)
        self.assertEqual(gains, {"R": 2.0, "G": 1.0, "B": 1.0})
        self.assertEqual(state.white_balance_gains, gains)

    def test_surface_reference_recomputes_from_that_surface(self):
        state, trans = wb_scene()
        collection = type("C", (), {})()
        collection.reflectors = [
            make_reflector(flat(0.5), source_file="leaf.tsv")]
        state.wb_reference_surface = "leaf.tsv"
        gains = state.recompute_white_balance(trans, collection)
        # Neutral grey scales all channels equally: same ratios as standard.
        self.assertEqual(gains, {"R": 2.0, "G": 1.0, "B": 1.0})
        self.assertEqual(state.wb_reference_surface, "leaf.tsv")

    def test_dangling_surface_reference_falls_back_and_clears(self):
        # Regression: a deleted surface file used to leave the stored gains
        # and the UI chip disagreeing.  Fallback must drop the reference too.
        state, trans = wb_scene()
        state.wb_reference_surface = "gone.tsv"
        gains = state.recompute_white_balance(trans, reflector_collection=None)
        self.assertEqual(gains, {"R": 2.0, "G": 1.0, "B": 1.0})
        self.assertIsNone(state.wb_reference_surface)

    def test_no_sensor_leaves_existing_gains_untouched(self):
        state = AppState()
        state.white_balance_gains = {"R": 3.0, "G": 2.0, "B": 1.0}
        gains = state.recompute_white_balance(flat(0.5))
        self.assertEqual(gains, {"R": 3.0, "G": 2.0, "B": 1.0})

    def test_set_white_balance_from_surface_records_the_source(self):
        state, trans = wb_scene()
        state.set_white_balance_from_surface(
            reflector=flat(0.5), transmission=trans, source_file="leaf.tsv")
        self.assertEqual(state.white_balance_gains, {"R": 2.0, "G": 1.0, "B": 1.0})
        self.assertEqual(state.wb_reference_surface, "leaf.tsv")

    def test_set_white_balance_from_surface_is_a_noop_without_sensor(self):
        state = AppState()
        state.set_white_balance_from_surface(flat(0.5), flat(0.5), "leaf.tsv")
        self.assertEqual(state.white_balance_gains, DEFAULT_WB_GAINS)
        self.assertIsNone(state.wb_reference_surface)

    def test_reset_restores_unity_and_clears_the_reference(self):
        state, trans = wb_scene()
        state.set_white_balance_from_surface(flat(0.5), trans, "leaf.tsv")
        state.reset_white_balance()
        self.assertEqual(state.white_balance_gains, DEFAULT_WB_GAINS)
        self.assertIsNone(state.wb_reference_surface)


class MyFiltersPersistence(TestCase):
    def test_add_writes_sorted_json_and_survives_a_new_state(self):
        with state_sandbox() as tmp:
            state = AppState()
            state.add_to_my_filters("Zeta (Z1, Foo)")
            state.add_to_my_filters("Alpha (A1, Foo)")
            on_disk = json.loads(
                (tmp / "my_filters.json").read_text(encoding="utf-8"))
            self.assertEqual(on_disk["my_filters"],
                             ["Alpha (A1, Foo)", "Zeta (Z1, Foo)"])
            reloaded = AppState()
            self.assertTrue(reloaded.is_my_filter("Alpha (A1, Foo)"))

    def test_add_is_idempotent_and_remove_updates_disk(self):
        with state_sandbox():
            state = AppState()
            state.add_to_my_filters("A (A1, Foo)")
            state.add_to_my_filters("A (A1, Foo)")
            self.assertEqual(len(state.get_my_filters()), 1)
            state.remove_from_my_filters("A (A1, Foo)")
            self.assertFalse(state.is_my_filter("A (A1, Foo)"))
            self.assertFalse(AppState().is_my_filter("A (A1, Foo)"))

    def test_corrupt_file_loads_as_empty_instead_of_crashing(self):
        with state_sandbox() as tmp:
            (tmp / "my_filters.json").write_text("{not json", encoding="utf-8")
            self.assertEqual(AppState().get_my_filters(), set())

    def test_add_manufacturer_returns_count_and_owning_only_that_brand(self):
        with state_sandbox():
            state = AppState()
            collection = make_collection([
                {"name": "A", "number": "A1", "manufacturer": "Foo",
                 "hex": "#101010", "curve": flat(0.5)},
                {"name": "B", "number": "B1", "manufacturer": "Foo",
                 "hex": "#202020", "curve": flat(0.4)},
                {"name": "C", "number": "C1", "manufacturer": "Bar",
                 "hex": "#303030", "curve": flat(0.3)},
            ])
            added = state.add_manufacturer_to_my_filters("Foo", collection)
            self.assertEqual(added, 2)
            self.assertTrue(state.is_my_filter(display_name("A", "A1", "Foo")))
            self.assertFalse(state.is_my_filter(display_name("C", "C1", "Bar")))
            removed = state.remove_manufacturer_from_my_filters("Foo", collection)
            self.assertEqual(removed, 2)
            self.assertEqual(state.get_my_filters(), set())


class DefaultReflectorPersistence(TestCase):
    def test_add_and_reload_round_trip(self):
        with state_sandbox():
            state = AppState()
            state.add_to_default_reflectors("plant/leaf.tsv")
            self.assertTrue(state.is_default_reflector("plant/leaf.tsv"))
            self.assertEqual(AppState().get_default_reflector_files(),
                             ["plant/leaf.tsv"])

    def test_seed_isdefault_is_idempotent(self):
        with state_sandbox():
            state = AppState()
            collection = type("C", (), {})()
            collection.reflectors = [
                make_reflector(flat(0.5), source_file="leaf.tsv",
                               IsDefault="yes")]
            state.seed_isdefault_reflectors(collection)
            state.seed_isdefault_reflectors(collection)
            self.assertEqual(state.get_default_reflector_files(), ["leaf.tsv"])

    def test_relative_and_absolute_forms_are_one_entry(self):
        # Old JSON saves mixed path forms; matching must normalize both.
        with state_sandbox():
            state = AppState()
            relative = "program/data/reflectors/plant/leaf.tsv"
            state.add_to_default_reflectors(str(REPO_ROOT / relative))
            state.add_to_default_reflectors(relative)  # same file, other form
            self.assertEqual(state.get_default_reflector_files(), [relative])
            self.assertTrue(
                state.is_default_reflector(str(REPO_ROOT / relative)))
            self.assertTrue(AppState().is_default_reflector(relative))


class SceneLookToggle(TestCase):
    def test_camera_look_is_on_by_default(self):
        self.assertTrue(AppState().scene_look)


class MutableDefaultSafety(TestCase):
    def test_wavelength_criteria_factory_gives_fresh_rows_per_state(self):
        with state_sandbox():
            s1, s2 = AppState(), AppState()
            s1.advanced_search_wavelength_criteria[0]["wavelength"] = 600
            self.assertEqual(
                s2.advanced_search_wavelength_criteria[0]["wavelength"], 550)

    def test_default_sort_choice_is_filter_number(self):
        self.assertEqual(AppState().advanced_search_sort, "Filter Number")


if __name__ == "__main__":
    unittest.main()
