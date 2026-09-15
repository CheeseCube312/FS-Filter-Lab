"""services.presenter — view models, the read-only guarantee, and search.

Locks: render paths never mutate AppState (snapshot diff), the metrics/WB
fallback messages, quick-search token AND-matching (regex-safe), advanced
search criteria (AND, UI-percent scale) and sort orders, and the Film Match
fallback notes.
"""
import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from models.core import ReflectorCollection
from services.presenter import (
    QE_ILLUMINANT_REQUIRED_MSG,
    prepare_filter_analysis,
    prepare_filter_rows,
    prepare_film_match,
    prepare_reference,
    prepare_reflectors_shelf,
    prepare_sensor_analysis,
    prepare_surfaces,
    search_filters,
    search_reflectors,
)
from services.state_manager import AppState
from tests.helpers import (
    TestCase,
    display_name,
    flat,
    make_app_data,
    make_bw,
    make_collection,
    make_reflector,
    make_sensor,
    snapshot_state,
    assert_state_unchanged,
    state_sandbox,
)


def shelf_collection():
    """Three filters with distinct brands, colors, and transmission levels."""
    return make_collection([
        {"name": "Oklahoma Yellow", "number": "Y1", "manufacturer": "Lee",
         "hex": "#ffdd00", "curve": flat(0.8)},
        {"name": "ND6", "number": "N1", "manufacturer": "Foo",
         "hex": "#204060", "curve": flat(0.25)},
        {"name": "ND3", "number": "N2", "manufacturer": "Foo",
         "hex": "#406080", "curve": flat(0.5)},
    ])


def rich_state(collection):
    """State with everything set — the fullest render path in the app."""
    state = AppState()
    state.selected_filters = [display_name("ND6", "N1", "Foo")]
    state.filter_multipliers = {display_name("ND6", "N1", "Foo"): 2}
    state.active_profile = make_sensor(r=2.0, g=1.0, b=1.0)
    state.illuminant = flat(2.0)
    state.illuminant_name = "D65"
    state.film_emulation_profile = make_bw()
    state.film_emulation_filters = [display_name("ND3", "N2", "Foo")]
    state.film_emulation_wb = True
    state.white_balance_gains = {"R": 2.0, "G": 1.0, "B": 1.0}
    state.wb_reference_surface = "leaf.tsv"
    state.selected_surfaces = ["leaf.tsv"]
    state.apply_white_balance = True
    return state


def reflector_app_data(collection):
    """AppData whose reflector library holds one flat 50% 'Leaf' surface."""
    data = make_app_data(collection)
    data.reflector_collection = ReflectorCollection(
        reflectors=[make_reflector(flat(0.5), source_file="leaf.tsv", Name="Leaf")],
        df=pd.DataFrame([{
            "Name": "Leaf", "Source File": "leaf.tsv",
            "Organization": "Uni", "Target Type": "foliage",
        }]),
        reflector_matrix=np.asarray([flat(0.5)]),
    )
    return data


class PresenterReadOnly(TestCase):
    def test_render_paths_never_mutate_app_state(self):
        with state_sandbox():
            collection = shelf_collection()
            data = reflector_app_data(collection)
            state = rich_state(collection)
            before = snapshot_state(state)
            prepare_filter_analysis(state, collection)
            prepare_sensor_analysis(state, data)
            prepare_reference(state, data)
            prepare_film_match(state, data)
            prepare_surfaces(state, data)
            prepare_reflectors_shelf(state, data)
            prepare_filter_rows(state, data, query="nd")
            search_filters(collection.df, collection.filter_matrix, state)
            after = snapshot_state(state)
            assert_state_unchanged(self, before, after)


class SurfacesPresenter(TestCase):
    def test_missing_sensor_or_illuminant_warns_with_no_chips(self):
        with state_sandbox():
            state = AppState()  # no profile
            vm = prepare_surfaces(state, reflector_app_data(shelf_collection()))
            self.assertEqual(vm.chips, [])
            self.assertEqual(vm.warning, QE_ILLUMINANT_REQUIRED_MSG)
            self.assertIsNone(vm.wb_source_name)

    def test_selected_surface_becomes_a_white_referenced_chip(self):
        # White referencing divides by the perfect-diffuser color behind the
        # same stack, so stack, sensor gains, and illuminant all cancel: a
        # flat 50% reflector must render as linear (0.5, 0.5, 0.5).
        with state_sandbox():
            state = AppState()
            state.active_profile = make_sensor(r=2.0, g=1.0, b=1.0)
            state.illuminant = flat(2.0)
            state.selected_surfaces = ["leaf.tsv"]
            state.wb_reference_surface = "leaf.tsv"
            vm = prepare_surfaces(state, reflector_app_data(shelf_collection()))
            self.assertEqual(len(vm.chips), 1)
            chip = vm.chips[0]
            self.assertEqual(chip.name, "Leaf")
            self.assertEqual(chip.source_file, "leaf.tsv")
            self.assertNanEqual(list(chip.linear_rgb), [0.5, 0.5, 0.5])
            self.assertTrue(chip.is_wb_ref)
            self.assertEqual(vm.wb_source_name, "Leaf")
            self.assertRegex(chip.hex_display, r"^#[0-9a-fA-F]{6}$")


class FilterAnalysisPresenter(TestCase):
    def test_no_sensor_qe_reports_the_missing_data_error(self):
        with state_sandbox():
            state = AppState()  # no profile
            vm = prepare_filter_analysis(state, shelf_collection())
            self.assertIsNotNone(vm.metrics)
            self.assertFalse(vm.metrics.has_valid_data)
            self.assertIn("no sensor QE data", vm.metrics.error_message)
            self.assertEqual(vm.metrics.label, "No Filter")

    def test_half_transmission_stack_shows_fifty_percent_and_one_stop(self):
        with state_sandbox():
            collection = make_collection([
                {"name": "Half", "number": "H1", "manufacturer": "Foo",
                 "hex": "#204060", "curve": flat(0.5)}])
            state = AppState()
            state.selected_filters = [display_name("Half", "H1", "Foo")]
            state.active_profile = make_sensor()
            vm = prepare_filter_analysis(state, collection)
            self.assertEqual(vm.metrics.label, "Single")
            self.assertEqual(vm.metrics.avg_transmission_pct, "50.0%")
            self.assertEqual(vm.metrics.effective_stops, "1.00")
            self.assertIsInstance(vm.filter_response_fig, go.Figure)


class SensorPresenter(TestCase):
    def test_missing_sensor_or_illuminant_shows_the_guidance_message(self):
        with state_sandbox():
            state = AppState()
            vm = prepare_sensor_analysis(state, make_app_data(shelf_collection()))
            self.assertEqual(vm.info_message, QE_ILLUMINANT_REQUIRED_MSG)
            self.assertIsNone(vm.wb_display)

    def test_full_scene_shows_gains_with_no_note_when_filters_selected(self):
        with state_sandbox():
            collection = shelf_collection()
            state = rich_state(collection)
            state.wb_reference_surface = None  # no surface reference active
            vm = prepare_sensor_analysis(state, make_app_data(collection))
            self.assertIsNone(vm.info_message)
            self.assertEqual(vm.wb_display.gains,
                             {"R": 2.0, "G": 1.0, "B": 1.0})
            self.assertEqual(vm.wb_display.mode_note, "")

    def test_no_filter_selected_note(self):
        with state_sandbox():
            collection = shelf_collection()
            state = rich_state(collection)
            state.selected_filters = []
            state.filter_multipliers = {}
            state.wb_reference_surface = None
            vm = prepare_sensor_analysis(state, make_app_data(collection))
            self.assertEqual(vm.wb_display.mode_note, " (No filter selected)")

    def test_surface_reference_note(self):
        with state_sandbox():
            collection = shelf_collection()
            state = rich_state(collection)
            state.selected_filters = []  # surface note wins over the empty note
            state.wb_reference_surface = "leaf.tsv"
            vm = prepare_sensor_analysis(state, make_app_data(collection))
            self.assertEqual(vm.wb_display.mode_note, " (from surface reference)")


class QuickSearch(TestCase):
    def rows(self, query, state=None, collection=None):
        collection = collection or shelf_collection()
        state = state or AppState()
        return prepare_filter_rows(state, make_app_data(collection), query=query)

    def names(self, rows):
        return [r.display_name for r in rows]

    def test_tokens_and_match_across_fields(self):
        # 'lee' hits the manufacturer, 'oklahoma' the name — both must match.
        result = self.names(self.rows("lee oklahoma"))
        self.assertEqual(result,
                         [display_name("Oklahoma Yellow", "Y1", "Lee")])

    def test_tokens_are_literal_not_regex(self):
        # '(' is a regex metacharacter; as a literal token it must match the
        # one name containing a paren instead of raising re.error.
        collection = make_collection([
            {"name": "ND (Old)", "number": "N0", "manufacturer": "Foo",
             "hex": "#0a0a0a", "curve": flat(0.4)},
            {"name": "ND6", "number": "N1", "manufacturer": "Foo",
             "hex": "#204060", "curve": flat(0.25)},
        ])
        result = self.names(self.rows("(", collection=collection))
        self.assertEqual(result, [display_name("ND (Old)", "N0", "Foo")])

    def test_owned_only_toggle_narrows_quick_results(self):
        with state_sandbox():
            state = AppState()
            state.add_to_my_filters(display_name("ND6", "N1", "Foo"))
            state.my_filters_only = True
            result = self.names(self.rows("", state=state))
            self.assertEqual(result, [display_name("ND6", "N1", "Foo")])

    def test_row_flags_carry_stack_membership_and_multiplier(self):
        with state_sandbox():
            state = AppState()
            state.selected_filters = [display_name("ND6", "N1", "Foo")]
            state.filter_multipliers = {display_name("ND6", "N1", "Foo"): 3}
            rows = prepare_filter_rows(state, make_app_data(shelf_collection()),
                                       query="ND6")
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0].in_stack)
            self.assertEqual(rows[0].multiplier, 3)
            self.assertFalse(rows[0].is_owned)


class AdvancedSearch(TestCase):
    def search(self, state, collection=None):
        collection = collection or shelf_collection()
        names = search_filters(collection.df, collection.filter_matrix, state)
        return list(names["Filter Name"])

    def test_manufacturer_filter(self):
        with state_sandbox():
            state = AppState()
            state.advanced_search_manufacturers = ["Lee"]
            self.assertEqual(self.search(state), ["Oklahoma Yellow"])

    def test_wavelength_criterion_uses_the_ui_percent_scale(self):
        with state_sandbox():
            state = AppState()
            state.advanced_search_wavelength_criteria = [
                {"wavelength": 550, "trans_min": 20, "trans_max": 60}]
            # ND6 25% and ND3 50% at 550 nm are both inside [20, 60].
            self.assertEqual(sorted(self.search(state)), ["ND3", "ND6"])

    def test_multiple_criteria_are_anded(self):
        with state_sandbox():
            state = AppState()
            state.advanced_search_wavelength_criteria = [
                {"wavelength": 550, "trans_min": 20, "trans_max": 30},
                {"wavelength": 700, "trans_min": 20, "trans_max": 30}]
            self.assertEqual(self.search(state), ["ND6"])

    def test_transmission_sort_is_descending_at_first_criterion(self):
        with state_sandbox():
            state = AppState()
            state.advanced_search_sort = "Trans @ 550 nm"
            state.advanced_search_wavelength_criteria = [
                {"wavelength": 550, "trans_min": 0, "trans_max": 100}]
            self.assertEqual(self.search(state),
                             ["Oklahoma Yellow", "ND3", "ND6"])

    def test_sort_label_wavelength_is_ignored_criterion_wavelength_wins(self):
        # The 'Trans @ ...' label is display text: the sort key is always the
        # FIRST CRITERION's wavelength, even when the label names another.
        with state_sandbox():
            state = AppState()
            state.advanced_search_sort = "Trans @ 551 nm"  # label lies
            state.advanced_search_wavelength_criteria = [
                {"wavelength": 550, "trans_min": 0, "trans_max": 100}]
            self.assertEqual(self.search(state),
                             ["Oklahoma Yellow", "ND3", "ND6"])

    def test_name_and_number_sorts(self):
        with state_sandbox():
            state = AppState()
            state.advanced_search_sort = "Filter Name"
            self.assertEqual(self.search(state), ["ND3", "ND6", "Oklahoma Yellow"])
            state.advanced_search_sort = "Filter Number"
            self.assertEqual(self.search(state),
                             ["ND6", "ND3", "Oklahoma Yellow"])  # N1, N2, Y1

    def test_hex_rainbow_sort_returns_all_rows_without_helper_columns(self):
        with state_sandbox():
            state = AppState()
            state.advanced_search_sort = "Hex-Rainbow"
            collection = shelf_collection()
            result = search_filters(collection.df, collection.filter_matrix, state)
            self.assertEqual(sorted(result["Filter Name"]),
                             ["ND3", "ND6", "Oklahoma Yellow"])
            self.assertNotIn("_hue", result.columns)


class ReflectorSearch(TestCase):
    def test_filters_by_each_facet_and_passes_everything_through_empty(self):
        df = pd.DataFrame([
            {"Name": "A", "Organization": "Uni", "Package Title": "Leaves",
             "Target Type": "Plant"},
            {"Name": "B", "Organization": "Corp", "Package Title": "Skins",
             "Target Type": "Animal"},
        ])
        self.assertEqual(list(search_reflectors(df, ["Uni"], [], [])["Name"]),
                         ["A"])
        self.assertEqual(list(search_reflectors(df, [], ["Leaves"], [])["Name"]),
                         ["A"])
        self.assertEqual(list(search_reflectors(df, [], [], ["Animal"])["Name"]),
                         ["B"])
        self.assertEqual(len(search_reflectors(df, [], [], [])), 2)


class FilmMatchPresenter(TestCase):
    def test_no_film_target_shows_pick_guidance(self):
        with state_sandbox():
            state = AppState()
            vm = prepare_film_match(state, make_app_data(shelf_collection()))
            self.assertIsNone(vm.overlay_fig)
            self.assertEqual(vm.deviations, [])
            self.assertIn("Pick a film target", vm.note)

    def test_film_without_sensor_or_illuminant_requires_them(self):
        with state_sandbox():
            state = AppState()
            state.film_emulation_profile = make_bw()
            vm = prepare_film_match(state, make_app_data(shelf_collection()))
            self.assertIn("sensor and an illuminant are required", vm.note)

    def test_full_scene_builds_overlay_and_per_channel_deviations(self):
        with state_sandbox():
            collection = shelf_collection()
            state = rich_state(collection)
            vm = prepare_film_match(state, make_app_data(collection))
            self.assertIsInstance(vm.overlay_fig, go.Figure)
            self.assertEqual([d.channel for d in vm.deviations],
                             ["R", "G", "B"])
            first = vm.deviations[0]
            self.assertTrue(first.peak_wavelength.endswith("nm"))
            self.assertTrue(first.peak_ratio.startswith("×"))
            self.assertTrue(first.rms_deviation.endswith("%"))

    def test_deviation_metrics_are_scale_invariant(self):
        # Deviations judge SHAPE: curves are peak-normalized first, so a pure
        # scale change (white-balancing the film target divides it by its own
        # gains) must not move peak_ratio or rms_deviation.
        with state_sandbox():
            collection = shelf_collection()
            state = rich_state(collection)
            with_wb = prepare_film_match(state, make_app_data(collection))
            state.film_emulation_wb = False
            without_wb = prepare_film_match(state, make_app_data(collection))
            wb_by_ch = {d.channel: d for d in with_wb.deviations}
            raw_by_ch = {d.channel: d for d in without_wb.deviations}
            for ch in ("R", "G", "B"):
                self.assertEqual(wb_by_ch[ch].peak_ratio,
                                 raw_by_ch[ch].peak_ratio)
                self.assertEqual(wb_by_ch[ch].rms_deviation,
                                 raw_by_ch[ch].rms_deviation)


if __name__ == "__main__":
    unittest.main()
