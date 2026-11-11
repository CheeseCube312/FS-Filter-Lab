"""services.presenter — view models, the read-only guarantee, and search.

Locks: render paths never mutate AppState (snapshot diff), the metrics/WB
fallback messages, quick-search token AND-matching (regex-safe), and advanced
search criteria (AND, UI-percent scale) and sort orders.
"""
import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from models.core import ColorNegativeFilmProfile, ReflectorCollection
from services.presenter import (
    QE_ILLUMINANT_REQUIRED_MSG,
    prepare_filter_analysis,
    prepare_filter_rows,
    prepare_header_status,
    prepare_reference,
    prepare_reflectors_shelf,
    prepare_scene_preview,
    prepare_sensor_analysis,
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
            prepare_scene_preview(state, data)
            prepare_reflectors_shelf(state, data)
            prepare_header_status(state, data)
            prepare_filter_rows(state, data, query="nd")
            search_filters(collection.df, collection.filter_matrix, state)
            after = snapshot_state(state)
            assert_state_unchanged(self, before, after)


class ReflectorsShelfPresenter(TestCase):
    def test_rows_carry_nan_honest_reflectance_sparklines(self):
        # The sparkline plots the raw matrix row: real samples as 0.5, and a
        # mid-spectrum NaN gap must survive as line breaks (never zero-filled).
        with state_sandbox():
            data = reflector_app_data(shelf_collection())
            curve = flat(0.5)
            curve[300:400] = np.nan
            data.reflector_collection.reflector_matrix = np.asarray([curve])
            vm = prepare_reflectors_shelf(AppState(), data)
            self.assertEqual(vm.total_matches, 1)
            self.assertEqual(len(vm.rows), 1)
            y = vm.rows[0].sparkline.data[0].y
            self.assertEqual(len(y), (len(curve) + 1) // 2)  # [::2] decimation
            self.assertTrue(np.isnan(y[150:200]).all())      # gap stays a gap
            self.assertTrue((y[:150] == 0.5).all())
            self.assertTrue((y[200:] == 0.5).all())


class ScenePreviewPresenter(TestCase):
    def test_missing_sensor_or_illuminant_warns_with_no_swatches(self):
        with state_sandbox():
            state = AppState()  # no profile
            vm = prepare_scene_preview(
                state, reflector_app_data(shelf_collection()))
            self.assertEqual(vm.swatches, [])
            self.assertEqual(vm.warning, QE_ILLUMINANT_REQUIRED_MSG)
            self.assertIsNone(vm.wb_source_name)
            self.assertTrue(vm.look_on)  # camera look is default-on

    def test_selected_surface_becomes_a_render_swatch(self):
        # White referencing divides by the perfect-diffuser color behind the
        # same stack, so stack, sensor gains, and illuminant all cancel: a
        # flat 50% reflector must render as linear (0.5, 0.5, 0.5) — the
        # camera look fixes neutral midtones.
        with state_sandbox():
            state = AppState()
            state.active_profile = make_sensor(r=2.0, g=1.0, b=1.0)
            state.illuminant = flat(2.0)
            state.selected_surfaces = ["leaf.tsv"]
            state.wb_reference_surface = "leaf.tsv"
            vm = prepare_scene_preview(
                state, reflector_app_data(shelf_collection()))
            self.assertEqual(len(vm.swatches), 1)
            swatch = vm.swatches[0]
            self.assertEqual(swatch.name, "Leaf")
            self.assertEqual(swatch.source_file, "leaf.tsv")
            self.assertNanEqual(list(swatch.display_linear), [0.5, 0.5, 0.5])
            self.assertTrue(swatch.is_wb_ref)
            self.assertEqual(vm.wb_source_name, "Leaf")
            self.assertRegex(swatch.display_hex, r"^#[0-9a-fA-F]{6}$")
            self.assertFalse(swatch.partial_data)

    def test_camera_look_toggle_switches_the_swatch_render(self):
        # 75% reflector: look OFF renders the plain white-referenced 0.75;
        # look ON lifts it onto the filmic smoothstep value.
        with state_sandbox():
            state = AppState()
            state.active_profile = make_sensor()
            state.illuminant = flat(2.0)
            state.selected_surfaces = ["leaf.tsv"]
            data = reflector_app_data(shelf_collection())
            data.reflector_collection.reflectors[0].values = flat(0.75)

            state.scene_look = False
            off = prepare_scene_preview(state, data).swatches[0]
            self.assertNanEqual(list(off.display_linear), [0.75, 0.75, 0.75])

            state.scene_look = True
            on = prepare_scene_preview(state, data).swatches[0]
            self.assertClose(on.display_linear[0], 0.75 ** 2 * (3 - 2 * 0.75))

    def test_negative_film_inverts_under_the_camera_look(self):
        with state_sandbox():
            state = AppState()
            state.active_profile = ColorNegativeFilmProfile(
                manufacturer="F", name="Neg",
                channel_data={"R": flat(1.0), "G": flat(1.0), "B": flat(1.0)})
            state.illuminant = flat(2.0)
            state.selected_surfaces = ["leaf.tsv"]
            data = reflector_app_data(shelf_collection())
            data.reflector_collection.reflectors[0].values = flat(0.75)
            vm = prepare_scene_preview(state, data)
            swatch = vm.swatches[0]
            self.assertTrue(vm.inverts_for_display)
            # 1 - 0.75 = 0.25 through the same tone curve.
            self.assertClose(swatch.display_linear[0], 0.25 ** 2 * (3 - 2 * 0.25))

    def test_partial_data_flagged_when_chain_weight_lands_on_gaps(self):
        # No reflectance data below 400 nm (grid positions 0..99); with a
        # full-grid chain, 100 of 801 weighted nanometers (~12.5%) fall in
        # the gap -> flagged.
        with state_sandbox():
            state = AppState()
            state.active_profile = make_sensor()
            state.illuminant = flat(2.0)
            state.selected_surfaces = ["leaf.tsv"]
            data = reflector_app_data(shelf_collection())
            curve = flat(0.5)
            curve[:100] = np.nan
            data.reflector_collection.reflectors[0].values = curve
            vm = prepare_scene_preview(state, data)
            swatch = vm.swatches[0]
            self.assertTrue(swatch.partial_data)
            self.assertIn("300–399 nm", swatch.coverage_note)


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

    def names(self, result):
        return [r.display_name for r in result.rows]

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
                                       query="ND6").rows
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0].in_stack)
            self.assertEqual(rows[0].multiplier, 3)
            self.assertFalse(rows[0].is_owned)


    def test_total_matches_counts_beyond_the_capped_rows(self):
        # 'nd' matches ND6 + ND3; a small limit caps the rows, not the count.
        vm = prepare_filter_rows(AppState(), make_app_data(shelf_collection()),
                                 query="nd", limit=1)
        self.assertEqual(len(vm.rows), 1)
        self.assertEqual(vm.total_matches, 2)


class HeaderStatusPresenter(TestCase):
    def vm(self, state):
        return prepare_header_status(state, make_app_data(shelf_collection()))

    def test_empty_state_shows_placeholders(self):
        self.assertEqual(
            self.vm(AppState()).segments,
            ["No receiver", "No illuminant", "Stack empty", "WB off"])

    def test_full_state_reports_receiver_illuminant_and_stack(self):
        state = AppState()
        state.active_profile = make_sensor()
        state.illuminant = flat(2.0)
        state.illuminant_name = "D65"
        state.selected_filters = [display_name("ND6", "N1", "Foo")]
        segments = self.vm(state).segments
        self.assertTrue(segments[0].startswith("Cam X"))
        self.assertEqual(segments[1], "D65")
        self.assertIn("1 filter ", segments[2])
        self.assertIn("stops", segments[2])

    def test_wb_surface_resolves_to_spectrum_name(self):
        with state_sandbox():
            state = AppState()
            state.apply_white_balance = True
            state.wb_reference_surface = "leaf.tsv"
            status = prepare_header_status(state, reflector_app_data(shelf_collection()))
            self.assertIn("WB: Leaf", status.segments)

    def test_wb_standard_when_no_reference_surface_is_set(self):
        state = AppState()
        state.apply_white_balance = True
        self.assertIn("WB: standard", self.vm(state).segments)


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


if __name__ == "__main__":
    unittest.main()
