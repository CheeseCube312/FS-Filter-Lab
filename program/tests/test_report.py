"""services.reporting — the PNG report smoke test + filename sanitizer.

One end-to-end report through generate_application_report into a throwaway
output tree: PNG magic bytes, a real file at the camera/illuminant path, and
the white-balanced variant (which regressed when report extraction dropped
its EPSILON/COLOR_MAP imports).  Plus the report-side filename sanitizer,
deliberately different from importing._sanitize_filename.
"""
import unittest
from pathlib import Path

import numpy as np

from models.core import FilterCollection  # noqa: F401  (type contract)
from services.reporting import (
    generate_application_report,
    generate_report_png,
    sanitize_filename_component,
)
from services.state_manager import AppState
from tests.helpers import (
    TestCase,
    display_name,
    flat,
    make_collection,
    make_sensor,
    report_sandbox,
)


def report_collection():
    return make_collection([
        {"name": "ND6", "number": "N1", "manufacturer": "Foo",
         "hex": "#204060", "curve": flat(0.25)},
        {"name": "ND3", "number": "N2", "manufacturer": "Foo",
         "hex": "#406080", "curve": flat(0.5)},
    ])


def report_state():
    state = AppState()
    state.selected_filters = [display_name("ND6", "N1", "Foo")]
    state.active_profile = make_sensor(manufacturer="Cam", name="X")
    state.illuminant = flat(2.0)
    state.illuminant_name = "D65"
    return state


class SanitizeFilenameComponent(TestCase):
    def test_docstring_example_invalid_chars_become_dashes(self):
        self.assertEqual(sanitize_filename_component("Filter: UV/IR-Cut"),
                         "Filter- UV-IR-Cut")

    def test_lowercase_and_max_len_kwargs(self):
        self.assertEqual(sanitize_filename_component("Cam Model X",
                                                     lowercase=True,
                                                     max_len=3), "cam")

    def test_all_invalid_characters_are_replaced(self):
        self.assertEqual(sanitize_filename_component('a<>:"/\\|?*b'), "a---------b")

    def test_deliberately_different_from_importing_sanitizer(self):
        # importing._sanitize_filename maps spaces -> underscores for import
        # files; the report sanitizer keeps spaces.  Unifying them would
        # orphan existing files — this difference is a contract.
        self.assertIn(" ", sanitize_filename_component("Cam X"))


class ApplicationReport(TestCase):
    def test_report_renders_png_bytes_and_a_file_in_the_output_tree(self):
        with report_sandbox() as reports:
            result = generate_application_report(
                report_state(), report_collection(), selected_camera="Cam X")
            self.assertIsNotNone(result)
            self.assertIn("bytes", result)
            self.assertIn("name", result)
            self.assertTrue(result["bytes"].startswith(b"\x89PNG"))
            self.assertGreater(len(result["bytes"]), 5000)
            written = reports / "Cam X" / "D65" / result["name"]
            self.assertTrue(written.exists(), f"missing {written}")
            self.assertEqual(written.read_bytes(), result["bytes"])
            self.assertTrue(result["name"].endswith(".png"))

    def test_report_renders_with_white_balance_applied(self):
        # Regression: the report extraction dropped the EPSILON/COLOR_MAP
        # imports, so the white-balanced sensor section crashed with
        # NameError.  This path must render.
        with report_sandbox():
            state = report_state()
            state.apply_white_balance = True
            state.white_balance_gains = {"R": 2.0, "G": 1.0, "B": 1.0}
            result = generate_application_report(state, report_collection())
            self.assertTrue(result["bytes"].startswith(b"\x89PNG"))

    def test_no_selected_filters_yields_no_report(self):
        with report_sandbox():
            state = report_state()
            state.selected_filters = []
            self.assertIsNone(
                generate_application_report(state, report_collection()))

    def test_direct_png_call_with_no_filters_returns_empty_dict(self):
        with report_sandbox():
            self.assertEqual(generate_report_png(
                selected_filters=[], selected_indices=[],
                active_transmission=np.ones(801),
                transmission_label="No Filter",
                effective_stops=0.0, avg_transmission=1.0,
                white_balance_gains={"R": 1.0, "G": 1.0, "B": 1.0},
                current_qe=None, camera_name="C", illuminant_name="D",
                filter_df=report_collection().df,
                display_to_index={}, filter_matrix=np.ones((1, 801)),
                masks=np.zeros((1, 801), dtype=bool),
                interp_grid=np.arange(300, 1101),
                sanitize_fn=sanitize_filename_component,
            ), {})


if __name__ == "__main__":
    unittest.main()
