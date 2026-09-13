"""
Regression suite for the S6 search-compute move.

Locks the advanced filter/reflector search logic that moved from
views/forms.py into services/presenter.py (search_filters,
search_reflectors and the _filter_by_* helpers). The UI layer keeps
rendering only; these tests exercise the compute against a synthetic
filter table and transmission matrix on the real INTERP_GRID.

Run from the program/ directory:

    python -m unittest discover -s tests -v

The suite is stdlib-only (unittest + numpy + pandas); no pytest dependency.
"""
import os
import sys
import unittest

# Make the program/ package root importable regardless of invocation cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from models.constants import INTERP_GRID
from services.presenter import (
    search_filters,
    search_reflectors,
    _filter_by_manufacturer,
    _filter_by_trans_at_wavelength,
    _filter_by_multiple_wavelength_criteria,
)
from services.state_manager import AppState


def _make_df():
    """Three-filter table with distinct manufacturers, colors, and names."""
    return pd.DataFrame({
        "Filter Name": ["Zeta", "Alpha", "Mike"],
        "Filter Number": ["003", "001", "002"],
        "Manufacturer": ["BrandB", "BrandA", "BrandA"],
        "Hex Color": ["#FF0000", "#00FF00", "#0000FF"],
    })


def _make_matrix(df):
    """Transmission matrix with per-row values at 550/600 nm; 1.0 elsewhere."""
    m = np.ones((len(df), len(INTERP_GRID)))
    m[:, int(np.where(INTERP_GRID == 550)[0][0])] = [0.9, 0.5, 0.1]  # Zeta, Alpha, Mike
    m[:, int(np.where(INTERP_GRID == 600)[0][0])] = [0.1, 0.5, 0.9]  # reversed
    return m


def _display_name(row):
    return f"{row['Filter Name']} ({row['Filter Number']}, {row['Manufacturer']})"


class TestFilterHelpers(unittest.TestCase):
    def setUp(self):
        self.df = _make_df()
        self.matrix = _make_matrix(self.df)

    def test_manufacturer_filter(self):
        out = _filter_by_manufacturer(self.df, ["BrandA"])
        self.assertEqual(sorted(out["Filter Name"]), ["Alpha", "Mike"])
        self.assertIs(self.df, _filter_by_manufacturer(self.df, []))

    def test_trans_at_wavelength_bounds_inclusive(self):
        out, tv = _filter_by_trans_at_wavelength(
            self.df, INTERP_GRID, self.matrix, 550, 0.1, 0.5
        )
        self.assertEqual(sorted(out["Filter Name"]), ["Alpha", "Mike"])
        self.assertTrue((tv >= 0.1).all() and (tv <= 0.5).all())

    def test_trans_at_wavelength_off_grid_keeps_all(self):
        out, tv = _filter_by_trans_at_wavelength(
            self.df, INTERP_GRID, self.matrix, 299, 0.0, 0.0
        )
        self.assertEqual(len(out), 3)
        self.assertTrue((tv == 0).all())

    def test_multiple_criteria_and_logic(self):
        criteria = [
            {"wavelength": 550, "trans_min": 0, "trans_max": 60},    # Alpha, Mike
            {"wavelength": 600, "trans_min": 40, "trans_max": 60},   # Alpha only
        ]
        out = _filter_by_multiple_wavelength_criteria(
            self.df, INTERP_GRID, self.matrix, criteria
        )
        self.assertEqual(list(out["Filter Name"]), ["Alpha"])

    def test_multiple_criteria_empty_returns_all(self):
        self.assertIs(
            self.df,
            _filter_by_multiple_wavelength_criteria(self.df, INTERP_GRID, self.matrix, []),
        )


class TestSearchFilters(unittest.TestCase):
    def setUp(self):
        self.df = _make_df()
        self.matrix = _make_matrix(self.df)
        self.state = AppState()

    def test_default_search_returns_all_sorted_by_number(self):
        out = search_filters(self.df, self.matrix, self.state)
        self.assertEqual(list(out["Filter Number"]), ["001", "002", "003"])

    def test_owned_only_matches_display_names(self):
        self.state.my_filters_only = True
        self.state._my_filters_set = {
            _display_name(self.df.iloc[2]),  # "Mike (002, BrandA)"
            "not-a-real-filter",
        }
        out = search_filters(self.df, self.matrix, self.state)
        self.assertEqual(list(out["Filter Name"]), ["Mike"])

    def test_manufacturer_selection(self):
        self.state.advanced_search_manufacturers = ["BrandB"]
        out = search_filters(self.df, self.matrix, self.state)
        self.assertEqual(list(out["Filter Name"]), ["Zeta"])

    def test_criteria_narrow_results(self):
        self.state.advanced_search_wavelength_criteria = [
            {"wavelength": 550, "trans_min": 0, "trans_max": 20},
        ]
        out = search_filters(self.df, self.matrix, self.state)
        self.assertEqual(list(out["Filter Name"]), ["Mike"])

    def test_sort_by_name(self):
        self.state.advanced_search_sort = "Filter Name"
        out = search_filters(self.df, self.matrix, self.state)
        self.assertEqual(list(out["Filter Name"]), ["Alpha", "Mike", "Zeta"])

    def test_sort_hex_rainbow_orders_by_hue(self):
        self.state.advanced_search_sort = "Hex-Rainbow"
        out = search_filters(self.df, self.matrix, self.state)
        # Red (hue 0.0) -> Green (~0.33) -> Blue (~0.67)
        self.assertEqual(list(out["Filter Name"]), ["Zeta", "Alpha", "Mike"])

    def test_sort_trans_at_lambda_descending(self):
        self.state.advanced_search_sort = "Trans @ λ"
        out = search_filters(self.df, self.matrix, self.state)
        self.assertEqual(list(out["Filter Name"]), ["Zeta", "Alpha", "Mike"])

    def test_sort_trans_off_grid_is_unsorted(self):
        self.state.advanced_search_sort = "Trans @ λ"
        self.state.advanced_search_wavelength_criteria = [
            {"wavelength": 299, "trans_min": 0, "trans_max": 100},  # off-grid
        ]
        out = search_filters(self.df, self.matrix, self.state)
        # No sort applies: original df order preserved
        self.assertEqual(list(out["Filter Name"]), ["Zeta", "Alpha", "Mike"])


class TestSearchReflectors(unittest.TestCase):
    def _make_reflectors(self):
        return pd.DataFrame({
            "Name": ["R1", "R2", "R3"],
            "Organization": ["OrgA", "OrgA", "OrgB"],
            "Package Title": ["P1", "P2", "P1"],
            "Target Type": ["T1", "T2", "T2"],
        })

    def test_no_filters_returns_copy_with_all_rows(self):
        df = self._make_reflectors()
        out = search_reflectors(df, [], [], [])
        self.assertEqual(len(out), 3)
        self.assertIsNot(out, df)

    def test_each_facet_filters(self):
        df = self._make_reflectors()
        self.assertEqual(list(search_reflectors(df, ["OrgA"], [], [])["Name"]), ["R1", "R2"])
        self.assertEqual(list(search_reflectors(df, [], ["P1"], [])["Name"]), ["R1", "R3"])
        self.assertEqual(list(search_reflectors(df, [], [], ["T2"])["Name"]), ["R2", "R3"])


if __name__ == "__main__":
    unittest.main()
