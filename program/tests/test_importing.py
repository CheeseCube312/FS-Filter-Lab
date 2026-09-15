"""services.importing — the six single-spectrum importers.

Round-trip focus: synthetic CSV in, loader-contract TSV out (column names,
0-100 file scale, filenames, comment headers), with the exact hand-computed
interpolation values the shared pipelines must produce.  Every write lands
in a throwaway tree (helpers.import_sandbox).
"""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from services import importing as imp
from tests.helpers import (
    GRID,
    TestCase,
    csv_upload,
    import_sandbox,
)


def read_tsv(path):
    """Loader's-eye view of a written TSV (metadata comments stripped)."""
    return pd.read_csv(path, sep="\t", comment="#")


def filter_meta(**overrides):
    meta = {"manufacturer": "Acme", "filter_name": "ND Filter",
            "filter_number": "ND6", "hex_color": "#204060"}
    meta.update(overrides)
    return meta


class FilterImportRoundTrip(TestCase):
    def test_happy_path_writes_loader_contract_tsv(self):
        with import_sandbox() as (root, folders, output):
            ok, msg = imp.import_filter_from_csv(
                csv_upload("400;50\n500;25\n600;10"), filter_meta(),
                False, False)
            self.assertTrue(ok, msg)
            path = (Path(output["filter_import"]) / "Acme"
                    / "Acme_ND6_ND_Filter.tsv")
            self.assertTrue(path.exists(), msg)
            df = read_tsv(path)
            self.assertEqual(list(df.columns),
                             ["Wavelength", "Transmittance", "hex_color",
                              "Manufacturer", "Name", "Filter Number"])
            self.assertEqual(df["Wavelength"].min(), 400)  # ceil(400/5)*5
            self.assertEqual(df["Wavelength"].max(), 600)  # floor(600/5)*5
            at_450 = float(df.loc[df["Wavelength"] == 450, "Transmittance"].iloc[0])
            self.assertClose(at_450, 37.5)  # 50 -> 25 across 400..600

    def test_round_trip_through_the_loader_scale_law(self):
        # Import writes 0-100; the filter loader divides by 100.  A value of
        # 50 in the CSV must become exactly 0.5 in the loaded matrix.
        from unittest import mock
        from services import data as data_mod
        with import_sandbox() as (root, folders, output):
            ok, msg = imp.import_filter_from_csv(
                csv_upload("400;50\n500;25\n600;10"), filter_meta(),
                False, False)
            self.assertTrue(ok, msg)
            with mock.patch.multiple(
                "services.data", DATA_FOLDERS=folders,
                CACHE_DIR=str(root / "cache"),
            ):
                collection = data_mod.load_filter_collection()
            self.assertEqual(len(collection.filters), 1)
            row = collection.filters[0].transmission
            self.assertClose(float(row[GRID == 400][0]), 0.5)
            self.assertClose(float(row[GRID == 500][0]), 0.25)

class FilterImportValidation(TestCase):
    def test_rejects_transmission_over_100_with_exact_message(self):
        with import_sandbox():
            ok, msg = imp.import_filter_from_csv(
                csv_upload("400;50\n500;150"), filter_meta(), False, False)
            self.assertFalse(ok)
            self.assertEqual(
                msg, "Transmission values seem too high (max: 150.00). Expected 0-100%.")

    def test_rejects_missing_metadata_keys_by_name(self):
        with import_sandbox():
            ok, msg = imp.import_filter_from_csv(
                csv_upload("400;50"), filter_meta(hex_color=""), False, False)
            self.assertFalse(ok)
            self.assertEqual(msg, "Missing required metadata: hex_color")

    def test_rejects_too_narrow_wavelength_range(self):
        with import_sandbox():
            ok, msg = imp.import_filter_from_csv(
                csv_upload("400;50\n420;25"), filter_meta(), False, False)
            self.assertFalse(ok)
            self.assertEqual(
                msg, "Wavelength range too narrow (400.0-420.0 nm). Need at least 50nm range.")

    def test_negative_values_are_clamped_to_zero(self):
        with import_sandbox() as (root, folders, output):
            ok, msg = imp.import_filter_from_csv(
                csv_upload("400;-10\n500;25\n600;10"), filter_meta(),
                False, False)
            self.assertTrue(ok, msg)
            path = Path(output["filter_import"]) / "Acme" / "Acme_ND6_ND_Filter.tsv"
            df = read_tsv(path)
            self.assertEqual(float(df["Transmittance"].min()), 0.0)


class FilterImportParsing(TestCase):
    def test_semicolon_separator_with_decimal_commas(self):
        # European CSV: ';' separator, ',' decimals -> 400.5 nm / 12.5%.
        with import_sandbox() as (root, folders, output):
            ok, msg = imp.import_filter_from_csv(
                csv_upload("400,5;12,5\n500,5;62,5"), filter_meta(),
                False, False)
            self.assertTrue(ok, msg)
            path = Path(output["filter_import"]) / "Acme" / "Acme_ND6_ND_Filter.tsv"
            df = read_tsv(path)
            self.assertEqual(int(df["Wavelength"].min()), 405)  # ceil(400.5/5)*5
            self.assertEqual(int(df["Wavelength"].max()), 500)  # floor(500.5/5)*5
            at_405 = float(df.loc[df["Wavelength"] == 405, "Transmittance"].iloc[0])
            self.assertClose(at_405, 14.75)  # 12.5 + (405-400.5)/100 * 50

    def test_interior_blank_in_one_column_keeps_pairs_aligned(self):
        # Regression: per-column dropna() desynced wavelength/value rows
        # whenever one column had an interior blank.  Row 2 has no wavelength;
        # its 99 must be dropped WITH its (missing) partner.
        with import_sandbox() as (root, folders, output):
            ok, msg = imp.import_filter_from_csv(
                csv_upload("400;50\n;99\n500;25\n600;10"), filter_meta(),
                False, False)
            self.assertTrue(ok, msg)
            path = Path(output["filter_import"]) / "Acme" / "Acme_ND6_ND_Filter.tsv"
            df = read_tsv(path)
            at_550 = float(df.loc[df["Wavelength"] == 550, "Transmittance"].iloc[0])
            self.assertClose(at_550, 17.5)  # midpoint of (500,25) and (600,10)
            self.assertFalse((df["Transmittance"] == 99.0).any())

    def test_extrapolation_extends_range_and_clamps_edges(self):
        with import_sandbox() as (root, folders, output):
            ok, msg = imp.import_filter_from_csv(
                csv_upload("400;50\n500;25\n600;10"), filter_meta(),
                True, True)
            self.assertTrue(ok, msg)
            path = (Path(output["filter_import"]) / "Acme"
                    / "Acme_ND6_ND_Filter_extrapolated_300_1100.tsv")
            self.assertTrue(path.exists(), msg)
            df = read_tsv(path)
            self.assertEqual(int(df["Wavelength"].min()), 300)
            self.assertEqual(int(df["Wavelength"].max()), 1100)
            at_300 = float(df.loc[df["Wavelength"] == 300, "Transmittance"].iloc[0])
            self.assertClose(at_300, 50.0)  # edge value extends, not NaN
            at_1100 = float(df.loc[df["Wavelength"] == 1100, "Transmittance"].iloc[0])
            self.assertClose(at_1100, 10.0)

    def test_no_extrapolation_blanks_outside_the_data(self):
        with import_sandbox() as (root, folders, output):
            imp.import_filter_from_csv(
                csv_upload("400;50\n500;25\n600;10"), filter_meta(),
                False, False)
            path = Path(output["filter_import"]) / "Acme" / "Acme_ND6_ND_Filter.tsv"
            df = read_tsv(path)
            below = df.loc[df["Wavelength"] < 400, "Transmittance"]
            self.assertTrue(below.isna().all())  # not extrapolated -> blank


class IlluminantImport(TestCase):
    def test_peak_is_normalized_to_100(self):
        with import_sandbox() as (root, folders, output):
            ok, msg = imp.import_illuminant_from_csv(
                csv_upload("400;10\n500;5"), "Test Lamp")
            self.assertTrue(ok, msg)
            path = Path(folders["illuminants"]) / "illuminant_Test_Lamp.tsv"
            self.assertTrue(path.exists(), msg)
            df = read_tsv(path)
            self.assertEqual(list(df.columns),
                             ["Wavelength (nm)", "Relative Power", "Description"])
            self.assertEqual(len(df), 801)  # INTERP_GRID rows
            self.assertClose(float(df["Relative Power"].max()), 100.0)
            at_300 = float(df.loc[df["Wavelength (nm)"] == 300,
                                  "Relative Power"].iloc[0])
            self.assertClose(at_300, 0.0)  # zero-filled beyond the data

    def test_rejects_empty_description(self):
        with import_sandbox():
            ok, msg = imp.import_illuminant_from_csv(
                csv_upload("400;10\n500;5"), "  ")
            self.assertFalse(ok)
            self.assertEqual(msg, "Description cannot be empty.")


class QEImport(TestCase):
    def test_six_column_xy_pairs_clamp_beyond_the_data(self):
        # Camera QE clamps to edge values (zero_edges=False); film/illuminant
        # zero-fill — this edge policy is the QE contract.
        csv = ("350;60;350;30;350;10\n"
               "400;80;400;50;400;20\n"
               "600;40;600;25;600;8\n")
        with import_sandbox() as (root, folders, output):
            ok, msg = imp.import_qe_from_csv(csv_upload(csv), "Canon", "EOS R")
            self.assertTrue(ok, msg)
            path = Path(folders["qe"]) / "Canon_EOS_R_QE.tsv"
            self.assertTrue(path.exists(), msg)
            df = read_tsv(path)
            self.assertEqual(list(df.columns),
                             ["Wavelength", "R", "G", "B", "Manufacturer", "Name"])
            at_300 = float(df.loc[df["Wavelength"] == 300, "R"].iloc[0])
            self.assertClose(at_300, 60.0)  # clamped to first data value
            at_400 = float(df.loc[df["Wavelength"] == 400, "R"].iloc[0])
            self.assertClose(at_400, 80.0)  # file keeps the 0-100 scale

    def test_rejects_wrong_column_count(self):
        with import_sandbox():
            ok, msg = imp.import_qe_from_csv(
                csv_upload("400;50\n500;25"), "Canon", "EOS R")
            self.assertFalse(ok)
            self.assertIn("Expected 6 columns", msg)


class FilmImport(TestCase):
    def test_bw_film_writes_comment_headers_and_v_column(self):
        with import_sandbox() as (root, folders, output):
            ok, msg = imp.import_film_from_csv(
                csv_upload("400;90\n500;45"), "Acme", "Pan 400", "bw",
                iso="400", process="D-76")
            self.assertTrue(ok, msg)
            path = Path(folders["film"]) / "Acme_Pan_400_film.tsv"
            text = path.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("# Manufacturer\tAcme"))
            self.assertIn("# Name\tPan 400", text)
            self.assertIn("# Type\tbw", text)
            self.assertIn("# ISO\t400", text)
            self.assertIn("# Process\tD-76", text)
            df = read_tsv(path)
            self.assertIn("V", df.columns)
            self.assertClose(float(df["V"].max()), 90.0)  # linear scale kept

    def test_log_sensitivity_is_unlogged_and_peak_normalized(self):
        # log10 values 2.0 and 1.0 -> 100 and 10 linear -> peak normalized: 100.
        with import_sandbox() as (root, folders, output):
            ok, msg = imp.import_film_from_csv(
                csv_upload("400;2.0\n500;1.0"), "Acme", "Log Pan", "bw",
                log_sensitivity=True)
            self.assertTrue(ok, msg)
            path = Path(folders["film"]) / "Acme_Log_Pan_film.tsv"
            df = read_tsv(path)
            self.assertClose(float(df["V"].max()), 100.0)

    def test_invalid_film_type_message_lists_valid_types(self):
        with import_sandbox():
            ok, msg = imp.import_film_from_csv(
                csv_upload("400;50\n500;25"), "Acme", "X", "nope")
            self.assertFalse(ok)
            self.assertEqual(
                msg, "Invalid film type 'nope'. Must be one of: bw, color_negative, color_slide.")

    def test_color_film_requires_six_columns(self):
        with import_sandbox():
            ok, msg = imp.import_film_from_csv(
                csv_upload("400;50\n500;25"), "Acme", "X", "color_negative")
            self.assertFalse(ok)
            self.assertEqual(
                msg, "Colour film CSV requires 6 columns (R_wl, R_val, G_wl, G_val, B_wl, B_val). Found 2.")


def reflector_meta(**overrides):
    meta = {"name": "Leaf", "description": "test surface",
            "category": "plant", "data_type": "Reflectance"}
    meta.update(overrides)
    return meta


class ReflectorImport(TestCase):
    def test_plant_import_writes_loader_contract_tsv(self):
        # The reflector loader (services.data) requires the value column to
        # be named 'Reflectance' no matter which data_type was imported;
        # metadata lives in the first row only, and the category picks the
        # plant/ subfolder.
        with import_sandbox() as (root, folders, output):
            ok, msg = imp.import_reflectance_absorption_from_csv(
                csv_upload("400;10\n600;40"), reflector_meta(), False, False)
            self.assertTrue(ok, msg)
            path = Path(folders["reflectors"]) / "plant" / "Leaf.tsv"
            self.assertTrue(path.exists(), msg)
            df = read_tsv(path)
            self.assertEqual(list(df.columns),
                             ["Wavelength", "Reflectance", "Name", "Description"])
            self.assertEqual(df["Name"].iloc[0], "Leaf")
            self.assertTrue(pd.isna(df["Name"].iloc[-1]))  # first row only
            at_500 = float(df.loc[df["Wavelength"] == 500, "Reflectance"].iloc[0])
            self.assertClose(at_500, 25.0)  # 10 -> 40 across 400..600

    def test_absorption_import_lands_in_other_with_extrapolation_suffix(self):
        with import_sandbox() as (root, folders, output):
            ok, msg = imp.import_reflectance_absorption_from_csv(
                csv_upload("400;10\n600;40"),
                reflector_meta(category="other", data_type="Absorption"),
                True, True)
            self.assertTrue(ok, msg)
            path = (Path(folders["reflectors"]) / "other"
                    / "Leaf_extrapolated_300_1100.tsv")
            self.assertTrue(path.exists(), msg)
            df = read_tsv(path)
            self.assertEqual(int(df["Wavelength"].min()), 300)
            self.assertEqual(int(df["Wavelength"].max()), 1100)


if __name__ == "__main__":
    unittest.main()
