"""services.data — loaders, the 0-100 -> 0-1 scale law, and the npz cache.

Locks the loader contracts: file scale, comment-header parsing, film dual-scale
heuristic, profile-class mapping, silent cache rebuilds (corrupt / stale /
foreign-format), and pack/unpack round-trip integrity.  All I/O lands in a
throwaway tree (helpers.data_sandbox).
"""
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from models.constants import DEFAULT_HEX_COLOR, MIN_DATA_POINTS
from models.core import (
    BWFilmProfile,
    ColorNegativeFilmProfile,
    ColorSlideFilmProfile,
)
from services import data as data_mod
from tests.helpers import (
    GRID,
    TestCase,
    data_sandbox,
    write_filter_tsv,
    write_film_tsv,
    write_illuminant_tsv,
    write_qe_tsv,
    write_reflector_tsv,
)


class FilterLoader(TestCase):
    def test_percent_file_scale_becomes_fractional_internal_scale(self):
        with data_sandbox() as (root, folders):
            write_filter_tsv(Path(folders["filters"]) / "a.tsv",
                             [300, 500, 700, 900], [50, 25, 10, 5])
            c = data_mod.load_filter_collection()
            self.assertEqual(len(c.filters), 1)
            row = c.filter_matrix[0]
            self.assertClose(float(row[GRID == 300][0]), 0.5)   # 50/100
            self.assertClose(float(row[GRID == 500][0]), 0.25)
            self.assertClose(float(row[GRID == 800][0]), 0.075)  # interp
            self.assertClose(float(row[GRID == 1100][0]), 0.0)   # beyond data -> 0

    def test_metadata_and_hex_fallback_rules(self):
        with data_sandbox() as (root, folders):
            write_filter_tsv(Path(folders["filters"]) / "a.tsv",
                             [400, 600], [50, 25], name="ND",
                             number="ND6", manufacturer="Foo", hex_color="red")
            c = data_mod.load_filter_collection()
            self.assertEqual(c.filters[0].hex_color, DEFAULT_HEX_COLOR)
            self.assertEqual(c.df.iloc[0]["Filter Name"], "ND")
            self.assertEqual(c.df.iloc[0]["Hex Color"], DEFAULT_HEX_COLOR)

    def test_lee_filename_flags_extrapolation_above_700(self):
        with data_sandbox() as (root, folders):
            write_filter_tsv(Path(folders["filters"]) / "LeeFilters_X.tsv",
                             [400, 600], [50, 25])
            c = data_mod.load_filter_collection()
            self.assertTrue(bool(c.df.iloc[0]["is_lee"]))
            i = int(np.where(GRID == 800)[0][0])
            self.assertTrue(c.extrapolated_masks[0][i])
            self.assertFalse(c.extrapolated_masks[0][10])  # 310 nm: not flagged

    def test_invalid_file_is_skipped_and_app_init_returns_none(self):
        with data_sandbox() as (root, folders):
            Path(folders["filters"], "broken.tsv").write_text(
                "Wavelength\tWrong\n400\t50\n", encoding="utf-8")
            self.assertEqual(len(data_mod.load_filter_collection().filters), 0)
            self.assertIsNone(data_mod.initialize_application_data())


class SensorProfileLoader(TestCase):
    def test_qe_percent_scale_and_display_name_key(self):
        with data_sandbox() as (root, folders):
            write_qe_tsv(Path(folders["qe"]) / "cam.tsv",
                         [400, 500, 600], [50, 80, 40], [30, 60, 20], [10, 30, 5])
            profiles, default_key = data_mod.load_sensor_profiles()
            self.assertIn("Brand M1", profiles)
            self.assertIsNone(default_key)
            r = profiles["Brand M1"].channel_data["R"]
            self.assertClose(float(r[GRID == 400][0]), 0.5)  # 50/100
            self.assertClose(float(r[GRID == 1100][0]), 0.0)  # beyond data -> 0

    def test_default_qe_filename_sets_default_key(self):
        with data_sandbox() as (root, folders):
            write_qe_tsv(Path(folders["qe"]) / "Default_QE.tsv",
                         [400, 600], [50, 40], [30, 20], [10, 5])
            _, default_key = data_mod.load_sensor_profiles()
            self.assertEqual(default_key, "Brand M1")

    def test_color_negative_film_maps_to_inverting_profile(self):
        with data_sandbox() as (root, folders):
            write_film_tsv(
                Path(folders["film"]) / "gold.tsv",
                {"Manufacturer": "Kodak", "Name": "Gold", "Type": "color_negative"},
                {"Wavelength": [400, 500, 600], "R": [80, 60, 40],
                 "G": [70, 50, 30], "B": [60, 40, 20]})
            profiles, _ = data_mod.load_sensor_profiles()
            profile = profiles["Kodak Gold"]
            self.assertIsInstance(profile, ColorNegativeFilmProfile)
            self.assertTrue(profile.is_film)
            self.assertTrue(profile.invert_for_display)  # negatives display 1-x
            self.assertClose(float(profile.channel_data["R"][GRID == 400][0]), 0.8)

    def test_film_dual_scale_heuristic_spares_fractions(self):
        # Max 0.9 <= 1.5: already fractional, must NOT be divided by 100.
        with data_sandbox() as (root, folders):
            write_film_tsv(
                Path(folders["film"]) / "frac.tsv",
                {"Manufacturer": "Acme", "Name": "Frac", "Type": "bw"},
                {"Wavelength": [400, 600], "V": [0.9, 0.8]})
            profiles, _ = data_mod.load_sensor_profiles()
            sensitivity = profiles["Acme Frac"].sensitivity
            self.assertClose(float(sensitivity[GRID == 400][0]), 0.9)

    def test_bw_film_maps_one_curve_to_all_three_channels(self):
        with data_sandbox() as (root, folders):
            write_film_tsv(
                Path(folders["film"]) / "pan.tsv",
                {"Manufacturer": "Acme", "Name": "Pan", "Type": "bw"},
                {"Wavelength": [400, 600], "V": [90, 80]})
            profile = data_mod.load_sensor_profiles()[0]["Acme Pan"]
            channels = profile.get_channels()
            self.assertIsInstance(profile, BWFilmProfile)
            self.assertIs(channels["R"], channels["G"])

    def test_slide_film_does_not_invert(self):
        with data_sandbox() as (root, folders):
            write_film_tsv(
                Path(folders["film"]) / "velvia.tsv",
                {"Manufacturer": "Fuji", "Name": "V50", "Type": "color_slide"},
                {"Wavelength": [400, 600], "R": [80, 60], "G": [70, 50],
                 "B": [60, 40]})
            profile = data_mod.load_sensor_profiles()[0]["Fuji V50"]
            self.assertIsInstance(profile, ColorSlideFilmProfile)
            self.assertFalse(profile.invert_for_display)

    def test_film_missing_required_metadata_is_skipped(self):
        with data_sandbox() as (root, folders):
            write_film_tsv(
                Path(folders["film"]) / "anon.tsv",
                {"Name": "NoMaker", "Type": "bw"},
                {"Wavelength": [400, 600], "V": [90, 80]})
            profiles, _ = data_mod.load_sensor_profiles()
            self.assertEqual(profiles, {})


def write_filter_source(path, value=50.0):
    write_filter_tsv(path, [300, 500, 700, 900], [value, 25, 10, 5])


class CacheLifecycle(TestCase):
    """The npz cache: silent rebuild on missing, corrupt, stale, foreign."""

    def cache_file(self, root):
        return root / "cache" / "filter_data.npz"

    def test_first_load_writes_one_npz_per_dataset(self):
        with data_sandbox() as (root, folders):
            write_filter_source(Path(folders["filters"]) / "a.tsv")
            data_mod.load_filter_collection()
            self.assertTrue(self.cache_file(root).exists())

    def test_valid_cache_serves_a_source_whose_mtime_is_older(self):
        with data_sandbox() as (root, folders):
            source = Path(folders["filters"]) / "a.tsv"
            write_filter_source(source, value=50.0)
            first = data_mod.load_filter_collection()
            # Rewrite the source with different values but an OLDER stamp:
            # the cache on disk is newer, so it must keep winning.
            write_filter_source(source, value=90.0)
            os.utime(source, (1_000_000, 1_000_000))
            second = data_mod.load_filter_collection()
            self.assertClose(float(second.filter_matrix[0][GRID == 300][0]), 0.5)
            self.assertNanEqual(second.filter_matrix, first.filter_matrix)

    def test_newer_source_mtime_invalidates_the_cache(self):
        with data_sandbox() as (root, folders):
            source = Path(folders["filters"]) / "a.tsv"
            write_filter_source(source, value=50.0)
            data_mod.load_filter_collection()
            write_filter_source(source, value=90.0)
            future = data_mod._newest_source_mtime(folders["filters"]) + 60
            os.utime(source, (future, future))
            rebuilt = data_mod.load_filter_collection()
            self.assertClose(float(rebuilt.filter_matrix[0][GRID == 300][0]), 0.9)

    def test_corrupt_cache_file_rebuilds_silently(self):
        with data_sandbox() as (root, folders):
            source = Path(folders["filters"]) / "a.tsv"
            write_filter_source(source)
            data_mod.load_filter_collection()
            future = data_mod._newest_source_mtime(folders["filters"]) + 60
            self.cache_file(root).write_bytes(b"not an npz at all")
            os.utime(self.cache_file(root), (future, future))
            c = data_mod.load_filter_collection()
            self.assertClose(float(c.filter_matrix[0][GRID == 300][0]), 0.5)

    def test_foreign_format_tag_never_loads(self):
        # A cache written by a different CACHE_FORMAT must be ignored even
        # when it is newer than every source (bumping the tag invalidates all).
        with data_sandbox() as (root, folders):
            source = Path(folders["filters"]) / "a.tsv"
            write_filter_source(source)
            foreign = self.cache_file(root)
            foreign.parent.mkdir(parents=True, exist_ok=True)
            data_mod._write_cache(foreign, {"filter_matrix": np.zeros((1, len(GRID)))},
                                  {"format": "fs-filterlab-npz-0", "df": {},
                                   "filters": []})
            future = data_mod._newest_source_mtime(folders["filters"]) + 60
            os.utime(foreign, (future, future))
            c = data_mod.load_filter_collection()
            self.assertClose(float(c.filter_matrix[0][GRID == 300][0]), 0.5)  # source value

    def test_pack_unpack_round_trip_survives_deleted_sources(self):
        with data_sandbox() as (root, folders):
            source = Path(folders["filters"]) / "a.tsv"
            write_filter_source(source)
            first = data_mod.load_filter_collection()
            expected = first.filter_matrix.copy()
            name = first.filters[0].name
            number = first.filters[0].number
            source.unlink()
            second = data_mod.load_filter_collection()  # cache-only rebuild
            self.assertNanEqual(second.filter_matrix, expected)
            self.assertEqual((second.filters[0].name, second.filters[0].number),
                             (name, number))


class IlluminantLoader(TestCase):
    def test_first_two_columns_power_and_description(self):
        with data_sandbox() as (root, folders):
            write_illuminant_tsv(Path(folders["illuminants"]) / "D65.tsv",
                                 [400, 600], [10, 5], description="daylight")
            illum, meta = data_mod.load_illuminant_collection()
            self.assertIn("D65", illum)
            self.assertClose(float(illum["D65"][GRID == 400][0]), 10.0)
            self.assertClose(float(illum["D65"][GRID == 1100][0]), 0.0)
            self.assertEqual(meta, {"D65": "daylight"})

    def test_description_column_is_optional(self):
        with data_sandbox() as (root, folders):
            write_illuminant_tsv(Path(folders["illuminants"]) / "LED.tsv",
                                 [400, 600], [1, 2], description="")
            illum, meta = data_mod.load_illuminant_collection()
            self.assertIn("LED", illum)
            self.assertEqual(meta.get("LED", ""), "")


class ReflectorLoader(TestCase):

    def test_too_few_valid_points_skips_the_file(self):
        self.assertGreater(MIN_DATA_POINTS, 1)  # assumption this test relies on
        with data_sandbox() as (root, folders):
            sparse = MIN_DATA_POINTS - 1
            write_reflector_tsv(
                Path(folders["reflectors"]) / "sparse.tsv",
                [400 + 10 * i for i in range(sparse)], [50.0] * sparse)
            c = data_mod.load_reflector_collection()
            self.assertEqual(len(c.reflectors), 0)


if __name__ == "__main__":
    unittest.main()
