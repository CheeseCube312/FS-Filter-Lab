"""
Regression suite for the npz data cache (roadmap S4).

Locks the S4 rework of services/data.py: the pickle cache (two .pkl files per
key) is replaced by a single npz file per key whose JSON meta blob records the
format tag; validity is a silent source-mtime comparison against the cache
file's own kernel mtime.
Typed pack/unpack serializer pairs round-trip every cached payload
(FilterCollection, QE tuple, illuminants, ReflectorCollection, film profiles)
and no pickle is involved anywhere in the pipeline.

Run from the program/ directory:

    python -m unittest discover -s tests -v

The suite is stdlib-only (unittest + numpy/pandas); no pytest dependency.
"""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

# Make the program/ package root importable regardless of invocation cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from models.core import (
    BWFilmProfile, ColorNegativeFilmProfile, ColorSlideFilmProfile,
    Filter, FilterCollection, RGBSensorProfile, ReflectorCollection,
    ReflectorSpectrum,
)
from services import data as data_mod

N = 8  # small fake grid size


# ---------------------------------------------------------------------------
# Synthetic payloads
# ---------------------------------------------------------------------------

def _filter_collection():
    rows = [
        {"Filter Number": "ND6", "Filter Name": "ND 0.6",
         "Manufacturer": "Foo", "Hex Color": "#112233", "is_lee": False},
        {"Filter Number": "X2", "Filter Name": "UV Cut",
         "Manufacturer": "Bar", "Hex Color": "#445566", "is_lee": True},
    ]
    matrix = np.linspace(0.0, 1.0, 2 * N).reshape(2, N)
    masks = np.array([[True] * N, [False] * N])
    filters = [
        Filter(name=r["Filter Name"], number=r["Filter Number"],
               manufacturer=r["Manufacturer"], hex_color=r["Hex Color"],
               transmission=matrix[i].copy(), extrapolated_mask=masks[i].copy())
        for i, r in enumerate(rows)
    ]
    return FilterCollection(filters=filters, df=pd.DataFrame(rows),
                            filter_matrix=matrix, extrapolated_masks=masks)


def _qe_payload():
    keys = ["Alpha Cam X", "Beta Works One"]
    qe = {k: {ch: np.arange(N, dtype=float) + i for ch in ("R", "G", "B")}
          for i, k in enumerate(keys)}
    return keys, qe, keys[0]


def _illum_payload():
    illum = {"D65": np.linspace(0.0, 2.0, N), "LED": np.ones(N)}
    return illum, {"D65": "Standard daylight"}


def _reflector_collection():
    matrix = np.linspace(1.0, 0.0, N)[None, :]
    spec = ReflectorSpectrum(name="Grey 18", values=matrix[0].copy(),
                             metadata={"Organization": "X", "source_file": "a.tsv"})
    df = pd.DataFrame([{"Name": "Grey 18", "Organization": "X"}])
    return ReflectorCollection(reflectors=[spec], df=df, reflector_matrix=matrix)


def _film_profiles():
    neg = ColorNegativeFilmProfile(
        manufacturer="Kodak", name="Gold 200", iso="200", process="C-41",
        channel_data={"R": np.linspace(0.0, 0.9, N),
                      "G": np.linspace(0.1, 1.0, N),
                      "B": np.linspace(0.2, 0.8, N)})
    bw = BWFilmProfile(manufacturer="ACME", name="Pan 100", notes="ortho",
                       sensitivity=np.linspace(0.0, 1.0, N))
    slide = ColorSlideFilmProfile(
        manufacturer="Fuji", name="Velvia 50", iso="50", process="E-6",
        channel_data={"R": np.ones(N), "G": np.full(N, 0.5), "B": np.zeros(N)})
    sensor = RGBSensorProfile(
        manufacturer="IO", name="Sensor S",
        channel_data={"R": np.arange(N, dtype=float), "G": np.ones(N)})
    return {"Kodak Gold 200": neg, "ACME Pan 100": bw,
            "Fuji Velvia 50": slide, "IO Sensor S": sensor}


# ---------------------------------------------------------------------------
# Minimal payload used to drive _cached_loader end-to-end
# ---------------------------------------------------------------------------

def _pack_marker(d):
    return {"arr": np.array([d["marker"]], dtype=float)}, {"n": d["marker"]}


def _unpack_marker(arrays, meta):
    return {"marker": int(arrays["arr"][0]), "n": meta["n"]}


# ---------------------------------------------------------------------------
# Cache file level: format + mtime validation, silent corruption handling
# ---------------------------------------------------------------------------

class CacheFileTest(unittest.TestCase):
    """_read_cache / _write_cache: validation rules and silent failures."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data_dir = Path(tmp.name) / "src"
        self.data_dir.mkdir()
        self.cache_file = Path(tmp.name) / "key.npz"

    def test_missing_file_is_none(self):
        self.assertIsNone(data_mod._read_cache(self.cache_file, self.data_dir))

    def test_roundtrip_arrays_and_meta(self):
        data_mod._write_cache(self.cache_file, {"a": np.arange(4)}, {"k": "v"})
        hit = data_mod._read_cache(self.cache_file, self.data_dir)
        self.assertIsNotNone(hit)
        arrays, meta = hit
        np.testing.assert_array_equal(arrays["a"], np.arange(4))
        self.assertEqual(meta["k"], "v")
        self.assertEqual(meta["format"], data_mod.CACHE_FORMAT)

    def test_rejects_wrong_format(self):
        stale_meta = json.dumps({"format": "ancient", "created": time.time() + 9})
        np.savez(self.cache_file, __meta__=np.array(stale_meta))
        self.assertIsNone(data_mod._read_cache(self.cache_file, self.data_dir))

    def test_rejects_stale_cache(self):
        data_mod._write_cache(self.cache_file, {"a": np.arange(2)}, {})
        future = time.time() + 60
        tsv = self.data_dir / "new.tsv"
        tsv.write_text("w\tt\n", encoding="utf-8")
        os.utime(tsv, (future, future))
        self.assertIsNone(data_mod._read_cache(self.cache_file, self.data_dir))

    def test_rejects_corrupt_file(self):
        self.cache_file.write_bytes(b"not an npz at all")
        self.assertIsNone(data_mod._read_cache(self.cache_file, self.data_dir))


# ---------------------------------------------------------------------------
# Loader level: hits, silent invalidation, silent recovery
# ---------------------------------------------------------------------------

class CachedLoaderTest(unittest.TestCase):
    """_cached_loader end-to-end against a temp cache dir."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache_dir = Path(tmp.name) / "cache"
        self.src_dir = Path(tmp.name) / "src"
        self.src_dir.mkdir()
        self.tsv = self.src_dir / "x.tsv"
        self.tsv.write_text("w\tt\n", encoding="utf-8")
        patcher = mock.patch("services.data.CACHE_DIR", str(self.cache_dir))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.calls = 0

    def _build(self):
        self.calls += 1
        return {"marker": self.calls}

    def test_miss_builds_and_writes_npz(self):
        out = data_mod._cached_loader("k", self.src_dir, self._build,
                                      _pack_marker, _unpack_marker)
        # A miss returns the freshly built payload as-is
        self.assertEqual(out, {"marker": 1})
        self.assertTrue((self.cache_dir / "k.npz").exists())

    def test_hit_skips_load_function(self):
        for _ in range(2):
            out = data_mod._cached_loader("k", self.src_dir, self._build,
                                          _pack_marker, _unpack_marker)
        self.assertEqual(self.calls, 1)  # built once, then served from cache
        self.assertEqual(out["marker"], 1)

    def test_newer_source_file_invalidates(self):
        data_mod._cached_loader("k", self.src_dir, self._build,
                                _pack_marker, _unpack_marker)
        future = time.time() + 60
        os.utime(self.tsv, (future, future))
        out = data_mod._cached_loader("k", self.src_dir, self._build,
                                      _pack_marker, _unpack_marker)
        self.assertEqual(self.calls, 2)  # silently rebuilt
        self.assertEqual(out["marker"], 2)

    def test_format_bump_invalidates(self):
        data_mod._cached_loader("k", self.src_dir, self._build,
                                _pack_marker, _unpack_marker)
        with mock.patch.object(data_mod, "CACHE_FORMAT", "other-format"):
            out = data_mod._cached_loader("k", self.src_dir, self._build,
                                          _pack_marker, _unpack_marker)
        self.assertEqual(self.calls, 2)
        self.assertEqual(out["marker"], 2)

    def test_corrupt_cache_rebuilds_silently(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "k.npz").write_bytes(b"garbage")
        out = data_mod._cached_loader("k", self.src_dir, self._build,
                                      _pack_marker, _unpack_marker)
        self.assertEqual(self.calls, 1)  # no exception raised
        self.assertEqual(out["marker"], 1)

    def test_unpack_failure_rebuilds_silently(self):
        data_mod._cached_loader("k", self.src_dir, self._build,
                                _pack_marker, _unpack_marker)
        state = {"raised": False}

        def bad_unpack(arrays, meta):
            if not state["raised"]:
                state["raised"] = True
                raise KeyError("boom")
            return _unpack_marker(arrays, meta)

        out = data_mod._cached_loader("k", self.src_dir, self._build,
                                      _pack_marker, bad_unpack)
        self.assertEqual(self.calls, 2)  # fell through to a fresh build
        self.assertEqual(out["marker"], 2)


# ---------------------------------------------------------------------------
# Serializer level: every cached payload survives pack -> npz -> unpack
# ---------------------------------------------------------------------------

class SerializerRoundTripTest(unittest.TestCase):
    """pack -> _write_cache -> _read_cache -> unpack reproduces each payload."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data_dir = Path(tmp.name) / "src"
        self.data_dir.mkdir()
        self.cache_file = Path(tmp.name) / "key.npz"

    def _roundtrip(self, obj, pack, unpack):
        arrays, meta = pack(obj)
        data_mod._write_cache(self.cache_file, arrays, meta)
        hit = data_mod._read_cache(self.cache_file, self.data_dir)
        self.assertIsNotNone(hit, "freshly written cache must be valid")
        return unpack(*hit)

    def test_filter_collection(self):
        fc = _filter_collection()
        out = self._roundtrip(fc, data_mod._pack_filter_collection,
                              data_mod._unpack_filter_collection)
        self.assertIsInstance(out, FilterCollection)
        self.assertTrue(out.df.equals(fc.df))
        self.assertEqual(list(out.df.dtypes.astype(str)),
                         list(fc.df.dtypes.astype(str)))  # incl. bool is_lee
        np.testing.assert_array_equal(out.filter_matrix, fc.filter_matrix)
        self.assertEqual(out.filter_matrix.dtype, fc.filter_matrix.dtype)
        np.testing.assert_array_equal(out.extrapolated_masks, fc.extrapolated_masks)
        self.assertEqual(out.extrapolated_masks.dtype, np.bool_)
        self.assertEqual([str(f) for f in out.filters], [str(f) for f in fc.filters])
        for got, want in zip(out.filters, fc.filters):
            np.testing.assert_array_equal(got.transmission, want.transmission)
            np.testing.assert_array_equal(got.extrapolated_mask, want.extrapolated_mask)
            self.assertEqual(got.hex_color, want.hex_color)

    def test_empty_filter_collection(self):
        empty = FilterCollection(
            filters=[], df=pd.DataFrame(),
            filter_matrix=np.empty((0, N)), extrapolated_masks=np.empty((0, N), dtype=bool))
        out = self._roundtrip(empty, data_mod._pack_filter_collection,
                              data_mod._unpack_filter_collection)
        self.assertEqual(out.filters, [])
        self.assertEqual(out.df.shape, (0, 0))
        self.assertEqual(out.filter_matrix.shape, (0, N))

    def test_qe_payload(self):
        keys, qe, default_key = _qe_payload()
        out = self._roundtrip((keys, qe, default_key), data_mod._pack_qe_payload,
                              data_mod._unpack_qe_payload)
        self.assertEqual(out[0], keys)
        self.assertEqual(out[2], default_key)
        for key in keys:
            for ch, arr in qe[key].items():
                np.testing.assert_array_equal(out[1][key][ch], arr)

    def test_illuminants(self):
        illum, descriptions = _illum_payload()
        out = self._roundtrip((illum, descriptions), data_mod._pack_illuminants,
                              data_mod._unpack_illuminants)
        self.assertEqual(list(out[0]), list(illum))  # order preserved
        for name, arr in illum.items():
            np.testing.assert_array_equal(out[0][name], arr)
        self.assertEqual(out[1], descriptions)

    def test_reflector_collection(self):
        rc = _reflector_collection()
        out = self._roundtrip(rc, data_mod._pack_reflector_collection,
                              data_mod._unpack_reflector_collection)
        self.assertIsInstance(out, ReflectorCollection)
        self.assertTrue(out.df.equals(rc.df))
        np.testing.assert_array_equal(out.reflector_matrix, rc.reflector_matrix)
        self.assertEqual([r.name for r in out.reflectors],
                         [r.name for r in rc.reflectors])
        self.assertEqual(out.reflectors[0].metadata, rc.reflectors[0].metadata)
        np.testing.assert_array_equal(out.reflectors[0].values,
                                      rc.reflectors[0].values)

    def test_film_profiles(self):
        profiles = _film_profiles()
        out = self._roundtrip(profiles, data_mod._pack_film_profiles,
                              data_mod._unpack_film_profiles)
        self.assertEqual(list(out), list(profiles))  # key order preserved
        for key, want in profiles.items():
            got = out[key]
            self.assertIs(type(got), type(want))  # exact subclass restored
            self.assertEqual(got.manufacturer, want.manufacturer)
            self.assertEqual(got.name, want.name)
            self.assertEqual(got.iso, want.iso)
            self.assertEqual(got.process, want.process)
            self.assertEqual(got.notes, want.notes)
            self.assertEqual(got.sensor_type, want.sensor_type)
            for ch, arr in want.get_channels().items():
                np.testing.assert_array_equal(got.get_channels()[ch], arr)


# ---------------------------------------------------------------------------
# The pickle path stays deleted
# ---------------------------------------------------------------------------

class PickleStaysDeletedTest(unittest.TestCase):
    def test_no_pickle_in_cache_module(self):
        source = Path(data_mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import pickle", source)
        self.assertNotIn(".pkl", source)
        self.assertNotIn("pickle.load", source)
        self.assertNotIn("pickle.dump", source)


if __name__ == "__main__":
    unittest.main()


