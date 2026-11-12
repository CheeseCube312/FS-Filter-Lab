"""Shared builders and redirects for the FS FilterLab test suite.

One place for: spectral curve builders, FilterCollection/AppData construction,
TSV/CSV writers, temp-dir redirects for every module-level path constant
(data, importing, state_manager, reporting), and NaN-aware assertions.

The sys.path bootstrap makes `services` / `models` importable regardless of
the invocation directory, so `python -m unittest discover -s tests` works
from program/ (the supported way to run the suite).
"""
import copy
import dataclasses
import io
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models.constants import INTERP_GRID  # noqa: E402
from models.core import (  # noqa: E402
    AppData,
    BWFilmProfile,
    ColorNegativeFilmProfile,
    ColorSlideFilmProfile,
    Filter,
    FilterCollection,
    RGBSensorProfile,
    ReflectorCollection,
    ReflectorSpectrum,
)

GRID = INTERP_GRID
N = len(GRID)


# ---------------------------------------------------------------------------
# Curve builders (spectra on the standard grid)
# ---------------------------------------------------------------------------

def flat(value):
    """Constant curve across the whole grid."""
    return np.full(N, float(value))


def band(lo, hi, inside=1.0, outside=0.0):
    """Step curve: `inside` between lo..hi nm, `outside` elsewhere."""
    return np.where((GRID >= lo) & (GRID <= hi), float(inside), float(outside))


def with_nan(curve, index):
    """Copy of `curve` with one wavelength set to NaN (the poisoned point)."""
    out = np.array(curve, dtype=float)
    out[index] = np.nan
    return out


def toy(n=4):
    """Distinct finite values on a tiny arbitrary grid (spectra oracles)."""
    return np.arange(float(n)) + 1.0


# ---------------------------------------------------------------------------
# Collection / profile / app-data builders (loader df schema)
# ---------------------------------------------------------------------------

def display_name(name, number, manufacturer):
    """The one display-name format every layer agrees on (Filter.__str__)."""
    return f"{name} ({number}, {manufacturer})"


def make_collection(specs):
    """FilterCollection from [{'name','number','manufacturer','hex','curve'}]."""
    filters = []
    for s in specs:
        filters.append(Filter(
            name=s["name"], number=s["number"],
            manufacturer=s.get("manufacturer", "Test"),
            hex_color=s.get("hex", "#112233"),
            transmission=np.asarray(s["curve"], dtype=float),
            extrapolated_mask=np.zeros(N, dtype=bool),
        ))
    df = pd.DataFrame([{
        "Filter Number": f.number, "Filter Name": f.name,
        "Manufacturer": f.manufacturer, "Hex Color": f.hex_color,
    } for f in filters])
    matrix = (np.vstack([f.transmission for f in filters])
              if filters else np.empty((0, N)))
    masks = np.zeros_like(matrix, dtype=bool)
    return FilterCollection(filters=filters, df=df,
                            filter_matrix=matrix, extrapolated_masks=masks)


def make_sensor(r=1.0, g=1.0, b=1.0, manufacturer="Cam", name="X"):
    """Digital sensor with flat per-channel QE curves (fractional scale)."""
    return RGBSensorProfile(
        manufacturer=manufacturer, name=name,
        channel_data={"R": flat(r), "G": flat(g), "B": flat(b)},
    )


def make_bw(sensitivity=1.0, manufacturer="ACME", name="Pan"):
    return BWFilmProfile(
        manufacturer=manufacturer, name=name, sensitivity=flat(sensitivity))


def make_app_data(collection):
    """Minimal real AppData around one filter collection."""
    return AppData(
        filter_collection=collection,
        default_key="",
        illuminants={},
        illuminant_metadata={},
        reflector_collection=ReflectorCollection(
            reflectors=[], df=pd.DataFrame(),
            reflector_matrix=np.empty((0, N))),
        sensor_profiles={},
    )


def make_reflector(values, source_file="leaf.tsv", **metadata):
    return ReflectorSpectrum(
        name=metadata.get("Name", source_file),
        values=np.asarray(values, dtype=float),
        metadata={"source_file": source_file, **metadata},
    )


# ---------------------------------------------------------------------------
# Temp-dir redirects for module-level path constants
# ---------------------------------------------------------------------------

_FOLDERS = {
    "filters": "filters_data", "qe": "QE_data", "film": "film_data",
    "illuminants": "illuminants", "reflectors": "reflectors",
}


@contextmanager
def data_sandbox():
    """Redirect services.data to a throwaway tree (DATA_FOLDERS + CACHE_DIR)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        folders = {k: str(root / name) for k, name in _FOLDERS.items()}
        for path in folders.values():
            os.makedirs(path, exist_ok=True)
        with mock.patch.multiple("services.data",
                                 DATA_FOLDERS=folders,
                                 CACHE_DIR=str(root / "cache")):
            yield root, folders


@contextmanager
def import_sandbox():
    """Redirect services.importing writes to a throwaway tree."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        folders = {k: str(root / name) for k, name in _FOLDERS.items()}
        output = {
            "reports": str(root / "output"),
            "filter_import": str(root / "filters_data"),
        }
        for path in list(folders.values()) + list(output.values()):
            os.makedirs(path, exist_ok=True)
        with mock.patch.multiple("services.importing",
                                 DATA_FOLDERS=folders,
                                 OUTPUT_FOLDERS=output):
            yield root, folders, output


@contextmanager
def state_sandbox():
    """Redirect the two JSON files AppState persists to.  Patch BEFORE
    constructing AppState (the files are read once at construction)."""
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.multiple(
            "services.state_manager",
            MY_FILTERS_FILE=str(Path(tmp) / "my_filters.json"),
            DEFAULT_REFLECTORS_FILE=str(Path(tmp) / "default_reflectors.json"),
        ):
            yield Path(tmp)


@contextmanager
def report_sandbox():
    """Redirect the report PNG output tree."""
    with tempfile.TemporaryDirectory() as tmp:
        reports = str(Path(tmp) / "output")
        os.makedirs(reports, exist_ok=True)
        with mock.patch.multiple("services.visualization",
                                 OUTPUT_FOLDERS={"reports": reports}):
            yield Path(reports)


# ---------------------------------------------------------------------------
# File writers (loader-contract schemas)
# ---------------------------------------------------------------------------

def _write_columns(path, columns, header=None):
    lines = [f"# {key}\t{value}" for key, value in (header or {}).items()]
    names = list(columns)
    lines.append("\t".join(names))
    length = len(next(iter(columns.values())))
    for i in range(length):
        lines.append("\t".join(str(columns[c][i]) for c in names))
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_filter_tsv(path, wavelengths, values, number="ND6", name="ND",
                     manufacturer="Foo", hex_color="#112233"):
    _write_columns(path, {
        "Wavelength": wavelengths, "Transmittance": values,
        "hex_color": [hex_color] * len(wavelengths),
        "Manufacturer": [manufacturer] * len(wavelengths),
        "Name": [name] * len(wavelengths),
        "Filter Number": [number] * len(wavelengths),
    })


def write_qe_tsv(path, wavelengths, r, g, b, manufacturer="Brand", model="M1"):
    _write_columns(path, {
        "Wavelength": wavelengths, "R": r, "G": g, "B": b,
        "Manufacturer": [manufacturer] * len(wavelengths),
        "Name": [model] * len(wavelengths),
    })


def write_film_tsv(path, header, columns):
    _write_columns(path, columns, header=header)


def write_illuminant_tsv(path, wavelengths, power, description="desc"):
    _write_columns(path, {
        "Wavelength (nm)": wavelengths, "Relative Power": power,
        "Description": [description] * len(wavelengths),
    })


def write_reflector_tsv(path, wavelengths, values, header=None):
    _write_columns(path, {"Wavelength": wavelengths, "Reflectance": values},
                   header=header)


def csv_upload(text):
    """Uploadable CSV as a seekable BytesIO (what _parse_csv expects)."""
    return io.BytesIO(text.encode("utf-8"))


# ---------------------------------------------------------------------------
# NaN-aware / state-snapshot assertions
# ---------------------------------------------------------------------------

def assert_nan_equal(testcase, actual, expected, msg=None):
    """Array equality where NaN == NaN (the 'same wavelengths spoiled' check)."""
    a, e = np.asarray(actual), np.asarray(expected)
    if a.dtype.kind in "fc" and e.dtype.kind in "fc":
        ok = np.array_equal(a, e, equal_nan=True)
    else:
        ok = np.array_equal(a, e)
    testcase.assertTrue(ok, msg or f"arrays differ:\n{a}\nvs\n{e}")


def snapshot_state(state):
    """Deep copy of every AppState field (arrays included)."""
    return copy.deepcopy({
        f.name: getattr(state, f.name)
        for f in type(state).__dataclass_fields__.values()
    })


def _values_match(a, b):
    """Deep equality that treats NaN == NaN inside arrays and dataclasses."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                and np.array_equal(a, b, equal_nan=True))
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        return (type(a) is type(b)
                and all(_values_match(getattr(a, f.name), getattr(b, f.name))
                        for f in dataclasses.fields(a)))
    if isinstance(a, dict):
        return (type(b) is type(a) and a.keys() == b.keys()
                and all(_values_match(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)):
        return (type(b) is type(a) and len(a) == len(b)
                and all(_values_match(x, y) for x, y in zip(a, b)))
    return a == b


def assert_state_unchanged(testcase, before, after):
    """Every field identical after presenters ran (render never writes)."""
    mutated = [key for key in before
               if not _values_match(before[key], after[key])]
    testcase.assertFalse(
        mutated, f"state fields mutated: {mutated}")


class TestCase(unittest.TestCase):
    """Base test case with NaN-aware comparison helpers."""

    def assertNanEqual(self, actual, expected, msg=None):
        assert_nan_equal(self, actual, expected, msg=msg)

    def assertClose(self, actual, expected, places=6, msg=None):
        if np.isnan(expected):
            self.assertTrue(np.isnan(actual), msg or "expected NaN")
        else:
            self.assertAlmostEqual(actual, expected, places=places, msg=msg)
