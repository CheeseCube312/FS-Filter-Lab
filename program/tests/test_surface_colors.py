"""White-referenced surface color math (UX_SPEC.md §7.1).

Property set under test:
- a near-black surface stays near-black (no auto-exposure amplification)
- the white anchor itself renders exactly white
- the surface chosen as WB reference renders exactly white by construction
- sRGB encoding is monotonic with 0->0 and 1->1
"""
import unittest

import numpy as np

from models.core import (
    AppData,
    FilterCollection,
    ReflectorCollection,
    ReflectorSpectrum,
    RGBSensorProfile,
)
from models.constants import INTERP_GRID
from services.calculations import (
    compute_reflector_color,
    compute_white_reference_color,
    reference_color_to_display,
    srgb_encode,
)
from services.presenter import prepare_surfaces
from services.state_manager import AppState


def _flat(x: float) -> np.ndarray:
    return np.full_like(INTERP_GRID, x, dtype=float)


def _make_app_data() -> AppData:
    """Minimal AppData: flat 0.5 QE sensor, flat illuminant, two reflectors."""
    from dataclasses import field as dc_field

    qe = {"R": _flat(0.5), "G": _flat(0.5), "B": _flat(0.5)}
    sensor = RGBSensorProfile(manufacturer="Test", name="Cam", channel_data=qe)

    gray = ReflectorSpectrum(name="Gray Card", values=_flat(0.5),
                             metadata={"source_file": "gray.tsv"})
    white = ReflectorSpectrum(name="Diffuse White", values=_flat(1.0),
                              metadata={"source_file": "white.tsv"})
    collection = ReflectorCollection(
        reflectors=[gray, white],
        df=None,
        reflector_matrix=np.vstack([gray.values, white.values]),
    )

    return AppData(
        filter_collection=FilterCollection(
            filters=[], df=None,
            filter_matrix=np.zeros((0, len(INTERP_GRID))),
            extrapolated_masks=np.zeros((0, len(INTERP_GRID)), dtype=bool),
        ),
        camera_keys=[],
        qe_data={},
        default_key="",
        illuminants={"Test": _flat(1.0)},
        illuminant_metadata={},
        reflector_collection=collection,
        sensor_profiles={sensor.display_name: sensor},
    )


class TestSRGBEncode(unittest.TestCase):
    def test_endpoints(self):
        self.assertEqual(float(srgb_encode(np.array([0.0]))[0]), 0.0)
        self.assertAlmostEqual(float(srgb_encode(np.array([1.0]))[0]), 1.0)

    def test_midpoint(self):
        self.assertAlmostEqual(float(srgb_encode(np.array([0.5]))[0]), 0.7353569830524495)

    def test_monotonic(self):
        lin = np.linspace(0.0, 1.0, 101)
        enc = srgb_encode(lin)
        self.assertTrue(np.all(np.diff(enc) >= 0))

    def test_clips_input(self):
        enc = srgb_encode(np.array([-0.5, 0.5, 2.0]))
        self.assertEqual(float(enc[0]), 0.0)
        self.assertEqual(float(enc[2]), 1.0)


class TestReferenceColorToDisplay(unittest.TestCase):
    def test_near_black_stays_near_black(self):
        """Regression: old auto-exposure turned [0.02, 0.001, 0.001] into red."""
        surface = np.array([0.02, 0.001, 0.001])
        white = np.array([1.0, 1.0, 1.0])
        display = reference_color_to_display(surface, white)
        self.assertTrue(np.all(display <= 0.05))
        self.assertAlmostEqual(float(display[0]), 0.02)

    def test_white_anchor_is_exactly_white(self):
        surface = np.array([0.8, 0.9, 0.7])
        display = reference_color_to_display(surface, surface.copy())
        self.assertTrue(np.allclose(display, 1.0))

    def test_clips_above_one(self):
        display = reference_color_to_display(np.array([2.0, 0.5, 0.1]), np.array([1.0, 1.0, 1.0]))
        self.assertEqual(float(display[0]), 1.0)

    def test_zero_white_reference_guarded(self):
        display = reference_color_to_display(np.array([0.5, 0.5, 0.5]), np.array([0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(display, 0.5))


class TestPrepareSurfaces(unittest.TestCase):
    def test_wb_reference_surface_renders_exactly_white(self):
        data = _make_app_data()
        state = AppState()
        state.select_sensor("Test Cam", data.sensor_profiles)
        state.select_illuminant("Test", data.illuminants)

        state.selected_surfaces = ["white.tsv", "gray.tsv"]
        state.set_white_balance_from_surface(
            reflector=_flat(1.0), transmission=_flat(1.0), source_file="white.tsv"
        )
        vm = prepare_surfaces(state, data)
        self.assertIsNone(vm.warning)
        by_name = {chip.name: chip for chip in vm.chips}
        self.assertEqual(by_name["Diffuse White"].hex_display.lower(), "#ffffff")
        self.assertTrue(by_name["Diffuse White"].is_wb_ref)
        # Gray card under white-WB: mid gray, NOT amplified to white
        self.assertNotEqual(by_name["Gray Card"].hex_display.lower(), "#ffffff")

    def test_warning_without_sensor_or_illuminant(self):
        data = _make_app_data()
        state = AppState()
        state.selected_surfaces = ["gray.tsv"]
        vm = prepare_surfaces(state, data)
        self.assertTrue(vm.warning)
        self.assertEqual(vm.chips, [])

    def test_white_reference_color_matches_reflector_color_of_ones(self):
        qe = {"R": _flat(0.5), "G": _flat(0.5), "B": _flat(0.5)}
        illum = _flat(1.0)
        trans = _flat(1.0)
        anchor = compute_white_reference_color(trans, qe, illum)
        manual = compute_reflector_color(_flat(1.0), trans, qe, illum)
        self.assertTrue(np.allclose(anchor, manual))


if __name__ == "__main__":
    unittest.main()
