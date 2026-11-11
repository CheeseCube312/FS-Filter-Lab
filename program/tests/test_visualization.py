"""services.visualization — the human-vision spectrum strip.

Locks the strip's hue-legend contract: core wavelengths render vivid,
correctly ordered CIE-1931 hues (no photopic dimming of the 450-480 nm
band), the visible band's edges taper to black, the hover text keeps the
true photopic brightness, and UV / IR aren't drawn at all — the strip
simply ends at the band edges.
"""
import unittest

import numpy as np
import plotly.graph_objects as go

from models.constants import (INTERP_GRID, SPECTRUM_STRIP, VISIBLE_MAX_NM,
                              VISIBLE_MIN_NM)
from services.visualization import (_add_human_spectrum_strip,
                                    _human_vision_strip_colors)
from tests.helpers import TestCase


def rgb_of(color: str):
    """Parse an 'rgb(r,g,b)' strip color into an (r, g, b) int tuple."""
    inner = color[4:-1] if color.startswith('rgb(') else color
    return tuple(int(p) for p in inner.split(','))


def strip_lookup():
    """Full-grid strip as {wavelength_nm: (color, hover)}."""
    colors, hovers = _human_vision_strip_colors(INTERP_GRID)
    return {int(nm): (colors[i], hovers[i])
            for i, nm in enumerate(INTERP_GRID)}


class HumanVisionStripColors(TestCase):
    """The human-vision strip is a hue legend, not a luminance map."""

    def test_core_hues_are_fully_saturated(self):
        lookup = strip_lookup()
        taper = SPECTRUM_STRIP['edge_taper_nm']
        # Inside the taper zone the peak channel is dimmed by design; in the
        # core band per-wavelength normalization pins it at 255, so no core
        # hue can fade toward black.
        for nm in range(380 + taper, 701 - taper):
            self.assertEqual(max(rgb_of(lookup[nm][0])), 255)

    def test_hue_ordering_across_the_visible_band(self):
        lookup = strip_lookup()
        r, g, b = rgb_of(lookup[460][0])   # blue: B at full scale
        self.assertEqual(b, 255)
        self.assertLessEqual(r, 60)
        self.assertEqual(g, 0)

        r, g, b = rgb_of(lookup[500][0])   # cyan-green: G at full scale
        self.assertEqual(g, 255)
        self.assertEqual(r, 0)
        self.assertGreaterEqual(b, 80)
        self.assertLessEqual(b, 190)

        r, g, b = rgb_of(lookup[610][0])   # red: R at full scale
        self.assertEqual((r, g, b), (255, 0, 0))

    def test_visible_edges_taper_to_dark(self):
        lookup = strip_lookup()
        # Band edges: the taper reaches exactly black at 380 / 700 nm, and
        # the hover's photopic truth agrees (0.0% at 380).
        self.assertEqual(rgb_of(lookup[380][0]), (0, 0, 0))
        self.assertEqual(rgb_of(lookup[700][0]), (0, 0, 0))
        self.assertIn("0.0% brightness", lookup[380][1])
        # Inside the ramps the hue survives, just dimmer, and stays blue- /
        # red-dominant (which also hides the CIE fit's unreliable edges).
        r, g, b = rgb_of(lookup[390][0])   # quarter-way up the violet ramp
        self.assertEqual(max(r, g, b), b)
        self.assertGreaterEqual(b, 100)
        self.assertLessEqual(g, 60)
        r, g, b = rgb_of(lookup[425][0])   # three-quarters up: clear violet
        self.assertGreaterEqual(b, 200)
        self.assertGreaterEqual(r, 70)
        self.assertLessEqual(g, 30)
        self.assertEqual(rgb_of(lookup[640][0]), (255, 0, 0))  # taper ends
        r, g, b = rgb_of(lookup[690][0])   # near the bottom of the red ramp
        self.assertGreater(r, g)
        self.assertLessEqual(r, 160)
        self.assertEqual(b, 0)

    def test_yellow_green_peak_at_555(self):
        lookup = strip_lookup()
        r, g, b = rgb_of(lookup[555][0])
        self.assertEqual(g, 255)
        self.assertEqual(b, 0)
        self.assertGreaterEqual(r, 40)
        self.assertLessEqual(r, 140)

    def test_hovers_keep_true_photopic_brightness(self):
        lookup = strip_lookup()
        self.assertIn("100.0% brightness", lookup[555][1])
        self.assertIn("human vision", lookup[460][1])
        v_460 = float(lookup[460][1].split("human vision · ")[1].split("%")[0])
        v_700 = float(lookup[700][1].split("human vision · ")[1].split("%")[0])
        # Equal-energy luminance truth survives in the hover, pixels aside:
        # 460 nm is genuinely dim (~6%), 555 nm is the peak, 700 nm near-zero.
        self.assertGreater(v_460, 0.0)
        self.assertLess(v_460, 10.0)
        self.assertLess(v_700, v_460)

    def test_uv_and_ir_are_not_drawn(self):
        # Outside the visible band the strip simply ends: no color, no
        # hover — the taper to black at the edges is the whole message.
        lookup = strip_lookup()
        for nm in (300, 379, 701, 1100):
            self.assertEqual(lookup[nm], ('', ''))

    def test_full_grid_produces_valid_colors_and_hovers(self):
        colors, hovers = _human_vision_strip_colors(INTERP_GRID)
        self.assertEqual(len(colors), len(INTERP_GRID))
        for color, hover in zip(colors, hovers):
            if not color:                      # UV / IR: not drawn
                self.assertEqual(hover, '')
                continue
            parts = rgb_of(color)
            self.assertEqual(len(parts), 3)
            self.assertTrue(all(0 <= c <= 255 for c in parts))
            self.assertTrue(hover)
        self.assertTrue(any(colors))           # the visible band is populated


class HumanVisionStripRendering(TestCase):
    """The strip builder translates the color contract into plotly marks."""

    def test_trace_covers_only_the_visible_band(self):
        fig = go.Figure()
        _add_human_spectrum_strip(fig, INTERP_GRID, 1.0)
        trace = fig.data[0]
        x = np.asarray(trace.x)
        self.assertEqual(int(x.min()), VISIBLE_MIN_NM)
        self.assertEqual(int(x.max()), VISIBLE_MAX_NM)
        self.assertEqual(len(x), VISIBLE_MAX_NM - VISIBLE_MIN_NM + 1)
        # One uniform marker size again: every drawn point is in-band.
        self.assertEqual(trace.marker.size, SPECTRUM_STRIP['marker_size'])


if __name__ == "__main__":
    unittest.main()