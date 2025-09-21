"""
matplotlib_curve.py

Provides minimal support for rendering filter transmission curves using Matplotlib,
primarily for use in static exports such as PDF or image generation.

--- Imports ---
- numpy: Used for array masking to distinguish between valid and extrapolated data segments.

--- Functions ---

add_filter_curve_to_matplotlib(ax, x, y, mask, label, color)
    - Adds a transmission curve to a given Matplotlib axis.
    - Solid lines represent valid data; dashed lines represent extrapolated regions (Lee Filters only).

--- Usage Context ---
Designed for static plotting scenarios where interactivity is not required, such as exporting figures for documentation or reports.
"""


import numpy as np


#matplotlib curve for exports
def add_filter_curve_to_matplotlib(ax, x, y, mask, label, color):
    ax.plot(
        x[~mask], y[~mask],
        label=label,
        linestyle='-', linewidth=1.75, color=color
    )
    if np.any(mask):
        ax.plot(
            x[mask], y[mask],
            linestyle='--', linewidth=1.0, color=color         
)
