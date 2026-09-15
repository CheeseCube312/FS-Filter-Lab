"""
Report generation for FS FilterLab.

- generate_application_report(): assembles metrics and renders the PNG report
- generate_report_png(): multi-panel matplotlib PNG renderer
- sanitize_filename_component(): safe filename fragments for report paths
"""
# Standard library imports
import io
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

# Local imports
from models.core import FilterCollection
from models.constants import (
    CHART_COLORS, CHART_LINE_STYLES, GRID_HEIGHT_RATIOS, INTERP_GRID,
    MPL_STYLE_CONFIG, OUTPUT_FOLDERS, REPORT_CONFIG, EPSILON,
)
from services.calculations import (
    compute_filter_transmission,
    compute_selected_filter_indices,
    compute_effective_stops,
)
from services.state_manager import AppState

logger = logging.getLogger(__name__)

COLOR_MAP = CHART_COLORS['rgb_colors']


def sanitize_filename_component(name: str, lowercase=False, max_len=None) -> str:
    """
    Sanitize a string for safe use in filenames across operating systems.

    Removes or replaces characters that are invalid in Windows, macOS, and Linux
    filenames, ensuring generated files can be saved and shared reliably.

    Args:
        name: The input string to sanitize
        lowercase: If True, convert to lowercase for consistency
        max_len: Maximum length limit for the output string

    Returns:
        Cleaned string safe for use in filenames

    Example:
        >>> sanitize_filename_component("Filter: UV/IR-Cut", lowercase=True)
        'filter- uv-ir-cut'
    """
    clean = re.sub(r'[<>:"/\\|?*]', "-", name).strip()
    if lowercase:
        clean = clean.lower()
    if max_len:
        clean = clean[:max_len]
    return clean


# =============================================================================
# MATPLOTLIB VISUALIZATION
# =============================================================================

def _create_filter_combo_info(selected_filters: List[str], df: Any, display_to_index: Dict[str, int]) -> Tuple[List, str]:
    """Create sorted (manufacturer, number, row) combo info and joined name."""
    combo = []
    for name in sorted(selected_filters):
        idx = display_to_index.get(name)
        row = df.iloc[idx]
        combo.append((row['Manufacturer'], row['Filter Number'], row))
    combo_name = ", ".join(f"{m} {n}" for m, n, _ in combo)
    return combo, combo_name


def _add_filter_swatches_section(ax0, selected_filters: List[str], df: Any, display_to_index: Dict[str, int]):
    """Add filter color swatches and labels to the report."""
    ax0.axis('off')
    y0 = 0.9
    counts = {f: selected_filters.count(f) for f in set(selected_filters)}
    
    for filter_name, filter_count in counts.items():
        filter_index = display_to_index[filter_name]
        filter_row = df.iloc[filter_index]
        filter_color = filter_row.get('Hex Color', '#000000')
        
        rect = Rectangle((0.0, y0-0.15), 0.03, 0.1, transform=ax0.transAxes,
                         facecolor=filter_color, edgecolor='black', lw=REPORT_CONFIG['swatch_line_width'])
        ax0.add_patch(rect)
        
        ax0.text(0.03, y0-0.1, f"{filter_row['Manufacturer']} – {filter_row['Filter Name']} (#{filter_row['Filter Number']}) ×{filter_count}",
                 transform=ax0.transAxes, fontsize=REPORT_CONFIG['font_sizes']['filter_label'], va='center')
        y0 -= 0.15


def _add_light_loss_section(ax1, label: str, stops: float, avg_trans: float):
    """Add light loss information section to the report."""
    ax1.axis('off')
    ax1.text(0.01, 0.7, 'Estimated Light Loss:', fontsize=REPORT_CONFIG['font_sizes']['section_header'], fontweight='bold')
    ax1.text(0.01, 0.3, f"{label} → {stops:.2f} stops (Avg: {avg_trans*100:.1f}%)", fontsize=REPORT_CONFIG['font_sizes']['section_header'])


def _add_transmission_plot_section(ax2, selected_indices: List[int], df: Any, filter_matrix: np.ndarray, 
                                  masks: np.ndarray, add_curve_fn: Callable, interp_grid: np.ndarray, 
                                  active_trans: np.ndarray):
    """Add transmission plot section to the report."""
    for filter_index in selected_indices:
        filter_row = df.iloc[filter_index]
        transmission_pct = np.clip(filter_matrix[filter_index], 1e-6, 1.0) * 100
        filter_mask = masks[filter_index]
        add_curve_fn(ax2, interp_grid, transmission_pct, filter_mask,
                     filter_row['Filter Name'], filter_row.get('Hex Color', '#000000'))
    
    if len(selected_indices) > 1:
        ax2.plot(interp_grid, active_trans * 100, color='black', lw=REPORT_CONFIG['combined_line_width'], label='Combined Filter')
    
    ax2.set_title('Filter Transmission (%)')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Transmission (%)')
    ax2.set_xlim(interp_grid.min(), interp_grid.max())
    ax2.set_ylim(0, 100)


def _add_white_balance_section(ax3, wb: Dict[str, float]):
    """Add white balance gains section to the report."""
    ax3.axis('off')
    ax3.text(0.01, 0.6, 'White Balance Gains (Green = 1):', fontsize=REPORT_CONFIG['font_sizes']['section_header'], fontweight='bold')
    
    # Convert gains back to raw intensities (relative to green)
    intensities = {
        'R': 1.0 / wb['R'] if wb['R'] != 0 else 0.0,
        'G': 1.0,
        'B': 1.0 / wb['B'] if wb['B'] != 0 else 0.0
    }
    
    ax3.text(0.01, 0.4, f"R: {intensities['R']:.3f}   G: {intensities['G']:.3f}   B: {intensities['B']:.3f}", fontsize=REPORT_CONFIG['font_sizes']['section_header'])


def _add_sensor_response_section(ax4, current_qe: Dict[str, np.ndarray], wb: Dict[str, float], 
                               active_trans: np.ndarray, interp_grid: np.ndarray, 
                               camera_name: str, illuminant_name: str,
                               apply_white_balance: bool = False):
    """Add sensor-weighted response section to the report."""
    maxresp = 0
    stack = {}
    
    # Plot in correct RGB order
    for ch in ['R', 'G', 'B']:
        qe = current_qe.get(ch)
        if qe is None:
            continue
        resp = np.nan_to_num(active_trans * qe) * 100
        if apply_white_balance:
            resp = resp / max(wb.get(ch, 1.0), EPSILON)
        ax4.plot(interp_grid, resp, label=f"{ch} Channel", lw=REPORT_CONFIG['channel_line_width'], color=COLOR_MAP[ch])
        maxresp = max(maxresp, np.nanmax(resp))
        stack[ch] = resp

    # Spectrum strip
    rgb_matrix = np.stack([
        stack.get('R', np.zeros_like(active_trans)),
        stack.get('G', np.zeros_like(active_trans)),
        stack.get('B', np.zeros_like(active_trans))
    ], axis=1)
    mv = np.nanmax(rgb_matrix)
    if mv > 0:
        rgb_matrix /= mv
    
    extent = [interp_grid.min(), interp_grid.max(), maxresp * 1.02, maxresp * 1.07]
    ax4.imshow(rgb_matrix[np.newaxis, :, :], aspect='auto', extent=extent)

    suffix = " (White-Balanced)" if apply_white_balance else " (Raw)"
    ax4.set_title(f'Sensor-Weighted Response{suffix}', fontsize=REPORT_CONFIG['font_sizes']['title'], fontweight='bold')
    subtitle = f"Quantum Efficiency: {camera_name or 'None'}   |   Illuminant: {illuminant_name or 'None'}"
    ax4.text(0.5, 0.98, subtitle, transform=ax4.transAxes, ha='center', va='bottom', fontsize=REPORT_CONFIG['font_sizes']['subtitle'])
    ax4.set_xlabel('Wavelength (nm)')
    ax4.set_ylabel('Response (%)')
    ax4.set_xlim(interp_grid.min(), interp_grid.max())
    ax4.set_ylim(0, extent[3] * 1.02)
    ax4.legend(loc='upper right', fontsize=REPORT_CONFIG['font_sizes']['legend'], bbox_to_anchor=(1.0, 0.95))


def setup_matplotlib_style():
    """Configure matplotlib: best-available seaborn grid style + custom config."""
    # Try different seaborn style variants in order of preference
    style_options = ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "whitegrid", "default"]
    
    for style in style_options:
        try:
            plt.style.use(style)
            break
        except OSError:
            continue
    
    # Apply custom configuration
    plt.rcParams.update(MPL_STYLE_CONFIG)


def add_filter_curve_to_matplotlib(ax, x, y, mask, label, color):
    """Add a filter transmission curve (plus dashed extrapolation) to matplotlib axes."""
    mpl_width = CHART_LINE_STYLES['matplotlib']['width']
    extrap = CHART_LINE_STYLES['extrapolated']

    # Plot main curve
    ax.plot(x, y, color=color, linewidth=mpl_width, label=label)
    
    # Add extrapolated regions if mask exists
    if mask is not None and np.any(mask):
        extrap_y = y.copy()
        extrap_y[~mask] = np.nan
        ax.plot(x, extrap_y, color=color, linewidth=mpl_width, 
                linestyle=extrap['style'], alpha=extrap['alpha'])


def generate_report_png(
    selected_filters: List[str],
    selected_indices: List[int],
    active_transmission: np.ndarray,
    transmission_label: str,
    effective_stops: float,
    avg_transmission: float,
    white_balance_gains: Dict[str, float],
    current_qe: Optional[Dict[str, np.ndarray]],
    camera_name: str,
    illuminant_name: str,
    filter_df: Any,
    display_to_index: Dict[str, int],
    filter_matrix: np.ndarray,
    masks: np.ndarray,
    interp_grid: np.ndarray,
    sanitize_fn: Callable[[str], str],
    apply_white_balance: bool = False,
) -> Dict[str, Any]:
    """Generate a multi-panel PNG report of the current filter configuration.

    All computations (transmission, stops, white balance) are done by the
    caller; this function only renders. Returns {'bytes', 'name'} or {}.
    """
    if not selected_filters:
        logger.warning("No filters selected — nothing to export.")
        return {}

    # Prepare filter combination data
    combo, combo_name = _create_filter_combo_info(
        selected_filters, filter_df, display_to_index
    )

    # Create figure with layout
    setup_matplotlib_style()
    fig = plt.figure(figsize=REPORT_CONFIG['figure_size'], dpi=REPORT_CONFIG['dpi'], constrained_layout=False)
    gs = GridSpec(5, 1, figure=fig, height_ratios=GRID_HEIGHT_RATIOS)

    # Build report sections
    _add_filter_swatches_section(
        fig.add_subplot(gs[0]), selected_filters,
        filter_df, display_to_index
    )
    _add_light_loss_section(fig.add_subplot(gs[1]), transmission_label, effective_stops, avg_transmission)
    _add_transmission_plot_section(
        fig.add_subplot(gs[2]), selected_indices, filter_df,
        filter_matrix, masks,
        add_filter_curve_to_matplotlib, interp_grid, active_transmission
    )
    _add_white_balance_section(fig.add_subplot(gs[3]), white_balance_gains)
    _add_sensor_response_section(
        fig.add_subplot(gs[4]), current_qe, white_balance_gains,
        active_transmission, interp_grid,
        camera_name, illuminant_name, apply_white_balance
    )

    # Finalize layout
    fig.suptitle("Filter Report", fontsize=REPORT_CONFIG['font_sizes']['main_title'], fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    # Render to PNG bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    png_bytes = buf.getvalue()

    fname = sanitize_fn(
        f"{camera_name}_{illuminant_name}_{combo_name}"
    ) + '.png'

    # Save to output folder
    output_dir = os.path.join(
        OUTPUT_FOLDERS['reports'],
        sanitize_fn(camera_name),
        sanitize_fn(illuminant_name),
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, fname), "wb") as f:
        f.write(png_bytes)

    logger.info("Report generated: %s", fname)
    return {'bytes': png_bytes, 'name': fname}


def generate_application_report(
    app_state: AppState,
    filter_collection: FilterCollection,
    selected_camera: Optional[str] = None,
) -> Optional[Dict]:
    """
    Generate a PNG report of the current filter configuration.

    Args:
        app_state: Current application state
        filter_collection: Available filters
        selected_camera: Name of selected camera (optional)

    Returns:
        Report result dict if successful, None otherwise.
    """
    # Get selected filter indices
    selected_indices = compute_selected_filter_indices(
        app_state.selected_filters,
        app_state.filter_multipliers,
        filter_collection
    )

    if not selected_indices:
        return None

    # Get filter transmission
    transmission, label = compute_filter_transmission(
        selected_indices, filter_collection.filter_matrix
    )

    # Resolve sensor QE and illuminant
    sensor_qe = (
        app_state.current_qe.get('G') if app_state.current_qe else None
    )
    illuminant = (
        app_state.illuminant if app_state.illuminant is not None
        else np.ones_like(INTERP_GRID)
    )

    # Compute metrics
    avg_trans, stops = (
        compute_effective_stops(transmission, sensor_qe, illuminant)
        if sensor_qe is not None else (0.0, 0.0)
    )
    # Mirror the on-screen numbers exactly: the action layer keeps
    # white_balance_gains in sync (standard gains on stack/sensor/illuminant
    # changes, surface-referenced gains after WB-from-surface). Recomputing
    # standard gains here would silently override a surface reference.
    wb = dict(app_state.white_balance_gains)

    camera_name = selected_camera or "UnknownCamera"
    illuminant_name = app_state.illuminant_name or "UnknownIlluminant"

    result = generate_report_png(
        selected_filters=app_state.selected_filters,
        selected_indices=selected_indices,
        active_transmission=transmission,
        transmission_label=label,
        effective_stops=stops,
        avg_transmission=avg_trans,
        white_balance_gains=wb,
        apply_white_balance=app_state.apply_white_balance,
        current_qe=app_state.current_qe,
        camera_name=camera_name,
        illuminant_name=illuminant_name,
        filter_df=filter_collection.df,
        display_to_index=filter_collection.get_display_to_index_map(),
        filter_matrix=filter_collection.filter_matrix,
        masks=filter_collection.extrapolated_masks,
        interp_grid=INTERP_GRID,
        sanitize_fn=sanitize_filename_component,
    )

    return result or None