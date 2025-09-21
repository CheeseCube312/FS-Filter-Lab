"""
plotly_utils.py

Provides reusable Plotly-based visualizations for filter and sensor response data in optical systems.

--- Imports ---
- plotly.graph_objects: Used for building interactive line plots and visual elements.
- numpy: Numerical operations including array masking and transformation.

--- Key Constants ---
- QE_COLORS: Standard color map for red, green, and blue sensor channels.

--- Main Functional Areas ---

1. create_filter_response_plot(...)
   - Plots transmittance curves of selected optical filters.
   - Supports extrapolated data visualization, combined filter overlay, and optional target profile.
   - Allows log-scale display in stops (log₂).

2. create_sensor_response_plot(...)
   - Visualizes sensor response curves by combining filter transmission with quantum efficiency (QE).
   - Includes filter and target comparisons with simulated white balance and RGB color bands.
   - Relies on externally provided helper functions for RGB conversion and QE × transmission calculations.

3. add_filter_curve_to_plotly(...)
   - Utility to add both valid and extrapolated filter curve segments to an existing Plotly figure.
   - Ensures extrapolated sections are visually distinct via dashed lines.

--- Usage Context ---
These functions are designed for integration into an interactive Streamlit application where dynamic filter and sensor visualizations are needed.

"""



import plotly.graph_objects as go
import numpy as np


# Define standard colors for QE channels
QE_COLORS = {"R": "red", "G": "green", "B": "blue"}

#Data preparation Filter Plotter
def create_filter_response_plot(
    interp_grid: np.ndarray,
    df,
    filter_matrix: np.ndarray,
    masks: np.ndarray,
    selected_indices: list[int],
    combined: np.ndarray | None,
    target_profile: dict | None,
    log_stops: bool = False
) -> go.Figure:
    """
    Create Plotly figure for filter transmission curves, including individual, combined, and target.
    """
    fig = go.Figure()

    # Individual filter traces
    for idx in selected_indices:
        row = df.iloc[idx]
        trans_curve = np.clip(filter_matrix[idx], 1e-6, 1.0)
        mask = masks[idx]
        y = np.log2(trans_curve) if log_stops else trans_curve * 100

        # main trace
        fig.add_trace(go.Scatter(
            x=interp_grid[~mask], y=y[~mask],
            name=f"{row['Filter Name']} ({row['Filter Number']})",
            mode="lines",
            line=dict(dash="solid", color=row.get('Hex Color', 'black'))
        ))
        # extrapolated
        if mask.any():
            fig.add_trace(go.Scatter(
                x=interp_grid[mask], y=y[mask],
                name=f"{row['Filter Name']} ({row['Filter Number']}) (Extrap)",
                mode="lines",
                line=dict(dash="dash", color=row.get('Hex Color', 'black')),
                showlegend=False
            ))

    # Combined trace
    if combined is not None:
        y_combined = np.log2(combined) if log_stops else combined * 100
        fig.add_trace(go.Scatter(
            x=interp_grid, y=y_combined,
            name="Combined Filter",
            mode="lines",
            line=dict(color="black", width=2)
        ))

    # Target trace
    if target_profile:
        valid = target_profile['valid']
        vals = target_profile['values']
        y_target = np.log2(vals) if log_stops else vals * 100
        fig.add_trace(go.Scatter(
            x=interp_grid[valid], y=y_target[valid],
            name=f"Target: {target_profile['name']}",
            mode="lines",
            line=dict(color="black", dash="dot", width=2)
        ))

    # Layout
    y_title = "Stops (log₂)" if log_stops else "Transmission (%)"
    fig.update_layout(
        title="Combined Filter Response",
        xaxis_title="Wavelength (nm)",
        yaxis_title=y_title,
        xaxis_range=(interp_grid.min(), interp_grid.max()),
        yaxis=dict(
            range=(-10, 0) if log_stops else (0, 100),
            tickvals=[-i for i in range(11)] if log_stops else None,
            ticktext=[f"-{i}" if i > 0 else "0" for i in range(11)] if log_stops else None
        ),
        showlegend=True
    )
    return fig


#Data preparation: Sensor response plot
def create_sensor_response_plot(
    interp_grid,
    trans_interp,
    qe_interp,
    visible_channels,
    white_balance_gains,
    apply_white_balance,
    target_profile=None,
    rgb_to_hex_fn=None,
    compute_sensor_weighted_rgb_response_fn=None,
):
    if rgb_to_hex_fn is None or compute_sensor_weighted_rgb_response_fn is None:
        raise ValueError("Required helper functions not provided.")

    responses, rgb_matrix, max_response = compute_sensor_weighted_rgb_response_fn(
        trans_interp, qe_interp, white_balance_gains, visible_channels
    )

    target_responses = None
    target_rgb_matrix = None
    target_max_response = 0

    if target_profile is not None:
        target_interp = target_profile["values"].copy()
        if np.nanmax(target_interp) > 1.5:
            target_interp /= 100.0
        target_responses, target_rgb_matrix, target_max_response = compute_sensor_weighted_rgb_response_fn(
            target_interp, qe_interp, white_balance_gains, visible_channels
        )

    fig = go.Figure()
    channel_colors = {"R": "red", "G": "green", "B": "blue"}

    for channel, show in visible_channels.items():
        if show:
            fig.add_trace(go.Scatter(
                x=interp_grid,
                y=responses[channel],
                name=f"{channel} Response (Filter){' (WB)' if apply_white_balance else ''}",
                mode="lines",
                line=dict(width=2, color=channel_colors.get(channel, "gray"))
            ))

    if target_responses is not None and len(target_responses) > 0:
        for channel, show in visible_channels.items():
            if show:
                fig.add_trace(go.Scatter(
                    x=interp_grid,
                    y=target_responses[channel],
                    name=f"{channel} Response (Target)",
                    mode="lines",
                    line=dict(width=2, color=channel_colors.get(channel, "gray"), dash="dot")
                ))

    combined_max_response = max(max_response, target_max_response)
    gradient_colors = [rgb_to_hex_fn(row) for row in rgb_matrix]
    spectrum_y = combined_max_response * 1.10

    for i in range(len(interp_grid) - 1):
        fig.add_trace(go.Scatter(
            x=[interp_grid[i], interp_grid[i + 1] + 1e-6],
            y=[spectrum_y, spectrum_y],
            mode="lines",
            line=dict(color=gradient_colors[i], width=15),
            showlegend=False,
            hoverinfo="skip"
        ))

    if target_responses is not None and len(target_responses) > 0:
        target_gradient_colors = [rgb_to_hex_fn(row) for row in target_rgb_matrix]
        target_spectrum_y = combined_max_response * 1.04

        for i in range(len(interp_grid) - 1):
            fig.add_trace(go.Scatter(
                x=[interp_grid[i], interp_grid[i + 1] + 1e-6],
                y=[target_spectrum_y, target_spectrum_y],
                mode="lines",
                line=dict(color=target_gradient_colors[i], width=10),
                showlegend=False,
                hoverinfo="skip"
            ))

        fig.add_annotation(
            x=interp_grid[-1],
            y=target_spectrum_y,
            text="Target Spectrum",
            showarrow=False,
            font=dict(size=10, color="white"),
            xanchor="right",
            bgcolor="rgba(0,0,0,0.4)",
        )

    fig.add_annotation(
        x=interp_grid[800],
        y=spectrum_y,
        text="Filter Spectrum",
        showarrow=False,
        font=dict(size=10, color="white"),
        xanchor="right",
        bgcolor="rgba(0,0,0,0.4)",
    )

    fig.update_layout(
        title="Effective Sensor Response (Transmission × QE) with Target Comparison",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Response (%)",
        xaxis_range=[300, 1100],
        yaxis_range=[0, combined_max_response * 1.15],
        showlegend=True,
        height=450
    )

    return fig


#Actual plotting part
def add_filter_curve_to_plotly(fig, x, y, mask, label, color):
    fig.add_trace(go.Scatter(
        x=x[~mask],
        y=y[~mask],
        name=label,
        mode="lines",
        line=dict(width=2, color=color)
    ))
    if np.any(mask):
        fig.add_trace(go.Scatter(
            x=x[mask],
            y=y[mask],
            name=label + " (extrapolated)",
            mode="lines",
            line=dict(width=1, dash="dash", color=color),
            showlegend=False  # Optional: prevent legend clutter
        ))