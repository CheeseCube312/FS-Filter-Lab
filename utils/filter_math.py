"""
filter_math.py

Functions:
- compute_active_transmission: Returns combined transmission for selected filters or identity if none selected.
- compute_combined_transmission: Multiplies transmission curves of selected filters to get raw combined output.
- compute_filter_transmission: Wraps combined transmission with label and output formatting for UI or display.
- compute_rgb_response_from_transmission_and_qe: Simulates RGB sensor response using filter transmission and sensor QE curves.

Requires:
- numpy
- INTERP_GRID from utils.constants
"""



import numpy as np
from utils.constants import INTERP_GRID


def compute_active_transmission(selected, selected_indices=None, filter_matrix=None):
    if selected and selected_indices and filter_matrix is not None:
        return 	compute_combined_transmission(selected_indices, filter_matrix, combine=True)
    return np.ones_like(INTERP_GRID)  # Identity transmission (no filter effect)


def compute_combined_transmission(indices, filter_matrix, combine=True):
    if combine and len(indices) > 1:
        stack = np.array([filter_matrix[i] for i in indices])
        combined = np.nanprod(stack, axis=0)
        combined[np.any(np.isnan(stack), axis=0)] = np.nan
        return combined
    return filter_matrix[indices[0]]


def compute_filter_transmission(selected_indices, filter_matrix, df):

    if len(selected_indices) > 1:
        trans = compute_combined_transmission(selected_indices, filter_matrix, combine=True)
        trans = np.clip(trans, 1e-6, 1.0)
        label = "Combined"
        combined = trans
    else:
        trans = filter_matrix[selected_indices[0]]
        label = df.iloc[selected_indices[0]]["Filter Name"]
        combined = None

    return trans, label, combined


def compute_rgb_response_from_transmission_and_qe(trans, current_qe, white_balance_gains, visible_channels):
    responses = {}
    rgb_stack = []

    # Check if trans is valid (non-empty, non-None, contains some finite values)
    if trans is None or len(trans) == 0 or not np.any(np.isfinite(trans)):
        # Return zeros if no valid transmission data
        zero_array = np.zeros_like(next(iter(current_qe.values()))) if current_qe else np.array([])
        for channel in ['R', 'G', 'B']:
            responses[channel] = zero_array
            rgb_stack.append(zero_array)
        return responses, np.stack(rgb_stack, axis=1) if rgb_stack else np.array([]), 0.0

    max_response = 0.0

    for channel in ['R', 'G', 'B']:
        qe_curve = current_qe.get(channel)
        if qe_curve is None or len(qe_curve) != len(trans):
            responses[channel] = np.zeros_like(trans)
            rgb_stack.append(responses[channel])
            continue

        gain = white_balance_gains.get(channel, 1.0)
        # Avoid division by zero in gain
        if gain == 0:
            gain = 1.0

        weighted = np.nan_to_num(trans * (qe_curve / 100)) / gain * 100
        max_response = max(max_response, np.nanmax(weighted))

        if visible_channels.get(channel, True):
            responses[channel] = weighted
        else:
            responses[channel] = np.zeros_like(weighted)

        rgb_stack.append(responses[channel])

    rgb_matrix = np.stack(rgb_stack, axis=1)
    max_val = np.nanmax(rgb_matrix)
    if max_val > 0:
        rgb_matrix = rgb_matrix / max_val
    rgb_matrix = np.clip(rgb_matrix, 1 / 255, 1.0)

    return responses, rgb_matrix, max_response
