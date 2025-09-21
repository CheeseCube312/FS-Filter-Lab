"""
metrics.py

Functions:
- compute_effective_stops: Calculates light-loss for filter, taking into account sensor quantum efficiency
- compute_white_balance_gains: calculates difference in RGB channel intensity for later white-balancing
- calculate_transmission_deviation_metrics: Compares actual transmission curve to a target and computes error metrics.

Requires:
- numpy
- streamlit (for UI warnings)
"""


import numpy as np
import streamlit as st

def compute_effective_stops(trans, sensor_qe):
    # Ensure inputs are numpy arrays
    trans = np.asarray(trans)
    sensor_qe = np.asarray(sensor_qe)
    
    # Find valid indices where neither is NaN
    valid = ~np.isnan(trans) & ~np.isnan(sensor_qe)
    
    # If no valid data, return NaNs immediately
    if not np.any(valid):
        return np.nan, np.nan
    
    clipped_trans = np.clip(trans[valid], 1e-6, 1.0)
    clipped_qe = sensor_qe[valid]
    
    # If all weights are zero, cannot compute weighted average
    if np.all(clipped_qe == 0):
        return np.nan, np.nan
    
    # Defensive: Check if clipped_trans or clipped_qe are empty before averaging
    if clipped_trans.size == 0 or clipped_qe.size == 0:
        return np.nan, np.nan
    
    # Weighted average transmission
    avg_trans = np.average(clipped_trans, weights=clipped_qe)
    
    # Prevent log2 of zero or negative (should be prevented by clipping but be safe)
    if avg_trans <= 0:
        return np.nan, np.nan
    
    effective_stops = -np.log2(avg_trans)
    
    return avg_trans, effective_stops


def compute_white_balance_gains(
    trans_interp: np.ndarray,
    current_qe: dict[str, np.ndarray],
    illum_curve: np.ndarray
) -> dict[str, float]:
    """
    Compute white balance gains normalized to green channel from transmission, QE, and illuminant.
    Returns a dict with gains such that green gain is exactly 1.0.
    """
    rgb_resp = {}
    for ch in ['R', 'G', 'B']:
        qe_curve = current_qe.get(ch)
        if qe_curve is None:
            rgb_resp[ch] = np.nan
            continue
        valid = ~np.isnan(trans_interp) & ~np.isnan(qe_curve) & ~np.isnan(illum_curve)
        if not valid.any():
            rgb_resp[ch] = np.nan
            continue
        rgb_resp[ch] = np.nansum(trans_interp[valid] * (qe_curve[valid] / 100) * illum_curve[valid])

    g = rgb_resp.get('G', np.nan)
    if not np.isnan(g) and g > 1e-6:
        # Normalize so green gain = 1.0
        return {ch: rgb_resp[ch] / g for ch in ['R', 'G', 'B']}
    
    st.warning("⚠️ Green channel too low — default white balance.")
    return {'R': 1.0, 'G': 1.0, 'B': 1.0}



def calculate_transmission_deviation_metrics(
    trans_curve: np.ndarray,
    target_profile: dict | None,  # allow None explicitly
    log_stops: bool = False
) -> dict:
    """
    Compute deviation metrics (MAE, Bias, Max Dev, RMSE) between trans_curve and target_profile.
    Returns an empty dict if target_profile is None or no valid overlap.
    """
    if target_profile is None:
        # No target given, return empty dict (or you could return None if preferred)
        return {}

    valid_t = ~np.isnan(trans_curve)
    valid_p = target_profile.get('valid')
    if valid_p is None:
        # Defensive: if 'valid' key missing, treat as no valid points
        return {}

    overlap = valid_t & valid_p
    if not overlap.any():
        return {}

    if log_stops:
        dev = np.log2(trans_curve[overlap]) - np.log2(target_profile['values'][overlap] / 100)
        unit = 'stops'
    else:
        dev = trans_curve[overlap] * 100 - target_profile['values'][overlap]
        unit = '%'

    mae = np.mean(np.abs(dev))
    bias = np.mean(dev)
    maxd = np.max(np.abs(dev))
    rmse = np.sqrt(np.mean(dev**2))

    return {'MAE': mae, 'Bias': bias, 'MaxDev': maxd, 'RMSE': rmse, 'Unit': unit}