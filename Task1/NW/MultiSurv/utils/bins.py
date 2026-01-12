# deephit_bins.py

import numpy as np
import torch

def compute_bin_edges(durations, num_bins=40, method="quantile", pad=0.1):
    """
    Compute global bin edges for DeepHit.

    Args:
        durations (array-like): observed durations (float, months).
        num_bins (int): number of bins (T).
        method (str): "quantile" (preferred) or "equal".
        pad (float): extend min/max range by this fraction.
    Returns:
        np.ndarray of shape [num_bins+1]
    """
    durations = np.asarray(durations, dtype=float)
    min_d, max_d = durations.min(), durations.max()

    if method == "quantile":
        qs = np.linspace(0, 1, num_bins+1)
        edges = np.quantile(durations, qs)
    else:  # equal-width bins
        edges = np.linspace(min_d, max_d, num_bins+1)

    # pad slightly to avoid clipping
    edges[0] = max(0.0, edges[0] * (1 - pad))
    edges[-1] = edges[-1] * (1 + pad)

    return edges


def durations_to_bins(durations, edges):
    """
    Convert durations to discrete bin indices (0..T-1).

    Args:
        durations (array-like): observed durations.
        edges (array-like): bin edges from compute_bin_edges.
    Returns:
        np.ndarray of bin indices
    """
    idx = np.digitize(durations, edges) - 1  # into 0..T-1
    idx = np.clip(idx, 0, len(edges) - 2)   # clamp
    return idx


def get_bin_centers(edges):
    """
    Bin centers for converting PMFs back to real time.
    """
    return 0.5 * (edges[:-1] + edges[1:])


def expected_time_from_pmf(pmf, edges):
    """
    Convert predicted PMFs into expected time-to-event.

    Args:
        pmf: [N, T] numpy or torch array (row-normalized).
        edges: bin edges [T+1].
    Returns:
        expected times [N]
    """
    centers = get_bin_centers(edges)

    if isinstance(pmf, torch.Tensor):
        centers = torch.as_tensor(centers, dtype=pmf.dtype, device=pmf.device)
        return (pmf * centers[None, :]).sum(dim=1)
    else:
        return (pmf * centers[None, :]).sum(axis=1)
