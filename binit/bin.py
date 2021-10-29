from typing import Optional, Tuple, List
import numpy as np
from collections import OrderedDict


def binned_array_regular_interval(
    arr: np.ndarray,
    binwidth: float,
    t_start: Optional[float] = None,
    t_stop: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bin an array of timestamps into bins at a regular interval

    Args:
        arr (np.ndarray): Input array of timestamps
        binwidth (float): Length of bins
        t_start (Optional[float], optional): Sets the first bin to start at this timepoint. Defaults to None.
        t_stop (Optional[float], optional): Sets the last bin to stop before this timepoint. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Bin edges, event counts
    """
    if t_start is None:
        t_start = np.min(arr)
    if t_stop is None:
        t_stop = np.max(arr)
    bins = np.arange(t_start, t_stop + binwidth, binwidth)
    values, edges = np.histogram(arr, bins=bins)
    return edges[1:], values


def binned_array_bins_provided(
    arr: np.ndarray, bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Bin an array of timestamps into pre-specified time bins

    Args:
        arr (np.ndarray): Input array of timestamps
        bins (np.ndarray): Time bins to use for binning

    Returns:
        Tuple[np.ndarray, np.ndarray]: Bin edges, event counts
    """
    values, edges = np.histogram(arr, bins=bins)
    return edges[1:], values


def binarize_array(arr: np.ndarray) -> np.ndarray:
    """Binarize an array

    Args:
        arr (np.ndarray): Input array

    Returns:
        np.ndarray: An array whose non-zero values have been replaced with 1
    """
    return np.where(arr != 0, 1, 0)


def which_bin(
    arr: np.ndarray,
    bin_edges: np.ndarray,
    time_before: Optional[float] = None,
    nan_vals_before_first_bin: bool = False,
    nan_vals_occuring_x_after_last_bin: Optional[float] = None,
) -> np.ndarray:
    """For each element of an input array, get the corresponding bin it would be binned into

    Args:
        arr (np.ndarray): Input array of timestamps
        bin_edges (np.ndarray): Array of bins
        time_before (Optional[float], optional): By default, values are binned into the closest preceding bin but if this value is specified, timestamps falling this value before a following bin are binned to that bin. Defaults to None.
        nan_vals_before_first_bin (bool, optional): If True, returns np.nan for values of input timestamps occuring before the after the first bin. Defaults to False.
        nan_vals_occuring_x_after_last_bin (Optional[float], optional): If specified, return np.nan for values occuring this latency after the final bin. Defaults to None.

    Returns:
        np.ndarray: Bin values
    """
    if time_before is not None:
        idx = np.digitize(arr, (bin_edges - time_before)) - 1
    else:
        idx = np.digitize(arr, bin_edges) - 1

    idx_to_use = idx.tolist()
    bin_values = (bin_edges[idx_to_use]).astype(float)
    if nan_vals_before_first_bin:
        nan_mask = idx < 0
        bin_values[nan_mask] = np.nan
    if nan_vals_occuring_x_after_last_bin is not None:
        latency_to_max = arr - np.max(bin_edges)
        bin_values[latency_to_max > nan_vals_occuring_x_after_last_bin] = np.nan
    return bin_values


def which_bin_idx(
    arr: np.ndarray,
    bin_edges: np.ndarray,
    time_before: Optional[float] = None,
    nan_vals_before_first_bin: bool = False,
    nan_vals_occuring_x_after_last_bin: Optional[float] = None,
) -> np.ndarray:
    """For each element of an input array, get the corresponding index of the bin it would be binned into

    Args:
        arr (np.ndarray): Input array of timestamps
        bin_edges (np.ndarray): Array of bins
        time_before (Optional[float], optional): By default, values are binned into the closest preceding bin but if this value is specified, timestamps falling this value before a following bin are binned to that bin. Defaults to None.
        nan_vals_before_first_bin (bool, optional): If True, returns np.nan for values of input timestamps occuring before the after the first bin. Defaults to False.
        nan_vals_occuring_x_after_last_bin (Optional[float], optional): If specified, return np.nan for values occuring this latency after the final bin. Defaults to None.

    Returns:
        np.ndarray: Bin values
    """
    if time_before is not None:
        idx = np.digitize(arr, (bin_edges - time_before)) - 1
    else:
        idx = np.digitize(arr, bin_edges) - 1

    idx = idx.astype(float)
    if nan_vals_before_first_bin:
        nan_mask = idx < 0
        idx[nan_mask] = np.nan
    if nan_vals_occuring_x_after_last_bin is not None:
        latency_to_max = arr - np.max(bin_edges)
        idx[latency_to_max > nan_vals_occuring_x_after_last_bin] = np.nan
    return idx


def split_by_bin(
    arr: np.ndarray,
    bins: np.ndarray,
    max_latency: Optional[float] = None,
    time_before: float = None,
) -> "OrderedDict[float, np.ndarray]":
    """Split an array of timestamps by bin and transform their values to be latencies to that bin.

    Args:
        arr (np.ndarray): Array of timestamps
        bins (np.ndarray): Array of bins
        max_latency (Optional[float], optional): Exclude timestamps occuring at a latency to the final bin greater than this value. Defaults to None.
        time_before (Optional[float], optional): By default, values are binned into the closest preceding bin but if this value is specified, timestamps falling this value before a following bin are binned to that bin. Defaults to None.

    Returns:
        OrderedDict[float, np.ndarray]: An ordered dict whose keys are the bins and whose values are arrays of vatency to these bins.
    """
    bin_vals = which_bin(
        arr,
        bins,
        nan_vals_occuring_x_after_last_bin=max_latency,
        time_before=time_before,
    )
    out = OrderedDict()
    for bin_val in np.unique(bin_vals[np.logical_not(np.isnan(bin_vals))]):
        out[bin_val] = arr[bin_vals == bin_val] - bin_val
    return out


def bin_array_around_event(
    arr: np.ndarray, timestamps_to_bin_around: np.ndarray, binsize: float
) -> np.ndarray:
    """Get counts of timestamps of one array occuring around an timestamps specified in another array.

    Args:
        arr (np.ndarray): Array of timestamps to be counted
        timestamps_to_bin_around (np.ndarray): Array of timestamps to bin around
        binsize (float): Size of the window around the timestamps to calculate counts over

    Returns:
        np.ndarray: Array of timestamp counts
    """
    bins = np.repeat(timestamps_to_bin_around, 2).astype(np.float64)
    bins[1::2] += binsize
    spikecounts = binned_array_bins_provided(arr, bins)
    return spikecounts[::2]
