"""Helpers for building token ranges."""

from typing import List, Tuple

import numpy as np

from .types import MetadataRow


def is_binary(unique_keys: np.ndarray, atol: float) -> bool:
    if unique_keys.size == 0 or unique_keys.size > 2:
        return False
    rounded_int = np.round(unique_keys).astype(int)
    if not np.all(np.abs(unique_keys - rounded_int) <= atol):
        return False
    return np.all(np.isin(rounded_int, (0, 1)))


def build_range_tokens(
    non_constant_values: np.ndarray,
    sample_count: int,
    variable_index: int,
    variable_name: str,
    start_token_id: int,
    is_constant: bool,
    is_binary_flag: bool,
    min_fraction: float,
    group_name: str,
    constant_keys: np.ndarray,
    gap_epsilon: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, List[MetadataRow]]:
    metadata_rows: List[MetadataRow] = []

    if non_constant_values.size == 0 or is_binary_flag:
        empty_float = np.empty(0, dtype=np.float64)
        empty_int = np.empty(0, dtype=np.int64)
        return (
            empty_float,
            empty_float,
            empty_int,
            empty_int,
            start_token_id,
            metadata_rows,
        )

    sorted_values = np.sort(non_constant_values)

    min_bin_size = max(1, int(np.ceil(min_fraction * sample_count))) if sample_count > 0 else 1

    split_points = np.sort(constant_keys.astype(np.float64)) if constant_keys.size else np.empty(0)
    segments: List[np.ndarray] = []
    start_idx = 0
    if split_points.size:
        for point in split_points:
            end_idx = int(np.searchsorted(sorted_values, point, side="left"))
            if end_idx > start_idx:
                segments.append(sorted_values[start_idx:end_idx])
            start_idx = int(np.searchsorted(sorted_values, point, side="right"))
        if start_idx < sorted_values.size:
            segments.append(sorted_values[start_idx:])
    else:
        segments.append(sorted_values)

    lower_bounds: List[float] = []
    upper_bounds: List[float] = []
    counts: List[int] = []
    averages: List[float] = []
    medians: List[float] = []

    for segment in segments:
        seg_len = segment.size
        if seg_len == 0:
            continue
        start = 0
        remaining = seg_len
        while remaining > 0:
            bin_count = max(1, int(np.ceil(remaining / min_bin_size)))
            if bin_count == 1:
                end = seg_len
            else:
                target = remaining / bin_count
                end = start + int(round(target))
                end = max(start + 1, min(seg_len, end))
                max_end = seg_len - (bin_count - 1)
                if end > max_end:
                    end = max_end
                while end < max_end and np.isclose(segment[end - 1], segment[end]):
                    end += 1
                end = min(end, max_end)
            if end <= start:
                end = min(seg_len, start + 1)
            lower = float(segment[start])
            upper = float(segment[end - 1])
            count = int(end - start)
            bin_values = segment[start:end]
            lower_bounds.append(lower)
            upper_bounds.append(upper)
            counts.append(count)
            averages.append(float(np.mean(bin_values)))
            medians.append(float(np.median(bin_values)))
            start = end
            remaining = seg_len - start

    if not counts:
        empty_float = np.empty(0, dtype=np.float64)
        empty_int = np.empty(0, dtype=np.int64)
        return empty_float, empty_float, empty_int, empty_int, start_token_id, metadata_rows

    lower_arr = np.asarray(lower_bounds, dtype=np.float64)
    upper_arr = np.asarray(upper_bounds, dtype=np.float64)
    count_arr = np.asarray(counts, dtype=np.int64)
    avg_arr = np.asarray(averages, dtype=np.float64)
    median_arr = np.asarray(medians, dtype=np.float64)

    if lower_arr.size > 1:
        for idx in range(lower_arr.size - 1):
            next_lower = lower_arr[idx + 1]
            target_upper = float(next_lower) - float(gap_epsilon)
            if target_upper > upper_arr[idx]:
                upper_arr[idx] = target_upper

    np.maximum(upper_arr, lower_arr, out=upper_arr)

    token_ids = np.empty(count_arr.size, dtype=np.int64)

    next_token_id = start_token_id
    for idx in range(count_arr.size):
        token_id = next_token_id
        next_token_id += 1
        token_ids[idx] = token_id
        metadata_rows.append(
            {
                "token_id": token_id,
                "variable_index": variable_index,
                "variable_name": variable_name,
                "token_type": "range",
                "value_average": float(avg_arr[idx]),
                "value_median": float(median_arr[idx]),
                "range_start": float(lower_arr[idx]),
                "range_end": float(upper_arr[idx]),
                "count": int(count_arr[idx]),
                "probability": float(count_arr[idx] / sample_count),
                "variable_sample_count": sample_count,
                "is_constant": bool(is_constant),
                "is_binary": bool(is_binary_flag),
                "variable_group": group_name,
                "duplicate_of": None,
            }
        )

    return lower_arr, upper_arr, count_arr, token_ids, next_token_id, metadata_rows


__all__ = ["build_range_tokens", "is_binary"]
