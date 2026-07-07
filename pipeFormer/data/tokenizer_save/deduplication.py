"""Duplicate detection helpers for the tokenizer."""

from typing import Dict, List, Optional, Union

import hashlib
import numpy as np

def column_signature(column: np.ndarray) -> bytes:
    mask = np.isfinite(column)
    normalised = np.array(column, copy=True)
    normalised[~mask] = 0.0
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(normalised.tobytes())
    hasher.update(mask.tobytes())
    return hasher.digest()


def find_identical_equipment_variables(
    combined: np.ndarray,
    variable_names: List[str],
    boundary_dims: int,
    signature_fn,
) -> Dict[int, int]:
    duplicates: Dict[int, int] = {}

    if combined.size == 0 or not variable_names:
        return duplicates

    start_idx = min(max(boundary_dims, 0), combined.shape[1])
    end_idx = combined.shape[1]
    if start_idx >= end_idx:
        return duplicates

    signature_map: Dict[bytes, List[int]] = {}
    for var_idx in range(start_idx, end_idx):
        column = combined[:, var_idx]
        signature = signature_fn(column)
        candidates = signature_map.setdefault(signature, [])
        canonical_idx: Optional[int] = None
        for candidate in candidates:
            if np.array_equal(column, combined[:, candidate], equal_nan=True):
                canonical_idx = candidate
                break
        if canonical_idx is None:
            candidates.append(var_idx)
            continue
        duplicates[var_idx] = canonical_idx

    return duplicates


def build_identical_pairs(
    duplicates: Dict[int, int],
    variable_names: List[str],
    get_group,
) -> List[Dict[str, Union[str, float]]]:
    pairs: List[Dict[str, Union[str, float]]] = []
    seen: set = set()
    for duplicate_idx, canonical_idx in duplicates.items():
        key = tuple(sorted((duplicate_idx, canonical_idx)))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(
            {
                "variable_a": variable_names[canonical_idx],
                "variable_b": variable_names[duplicate_idx],
                "match_ratio": 1.0,
                "group": get_group(duplicate_idx),
            }
        )
    return pairs


__all__ = [
    "build_identical_pairs",
    "column_signature",
    "find_identical_equipment_variables",
]
