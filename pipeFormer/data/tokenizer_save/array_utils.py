"""Utilities for handling array inputs to the tokenizer."""

from typing import Iterable, List, Sequence, Union

import numpy as np
import torch

from .types import ArrayLike


def _iter_items(item: Union[ArrayLike, None]) -> Iterable[Union[np.ndarray, torch.Tensor]]:
    if item is None:
        return []
    if isinstance(item, (np.ndarray, torch.Tensor)):
        return [item]
    if isinstance(item, Sequence):
        return item
    raise TypeError(f"Unsupported data type for tokenizer input: {type(item)}")


def to_numpy_2d(array: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy().astype(np.float64, copy=False)
    if isinstance(array, np.ndarray):
        return array.astype(np.float64, copy=False)
    raise TypeError(f"Unsupported array type: {type(array)}")


def collect_and_stack(primary: ArrayLike, extra: ArrayLike = None) -> np.ndarray:
    arrays: List[np.ndarray] = []

    for collection in (_iter_items(primary), _iter_items(extra)):
        for arr in collection:
            arrays.append(to_numpy_2d(arr))

    if not arrays:
        raise ValueError("No data provided for tokenizer fitting.")

    feature_dim = arrays[0].shape[1]
    for arr in arrays:
        if arr.ndim != 2:
            raise ValueError(f"Expected 2-D arrays, got shape {arr.shape}.")
        if arr.shape[1] != feature_dim:
            raise ValueError(
                f"Mismatched feature dimensions: {arr.shape[1]} vs {feature_dim}."
            )

    combined = np.concatenate(arrays, axis=0).astype(np.float64, copy=False)
    if np.isnan(combined).any():
        raise ValueError("Tokenizer input contains NaN values; please clean the data before tokenization.")
    return combined


__all__ = ["collect_and_stack", "to_numpy_2d"]
