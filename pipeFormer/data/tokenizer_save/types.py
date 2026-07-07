"""Common type definitions for the tokenizer package."""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Union

import numpy as np
import torch


ArrayLike = Union[
    np.ndarray,
    torch.Tensor,
    Sequence[Union[np.ndarray, torch.Tensor]],
]
MetadataValue = Union[int, float, str, bool, None]
MetadataRow = Dict[str, MetadataValue]


@dataclass
class VariableTokens:
    """Tokenisation details for a single variable."""

    variable_name: str
    variable_index: int
    constant_keys: np.ndarray
    constant_values: np.ndarray
    constant_token_ids: np.ndarray
    constant_counts: np.ndarray
    range_lower_bounds: np.ndarray
    range_upper_bounds: np.ndarray
    range_token_ids: np.ndarray
    range_counts: np.ndarray
    is_binary: bool
    is_constant: bool
    sample_count: int
    constant_map: Dict[float, int] = field(default_factory=dict)

    def serialize(self) -> Dict[str, Union[str, int, float, List[float], List[int]]]:
        """Convert the dataclass to a JSON serialisable dict."""

        return {
            "variable_name": self.variable_name,
            "variable_index": self.variable_index,
            "constant_keys": self.constant_keys.tolist(),
            "constant_values": self.constant_values.tolist(),
            "constant_token_ids": self.constant_token_ids.tolist(),
            "constant_counts": self.constant_counts.tolist(),
            "range_lower_bounds": self.range_lower_bounds.tolist(),
            "range_upper_bounds": self.range_upper_bounds.tolist(),
            "range_token_ids": self.range_token_ids.tolist(),
            "range_counts": self.range_counts.tolist(),
            "is_binary": bool(self.is_binary),
            "is_constant": bool(self.is_constant),
            "sample_count": int(self.sample_count),
        }


__all__ = [
    "ArrayLike",
    "MetadataRow",
    "MetadataValue",
    "VariableTokens",
]
