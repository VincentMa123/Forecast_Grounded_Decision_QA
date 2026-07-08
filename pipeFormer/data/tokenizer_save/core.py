"""Implementation of the DataTokenizer class."""

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from .array_utils import collect_and_stack
from .deduplication import (
    build_identical_pairs,
    column_signature,
    find_identical_equipment_variables,
)
from .metadata import TokenMetadataStore
from .node_utils import (
    compute_group_from_index,
    get_variable_group,
    group_variables_by_node,
    load_node_connections,
    load_variable_names,
)
from .range_utils import build_range_tokens, is_binary
from .types import ArrayLike, MetadataRow, VariableTokens


logger = logging.getLogger(__name__)


class DataTokenizer:
    """Tokenizer that produces a shared vocabulary across all variables."""

    CONFIG_VERSION = 1

    def __init__(
        self,
        data_dir: Union[str, Path],
        vocab_size: int = 0,
        boundary_dims: Optional[int] = None,
        equipment_dims: Optional[int] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.config_filename = "data_tokenizer_config.json"
        self.stats_dir: Path = Path()
        self.config_path: Path = Path()
        self.log_path: Path = Path()
        self.set_stats_directory(self.data_dir / "tokenizer_save")
        self._boundary_dims_explicit = boundary_dims is not None
        self._equipment_dims_explicit = equipment_dims is not None
        self.boundary_dims = int(boundary_dims) if boundary_dims is not None else 0
        self.equipment_dims = int(equipment_dims) if equipment_dims is not None else 0
        self.total_dims = self.boundary_dims + self.equipment_dims

        # Tokenisation hyper-parameters
        self.constant_freq_threshold: float = 0.02
        self.constant_variable_threshold: float = 0.999
        self.quantile_step: float = 0.02  # 分位数步长
        self.quantile_method: str = "linear"
        self.range_gap_epsilon: float = 1e-9
        self.set_round_gap(0.3)
        self._load_config_from_file()

        # Learned state
        self.variable_names: Optional[List[str]] = None
        self._variable_name_to_index: Dict[str, int] = {}
        self.variable_token_configs: List[VariableTokens] = []
        self._metadata_store = TokenMetadataStore()
        self.vocab_size: int = int(vocab_size)
        self.constant_variables: List[str] = []
        self.binary_variables: List[str] = []
        self.identical_pairs: List[Dict[str, Union[str, float]]] = []
        try:
            self.node_connections: Dict[str, List[str]] = load_node_connections(self.data_dir)
        except RuntimeError as exc:
            logger.warning("Failed to load node connectivity: %s", exc)
            self.node_connections = {}
        self.node_variable_names: Dict[str, List[str]] = {}
        self._shared_token_sources: Dict[int, int] = {}
        self._token_decode_maps: Dict[bool, Optional[List[Dict[int, float]]]] = {
            True: None,
            False: None,
        }
        self._token_value_tables: Dict[Tuple[bool, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._cached_token_value_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None

        self.fitted: bool = False

        logger.info(
            "DataTokenizer initialised (boundary_dims=%d, equipment_dims=%d)",
            self.boundary_dims,
            self.equipment_dims,
        )

    def set_round_gap(self, value: float) -> None:
        gap = float(value)
        if gap <= 0:
            raise ValueError("round_gap must be positive")
        self.round_gap = gap
        self.round_atol = max(gap / 2.0, 1e-12)

    def set_stats_directory(self, directory: Union[str, Path], *, load_config: bool = False) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.stats_dir = path
        self.config_path = self.stats_dir / self.config_filename
        self.log_path = self.stats_dir / "tokenizer_output.txt"

        has_handler = any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", None) == str(self.log_path)
            for handler in logger.handlers
        )
        if not has_handler:
            try:
                file_handler = logging.FileHandler(self.log_path, mode="a", encoding="utf-8")
            except OSError as exc:
                logger.debug("Tokenizer log file is unavailable (%s); continuing without file logging.", exc)
            else:
                file_handler.setLevel(logging.INFO)
                formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

        if load_config:
            self._load_config_from_file()

    def _round_values(self, values: np.ndarray) -> np.ndarray:
        gap = self.round_gap
        array = np.asarray(values, dtype=np.float64)
        if gap == 0:
            return array
        return np.round(array / gap) * gap

    def _round_scalar(self, value: float) -> float:
        gap = self.round_gap
        if gap == 0:
            return float(value)
        return float(np.round(float(value) / gap) * gap)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        data: ArrayLike,
        *,
        variable_names: Optional[List[str]] = None,
        extra_data: Optional[ArrayLike] = None,
        progress: bool = False,
    ) -> None:
        combined = collect_and_stack(data, extra_data)
        num_samples, num_variables = combined.shape

        if variable_names is not None:
            if len(variable_names) != num_variables:
                raise ValueError(
                    "variable_names length (%d) does not match data dimension (%d)."
                    % (len(variable_names), num_variables)
                )
            self.variable_names = list(variable_names)
        else:
            try:
                self.variable_names = load_variable_names(self.data_dir, num_variables)
            except RuntimeError as exc:
                logger.warning("%s", exc)
                self.variable_names = [f"feature_{idx:04d}" for idx in range(num_variables)]

        self.total_dims = num_variables
        if self.boundary_dims >= self.total_dims:
            if self.boundary_dims != self.total_dims:
                logger.warning(
                    "boundary_dims (%d) exceeds total variables (%d); clamping.",
                    self.boundary_dims,
                    self.total_dims,
                )
            self.boundary_dims = self.total_dims
            self.equipment_dims = 0
        else:
            self.equipment_dims = max(self.total_dims - self.boundary_dims, 0)
        self._variable_name_to_index = {
            name: idx for idx, name in enumerate(self.variable_names)
        }
        self.node_variable_names = group_variables_by_node(self.variable_names)

        self.variable_token_configs = []
        self._token_decode_maps = {True: None, False: None}
        self._token_value_tables = {}
        self._cached_token_value_stats = None
        self._metadata_store.reset()
        self.constant_variables = []
        self.binary_variables = []
        self.identical_pairs = []
        self._shared_token_sources.clear()

        next_token_id = 0

        duplicate_equipment_map = find_identical_equipment_variables(
            combined,
            self.variable_names,
            self.boundary_dims,
            column_signature,
        )

        iterator: Iterable[int]
        if progress:
            try:
                from tqdm import tqdm  # type: ignore

                iterator = tqdm(range(num_variables), desc="Tokenising vars", unit="var")
            except Exception:  # pragma: no cover - tqdm import best effort
                logger.warning("tqdm unavailable; continuing without progress bar.")
                iterator = range(num_variables)
        else:
            iterator = range(num_variables)

        for var_idx in iterator:
            column = combined[:, var_idx]
            variable_name = self.variable_names[var_idx]

            shared_source = duplicate_equipment_map.get(var_idx)
            if shared_source is not None:
                if shared_source >= var_idx:
                    raise ValueError(
                        f"Shared token source {shared_source} must precede duplicate {var_idx}."
                    )
                canonical_cfg = self.variable_token_configs[shared_source]
                alias_tokens = VariableTokens(
                    variable_name=variable_name,
                    variable_index=var_idx,
                    constant_keys=canonical_cfg.constant_keys,
                    constant_values=canonical_cfg.constant_values,
                    constant_token_ids=canonical_cfg.constant_token_ids,
                    constant_counts=canonical_cfg.constant_counts,
                    range_lower_bounds=canonical_cfg.range_lower_bounds,
                    range_upper_bounds=canonical_cfg.range_upper_bounds,
                    range_token_ids=canonical_cfg.range_token_ids,
                    range_counts=canonical_cfg.range_counts,
                    is_binary=canonical_cfg.is_binary,
                    is_constant=canonical_cfg.is_constant,
                    sample_count=canonical_cfg.sample_count,
                    constant_map=canonical_cfg.constant_map,
                )
                self.variable_token_configs.append(alias_tokens)
                canonical_rows = self._metadata_store.rows_for_variable(shared_source)
                if not canonical_rows:
                    raise KeyError(
                        f"Missing metadata for canonical variable index {shared_source}"
                    )
                alias_metadata = self._metadata_store.alias_rows(
                    canonical_rows,
                    alias_index=var_idx,
                    alias_name=variable_name,
                    canonical_name=self.variable_names[shared_source],
                )
                self._metadata_store.register_many(alias_metadata)
                if alias_tokens.is_constant:
                    self.constant_variables.append(variable_name)
                if alias_tokens.is_binary:
                    self.binary_variables.append(variable_name)
                self._shared_token_sources[var_idx] = shared_source
                continue

            (
                variable_tokens,
                new_metadata,
                next_token_id,
            ) = self._tokenize_single_variable(
                column,
                variable_name,
                var_idx,
                next_token_id,
            )
            self.variable_token_configs.append(variable_tokens)
            self._metadata_store.register_many(new_metadata)
            if variable_tokens.is_constant:
                self.constant_variables.append(variable_name)
            if variable_tokens.is_binary:
                self.binary_variables.append(variable_name)

        self.vocab_size = next_token_id
        self.identical_pairs = build_identical_pairs(
            duplicate_equipment_map,
            self.variable_names,
            lambda idx: get_variable_group(idx, self.boundary_dims, self.equipment_dims),
        )
        for pair in self.identical_pairs:
            logger.info(
                "Identical %s variables detected: %s ↔ %s (shared tokens).",
                pair.get("group", "unknown"),
                pair.get("variable_a"),
                pair.get("variable_b"),
            )
        self.fitted = True

        logger.info(
            "Tokenizer fitted on %d samples across %d variables. "
            "Total tokens (including special): %d. Constant vars: %d. Binary vars: %d.",
            num_samples,
            num_variables,
            self.vocab_size,
            len(self.constant_variables),
            len(self.binary_variables),
        )

    def transform_to_tokens(self, data: ArrayLike) -> ArrayLike:
        if not self.fitted:
            raise RuntimeError("Tokenizer not fitted. Call fit() or load_stats().")

        arrays = collect_and_stack(data) # 转换到[-1 6712]
        if arrays.shape[-1] != self.total_dims:
            raise ValueError(
                "Expected last dimension %d, received %d."
                % (self.total_dims, arrays.shape[-1])
            )

        original_shape = arrays.shape
        flat = arrays.reshape(-1, self.total_dims)

        if flat.size == 0:
            tokens = np.empty_like(flat, dtype=np.int64)
            fast_path = True
        elif flat.shape[0] <= 8:
            token_rows = [self._encode_row_tokens(row) for row in flat]
            tokens = np.stack(token_rows, axis=0)
            fast_path = True
        else:
            tokens = np.full(flat.shape, -1, dtype=np.int64)
            fast_path = False

            for cfg in self.variable_token_configs:
                col = flat[:, cfg.variable_index]
                token_col = tokens[:, cfg.variable_index]
                assigned = np.zeros(col.shape, dtype=bool)

                if cfg.constant_keys.size > 0:
                    rounded = self._round_values(col)
                    for key, token_id in cfg.constant_map.items():
                        mask = np.isclose(rounded, key, atol=self.round_atol)
                        if np.any(mask):
                            token_col[mask] = token_id
                            assigned |= mask

                valid_mask = ~assigned
                if cfg.range_token_ids.size > 0 and np.any(valid_mask):
                    values = col[valid_mask]
                    upper_bounds = cfg.range_upper_bounds
                    if upper_bounds.size > 0:
                        idx = np.searchsorted(upper_bounds, values, side="left")
                        idx = np.clip(idx, 0, upper_bounds.size - 1)  # 异常值溢出
                        token_col[valid_mask] = cfg.range_token_ids[idx]
                        assigned[valid_mask] = True

                tokens[:, cfg.variable_index] = token_col

            for cfg in self.variable_token_configs:
                if cfg.is_constant:
                    continue
                if np.any(tokens[:, cfg.variable_index] < 0):
                    raise ValueError(
                        f"Tokenizer transform produced unset token ids for variable index {cfg.variable_index}; check value ranges."
                    )

        tokens = tokens.reshape(original_shape[:-1] + (self.total_dims,))

        if fast_path and np.any(tokens < 0):
            raise ValueError("Tokenizer fast path produced unset token ids")

        if isinstance(data, torch.Tensor):
            return torch.from_numpy(tokens).to(device=data.device, dtype=torch.long)
        return tokens

    def _encode_row_tokens(self, row: np.ndarray) -> np.ndarray:
        tokens = np.empty(self.total_dims, dtype=np.int64)
        for cfg in self.variable_token_configs:
            value = float(row[cfg.variable_index])
            tokens[cfg.variable_index] = self._encode_value(cfg, value)
        return tokens

    def _encode_value(self, cfg: VariableTokens, value: float) -> int:
        if cfg.constant_map:
            rounded = float(self._round_values(np.asarray([value], dtype=np.float64))[0])
            token_id = cfg.constant_map.get(rounded)
            if token_id is not None:
                return int(token_id)

        if cfg.range_token_ids.size > 0 and cfg.range_upper_bounds.size > 0:
            idx = np.searchsorted(cfg.range_upper_bounds, value, side="left")
            idx = max(0, min(idx, cfg.range_token_ids.size - 1))
            return int(cfg.range_token_ids[idx])

        if cfg.constant_token_ids.size > 0:
            return int(cfg.constant_token_ids[0])

        raise ValueError(
            f"无法为变量索引 {cfg.variable_index} 的值 {value} 找到 token 映射"
        )

    def _get_token_decode_maps(self, use_median: bool) -> List[Dict[int, float]]:
        cached = self._token_decode_maps.get(use_median)
        if cached is not None:
            return cached

        maps: List[Dict[int, float]] = [dict() for _ in range(self.total_dims)]

        for cfg in self.variable_token_configs:
            mapping = maps[cfg.variable_index]

            if cfg.constant_token_ids.size:
                for token_id, value in zip(cfg.constant_token_ids, cfg.constant_values):
                    mapping[int(token_id)] = float(value)

            if cfg.range_token_ids.size:
                lower = cfg.range_lower_bounds.astype(np.float64, copy=False)
                upper = cfg.range_upper_bounds.astype(np.float64, copy=False)
                if lower.size and upper.size and lower.size == upper.size:
                    centres = (lower + upper) * 0.5
                elif lower.size:
                    centres = lower
                elif upper.size:
                    centres = upper
                else:
                    centres = np.zeros_like(cfg.range_token_ids, dtype=np.float64)

                for idx, token_id in enumerate(cfg.range_token_ids):
                    if idx < centres.size:
                        mapped_value = float(centres[idx])
                    else:
                        mapped_value = float(centres[-1])
                    mapping[int(token_id)] = mapped_value

        self._token_decode_maps[use_median] = maps
        return maps

    def _get_token_value_table(
        self,
        use_median: bool,
        vocab_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.total_dims <= 0:
            raise ValueError("Tokenizer has no variables configured.")

        size = int(vocab_size) if vocab_size is not None else int(max(self.vocab_size, 0))
        if size <= 0:
            raise ValueError("Tokenizer vocab size is unavailable; ensure tokenizer is fitted.")

        key = (bool(use_median), size)
        cached = self._token_value_tables.get(key)
        if cached is not None:
            return cached

        table = np.zeros((self.total_dims, size), dtype=np.float64)
        valid = np.zeros((self.total_dims, size), dtype=bool)

        decode_maps = self._get_token_decode_maps(use_median)
        for var_idx, mapping in enumerate(decode_maps):
            if not mapping:
                continue
            for token_id, value in mapping.items():
                if 0 <= token_id < size:
                    table[var_idx, token_id] = float(value)
                    valid[var_idx, token_id] = True
                else:
                    raise ValueError(f"Token id {token_id} is out of range for variable index {var_idx}.")

        result = (table, valid)
        self._token_value_tables[key] = result
        return result

    @staticmethod
    def _restrict_top_probability_mass(
        probabilities: np.ndarray,
        *,
        keep_mass: float,
    ) -> np.ndarray:
        if probabilities.size == 0:
            return probabilities
        if not (0.0 < keep_mass <= 1.0):
            raise ValueError("keep_mass must lie in (0, 1].")

        vocab_dim = probabilities.shape[-1]
        if vocab_dim <= 1:
            return probabilities

        flat = probabilities.reshape(-1, vocab_dim)
        sorted_idx = np.argsort(flat, axis=-1)[:, ::-1]
        sorted_probs = np.take_along_axis(flat, sorted_idx, axis=-1)

        cumulative = np.cumsum(sorted_probs, axis=-1)
        threshold = float(keep_mass)
        has_threshold = cumulative >= threshold
        any_threshold = np.any(has_threshold, axis=-1)
        first_idx = np.argmax(has_threshold, axis=-1)
        first_idx = np.where(any_threshold, first_idx, vocab_dim - 1)

        idx_range = np.arange(vocab_dim, dtype=np.int64)
        keep_sorted = idx_range <= first_idx[:, None]

        trimmed = np.zeros_like(flat)
        np.put_along_axis(
            trimmed,
            sorted_idx,
            sorted_probs * keep_sorted,
            axis=-1,
        )

        renorm = trimmed.sum(axis=-1, keepdims=True)
        zero_mask = renorm.squeeze(axis=-1) <= 0
        if np.any(zero_mask):
            trimmed[zero_mask] = flat[zero_mask]
            renorm[zero_mask] = np.sum(trimmed[zero_mask], axis=-1, keepdims=True)

        renorm = np.clip(renorm, 1e-12, None)
        trimmed = trimmed / renorm
        return trimmed.reshape(probabilities.shape)

    def tokens_to_values(
        self,
        tokens: Optional[ArrayLike],
        *,
        use_median: bool = False,
        decode_mode: str = "hard",
        token_probabilities: Optional[ArrayLike] = None,
        token_logits: Optional[ArrayLike] = None,
        temperature: float = 1.0,
        variable_indices: Optional[Union[int, Iterable[int]]] = None,
    ) -> ArrayLike:
        if not self.fitted:
            raise RuntimeError("Tokenizer not fitted. Call fit() or load_stats().")

        mode = str(decode_mode).lower()
        if mode == "hard":
            if tokens is None:
                raise ValueError("tokens cannot be None when decode_mode='hard'")
            return self._tokens_to_values_hard(tokens, use_median=use_median)
        if mode == "soft":
            return self._tokens_to_values_soft(
                token_probabilities=token_probabilities,
                token_logits=token_logits,
                temperature=temperature,
                variable_indices=variable_indices,
                use_median=use_median,
            )
        raise ValueError(f"Unsupported decode_mode '{decode_mode}'. Expected 'hard' or 'soft'.")

    def _tokens_to_values_hard(
        self,
        tokens: ArrayLike,
        *,
        use_median: bool,
    ) -> ArrayLike:
        is_torch = isinstance(tokens, torch.Tensor)
        if is_torch:
            tokens_array = tokens.detach().cpu().numpy()
        else:
            tokens_array = np.asarray(tokens)

        if tokens_array.shape[-1] != self.total_dims:
            raise ValueError(
                "Expected last dimension %d, received %d."
                % (self.total_dims, tokens_array.shape[-1])
            )

        original_shape = tokens_array.shape
        flat_tokens = tokens_array.reshape(-1, self.total_dims)
        decoded_flat = np.full(flat_tokens.shape, np.nan, dtype=np.float64)

        decode_maps = self._get_token_decode_maps(use_median)

        for var_idx in range(self.total_dims):
            mapping = decode_maps[var_idx]
            if not mapping:
                raise ValueError(f"No mapping found for variable index {var_idx}.")
            column = flat_tokens[:, var_idx].astype(np.int64, copy=False)

            def _map_token(tok: int) -> float:
                return mapping.get(int(tok), 0.0)

            vectorized = np.vectorize(_map_token, otypes=[np.float64])
            decoded_flat[:, var_idx] = vectorized(column)

        decoded = decoded_flat.reshape(original_shape).astype(np.float32, copy=False)

        if is_torch:
            return torch.from_numpy(decoded).to(tokens.device, dtype=torch.float32)
        return decoded

    def _tokens_to_values_soft(
        self,
        *,
        token_probabilities: Optional[ArrayLike],
        token_logits: Optional[ArrayLike],
        temperature: float,
        variable_indices: Optional[Union[int, Iterable[int]]],
        use_median: bool,
    ) -> ArrayLike:
        if token_logits is not None and token_probabilities is not None:
            raise ValueError("Provide either token_logits or token_probabilities, not both.")

        source_tensor: Optional[torch.Tensor] = None
        if isinstance(token_logits, torch.Tensor):
            source_tensor = token_logits
        elif isinstance(token_probabilities, torch.Tensor):
            source_tensor = token_probabilities

        if token_logits is not None:
            if temperature <= 0:
                raise ValueError("temperature must be positive when using logits for soft decoding.")
            logits_array = (
                token_logits.detach().cpu().numpy()
                if isinstance(token_logits, torch.Tensor)
                else np.asarray(token_logits, dtype=np.float64)
            )
            logits_array = np.asarray(logits_array, dtype=np.float64)
            if logits_array.ndim < 2:
                raise ValueError("token_logits must be at least 2-dimensional (…, vocab).")
            shifted = logits_array / float(temperature)
            shifted -= np.max(shifted, axis=-1, keepdims=True)
            exp_values = np.exp(shifted)
            sums = np.sum(exp_values, axis=-1, keepdims=True)
            if np.any(sums <= 0):
                raise ValueError("Softmax normalization encountered zero partition sum.")
            probabilities = exp_values / sums
        else:
            if token_probabilities is None:
                raise ValueError("token_probabilities or token_logits must be provided for soft decoding.")
            if temperature != 1.0:
                raise ValueError("temperature can only be applied when token_logits are provided.")
            probabilities = (
                token_probabilities.detach().cpu().numpy()
                if isinstance(token_probabilities, torch.Tensor)
                else np.asarray(token_probabilities, dtype=np.float64)
            )
            probabilities = np.asarray(probabilities, dtype=np.float64)

        if probabilities.ndim < 2:
            raise ValueError("token probabilities must be at least 2-dimensional (…, vocab).")

        if not np.all(np.isfinite(probabilities)):
            raise ValueError("token probabilities contain non-finite values.")
        if np.any(probabilities < 0):
            raise ValueError("token probabilities must be non-negative.")

        original_shape = probabilities.shape
        vocab_dim = probabilities.shape[-1]
        if self.vocab_size > 0 and vocab_dim != self.vocab_size:
            raise ValueError(
                "Probability vocab size mismatch: expected %d, received %d."
                % (self.vocab_size, vocab_dim)
            )

        target_shape = original_shape[:-1]

        if probabilities.ndim == 2:
            probabilities = probabilities[:, np.newaxis, :]
        num_vars_in_input = probabilities.shape[-2]

        if variable_indices is None:
            if num_vars_in_input != self.total_dims:
                raise ValueError(
                    "token probabilities missing variable axis; provide variable_indices explicitly."
                )
            indices = np.arange(self.total_dims, dtype=np.int64)
        else:
            if isinstance(variable_indices, (int, np.integer)):
                indices_array = np.array([int(variable_indices)], dtype=np.int64)
            else:
                indices_array = np.asarray(list(variable_indices), dtype=np.int64)
            if indices_array.ndim != 1:
                raise ValueError("variable_indices must be a 1-D collection of integers.")
            if indices_array.size != num_vars_in_input:
                raise ValueError(
                    "variable_indices size (%d) does not match probability tensor variable dimension (%d)."
                    % (indices_array.size, num_vars_in_input)
                )
            if np.any((indices_array < 0) | (indices_array >= self.total_dims)):
                raise ValueError("variable_indices contains out-of-range variable ids.")
            indices = indices_array

        sums = np.sum(probabilities, axis=-1)
        if np.any(np.abs(sums - 1.0) > 1e-3):
            raise ValueError("token probabilities must sum to 1 along the vocab dimension.")

        probabilities = self._restrict_top_probability_mass(
            probabilities,
            keep_mass=0.99,
        )

        value_table, valid_mask = self._get_token_value_table(use_median, vocab_dim)
        selected_values = value_table[indices]
        selected_valid = valid_mask[indices]

        broadcast_shape = (1,) * (probabilities.ndim - 2) + selected_values.shape
        expanded_values = selected_values.reshape(broadcast_shape)
        expanded_valid = selected_valid.reshape(broadcast_shape)

        if np.any(~expanded_valid):
            invalid_mass = np.sum(
                probabilities * (~expanded_valid).astype(np.float64),
                axis=-1,
            )
            if np.max(invalid_mass) > 1e-6:
                logger.warning(
                    "Soft decoding received probability mass on undefined tokens; renormalising (max mass %.4f).",
                    float(np.max(invalid_mass)),
                )
            probabilities = np.where(expanded_valid, probabilities, 0.0)
            renorm = np.sum(probabilities, axis=-1, keepdims=True)
            if np.any(renorm <= 0):
                raise ValueError("All probability mass was placed on undefined tokens.")
            probabilities = probabilities / renorm

        expectations = np.sum(probabilities * expanded_values, axis=-1)
        expectations = expectations.astype(np.float32, copy=False)
        expectations = expectations.reshape(target_shape)

        if source_tensor is not None:
            device = source_tensor.device
            return torch.from_numpy(expectations).to(device=device, dtype=torch.float32)
        return expectations

    def _current_config_dict(self) -> Dict[str, Union[int, float, str]]:
        return {
            "version": self.CONFIG_VERSION,
            "boundary_dims": int(self.boundary_dims),
            "equipment_dims": int(self.equipment_dims),
            "constant_freq_threshold": float(self.constant_freq_threshold),
            "constant_variable_threshold": float(self.constant_variable_threshold),
            "quantile_step": float(self.quantile_step),
            "quantile_method": str(self.quantile_method),
            "range_gap_epsilon": float(self.range_gap_epsilon),
            "round_gap": float(self.round_gap),
        }

    def _save_config_to_file(self) -> None:
        config = self._current_config_dict()
        try:
            with self.config_path.open("w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2, sort_keys=True)
        except Exception as exc:
            logger.warning("Failed to write tokenizer config %s: %s", self.config_path, exc)

    def _load_config_from_file(self) -> None:
        if not self.config_path.exists():
            return
        try:
            with self.config_path.open("r", encoding="utf-8") as handle:
                config_data = json.load(handle)
        except Exception as exc:
            logger.warning(
                "Failed to load tokenizer config %s: %s",
                self.config_path,
                exc,
            )
            return

        updated_keys = []
        if "constant_freq_threshold" in config_data:
            self.constant_freq_threshold = float(config_data["constant_freq_threshold"])
            updated_keys.append("constant_freq_threshold")
        if "constant_variable_threshold" in config_data:
            self.constant_variable_threshold = float(
                config_data["constant_variable_threshold"]
            )
            updated_keys.append("constant_variable_threshold")
        boundary_override = config_data.get("boundary_dims")
        if boundary_override is not None and not self._boundary_dims_explicit:
            try:
                boundary_value = int(boundary_override)
                if boundary_value < 0:
                    raise ValueError("boundary_dims must be non-negative")
                self.boundary_dims = boundary_value
                updated_keys.append("boundary_dims")
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid boundary_dims value in %s: %s",
                    self.config_path,
                    boundary_override,
                )
        equipment_override = config_data.get("equipment_dims")
        if equipment_override is not None and not self._equipment_dims_explicit:
            try:
                equipment_value = int(equipment_override)
                if equipment_value < 0:
                    raise ValueError("equipment_dims must be non-negative")
                self.equipment_dims = equipment_value
                updated_keys.append("equipment_dims")
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid equipment_dims value in %s: %s",
                    self.config_path,
                    equipment_override,
                )
        if "quantile_step" in config_data:
            self.quantile_step = float(config_data["quantile_step"])
            updated_keys.append("quantile_step")
        if "quantile_method" in config_data:
            self.quantile_method = str(config_data["quantile_method"])
            updated_keys.append("quantile_method")
        if "range_gap_epsilon" in config_data:
            self.range_gap_epsilon = float(config_data["range_gap_epsilon"])
            updated_keys.append("range_gap_epsilon")
        if "round_gap" in config_data:
            self.set_round_gap(float(config_data["round_gap"]))
            updated_keys.append("round_gap")

        self.total_dims = self.boundary_dims + self.equipment_dims

        if updated_keys:
            logger.info(
                "Loaded tokenizer config overrides from %s: %s",
                self.config_path,
                ", ".join(updated_keys),
            )

    def save_stats(
        self,
        *,
        stats_filename: Optional[str] = None,
    ) -> str:
        if not self.fitted:
            raise RuntimeError("Tokenizer not fitted; nothing to save.")

        stats_filename = stats_filename or "token_stats.csv"
        stats_path = self.stats_dir / stats_filename

        df = pd.DataFrame(self.token_metadata)
        if "token_id" in df.columns:
            df = df.sort_values("token_id", kind="mergesort").reset_index(drop=True)
        df.to_csv(stats_path, index=False, float_format="%.10f")

        self._save_config_to_file()
        logger.info("Tokenizer stats saved to %s", stats_path)
        logger.info("Tokenizer config saved to %s", self.config_path)
        return str(stats_path)

    def load_stats(
        self,
        *,
        stats_filename: Optional[str] = None,
    ) -> bool:
        stats_filename = stats_filename or "token_stats.csv"
        stats_path = self.stats_dir / stats_filename

        if not stats_path.exists():
            logger.error("Tokenizer stats file not found: %s", stats_path)
            return False

        try:
            df = pd.read_csv(stats_path)
        except Exception as exc:
            logger.error("Failed to load tokenizer CSV %s: %s", stats_path, exc)
            return False

        if df.empty:
            logger.error("Tokenizer stats CSV %s is empty.", stats_path)
            return False

        if "variable_index" not in df.columns or "variable_name" not in df.columns:
            raise ValueError("Tokenizer stats CSV missing required columns.")

        if "duplicate_of" not in df.columns:
            df["duplicate_of"] = None
        else:
            df["duplicate_of"] = df["duplicate_of"].apply(
                lambda value: value if isinstance(value, str) and value else None
            )

        valid_rows = df[df["variable_index"] >= 0].copy()
        if valid_rows.empty:
            raise ValueError("Tokenizer stats CSV does not contain variable entries.")

        variable_indices = sorted(int(idx) for idx in valid_rows["variable_index"].unique())
        max_index = max(variable_indices)

        variable_names: List[Optional[str]] = [None] * (max_index + 1)
        token_configs: List[VariableTokens] = []
        canonical_by_name: Dict[str, VariableTokens] = {}

        for var_idx in variable_indices:
            group = valid_rows[valid_rows["variable_index"] == var_idx]
            if group.empty:
                continue
            variable_name = str(group["variable_name"].iloc[0])
            duplicate_values = {val for val in group["duplicate_of"] if isinstance(val, str) and val}
            canonical_name = duplicate_values.pop() if duplicate_values else None
            sample_count = int(group["variable_sample_count"].dropna().iloc[0]) if "variable_sample_count" in group.columns and not group["variable_sample_count"].dropna().empty else int(group["count"].sum())
            is_constant_flag = bool(group["is_constant"].iloc[0]) if "is_constant" in group.columns else False
            is_binary_flag = bool(group["is_binary"].iloc[0]) if "is_binary" in group.columns else False

            if canonical_name:
                canonical_tokens = canonical_by_name.get(canonical_name)
                if canonical_tokens is None:
                    raise ValueError(
                        f"Tokenizer stats reference missing canonical variable '{canonical_name}' for duplicate '{variable_name}'."
                    )
                alias_tokens = VariableTokens(
                    variable_name=variable_name,
                    variable_index=var_idx,
                    constant_keys=canonical_tokens.constant_keys,
                    constant_values=canonical_tokens.constant_values,
                    constant_token_ids=canonical_tokens.constant_token_ids,
                    constant_counts=canonical_tokens.constant_counts,
                    range_lower_bounds=canonical_tokens.range_lower_bounds,
                    range_upper_bounds=canonical_tokens.range_upper_bounds,
                    range_token_ids=canonical_tokens.range_token_ids,
                    range_counts=canonical_tokens.range_counts,
                    is_binary=canonical_tokens.is_binary,
                    is_constant=canonical_tokens.is_constant,
                    sample_count=sample_count,
                    constant_map=canonical_tokens.constant_map,
                )
                token_configs.append(alias_tokens)
                canonical_by_name[variable_name] = alias_tokens
            else:
                constant_rows = group[group["token_type"].isin(["constant", "binary"])].copy() if "token_type" in group.columns else group.iloc[0:0]
                range_rows = group[group["token_type"] == "range"].copy() if "token_type" in group.columns else group.iloc[0:0]

                constant_stub = constant_rows[constant_rows["token_id"] < 0]
                constant_token_rows = constant_rows[constant_rows["token_id"] >= 0].sort_values("token_id")

                if is_constant_flag and not constant_token_rows.empty and constant_stub.empty:
                    constant_stub = constant_token_rows.iloc[:0]

                if is_constant_flag and not constant_stub.empty and constant_token_rows.empty:
                    dominant_row = constant_stub.iloc[0]
                    dominant_value = float(dominant_row["value_average"])
                    dominant_key = self._round_scalar(dominant_value)
                    dominant_count = int(dominant_row["count"])
                    constant_keys = np.array([dominant_key], dtype=np.float64)
                    constant_values = np.array([dominant_value], dtype=np.float64)
                    constant_counts = np.array([dominant_count], dtype=np.int64)
                    constant_token_ids = np.empty(0, dtype=np.int64)
                    constant_map: Dict[float, int] = {}
                else:
                    constant_values_series = constant_token_rows["value_average"] if "value_average" in constant_token_rows.columns else pd.Series(dtype=float)
                    constant_keys = np.asarray(
                        [self._round_scalar(float(val)) for val in constant_values_series],
                        dtype=np.float64,
                    )
                    constant_values = np.asarray(
                        [float(val) for val in constant_values_series],
                        dtype=np.float64,
                    )
                    constant_counts = np.asarray(
                        constant_token_rows["count"].astype(np.int64),
                        dtype=np.int64,
                    )
                    constant_token_ids = np.asarray(
                        constant_token_rows["token_id"].astype(np.int64),
                        dtype=np.int64,
                    )
                    constant_map = {
                        self._round_scalar(float(val)): int(token)
                        for val, token in zip(constant_values, constant_token_ids)
                    }

                if not range_rows.empty:
                    range_rows = range_rows.sort_values("token_id")

                range_lower_bounds = np.asarray(
                    range_rows["range_start"], dtype=np.float64
                ) if not range_rows.empty else np.empty(0, dtype=np.float64)
                range_upper_bounds = np.asarray(
                    range_rows["range_end"], dtype=np.float64
                ) if not range_rows.empty else np.empty(0, dtype=np.float64)
                range_counts = np.asarray(
                    range_rows["count"].astype(np.int64), dtype=np.int64
                ) if not range_rows.empty else np.empty(0, dtype=np.int64)
                range_token_ids = np.asarray(
                    range_rows["token_id"].astype(np.int64), dtype=np.int64
                ) if not range_rows.empty else np.empty(0, dtype=np.int64)

                variable_tokens = VariableTokens(
                    variable_name=variable_name,
                    variable_index=var_idx,
                    constant_keys=constant_keys,
                    constant_values=constant_values,
                    constant_token_ids=constant_token_ids,
                    constant_counts=constant_counts,
                    range_lower_bounds=range_lower_bounds,
                    range_upper_bounds=range_upper_bounds,
                    range_token_ids=range_token_ids,
                    range_counts=range_counts,
                    is_binary=is_binary_flag,
                    is_constant=is_constant_flag,
                    sample_count=sample_count,
                    constant_map=constant_map,
                )
                token_configs.append(variable_tokens)
                canonical_by_name[variable_name] = variable_tokens

            if var_idx >= len(variable_names):
                variable_names.extend([None] * (var_idx - len(variable_names) + 1))
            variable_names[var_idx] = variable_name

        if any(name is None for name in variable_names):
            missing = [idx for idx, name in enumerate(variable_names) if name is None]
            raise ValueError(
                f"Tokenizer stats missing entries for variable indices: {missing[:5]}"
            )

        self.variable_names = [str(name) for name in variable_names]
        self.total_dims = len(self.variable_names)
        if self.boundary_dims >= self.total_dims:
            if self.boundary_dims != self.total_dims:
                logger.warning(
                    "boundary_dims (%d) exceeds total variables (%d) when loading; clamping.",
                    self.boundary_dims,
                    self.total_dims,
                )
            self.boundary_dims = self.total_dims
            self.equipment_dims = 0
        else:
            self.equipment_dims = max(self.total_dims - self.boundary_dims, 0)

        self._variable_name_to_index = {
            name: idx for idx, name in enumerate(self.variable_names)
        }
        self.node_variable_names = group_variables_by_node(self.variable_names)

        self.variable_token_configs = token_configs
        self._token_decode_maps = {True: None, False: None}
        self._token_value_tables = {}

        positive_token_ids = df.loc[df["token_id"] >= 0, "token_id"]
        self.vocab_size = (
            int(positive_token_ids.max()) + 1 if not positive_token_ids.empty else 0
        )

        if "variable_group" not in df.columns:
            df["variable_group"] = df["variable_index"].apply(
                lambda idx: compute_group_from_index(
                    idx, self.boundary_dims, self.equipment_dims
                )
            )
        else:
            df.loc[df["variable_index"] < 0, "variable_group"] = "special"

        records = df.to_dict("records")
        self._metadata_store.reset()
        self._cached_token_value_stats = None
        for row in records:
            duplicate_of = row.get("duplicate_of")
            if duplicate_of is None or (
                isinstance(duplicate_of, float) and np.isnan(duplicate_of)
            ):
                row["duplicate_of"] = None
            elif isinstance(duplicate_of, str) and not duplicate_of:
                row["duplicate_of"] = None
            self._metadata_store.register(dict(row))

        self.constant_variables = [
            cfg.variable_name for cfg in self.variable_token_configs if cfg.is_constant
        ]
        self.binary_variables = [
            cfg.variable_name for cfg in self.variable_token_configs if cfg.is_binary
        ]
        self._shared_token_sources = {}
        duplicates: Dict[int, int] = {}
        seen_pairs: set = set()
        for var_idx, rows in self._metadata_store.by_variable.items():
            if var_idx < 0 or not rows:
                continue
            duplicate_of = rows[0].get("duplicate_of")
            if isinstance(duplicate_of, float) and np.isnan(duplicate_of):
                duplicate_of = None
            if isinstance(duplicate_of, str) and duplicate_of:
                canonical_idx = self._variable_name_to_index.get(duplicate_of)
                if canonical_idx is None:
                    continue
                self._shared_token_sources[var_idx] = canonical_idx
                duplicates[var_idx] = canonical_idx
                pair_key = tuple(sorted((canonical_idx, var_idx)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
        self.identical_pairs = build_identical_pairs(
            duplicates,
            self.variable_names,
            lambda idx: get_variable_group(idx, self.boundary_dims, self.equipment_dims),
        )
        self.fitted = True
        if not self.config_path.exists():
            self._save_config_to_file()
        logger.info("Tokenizer stats loaded from %s", stats_path)
        return True

    def get_stats_summary(self) -> Dict[str, Union[int, float]]:
        if not self.fitted:
            raise RuntimeError("Tokenizer statistics unavailable.")
        total_vars = self.total_dims
        non_special_tokens = max(self.vocab_size, 0)
        tokens_per_var = (
            non_special_tokens / total_vars if total_vars > 0 else float("nan")
        )
        return {
            "vocab_size": self.vocab_size,
            "num_variables": total_vars,
            "constant_variables": len(self.constant_variables),
            "binary_variables": len(self.binary_variables),
            "avg_tokens_per_variable": tokens_per_var,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _tokenize_single_variable(
        self,
        column: np.ndarray,
        variable_name: str,
        variable_index: int,
        next_token_id: int,
    ) -> Tuple[VariableTokens, List[MetadataRow], int]:
        metadata_rows: List[MetadataRow] = []

        values = column
        sample_count = int(values.size)

        if sample_count == 0:
            variable_tokens = VariableTokens(
                variable_name=variable_name,
                variable_index=variable_index,
                constant_keys=np.empty(0, dtype=np.float64),
                constant_values=np.empty(0, dtype=np.float64),
                constant_token_ids=np.empty(0, dtype=np.int64),
                constant_counts=np.empty(0, dtype=np.int64),
                range_lower_bounds=np.empty(0, dtype=np.float64),
                range_upper_bounds=np.empty(0, dtype=np.float64),
                range_token_ids=np.empty(0, dtype=np.int64),
                range_counts=np.empty(0, dtype=np.int64),
                is_binary=False,
                is_constant=True,
                sample_count=0,
                constant_map={},
            )
            logger.warning("Variable %s has no observations.", variable_name)
            return variable_tokens, metadata_rows, next_token_id

        rounded = self._round_values(values)
        unique_keys, inverse_indices, counts = np.unique(
            rounded, return_inverse=True, return_counts=True
        )
        if unique_keys.size:
            order = np.argsort(inverse_indices, kind="mergesort")
            sorted_values = values[order]
            unique_medians = np.empty_like(unique_keys, dtype=np.float64)
            start = 0
            for idx, count in enumerate(counts):
                end = start + int(count)
                chunk = sorted_values[start:end]
                if chunk.size:
                    unique_medians[idx] = float(np.median(chunk))
                else:
                    unique_medians[idx] = float(unique_keys[idx])
                start = end
        else:
            unique_medians = np.empty(0, dtype=np.float64)
        sums = np.zeros_like(unique_keys, dtype=np.float64)
        np.add.at(sums, inverse_indices, values)
        means = sums / counts
        proportions = counts / sample_count if sample_count > 0 else np.zeros_like(counts)

        is_binary_flag = is_binary(unique_keys, self.round_atol)
        is_constant_flag = bool(
            unique_keys.size == 1
            or (counts.max() / sample_count) >= self.constant_variable_threshold
        )

        dominant_idx = int(np.argmax(counts)) if counts.size else 0
        if is_constant_flag:
            dominant_key = float(unique_keys[dominant_idx]) if unique_keys.size else 0.0
            dominant_value = float(means[dominant_idx]) if means.size else 0.0
            dominant_count = int(counts[dominant_idx]) if counts.size else 0
            constant_keys = np.array([dominant_key], dtype=np.float64)
            constant_values = np.array([dominant_value], dtype=np.float64)
            constant_counts = np.array([dominant_count], dtype=np.int64)
        else:
            if is_binary_flag:
                constant_mask = np.isin(unique_keys, unique_keys)
            else:
                constant_mask = proportions >= self.constant_freq_threshold
            constant_keys = unique_keys[constant_mask]
            constant_values = means[constant_mask]
            constant_counts = counts[constant_mask]
            constant_keys = constant_keys.astype(np.float64, copy=False)
            constant_values = constant_values.astype(np.float64, copy=False)
            constant_counts = constant_counts.astype(np.int64, copy=False)

        residual_count_est = max(sample_count - int(constant_counts.sum()), 0)
        residual_pct = (
            (residual_count_est / sample_count) * 100.0 if sample_count else 0.0
        )
        if self.round_gap >= 1:
            decimals = 2
        else:
            decimals = int(np.ceil(-np.log10(self.round_gap))) + 2
        decimals = max(0, min(12, decimals))
        if constant_keys.size > 5:
            preview_keys = np.round(constant_keys[:5], decimals).tolist()
            keys_repr = f"{preview_keys}..."
        else:
            keys_repr = np.round(constant_keys, decimals).tolist()
        logger.info(
            "Variable %s constant_keys=%s (count=%d, residual≈%.4f%% of %d)",
            variable_name,
            keys_repr,
            int(constant_keys.size),
            residual_pct,
            sample_count,
        )

        group_name = get_variable_group(
            variable_index, self.boundary_dims, self.equipment_dims
        )
        if is_constant_flag:
            dominant_value = float(constant_values[0]) if constant_values.size else np.nan
            dominant_key = float(constant_keys[0]) if constant_keys.size else float(dominant_value)
            dominant_median = (
                float(unique_medians[dominant_idx]) if unique_medians.size else float(dominant_value)
            )
            token_id = next_token_id
            next_token_id += 1
            constant_token_ids = np.array([token_id], dtype=np.int64)
            rounded_key = self._round_scalar(dominant_key)
            constant_map = {rounded_key: token_id}
            half_gap = self.round_gap / 2.0
            lower_bound = dominant_key - half_gap
            upper_bound = dominant_key + half_gap
            shrink = min(self.range_gap_epsilon, half_gap)
            upper_bound = max(lower_bound, upper_bound - shrink)
            coverage = (
                float(constant_counts.sum()) / sample_count if sample_count else 0.0
            )
            metadata_rows.append(
                {
                    "token_id": token_id,
                    "variable_index": variable_index,
                    "variable_name": variable_name,
                    "token_type": "constant",
                    "value_average": float(dominant_value),
                    "value_median": float(dominant_median),
                    "range_start": lower_bound,
                    "range_end": upper_bound,
                    "count": int(constant_counts.sum()),
                    "probability": float(coverage),
                    "variable_sample_count": sample_count,
                    "is_constant": True,
                    "is_binary": False,
                    "variable_group": group_name,
                    "duplicate_of": None,
                }
            )
        else:
            constant_token_ids = np.empty(constant_keys.size, dtype=np.int64)
            constant_map = {}
            constant_medians = (
                unique_medians[constant_mask]
                if unique_medians.size and constant_mask.size
                else np.full(constant_keys.size, np.nan, dtype=np.float64)
            )
            for idx, (key, value, count) in enumerate(
                zip(constant_keys, constant_values, constant_counts)
            ):
                token_id = next_token_id
                next_token_id += 1
                constant_token_ids[idx] = token_id
                center = float(key)
                constant_map[self._round_scalar(center)] = token_id
                half_gap = self.round_gap / 2.0
                lower_bound = center - half_gap
                upper_bound = center + half_gap
                shrink = min(self.range_gap_epsilon, half_gap)
                upper_bound = max(lower_bound, upper_bound - shrink)
                median_raw = constant_medians[idx] if constant_medians.size else np.nan
                median_value = float(value) if np.isnan(median_raw) else float(median_raw)
                metadata_rows.append(
                    {
                        "token_id": token_id,
                        "variable_index": variable_index,
                        "variable_name": variable_name,
                        "token_type": "binary" if is_binary_flag else "constant",
                        "value_average": float(value),
                        "value_median": median_value,
                        "range_start": lower_bound,
                        "range_end": upper_bound,
                        "count": int(count),
                        "probability": float(count / sample_count),
                        "variable_sample_count": sample_count,
                        "is_constant": bool(is_constant_flag),
                        "is_binary": bool(is_binary_flag),
                        "variable_group": group_name,
                        "duplicate_of": None,
                    }
                )

        residual_count_local = 0
        residual_pct_local = 0.0

        if is_constant_flag:
            residual_count = sample_count - int(constant_counts.sum())
            residual_count_local = residual_count
            residual_pct_local = (
                (residual_count / sample_count) * 100.0 if sample_count else 0.0
            )
            range_lower_bounds = np.empty(0, dtype=np.float64)
            range_upper_bounds = np.empty(0, dtype=np.float64)
            range_counts = np.empty(0, dtype=np.int64)
            range_token_ids = np.empty(0, dtype=np.int64)
            range_metadata: List[MetadataRow] = []
            if residual_count > 0:
                logger.info(
                    "Constant variable %s: ignoring %d residual samples (~%.4f%%) below dominance threshold.",
                    variable_name,
                    residual_count,
                    residual_pct_local,
                )
        else:
            non_constant_mask = ~np.isin(rounded, constant_keys)
            non_constant_values = values[non_constant_mask]

            (
                range_lower_bounds,
                range_upper_bounds,
                range_counts,
                range_token_ids,
                next_token_id,
                range_metadata,
            ) = build_range_tokens(
                non_constant_values=non_constant_values,
                sample_count=sample_count,
                variable_index=variable_index,
                variable_name=variable_name,
                start_token_id=next_token_id,
                is_constant=is_constant_flag,
                is_binary_flag=is_binary_flag,
                min_fraction=self.quantile_step,
                group_name=group_name,
                constant_keys=constant_keys,
                gap_epsilon=self.range_gap_epsilon,
            )
            metadata_rows.extend(range_metadata)

        variable_tokens = VariableTokens(
            variable_name=variable_name,
            variable_index=variable_index,
            constant_keys=constant_keys.astype(np.float64, copy=False),
            constant_values=constant_values.astype(np.float64, copy=False),
            constant_token_ids=constant_token_ids,
            constant_counts=constant_counts.astype(np.int64, copy=False),
            range_lower_bounds=range_lower_bounds,
            range_upper_bounds=range_upper_bounds,
            range_token_ids=range_token_ids,
            range_counts=range_counts,
            is_binary=is_binary_flag,
            is_constant=is_constant_flag,
            sample_count=sample_count,
            constant_map=constant_map,
        )

        if is_constant_flag:
            coverage_pct = (
                (float(constant_counts.sum()) / sample_count) * 100.0 if sample_count else 0.0
            )
            if residual_count_local > 0:
                logger.info(
                    "Constant variable detected: %s (value≈%.6f, coverage=%.2f%%, residuals ignored=%.4f%%)",
                    variable_name,
                    float(constant_values[0]),
                    coverage_pct,
                    residual_pct_local,
                )
            else:
                logger.info(
                    "Constant variable detected: %s (value≈%.6f, coverage=%.2f%%)",
                    variable_name,
                    float(constant_values[0]),
                    coverage_pct,
                )
        elif is_binary_flag:
            logger.info("Binary flag variable detected: %s", variable_name)

        return variable_tokens, metadata_rows, next_token_id

    @property
    def token_metadata(self) -> List[MetadataRow]:
        return self._metadata_store.rows

    def get_token_value_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return per-token medians and half widths derived from metadata."""

        if not self.fitted:
            raise RuntimeError("Tokenizer statistics unavailable; call load() first.")

        cached = self._cached_token_value_stats
        if cached is not None:
            return cached

        vocab = int(self.vocab_size)
        if vocab <= 0:
            raise ValueError("Tokenizer vocab size must be positive before extracting value stats.")

        medians = np.zeros(vocab, dtype=np.float64)
        half_widths = np.zeros(vocab, dtype=np.float64)
        filled = np.zeros(vocab, dtype=bool)

        rows = self.token_metadata
        if not rows:
            raise ValueError("Tokenizer metadata is empty; cannot derive value statistics.")

        for row in rows:
            token_id = int(row.get("token_id", -1))
            if token_id < 0 or token_id >= vocab:
                continue
            range_start = row.get("range_start")
            range_end = row.get("range_end")
            value_median = row.get("value_median")
            if range_start is None or range_end is None or value_median is None:
                raise ValueError(
                    f"Metadata for token_id={token_id} is missing range_start, range_end, or value_median"
                )
            lower = float(range_start)
            upper = float(range_end)
            half = 0.5 * (upper - lower)
            if not np.isfinite(half) or half < 0:
                raise ValueError(
                    f"Invalid half width {half} for token_id={token_id}; range_start={lower}, range_end={upper}"
                )
            # constant / binary tokens may have zero width; clamp tiny spans to zero
            if half < 1e-4:
                half = 0.0
            median_val = float(value_median)
            if filled[token_id]:
                if not (np.isclose(medians[token_id], median_val) and np.isclose(half_widths[token_id], half)):
                    raise ValueError(f"Conflicting metadata entries detected for token_id={token_id}")
                continue
            medians[token_id] = median_val
            half_widths[token_id] = half
            filled[token_id] = True

        if not np.all(filled):
            missing = np.nonzero(~filled)[0]
            if missing.size:
                raise ValueError(
                    f"Token value stats incomplete; missing entries for token ids: {missing[:5].tolist()}"
                )

        self._cached_token_value_stats = (medians, half_widths)
        return self._cached_token_value_stats
