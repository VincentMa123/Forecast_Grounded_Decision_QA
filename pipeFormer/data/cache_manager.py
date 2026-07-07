import os
import hashlib
import pickle
import math
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm
import json
import pyarrow as pa
import pyarrow.parquet as pq
# 使用相对导入避免循环导入问题
from .processor import DataProcessor
from .tokenizer_save import DataTokenizer
from .topology_attention_index import TopologyAttentionIndexBuilder
logger = logging.getLogger(__name__)

class CacheManager:
    """
    高效缓存管理器，用于预处理和快速加载fluid动力学数据。

    Features:
    - Parquet格式存储，快速读写
    - 数据完整性验证
    - 版本控制和自动更新检测
    - 内存优化的批量处理
    - 存储原始样本数据，动态创建序列
    """

    DEFAULT_BOUNDARY_DIMS = 538

    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        static_dir: Optional[str] = None,
        variable_names: Optional[List[str]] = None,
    ):
        """
        初始化CacheManager。

        Args:
            data_dir: 原始数据目录路径
            cache_dir: 缓存目录路径，默认为data_dir/cache
            static_dir: 静态拓扑/变量文件目录，默认 data/static/full
            variable_names: 可选的变量名列表，覆盖静态目录中的配置
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.static_dir = self._resolve_static_dir(static_dir)

        # 变量选择配置
        self._full_var_to_index: Optional[Dict[str, int]] = None
        self.active_variable_names: List[str] = []
        self.active_global_indices: List[int] = []
        self.active_boundary_dims: int = self.DEFAULT_BOUNDARY_DIMS
        self.active_equipment_dims: int = 0

        self._refresh_active_schema(variable_names)

        # 缓存文件路径
        self.train_cache_path = self.cache_dir / "train_sequences.parquet"
        self.val_cache_path = self.cache_dir / "val_sequences.parquet"
        self.train_tokens_path = self.cache_dir / "train_tokens.parquet"
        self.val_tokens_path = self.cache_dir / "val_tokens.parquet"
        self.metadata_cache_path = self.cache_dir / "metadata.json"
        self.mask_cache_path = self.cache_dir / "prediction_mask.npy"

        # 当前token缓存配置
        self.token_dtype: Optional[str] = None

        # 数据版本信息
        self.data_hash = self._compute_data_hash()
        self.variable_hash = self._hash_variable_names(self.active_variable_names)

        logger.info(
            "CacheManager initialized: cache_dir=%s, static_dir=%s, dims=%d "
            "(boundary=%d, equipment=%d), data_hash=%s, variable_hash=%s",
            self.cache_dir,
            self.static_dir,
            len(self.active_variable_names),
            self.active_boundary_dims,
            self.active_equipment_dims,
            self.data_hash[:8],
            self.variable_hash[:8],
        )

    def _compute_data_hash(self) -> str:
        """计算数据目录的hash值用于版本控制。"""
        hash_md5 = hashlib.md5()

        # 计算dataset目录结构的hash
        dataset_dir = self.data_dir / "dataset"
        if dataset_dir.exists():
            for root, dirs, files in os.walk(dataset_dir):
                dirs.sort()  # 确保一致的遍历顺序
                for file in sorted(files):
                    if file.endswith('.csv'):
                        file_path = Path(root) / file
                        # 只考虑文件修改时间和大小，不读取内容以提高速度
                        stat = file_path.stat()
                        hash_md5.update(f"{file_path.relative_to(dataset_dir)}:{stat.st_mtime}:{stat.st_size}".encode())

        return hash_md5.hexdigest()

    def _resolve_static_dir(self, static_dir: Optional[str]) -> Path:
        """Resolve the static directory containing topology artifacts."""
        if static_dir is None:
            candidate = self.data_dir / "static" / "full"
        else:
            candidate = Path(static_dir)
        return candidate.resolve()

    def _hash_variable_names(self, variable_names: List[str]) -> str:
        """Compute a stable hash for the active variable ordering."""
        hasher = hashlib.md5()
        for name in variable_names:
            hasher.update(name.encode("utf-8"))
            hasher.update(b",")
        return hasher.hexdigest()

    def _percentage_close(self, cached_value: Optional[float], target_value: float) -> bool:
        if cached_value is None:
            return False
        return math.isclose(float(cached_value), float(target_value), rel_tol=1e-6, abs_tol=1e-6)

    def _refresh_active_schema(self, variable_names_override: Optional[List[str]]) -> None:
        """
        Update active variable configuration (names, global indices, dim breakdown).
        """
        var_names, global_indices = self._load_variable_schema(variable_names_override)
        self.active_variable_names = var_names
        self.active_global_indices = global_indices
        boundary_count = sum(1 for idx in global_indices if idx < self.DEFAULT_BOUNDARY_DIMS)
        self.active_boundary_dims = boundary_count
        self.active_equipment_dims = max(len(var_names) - boundary_count, 0)

    def _load_variable_schema(
        self,
        variable_names_override: Optional[List[str]],
    ) -> Tuple[List[str], List[int]]:
        """
        Load variable names and their corresponding global indices.
        """
        variable_names: Optional[List[str]] = None
        global_indices: Optional[List[int]] = None

        # 1) Prefer explicit override
        if variable_names_override:
            variable_names = list(variable_names_override)

        # 2) Attempt to load from static directory mapping
        mapping_path = self.static_dir / "index_variable_mapping.csv"
        if mapping_path.exists():
            try:
                mapping_df = pd.read_csv(mapping_path)
                if "index" in mapping_df.columns:
                    mapping_df = mapping_df.sort_values("index")
                if variable_names is None and "variable_name" in mapping_df.columns:
                    variable_names = mapping_df["variable_name"].astype(str).tolist()
                if "global_index" in mapping_df.columns:
                    global_indices = mapping_df["global_index"].astype(int).tolist()
            except Exception as exc:
                logger.warning("Failed to load static mapping from %s: %s", mapping_path, exc)

        # 3) Fallback to global variable names if still missing
        if variable_names is None:
            full_map = self._get_full_variable_mapping()
            ordered = sorted(full_map.items(), key=lambda item: item[1])
            variable_names = [name for name, _ in ordered]
            global_indices = [idx for _, idx in ordered]

        # 4) Ensure global indices available
        if global_indices is None:
            full_map = self._get_full_variable_mapping()
            missing = []
            resolved_indices: List[int] = []
            for name in variable_names:
                idx = full_map.get(name)
                if idx is None:
                    missing.append(name)
                else:
                    resolved_indices.append(idx)
            if missing:
                raise KeyError(
                    f"Missing global indices for variables: {missing[:5]} (total {len(missing)})"
                )
            global_indices = resolved_indices

        if len(variable_names) != len(global_indices):
            raise ValueError(
                f"Variable names ({len(variable_names)}) and global indices "
                f"({len(global_indices)}) length mismatch."
            )

        return variable_names, global_indices

    def _get_full_variable_mapping(self) -> Dict[str, int]:
        """Compute or retrieve the full graph variable->index mapping."""
        if self._full_var_to_index is None:
            builder = TopologyAttentionIndexBuilder(str(self.data_dir), static_dir=str(self.static_dir))
            self._full_var_to_index = builder._build_variable_index_mapping()
        return self._full_var_to_index

    def _is_cache_valid(self, require_tokens: bool = True) -> bool:
        """
        检查缓存是否有效。

        Returns:
            True if cache is valid, False otherwise
        """
        if not self.metadata_cache_path.exists():
            return False

        try:
            with open(self.metadata_cache_path, 'r') as f:
                metadata = json.load(f)

            # 检查数据hash是否匹配，这里说的是csv的hash，但是我多了不少训练的csv
            cached_hash = metadata.get('data_hash', '')
            if False and (cached_hash != self.data_hash):
                logger.info(f"Cache invalid: data hash mismatch {cached_hash[:8]} vs {self.data_hash[:8]}")
                return False

            cached_variable_hash = metadata.get('variable_hash')
            expected_hash = self.variable_hash
            if cached_variable_hash is None:
                logger.info("Cache invalid: metadata missing variable_hash")
                return False
            if expected_hash and cached_variable_hash != expected_hash:
                logger.info(
                    "Cache invalid: variable hash mismatch %s vs %s",
                    cached_variable_hash[:8],
                    expected_hash[:8],
                )
                return False

            cached_total_dims = int(metadata.get('total_dims', -1))
            if cached_total_dims != len(self.active_variable_names):
                logger.info(
                    "Cache invalid: total_dims mismatch %s vs %s",
                    cached_total_dims,
                    len(self.active_variable_names),
                )
                return False

            # 检查必要的缓存文件是否存在
            required_files = [self.train_cache_path, self.val_cache_path, self.mask_cache_path]
            for file_path in required_files:
                if not file_path.exists():
                    logger.info(f"Cache invalid: missing file {file_path}")
                    return False

            if require_tokens and metadata.get('contains_tokens', False):
                token_files = [self.train_tokens_path, self.val_tokens_path]
                for token_file in token_files:
                    if not token_file.exists():
                        logger.info(f"Cache invalid: missing token file {token_file}")
                        return False

            logger.info("Cache is valid")
            return True

        except Exception as e:
            logger.warning(f"Error checking cache validity: {e}")
            return False

    def _save_sequences_to_parquet(self, samples_data: List[Dict],
                                  file_path: Path) -> None:
        """
        将原始样本数据保存为Parquet格式。

        Args:
            samples_data: List of sample dictionaries, each containing:
                - 'sample_id': str
                - 'data': np.ndarray of shape [1439, 6712]
                - 'start_time': timestamp of first minute
                - 'end_time': timestamp of last minute
                - 'has_nan': bool indicating if data contains NaN
            file_path: Path to save parquet file
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Preparing {len(samples_data)} samples for Parquet format...")

        data_rows: List[Dict[str, Any]] = []
        if samples_data:
            for i, sample in enumerate(tqdm(
                samples_data, total=len(samples_data),
                desc=f"Preparing samples for {file_path.name}", unit="sample"
            )):
                # 检查NaN值
                data_array = sample['data']
                has_nan = np.isnan(data_array).any()

                if has_nan:
                    nan_count = np.isnan(data_array).sum()
                    logger.warning(
                        "Sample %s contains %d NaN values (%.2f%%)",
                        sample['sample_id'],
                        nan_count,
                        nan_count / data_array.size * 100 if data_array.size else 0.0,
                    )

                # 将numpy数组序列化
                data_bytes = data_array.astype(np.float32).tobytes()

                data_rows.append({
                    'sample_id': sample['sample_id'],
                    'data': data_bytes,
                    'shape_0': data_array.shape[0],  # 1439
                    'shape_1': data_array.shape[1],  # 6712
                    'start_time': sample['start_time'],
                    'end_time': sample['end_time'],
                    'has_nan': has_nan
                })
        else:
            logger.warning(f"No samples to save to {file_path}; writing empty placeholder parquet.")

        if data_rows:
            df = pd.DataFrame(data_rows)
        else:
            df = pd.DataFrame({
                'sample_id': pd.Series(dtype='object'),
                'data': pd.Series(dtype='object'),
                'shape_0': pd.Series(dtype='int32'),
                'shape_1': pd.Series(dtype='int32'),
                'start_time': pd.Series(dtype='datetime64[ns]'),
                'end_time': pd.Series(dtype='datetime64[ns]'),
                'has_nan': pd.Series(dtype='bool'),
            })

        logger.info(f"Saving {len(df)} samples to {file_path}...")
        schema_fields = [
            pa.field('sample_id', pa.string()),
            pa.field('data', pa.binary()),
            pa.field('shape_0', pa.int32()),
            pa.field('shape_1', pa.int32()),
            pa.field('start_time', pa.timestamp('ns')),
            pa.field('end_time', pa.timestamp('ns')),
            pa.field('has_nan', pa.bool_())
        ]

        schema = pa.schema(schema_fields)
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        pq.write_table(table, file_path, compression='snappy')

        logger.info(f"Successfully saved {len(samples_data)} samples to {file_path}")

    def _save_tokens_to_parquet(self, samples_data: List[Dict], file_path: Path) -> None:
        """保存token id数据到独立的Parquet文件。"""
        if self.token_dtype is None:
            raise ValueError("token_dtype is unset while attempting to save token cache")

        token_rows: List[Dict[str, Any]] = []
        dtype_np = np.dtype(self.token_dtype)

        for sample in samples_data:
            tokens_array = sample.get('tokens')
            if tokens_array is None:
                continue
            if tokens_array.dtype != dtype_np:
                tokens_array = tokens_array.astype(dtype_np, copy=False)
            token_rows.append({
                'sample_id': sample['sample_id'],
                'tokens': tokens_array.tobytes(),
                'shape_0': tokens_array.shape[0],
                'shape_1': tokens_array.shape[1]
            })

        if not token_rows:
            logger.warning(f"No token rows produced for {file_path}; writing empty token cache.")

        df = pd.DataFrame(token_rows, columns=['sample_id', 'tokens', 'shape_0', 'shape_1'])
        logger.info(f"Saving token ids for {len(df)} samples to {file_path}...")
        schema = pa.schema([
            pa.field('sample_id', pa.string()),
            pa.field('tokens', pa.binary()),
            pa.field('shape_0', pa.int32()),
            pa.field('shape_1', pa.int32())
        ])
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        pq.write_table(table, file_path, compression='snappy')
        logger.info(f"Successfully saved token ids to {file_path}")

    def _count_parquet_rows(self, file_path: Path) -> int:
        parquet_file = pq.ParquetFile(file_path)
        return parquet_file.metadata.num_rows

    def _load_tokens_from_parquet(self, file_path: Path, token_dtype: Optional[str]) -> Dict[str, np.ndarray]:
        """加载token id Parquet文件。"""
        if token_dtype is None or not file_path.exists():
            return {}

        dtype_np = np.dtype(token_dtype)
        table = pq.read_table(file_path, memory_map=True)
        df = table.to_pandas()
        tokens_map: Dict[str, np.ndarray] = {}

        for _, row in df.iterrows():
            shape = (int(row['shape_0']), int(row['shape_1']))
            token_array = np.frombuffer(row['tokens'], dtype=dtype_np).reshape(shape)
            tokens_map[str(row['sample_id'])] = token_array

        logger.info(f"Loaded token ids for {len(tokens_map)} samples from {file_path}")
        return tokens_map

    def _load_sequences_from_parquet(self, file_path: Path) -> Tuple[List[Dict], Dict]:
        """
        从Parquet格式加载原始样本数据.

        Returns:
            Tuple of (samples_data, metadata)
            - samples_data: List of sample dictionaries
            - metadata: Dict containing cache metadata
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Cache file not found: {file_path}")

        logger.info(f"Loading samples from {file_path}...")

        # 读取Parquet文件
        table = pq.read_table(file_path, memory_map=True)
        df = table.to_pandas()

        samples_data = []

        logger.info(f"Loading {len(df)} samples from Parquet format...")
        for _, row in tqdm(df.iterrows(), total=len(df),
                          desc=f"Loading samples from {file_path.name}", unit="sample"):
            # 反序列化numpy数组
            shape = (row['shape_0'], row['shape_1'])
            data_array = np.frombuffer(row['data'], dtype=np.float32).reshape(shape)

            samples_data.append({
                'sample_id': row['sample_id'],
                'data': data_array,  # [1439, 6712]
                'start_time': row['start_time'],
                'end_time': row['end_time'],
                'has_nan': row.get('has_nan', False)
            })

        logger.info(f"Successfully loaded {len(samples_data)} samples from {file_path}")
        return samples_data, {'format_version': '2.1'}

    def _transform_tokens(self, tokenizer: 'DataTokenizer', data: np.ndarray, sample_id: str) -> np.ndarray:
        token_array = tokenizer.transform_to_tokens(data)
        if isinstance(token_array, torch.Tensor):
            token_array = token_array.cpu().numpy()
        token_array = np.asarray(token_array)
        if token_array.ndim != 2:
            raise ValueError(f"Tokenizer output for sample {sample_id} must be 2D, got shape {token_array.shape}")

        if token_array.size > 0 and np.any(token_array < 0):
            token_array = token_array.copy()
            token_array[token_array < 0] = 0

        if token_array.size > 0:
            max_token_value = int(token_array.max())
        else:
            max_token_value = 0

        vocab_size = int(getattr(tokenizer, 'vocab_size', 0) or 0)
        if vocab_size <= np.iinfo(np.uint16).max and max_token_value <= np.iinfo(np.uint16).max:
            dtype_str = 'uint16'
        elif max_token_value <= np.iinfo(np.uint32).max:
            dtype_str = 'uint32'
        else:
            dtype_str = 'int64'

        if self.token_dtype is None:
            self.token_dtype = dtype_str
        elif self.token_dtype != dtype_str:
            raise ValueError(
                f"Inconsistent token dtype encountered: existing={self.token_dtype}, new={dtype_str}"
            )

        target_dtype = np.dtype(self.token_dtype)
        if token_array.dtype != target_dtype:
            token_array = token_array.astype(target_dtype, copy=False)
        return token_array

    def _attach_tokens(self, samples: List[Dict], tokenizer: 'DataTokenizer') -> None:
        for sample in samples:
            sample['tokens'] = self._transform_tokens(tokenizer, sample['data'], sample['sample_id'])

    def _clear_token_cache(self) -> None:
        for path in [self.train_tokens_path, self.val_tokens_path]:
            if path.exists():
                try:
                    path.unlink()
                    logger.info("Removed stale token cache file: %s", path)
                except Exception as exc:
                    logger.warning("Failed to remove token cache %s: %s", path, exc)

    def _default_predict_flag(self, global_index: int) -> int:
        """Return 0 for boundary variables and 1 for equipment variables."""
        return 0 if int(global_index) < self.DEFAULT_BOUNDARY_DIMS else 1

    def _build_prediction_mask_for_active_variables(
        self,
        processor: 'DataProcessor',
        selected_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Build a prediction mask aligned with the currently active variable ordering.
        Prefers static_dir/prediction_mask.csv when present; otherwise falls back to
        processor defaults filtered by selected_indices.
        """
        selected_indices = np.asarray(selected_indices, dtype=np.int64)
        if selected_indices.ndim != 1:
            raise ValueError(f"selected_indices必须是一维数组，当前shape={selected_indices.shape}")
        if len(self.active_variable_names) != selected_indices.shape[0]:
            raise ValueError(
                "active_variable_names与selected_indices长度不一致，"
                f"{len(self.active_variable_names)} vs {selected_indices.shape[0]}"
            )

        static_mask_path = self.static_dir / "prediction_mask.csv"
        if static_mask_path.exists():
            mask_df = pd.read_csv(static_mask_path)
            if 'predict' not in mask_df.columns:
                raise ValueError(f"{static_mask_path} 缺少 predict 列")

            predict_series = mask_df['predict'].fillna(0).astype(int)

            if 'variable_name' not in mask_df.columns:
                if len(predict_series) != len(self.active_variable_names):
                    raise ValueError(
                        f"{static_mask_path} 中的预测掩码长度 {len(predict_series)} "
                        f"与当前激活变量数 {len(self.active_variable_names)} 不一致。"
                    )
                logger.info(
                    "prediction_mask.csv 未提供 variable_name 列，假定其顺序与 index_variable_mapping 一致。"
                )
                return predict_series.to_numpy(dtype=np.int32)

            mask_lookup: Dict[str, int] = {}
            for _, row in mask_df.iterrows():
                name = str(row['variable_name'])
                mask_lookup[name] = 1 if int(row['predict']) else 0

            mask_values: List[int] = []
            missing_variables: List[str] = []
            for name, global_idx in zip(self.active_variable_names, selected_indices):
                value = mask_lookup.get(name)
                if value is None:
                    missing_variables.append(name)
                    value = self._default_predict_flag(global_idx)
                mask_values.append(value)

            extra_vars = set(mask_lookup.keys()) - set(self.active_variable_names)
            if extra_vars:
                logger.info(
                    "prediction_mask.csv 包含 %d 个当前子图未使用的变量，示例: %s",
                    len(extra_vars),
                    ", ".join(list(extra_vars)[:5]),
                )

            if missing_variables:
                preview = ", ".join(missing_variables[:5])
                logger.warning(
                    "prediction_mask.csv 缺少 %d 个变量，按照boundary规则填充，示例: %s",
                    len(missing_variables),
                    preview,
                )

            return np.asarray(mask_values, dtype=np.int32)

        base_mask = np.asarray(processor.create_prediction_mask(), dtype=np.int32).reshape(-1)
        if base_mask.ndim != 1:
            raise ValueError(f"processor.create_prediction_mask() 必须返回一维数组，当前shape={base_mask.shape}")

        max_required_index = int(selected_indices.max())
        if max_required_index >= base_mask.shape[0]:
            raise ValueError(
                "Prediction mask长度不足，"
                f"需要覆盖global index {max_required_index}，但只有 {base_mask.shape[0]}"
            )

        return base_mask[selected_indices]

    def build_cache(
        self,
        processor: 'DataProcessor',
        tokenizer: Optional['DataTokenizer'],
        strict_nan_check: bool = False,
        sample_percentage: float = 1.0,
        variable_names: Optional[List[str]] = None,
        save_sequences: bool = True,
    ) -> None:
        """
        构建缓存文件。

        Args:
            processor: 数据处理器
            strict_nan_check: 如果为True，发现NaN时会抛出异常；如果为False，仅记录警告
            sample_percentage: 使用训练数据的百分比，范围(0.0, 1.0]，默认1.0使用全部数据
            variable_names: 可选变量名列表，用于构建子图缓存
        """
        if not 0.0 < sample_percentage <= 1.0:
            raise ValueError(f"sample_percentage必须在(0.0, 1.0]范围内，当前值：{sample_percentage}")

        logger.info(f"Preparing cache for {sample_percentage*100:.1f}% of samples...")

        # 重置token dtype记录
        self.token_dtype = None

        # 刷新变量配置
        self._refresh_active_schema(variable_names)
        self.variable_hash = self._hash_variable_names(self.active_variable_names)

        selected_indices = np.asarray(self.active_global_indices, dtype=np.int64)
        if selected_indices.size == 0:
            raise ValueError("Active variable index list is empty; cannot build cache.")

        # 对Tokenizer进行子图对齐，确保token维度一致
        generate_tokens = tokenizer is not None
        aligned_tokenizer: Optional['DataTokenizer'] = None
        if generate_tokens:
            aligned_tokenizer = tokenizer
            expected_dims = len(selected_indices)
            if getattr(tokenizer, "total_dims", expected_dims) != expected_dims:
                raise RuntimeError(
                    f"Tokenizer dimensions {tokenizer.total_dims} do not match active variable count {expected_dims}. "
                    "请先在目标静态目录生成对应的tokenizer。"
                )

        all_train_dirs = processor.get_sample_directories('train')
        total_available_samples = len(all_train_dirs)

        has_sequence_cache = self.train_cache_path.exists() and self.val_cache_path.exists()
        cached_metadata: Optional[Dict[str, Any]] = None
        if self.metadata_cache_path.exists():
            with open(self.metadata_cache_path, 'r') as meta_file:
                cached_metadata = json.load(meta_file)

        reuse_sequences = False
        if has_sequence_cache:
            if cached_metadata is not None:
                reuse_sequences = (
                    # cached_metadata.get('data_hash') == self.data_hash and 
                    cached_metadata.get('variable_hash') == self.variable_hash
                    and int(cached_metadata.get('total_dims', -1)) == len(self.active_variable_names)
                    and self._percentage_close(cached_metadata.get('sample_percentage'), sample_percentage)
                )
            else:
                cached_total_rows = self._count_parquet_rows(self.train_cache_path) + self._count_parquet_rows(self.val_cache_path)
                expected_count = int(total_available_samples * sample_percentage)
                reuse_sequences = cached_total_rows == expected_count

            if reuse_sequences:
                logger.info("Detected compatible cached sequences; reusing existing Parquet data.")
            else:
                logger.info("Existing sequence cache found but configuration mismatch; rebuilding from raw data.")

        train_samples_data: List[Dict]
        val_samples_data: List[Dict]

        if reuse_sequences:
            train_samples_data, _ = self._load_sequences_from_parquet(self.train_cache_path)
            val_samples_data, _ = self._load_sequences_from_parquet(self.val_cache_path)
            expected_dim = len(self.active_variable_names)

            for sample in train_samples_data:
                if sample['data'].shape[1] != expected_dim:
                    raise ValueError(
                        f"Cached train sequence width {sample['data'].shape[1]} does not match active variable count {expected_dim}."
                    )
            for sample in val_samples_data:
                if sample['data'].shape[1] != expected_dim:
                    raise ValueError(
                        f"Cached validation sequence width {sample['data'].shape[1]} does not match active variable count {expected_dim}."
                    )

            if strict_nan_check:
                train_has_nan = any(sample['has_nan'] for sample in train_samples_data)
                val_has_nan = any(sample['has_nan'] for sample in val_samples_data)
                if train_has_nan or val_has_nan:
                    raise ValueError("Cached sequences contain NaN values; rebuild from raw data or disable strict_nan_check.")

            if generate_tokens:
                self._attach_tokens(train_samples_data, aligned_tokenizer)
                self._attach_tokens(val_samples_data, aligned_tokenizer)

            logger.info(
                "Reused %d cached samples (%d train / %d val).",
                len(train_samples_data) + len(val_samples_data),
                len(train_samples_data),
                len(val_samples_data),
            )
        else:
            total_samples_to_use = int(total_available_samples * sample_percentage)
            selected_sample_dirs = all_train_dirs[:total_samples_to_use]

            if total_available_samples > 0:
                logger.info(
                    "Using %d out of %d available samples (%.1f%%).",
                    len(selected_sample_dirs),
                    total_available_samples,
                    sample_percentage * 100.0,
                )
            else:
                logger.info("Dataset does not contain training samples; proceeding with empty cache.")

            train_size = int(0.9 * len(selected_sample_dirs))
            train_sample_dirs = selected_sample_dirs[:train_size]
            val_sample_dirs = selected_sample_dirs[train_size:]

            train_samples_data = self._process_samples(
                processor,
                train_sample_dirs,
                strict_nan_check,
                "train",
                aligned_tokenizer if generate_tokens else None,
                selected_indices,
            )

            val_samples_data = self._process_samples(
                processor,
                val_sample_dirs,
                strict_nan_check,
                "validation",
                aligned_tokenizer if generate_tokens else None,
                selected_indices,
            )

        # 保存到Parquet文件
        if save_sequences and not reuse_sequences:
            self._save_sequences_to_parquet(train_samples_data, self.train_cache_path)
            self._save_sequences_to_parquet(val_samples_data, self.val_cache_path)
        elif save_sequences:
            logger.info("Sequence parquet already present; reuse without rewriting.")
        else:
            logger.info("Skipping sequence parquet export (save_sequences=False).")

        if generate_tokens and self.token_dtype is not None:
            self._save_tokens_to_parquet(train_samples_data, self.train_tokens_path)
            self._save_tokens_to_parquet(val_samples_data, self.val_tokens_path)
        else:
            self._clear_token_cache()

        # 保存预测mask
        prediction_mask = self._build_prediction_mask_for_active_variables(processor, selected_indices)
        np.save(self.mask_cache_path, prediction_mask)

        # tokenizer信息
        tokenizer_vocab_size = int(getattr(aligned_tokenizer, 'vocab_size', 0) or 0) if aligned_tokenizer is not None else None
        tokenizer_stats_dir: Optional[str] = None
        if aligned_tokenizer is not None and hasattr(aligned_tokenizer, 'stats_dir') and aligned_tokenizer.stats_dir is not None:
            try:
                tokenizer_stats_dir = str(Path(aligned_tokenizer.stats_dir).resolve())
            except Exception:
                tokenizer_stats_dir = str(aligned_tokenizer.stats_dir)

        # 保存元数据
        metadata = {
            'data_hash': self.data_hash,
            'cache_format_version': '2.1',
            'created_at': pd.Timestamp.now().isoformat(),
            'sample_percentage': sample_percentage,
            'train_samples': len(train_samples_data),
            'val_samples': len(val_samples_data),
            'total_dims': len(self.active_variable_names),
            'boundary_dims': self.active_boundary_dims,
            'equipment_dims': self.active_equipment_dims,
            'data_shape_per_sample': [1439, len(self.active_variable_names)],  # 每个样本的数据形状
            'max_sequence_length': 1438,  # 最大可能的序列长度（样本长度-1）
            'token_dtype': self.token_dtype if generate_tokens else None,
            'contains_tokens': generate_tokens and self.token_dtype is not None,
            'tokenizer_vocab_size': tokenizer_vocab_size,
            'tokenizer_stats_dir': tokenizer_stats_dir,
            'variable_hash': self.variable_hash,
            'static_dir': str(self.static_dir),
            'variable_names': self.active_variable_names,
        }
        if generate_tokens and self.token_dtype is not None:
            metadata['token_cache_files'] = {
                'train': self.train_tokens_path.name,
                'val': self.val_tokens_path.name
            }

        with open(self.metadata_cache_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Cache built successfully: {len(train_samples_data)} train, "
                   f"{len(val_samples_data)} val")
        logger.info(
            "Each sample stores [1439, %d] raw data (boundary=%d, equipment=%d)",
            len(self.active_variable_names),
            self.active_boundary_dims,
            self.active_equipment_dims,
        )
        logger.info(f"Max possible sequence length: 1438 (sample_length - 1)")

    def _process_samples(
        self,
        processor: 'DataProcessor',
        samples: List[Path],
        strict_nan_check: bool,
        split_name: str,
        tokenizer: Optional['DataTokenizer'],
        selected_indices: Optional[np.ndarray],
    ) -> List[Dict]:
        """
        处理样本数据。

        Returns:
            List of sample dictionaries with raw data
        """
        all_samples_data = []

        logger.info(f"Processing {len(samples)} {split_name} samples...")

        for sample_dir in tqdm(samples, desc=f"Processing {split_name} samples", unit="sample"):
            try:
                # 加载原始样本数据 (不创建序列)
                raw_data, metadata = processor.load_raw_sample_data(sample_dir)

                if raw_data is None:
                    logger.warning(f"Failed to load data from {sample_dir.name}")
                    continue

                # 检查NaN并根据strict_nan_check决定是否抛出异常
                if metadata.get('has_nan', False):
                    if strict_nan_check:
                        raise ValueError(f"NaN values found in {split_name} data from sample "
                                       f"{sample_dir.name}. Use strict_nan_check=False to continue.")

                # 添加到结果列表
                filtered_data = raw_data
                if selected_indices is not None:
                    filtered_data = raw_data[:, selected_indices]

                token_array = None
                if tokenizer is not None:
                    token_array = self._transform_tokens(tokenizer, filtered_data, metadata['sample_id'])

                sample_entry = {
                    'sample_id': metadata['sample_id'],
                    'data': filtered_data,  # [1439, V]
                    'start_time': metadata['start_time'],
                    'end_time': metadata['end_time'],
                    'has_nan': metadata.get('has_nan', False)
                }
                if token_array is not None:
                    sample_entry['tokens'] = token_array

                all_samples_data.append(sample_entry)

            except Exception as e:
                logger.error(f"Error processing sample {sample_dir.name}: {e}")
                if strict_nan_check:
                    raise
                continue

        # 总结处理结果
        samples_with_nan = sum(1 for s in all_samples_data if s['has_nan'])
        if samples_with_nan > 0:
            logger.warning(f"NaN Summary for {split_name} data:")
            logger.warning(f"  - Samples with NaN: {samples_with_nan}/{len(all_samples_data)}")
        else:
            logger.info(f"✓ No NaN values found in {split_name} data - all {len(all_samples_data)} samples are clean!")

        logger.info(f"Processed {len(all_samples_data)} samples for {split_name}")
        return all_samples_data

    def load_cached_data(self, split: str, include_tokens: bool = True) -> Tuple[List[Dict], Dict, Optional[np.ndarray]]:
        """
        从磁盘缓存加载指定划分的样本、元数据和预测掩码。

        Args:
            split: `'train'` 或 `'val'`，决定读取的缓存文件。
            include_tokens: 是否尝试加载token缓存；False 时仅返回原始序列。

        Returns:
            samples_data: List[Dict]，例如 `{'sample_id': 'train_001', 'data': ndarray((T, V)), ...}`，
                当存在token缓存时还会附带 `tokens`。
            cache_metadata: Dict，合并 Parquet 读出的元数据与 `metadata.json` 中的缓存配置。
            prediction_mask: `np.ndarray` 或 `None`，如 `ndarray(shape=(V,), dtype=int32)`，表示选定变量维度上的掩码。
        """
        if not self._is_cache_valid(require_tokens=include_tokens):
            raise ValueError("Cache is invalid or missing. Please build cache first.")

        # 选择对应的缓存文件
        if split == 'train':
            cache_path = self.train_cache_path
            tokens_path = self.train_tokens_path
        elif split == 'val':
            cache_path = self.val_cache_path
            tokens_path = self.val_tokens_path
        else:
            raise ValueError(f"Invalid split: {split}")

        # 加载样本数据
        token_dtype = None
        metadata_contents: Optional[Dict[str, Any]] = None
        if self.metadata_cache_path.exists():
            try:
                with open(self.metadata_cache_path, 'r') as f:
                    metadata = json.load(f)
                token_dtype = metadata.get('token_dtype')
                if token_dtype:
                    self.token_dtype = token_dtype
                metadata_contents = metadata
            except Exception as exc:
                logger.warning(f"Failed to read token dtype from metadata: {exc}")

        samples_data, cache_metadata = self._load_sequences_from_parquet(cache_path)
        if metadata_contents:
            cache_metadata.update(metadata_contents)
        if include_tokens and token_dtype:
            tokens_map = self._load_tokens_from_parquet(tokens_path, token_dtype)
            if tokens_map:
                for sample in samples_data:
                    sample_tokens = tokens_map.get(str(sample['sample_id']))
                    if sample_tokens is not None:
                        sample['tokens'] = sample_tokens
        if include_tokens:
            cache_metadata['token_dtype'] = token_dtype
            if 'contains_tokens' not in cache_metadata:
                cache_metadata['contains_tokens'] = bool(token_dtype)
        else:
            cache_metadata['token_dtype'] = None
            cache_metadata['contains_tokens'] = False

        # 加载预测mask
        prediction_mask = None
        if self.mask_cache_path.exists():
            prediction_mask = np.load(self.mask_cache_path)

        return samples_data, cache_metadata, prediction_mask

    def get_cache_info(self) -> Dict:
        """获取缓存信息。"""
        if not self.metadata_cache_path.exists():
            return {'cache_exists': False}

        try:
            with open(self.metadata_cache_path, 'r') as f:
                metadata = json.load(f)

            metadata['cache_exists'] = True
            metadata['cache_valid'] = self._is_cache_valid()
            metadata['cache_dir'] = str(self.cache_dir)

            # 添加文件大小信息
            cache_files = {
                'train': self.train_cache_path,
                'val': self.val_cache_path,
                'train_tokens': self.train_tokens_path,
                'val_tokens': self.val_tokens_path,
                'mask': self.mask_cache_path
            }

            file_sizes = {}
            for name, path in cache_files.items():
                if path.exists():
                    size_mb = path.stat().st_size / (1024 * 1024)
                    file_sizes[f'{name}_size_mb'] = round(size_mb, 2)

            metadata.update(file_sizes)
            var_names = metadata.get('variable_names')
            if isinstance(var_names, list):
                metadata['variable_names_count'] = len(var_names)
                metadata['variable_names_preview'] = var_names[:10]
                if len(var_names) > 32:
                    metadata.pop('variable_names', None)
            return metadata

        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return {'cache_exists': True, 'error': str(e)}

    def clear_cache(self, preserve_sequences: bool = False, preserve_tokens: bool = False) -> None:
        """清除缓存文件，可选择保留序列或token文件。"""
        targets = [
            (self.train_cache_path, preserve_sequences),
            (self.val_cache_path, preserve_sequences),
            (self.train_tokens_path, preserve_tokens),
            (self.val_tokens_path, preserve_tokens),
            (self.metadata_cache_path, False),
            (self.mask_cache_path, False),
        ]

        for cache_file, preserve in targets:
            if preserve:
                continue
            if cache_file.exists():
                cache_file.unlink()
                logger.info("Removed %s", cache_file)

        logger.info("Cache cleared successfully")
