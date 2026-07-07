import json
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import logging
from tensordict import TensorDict
from tqdm import tqdm

from .processor import DataProcessor
from .normalizer import DataNormalizer
from .tokenizer_save import DataTokenizer, load_tokenizer as load_tokenizer_from_stats
from .cache_manager import CacheManager
from .topology_attention_index import load_attention_indices
from .dataset_utils import _to_writable_contiguous_float32

logger = logging.getLogger(__name__)

class FluidDataset(Dataset):
    """
    PyTorch Dataset for gas pipeline network fluid dynamics data.
    
    Data Format:
    - Input: [B, T, V] where B=batch, T=time_steps (default 3), V=variates (6712)
    - Target: [B, T, V] same format, time-shifted by configurable time_step_offset minutes
    - Mask: [V] prediction mask, boundary=0, equipment=1
    
    Features:
    - Boundary conditions (538 dims) + Equipment predictions (6174 dims) = 6712 total
    - 30-min boundary data interpolated to 1-min intervals
    - Autoregressive structure: each minute predicts next minute
    - TensorDict packaging for structured data handling
    """
    
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 sequence_length: int = 3,
                 use_cache: bool = True,
                 force_rebuild_cache: bool = False,
                 max_sequences_per_sample: Optional[int] = None,
                 cache_dir: Optional[str] = None,
                 static_dir: Optional[str] = None,
                 predict_variable_name: Optional[str] = None,
                 tokenizer: Optional[DataTokenizer] = None,
                 time_step_offset: int = 1):
        """
        Initialize FluidDataset.

        Args:
            data_dir: Path to data directory
            split: 'train', 'val', or 'test'
            sequence_length: Length of time series sequences (default: 3 for 3 minutes)
            use_cache: Whether to use precomputed cache for fast loading
            force_rebuild_cache: Force rebuilding cache even if it exists
            max_sequences_per_sample: Maximum sequences per sample (for memory control)
            cache_dir: Custom cache directory path (default: data/cache)
            tokenizer: Optional tokenizer aligned with this dataset (auto-aligned if provided)
            time_step_offset: Number of minutes to shift targets forward relative to inputs
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        if time_step_offset < 1:
            raise ValueError("time_step_offset must be >= 1")
        self.time_step_offset = int(time_step_offset)
        self.use_cache = use_cache
        self.force_rebuild_cache = force_rebuild_cache
        self.max_sequences_per_sample = max_sequences_per_sample
        if static_dir is not None:
            self.static_dir = Path(static_dir).resolve()
        else:
            self.static_dir = (self.data_dir / "static" / "full").resolve()
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir).resolve()
        else:
            self.cache_dir = (self.static_dir / "cache").resolve()
        self.is_full_graph = self.static_dir.name == "full"
        self.predict_variable_name = predict_variable_name

        # Initialize data processor and cache manager
        self.processor = DataProcessor(str(self.data_dir))
        self.cache_manager = (
            CacheManager(str(self.data_dir), cache_dir=str(self.cache_dir), static_dir=str(self.static_dir))
            if use_cache else None
        )
        
        # Graph hyperparameters (may override defaults)
        self.graph_hyperparameters: Dict[str, Any] = {}
        self.boundary_dims: Optional[int] = None
        self.equipment_dims: Optional[int] = None
        self.total_dims: int = 0

        # Prediction mask (fixed across all samples)
        self.prediction_mask = None

        # Topology attention indices [total_dims, 64]
        self.attention_indices = None
        self.variable_names = None

        # Variable mapping (index <-> variable_name)
        self.index_to_variable = {}
        self.variable_to_index = {}
        self.cache_contains_tokens: bool = False
        self.prediction_mask_override: Optional[np.ndarray] = None
        self.tokenizer: Optional[DataTokenizer] = None
        self.token_vocab_size: Optional[int] = None
        # Static tensors reused across samples to reduce per-item CPU work
        self._mask_tensor: Optional[torch.Tensor] = None
        self._attention_indices_tensor: Optional[torch.Tensor] = None

        # Load graph metadata and apply overrides
        self._load_graph_hyperparameters()
        if self.boundary_dims is not None:
            try:
                self.processor.boundary_dims = int(self.boundary_dims)
            except Exception:
                logger.warning("Failed to apply boundary_dims=%s to processor.", self.boundary_dims)
        if self.equipment_dims is not None:
            try:
                self.processor.total_prediction_dims = int(self.equipment_dims)
            except Exception:
                logger.warning("Failed to apply equipment_dims=%s to processor.", self.equipment_dims)

        # Load variable mapping first (handles static graph selection and dims override)
        self._load_variable_mapping()

        if self.total_dims == 0:
            # Fall back to processor defaults when metadata is missing
            self.total_dims = getattr(self.processor, "boundary_dims", 538) + getattr(
                self.processor, "total_prediction_dims", 6174
            )
        if self.boundary_dims is None:
            self.boundary_dims = getattr(self.processor, "boundary_dims", None)
        if self.equipment_dims is None:
            self.equipment_dims = getattr(self.processor, "total_prediction_dims", None)

        # Load topology attention indices
        self._load_topology_indices()

        # Load and prepare data
        self._load_data()

        # Materialize static tensors once (mask and attention indices)
        self._prepare_static_tensors()

        # Attach tokenizer once: use provided instance, otherwise load from static_dir
        if tokenizer is not None:
            self.set_tokenizer(tokenizer)
        else:
            self._load_tokenizer_from_static_dir()
        
        # Get the correct sequence count based on which loading method was used
        seq_count = 0
        if hasattr(self, 'total_sequences'):
            seq_count = self.total_sequences
        elif hasattr(self, 'all_sequences'):
            seq_count = len(self.all_sequences)
        
        logger.info(f"FluidDataset initialized: {self.split} split, {seq_count} sequences, "
                   f"sequence_length={self.sequence_length}, total_dims={self.total_dims}, use_cache={use_cache}")
    
    def _prepare_static_tensors(self) -> None:
        """Create static tensors for mask and attention indices once per dataset."""
        # Prepare prediction mask tensor
        try:
            if self.prediction_mask is None:
                self.prediction_mask = self.processor.create_prediction_mask()
            if not isinstance(self.prediction_mask, np.ndarray):
                self.prediction_mask = np.asarray(self.prediction_mask, dtype=np.int32)
            self._mask_tensor = torch.from_numpy(self.prediction_mask).to(dtype=torch.int32)
        except Exception as exc:
            logger.warning(f"Failed to prepare static prediction mask tensor: {exc}")
            self._mask_tensor = None

        # Prepare attention indices tensor
        try:
            if self.attention_indices is None:
                # Default to self-attention only if indices are missing
                self.attention_indices = np.zeros((self.total_dims, 64), dtype=np.int32)
                for i in range(self.total_dims):
                    self.attention_indices[i, 0] = i
            self._attention_indices_tensor = torch.from_numpy(self.attention_indices).long()
        except Exception as exc:
            logger.warning(f"Failed to prepare static attention indices tensor: {exc}")
            self._attention_indices_tensor = None
    
    def _load_topology_indices(self):
        """Load topology attention indices for graph attention."""
        result = load_attention_indices(str(self.data_dir), static_dir=str(self.static_dir))
        if result is not None:
            self.attention_indices, self.variable_names = result
        else:
            raise ValueError("No topology attention indices found in %s.", self.static_dir)
        logger.info(
            "Loaded topology attention indices from %s: shape %s",
            self.static_dir,
            getattr(self.attention_indices, 'shape', None),
        )

    def _load_variable_mapping(self):
        """Load variable mapping from the static directory."""
        mapping_file = self.static_dir / "index_variable_mapping.csv"
        self.index_to_variable.clear()
        self.variable_to_index.clear()

        if mapping_file.exists():
            try:
                import pandas as pd

                mapping_df = pd.read_csv(mapping_file)
                if 'index' not in mapping_df.columns or 'variable_name' not in mapping_df.columns:
                    raise ValueError(f"{mapping_file} 缺少必要的列: index, variable_name")

                mapping_df = mapping_df.sort_values('index')
                variable_names = mapping_df['variable_name'].astype(str).tolist()

                mapping_dims = len(variable_names)
                if self.total_dims and self.total_dims != mapping_dims:
                    logger.info(
                        "Overriding total_dims from metadata (%d) with mapping size %d",
                        self.total_dims,
                        mapping_dims,
                    )
                self.total_dims = mapping_dims
                for idx, name in enumerate(variable_names):
                    self.index_to_variable[idx] = name
                    self.variable_to_index[name] = idx

                logger.info(
                    "Loaded variable mapping (%d variables) from %s",
                    self.total_dims,
                    mapping_file,
                )

                if self.predict_variable_name is None and not self.is_full_graph:
                    try:
                        base = self.static_dir.name
                        center = base.rsplit('_', 1)[0]
                    except Exception:
                        center = None
                    if center is not None:
                        for name in variable_names:
                            if name.startswith(center + ":") or name.startswith(center + "_"):
                                self.predict_variable_name = name
                                break

            except Exception as exc:
                logger.error("Error loading variable mapping from %s: %s", mapping_file, exc)
                self._create_default_mapping()
        else:
            logger.warning("Mapping file not found at %s; using default sequential mapping.", mapping_file)
            self._create_default_mapping()

        # Attempt to load custom prediction mask when static configuration provides it
        self.prediction_mask_override = self._load_custom_prediction_mask()

    def _create_default_mapping(self):
        """Create default variable mapping when CSV is not available."""
        for i in range(self.total_dims):
            variable_name = f"var_{i}"
            self.index_to_variable[i] = variable_name
            self.variable_to_index[variable_name] = i

    def _load_graph_hyperparameters(self) -> None:
        """Load graph hyperparameters (dimensions, tokenizer metadata) from static_dir."""
        hyper_path = self.static_dir / "graph_hyperparameters.json"
        if not hyper_path.exists():
            logger.debug("Graph hyperparameter file not found at %s", hyper_path)
            return

        try:
            with hyper_path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception as exc:
            logger.warning("Failed to load graph hyperparameters from %s: %s", hyper_path, exc)
            return

        if not isinstance(data, dict):
            logger.warning("Graph hyperparameters in %s are not a JSON object.", hyper_path)
            return

        self.graph_hyperparameters = data

        variables_meta = data.get("variables", {}) if isinstance(data.get("variables"), dict) else {}
        total_vars = variables_meta.get("total_variables")
        boundary_vars = variables_meta.get("boundary_variables")
        equipment_vars = variables_meta.get("equipment_variables")

        if total_vars is not None:
            try:
                self.total_dims = int(total_vars)
            except Exception:
                logger.warning("Invalid total_variables value in %s: %s", hyper_path, total_vars)

        if boundary_vars is not None:
            try:
                self.boundary_dims = int(boundary_vars)
            except Exception:
                logger.warning("Invalid boundary_variables value in %s: %s", hyper_path, boundary_vars)

        if equipment_vars is not None:
            try:
                self.equipment_dims = int(equipment_vars)
            except Exception:
                logger.warning("Invalid equipment_variables value in %s: %s", hyper_path, equipment_vars)

        tokenizer_meta = data.get("tokenizer", {}) if isinstance(data.get("tokenizer"), dict) else {}
        vocab_size = tokenizer_meta.get("vocab_size")
        if vocab_size is not None:
            try:
                self.token_vocab_size = int(vocab_size)
            except Exception:
                logger.warning("Invalid tokenizer vocab_size in %s: %s", hyper_path, vocab_size)

    def _load_tokenizer_from_static_dir(self) -> Optional[DataTokenizer]:
        """Load tokenizer statistics from static_dir/tokenizer_save if available."""
        if self.tokenizer is not None:
            return self.tokenizer

        stats_dir = self.static_dir / "tokenizer_save"
        if not stats_dir.exists():
            logger.debug("Tokenizer stats directory not found at %s", stats_dir)
            return None

        tokenizer = load_tokenizer_from_stats(str(self.static_dir))
        if tokenizer is None or not getattr(tokenizer, "fitted", False):
            logger.warning("Tokenizer statistics not found or invalid under %s", stats_dir)
            return None

        try:
            self.set_tokenizer(tokenizer)
        except RuntimeError as exc:
            logger.error("Tokenizer loaded from %s does not match dataset dims: %s", stats_dir, exc)
            return None

        return self.tokenizer

    def _ensure_cache_tokenizer(self) -> DataTokenizer:
        """Load tokenizer stats used for cache token generation."""
        if self.tokenizer is not None:
            return self.tokenizer

        tokenizer = self._load_tokenizer_from_static_dir()
        if tokenizer is None:
            raise RuntimeError(
                "Tokenizer statistics not found or invalid. Run data/compute_tokenizer_stats.py with the desired static_dir before building cache with token ids."
            )
        return tokenizer

    def set_tokenizer(self, tokenizer: DataTokenizer) -> None:
        """
        Align and attach a tokenizer to this dataset for token id generation.
        """
        if tokenizer is None:
            raise ValueError("tokenizer cannot be None when calling set_tokenizer.")
        expected_dims = getattr(self, "total_dims", None)
        if expected_dims:
            if tokenizer.total_dims != expected_dims:
                raise RuntimeError(
                    f"Tokenizer dims ({tokenizer.total_dims}) do not match expected dims ({expected_dims})."
                )
        self.tokenizer = tokenizer
        self.token_vocab_size = tokenizer.vocab_size
        if self.boundary_dims is None:
            self.boundary_dims = getattr(tokenizer, "boundary_dims", None)
        if self.equipment_dims is None:
            self.equipment_dims = getattr(tokenizer, "equipment_dims", None)
        logger.info(
            "Attached tokenizer to dataset (dims=%d, vocab=%d)",
            tokenizer.total_dims,
            tokenizer.vocab_size,
        )

    def _default_predict_flag(self, local_index: int) -> int:
        """Determine the default predict flag for a variable."""
        boundary_dims = (
            self.boundary_dims
            if self.boundary_dims is not None
            else getattr(self.processor, 'boundary_dims', 538)
        )
        if 0 <= local_index < boundary_dims:
            return 0
        return 1

    def _load_custom_prediction_mask(self) -> Optional[np.ndarray]:
        """Load customized prediction mask from the static directory if available."""
        mask_root = self.static_dir
        if mask_root is None:
            return None

        mask_path = Path(mask_root) / "prediction_mask.csv"
        if not mask_path.exists():
            return None

        try:
            import pandas as pd
            mask_df = pd.read_csv(mask_path)
            if 'variable_name' not in mask_df.columns or 'predict' not in mask_df.columns:
                logger.warning(f"prediction_mask.csv missing required columns in {mask_path}")
                return None

            mask_map = {}
            for _, row in mask_df.iterrows():
                var_name = str(row['variable_name'])
                try:
                    predict_flag = int(row['predict'])
                except Exception:
                    predict_flag = self._default_predict_flag(0)
                mask_map[var_name] = 1 if predict_flag else 0

            mask_values = []
            missing_vars = []
            for local_idx in range(self.total_dims):
                var_name = self.index_to_variable.get(local_idx)
                if var_name is None:
                    missing_vars.append(local_idx)
                    mask_values.append(0)
                    continue
                if var_name in mask_map:
                    mask_values.append(1 if mask_map[var_name] else 0)
                else:
                    mask_values.append(self._default_predict_flag(local_idx))
            if missing_vars:
                logger.warning(f"Missing variable names for local indices {missing_vars[:5]} (total {len(missing_vars)}); defaulting predict flag to 0")

            extra_vars = set(mask_df['variable_name']) - set(self.index_to_variable.values())
            if extra_vars:
                logger.info(f"prediction_mask.csv contains {len(extra_vars)} extra variables not in this static graph; ignoring them.")

            mask_array = np.array(mask_values, dtype=np.int32)
            return mask_array
        except Exception as e:
            logger.warning(f"Failed to load custom prediction mask from {mask_path}: {e}")
            return None

    def _select_prediction_mask(self, base_mask: Optional[np.ndarray]) -> np.ndarray:
        """Resolve final prediction mask considering custom overrides and defaults."""
        # Prefer custom override when valid
        if self.prediction_mask_override is not None:
            if self.prediction_mask_override.shape[0] == self.total_dims:
                mask = self.prediction_mask_override
            else:
                logger.warning(f"Custom prediction mask length {self.prediction_mask_override.shape[0]} "
                               f"does not match total_dims {self.total_dims}; falling back to base mask.")
                mask = None
        else:
            mask = None

        if mask is None and base_mask is not None:
            mask = np.array(base_mask, dtype=np.int32)
            if mask.shape[0] != self.total_dims:
                logger.warning(
                    "Base prediction mask length %s mismatch with total_dims %s; discarding base mask.",
                    mask.shape[0],
                    self.total_dims,
                )
                mask = None

        if mask is None:
            raise ValueError("Prediction mask is None，不应该执行到这里")

        mask = np.array(mask, dtype=np.int32)
        if mask.shape[0] != self.total_dims:
            raise ValueError(f"Prediction mask dimension mismatch: expected {self.total_dims}, got {mask.shape[0]}")
        return mask

    def _load_data(self):
        """Load and prepare dataset using cache if available (v2.1 - raw samples)."""

        # 尝试使用缓存快速加载
        if self.use_cache and self.cache_manager:
            try:
                # 检查是否需要重建缓存
                if self.force_rebuild_cache or not self.cache_manager._is_cache_valid():
                    logger.info("Building cache (v2.1) for fast loading...")
                    cache_tokenizer = self._ensure_cache_tokenizer()
                    self.cache_manager.build_cache(self.processor, cache_tokenizer)

                # 从缓存加载数据 (v2.1 - raw samples)
                logger.info(f"Loading {self.split} data from cache (v2.1)...")
                samples_data, cache_metadata, prediction_mask = self.cache_manager.load_cached_data(self.split)

                # 存储样本级别的数据
                self.samples = []
                global_seq_idx = 0

                self.cache_contains_tokens = bool(cache_metadata.get('contains_tokens', False))

                metadata_total_dims = int(cache_metadata.get('total_dims', self.total_dims))
                if metadata_total_dims > 0 and metadata_total_dims != self.total_dims:
                    logger.info(
                        "Updating dataset total_dims from %d to %d based on cache metadata.",
                        self.total_dims,
                        metadata_total_dims,
                    )
                    self.total_dims = metadata_total_dims

                metadata_var_names = cache_metadata.get('variable_names')
                if isinstance(metadata_var_names, list) and len(metadata_var_names) == self.total_dims:
                    self.index_to_variable = {idx: str(name) for idx, name in enumerate(metadata_var_names)}
                    self.variable_to_index = {str(name): idx for idx, name in enumerate(metadata_var_names)}

                metadata_vocab = cache_metadata.get('tokenizer_vocab_size')
                if metadata_vocab is not None:
                    try:
                        self.token_vocab_size = int(metadata_vocab)
                    except Exception:
                        self.token_vocab_size = None
                if not self.cache_contains_tokens:
                    self.token_vocab_size = None

                for sample in samples_data:
                    # 计算该样本可生成的序列数
                    data_array = np.asarray(sample['data'])
                    required_span = self.sequence_length + self.time_step_offset
                    num_sequences = data_array.shape[0] - required_span + 1  # 可生成的滑窗数量

                    if num_sequences <= 0:
                        logger.warning(
                            f"Sample {sample['sample_id']} too short for sequence_length={self.sequence_length} with time_step_offset={self.time_step_offset}"
                        )
                        continue

                    # 应用max_sequences_per_sample限制
                    if self.max_sequences_per_sample is not None:
                        num_sequences = min(num_sequences, self.max_sequences_per_sample)

                    sample_tokens = sample.get('tokens')
                    token_array = None
                    if sample_tokens is None:
                        if self.cache_contains_tokens:
                            raise RuntimeError(
                                "缓存元数据指示包含tokens，但样本缺失token信息。请重新构建缓存并确保生成tokens。"
                            )
                    else:
                        token_array = np.asarray(sample_tokens)
                        if token_array.shape[0] != data_array.shape[0] or token_array.shape[1] != data_array.shape[1]:
                            raise RuntimeError(
                                f"Token array shape mismatch for sample {sample['sample_id']}: "
                                f"data {data_array.shape}, tokens {token_array.shape}"
                            )

                    if data_array.shape[1] != self.total_dims:
                        logger.warning(
                            "Sample %s dimension mismatch (expected %d, got %d); updating dataset dims.",
                            sample['sample_id'],
                            self.total_dims,
                            data_array.shape[1],
                        )
                        self.total_dims = data_array.shape[1]

                    self.samples.append({
                        'sample_id': sample['sample_id'],
                        'data': data_array,  # [1439, V]
                        'tokens': token_array,
                        'num_sequences': num_sequences,
                        'start_idx': global_seq_idx,
                        'end_idx': global_seq_idx + num_sequences,
                        'start_time': sample['start_time'],
                        'end_time': sample['end_time'],
                        'time_step_offset': self.time_step_offset
                    })

                    global_seq_idx += num_sequences

                # Filter prediction mask if subset in use
                self.prediction_mask = self._select_prediction_mask(prediction_mask)
                self.total_sequences = global_seq_idx

                logger.info(
                    f"Loaded {len(self.samples)} samples with {self.total_sequences} total sequences from cache (v2.1) for {self.split} split"
                )
                return

            except Exception as e:
                logger.warning(f"Cache loading failed, falling back to raw data loading: {e}")

        # 回退到原始数据加载方式
        logger.info("Loading data from raw files...")
        self._load_data_from_raw_files()
    
    def _load_data_from_raw_files(self):
        """Load data from raw CSV files (fallback method)."""
        self.cache_contains_tokens = False
        # Get sample directories
        if self.split == 'train':
            all_samples = self.processor.get_sample_directories('train')
            # Split train data: use first 90% for train, last 10% for validation
            train_size = int(0.9 * len(all_samples))
            self.sample_dirs = all_samples[:train_size]
        elif self.split == 'val':
            all_samples = self.processor.get_sample_directories('train')
            train_size = int(0.9 * len(all_samples))
            self.sample_dirs = all_samples[train_size:]
        elif self.split == 'test':
            self.sample_dirs = self.processor.get_sample_directories('test')
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        
        # Load all sequences from all samples
        self._load_all_sequences()
        
        # Validate that we have data
        if len(self.all_sequences) == 0:
            raise ValueError(f"No sequences found for split: {self.split}")
            
        logger.info(f"Loaded {len(self.all_sequences)} sequences from {len(self.sample_dirs)} samples for {self.split} split")
    
    def _load_all_sequences(self):
        """Load all sequences from all samples."""
        self.all_sequences = []
        self.sequence_metadata = []
        base_prediction_mask = None
        
        # Add progress bar for sample loading
        sample_dirs_pbar = tqdm(self.sample_dirs, desc=f"Loading {self.split} samples", unit="sample")
        for sample_dir in sample_dirs_pbar:
            try:
                # Load sequences and prediction mask from this sample
                sequences, prediction_mask = self.processor.load_combined_sample_data(
                    sample_dir, self.sequence_length, self.time_step_offset)
                
                if not sequences:
                    logger.warning(f"No sequences loaded from {sample_dir.name}")
                    continue
                
                # Store prediction mask (should be same for all samples)
                if base_prediction_mask is None and prediction_mask is not None:
                    base_prediction_mask = prediction_mask
                
                # Limit sequences per sample if specified
                if self.max_sequences_per_sample is not None:
                    sequences = sequences[:self.max_sequences_per_sample]
                
                # Add sequences with metadata
                for input_seq, target_seq, start_time, end_time in sequences:
                    self.all_sequences.append((input_seq, target_seq))
                    self.sequence_metadata.append({
                        'sample_id': sample_dir.name,
                        'start_time': start_time,
                        'end_time': end_time,
                        'time_step_offset': self.time_step_offset
                    })
                    
                logger.debug(f"Loaded {len(sequences)} sequences from {sample_dir.name}")
                sample_dirs_pbar.set_postfix({"sequences": len(self.all_sequences), "current_sample": sample_dir.name})
                
            except Exception as e:
                logger.error(f"Error loading sequences from {sample_dir.name}: {e}")
                continue
        
        # Create default mask if none was loaded
        if self.prediction_mask is None:
            if base_prediction_mask is None:
                logger.warning("No prediction mask loaded from raw data; using default mask")
            self.prediction_mask = self._select_prediction_mask(base_prediction_mask)
    
    
    def __len__(self) -> int:
        """Return number of sequences in dataset."""
        # v2.1: return total sequences across all samples
        if hasattr(self, 'total_sequences'):
            return self.total_sequences
        # Fallback for raw file loading
        if hasattr(self, 'all_sequences'):
            return len(self.all_sequences)
        return 0
    
    def __getitem__(self, idx: int) -> TensorDict:
        """
        Get a single sequence sample from the dataset (v2.1 - dynamic sequence creation).

        Args:
            idx: Global sequence index

        Returns:
            TensorDict containing:
            - 'input': Input tensor [T, V]
            - 'target': Target tensor [T, V]
            - 'mask': Prediction mask [V]
            - 'attention_indices': [V, 64]
            - 'metadata': Sample metadata dict
        """
        # v2.1: Dynamic sequence creation from raw samples
        input_tokens_arr = None
        target_tokens_arr = None

        if hasattr(self, 'samples') and self.samples:
            # Binary search to find the sample containing this sequence index
            sample = None
            for s in self.samples:
                if s['start_idx'] <= idx < s['end_idx']:
                    sample = s
                    break

            if sample is None:
                raise IndexError(f"Sequence index {idx} out of range")

            # Calculate offset within the sample
            offset = idx - sample['start_idx']

            # Extract sequence window from raw sample data
            data = sample['data']  # [1439, 6712]
            input_seq = data[offset:offset+self.sequence_length]  # [T, 6712]
            target_start = offset + self.time_step_offset
            target_end = target_start + self.sequence_length
            target_seq = data[target_start:target_end]  # [T, 6712]

            sample_tokens = sample.get('tokens')
            if sample_tokens is None:
                if self.cache_contains_tokens:
                    raise RuntimeError(
                        f"Sample {sample['sample_id']} is missing token ids despite cache requiring them. Rebuild cache."
                    )
                input_tokens_arr = None
                target_tokens_arr = None
            else:
                try:
                    input_tokens_arr = np.asarray(sample_tokens[offset:offset+self.sequence_length])
                    target_tokens_arr = np.asarray(sample_tokens[target_start:target_end])
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to slice cached tokens for sample {sample['sample_id']} at offset {offset}: {exc}"
                    )

            # Create metadata
            metadata = {
                'sample_id': sample['sample_id'],
                'start_time': sample['start_time'],
                'end_time': sample['end_time'],
                'sequence_offset': offset,
                'time_step_offset': self.time_step_offset
            }

        else:
            # Fallback: use pre-loaded sequences (for raw file loading)
            input_seq, target_seq = self.all_sequences[idx]
            metadata = self.sequence_metadata[idx]
            metadata = dict(metadata)
            metadata['time_step_offset'] = self.time_step_offset

            # Convert to numpy if needed
            if not isinstance(input_seq, np.ndarray):
                input_seq = np.array(input_seq, dtype=np.float32)
            if not isinstance(target_seq, np.ndarray):
                target_seq = np.array(target_seq, dtype=np.float32)

        input_tokens_tensor = None
        target_tokens_tensor = None
        if hasattr(self, 'samples') and self.samples and input_tokens_arr is not None and target_tokens_arr is not None:
            input_tokens_np = np.ascontiguousarray(input_tokens_arr, dtype=np.int64)
            target_tokens_np = np.ascontiguousarray(target_tokens_arr, dtype=np.int64)
            input_tokens_tensor = torch.from_numpy(input_tokens_np)
            target_tokens_tensor = torch.from_numpy(target_tokens_np)

        # Ensure float32 type, writable memory, and contiguous layout for tensor conversion
        input_np = np.asarray(input_seq, dtype=np.float32)
        target_np = np.asarray(target_seq, dtype=np.float32)
        input_tensor = _to_writable_contiguous_float32(input_np)      # [T, V]
        target_tensor = _to_writable_contiguous_float32(target_np)    # [T, V]

        # Clone reusable static tensors to avoid shared zero-stride views downstream
        if self._mask_tensor is not None:
            mask_tensor = self._mask_tensor.clone()
        else:
            mask_tensor = torch.from_numpy(np.ascontiguousarray(self.prediction_mask, dtype=np.int32))

        if self._attention_indices_tensor is not None:
            attention_indices_tensor = self._attention_indices_tensor.clone()
        else:
            attention_indices_tensor = torch.from_numpy(
                np.ascontiguousarray(self.attention_indices, dtype=np.int32)
            ).long()

        # Add static equipment data if available
        static_data = {}
        if self.processor.pipe_features is not None:
            static_data['pipe_features'] = torch.from_numpy(self.processor.pipe_features).float()  # [num_pipes, 9]
            static_data['pipe_names'] = list(self.processor.static_equipment_data['pipe_names'])

        if self.processor.compressor_features is not None:
            static_data['compressor_features'] = torch.from_numpy(self.processor.compressor_features).float()  # [num_compressors, pca_dim]
            static_data['compressor_names'] = list(self.processor.static_equipment_data['compressor_names'])

        # Create TensorDict
        tdict = {
            'input': input_tensor,
            'target': target_tensor,
            'mask': mask_tensor,
            'attention_indices': attention_indices_tensor,
            'static_data': static_data,
            'metadata': metadata
        }

        if input_tokens_tensor is not None and target_tokens_tensor is not None:
            tdict['input_tokens'] = input_tokens_tensor
            tdict['target_tokens'] = target_tokens_tensor

        batch = TensorDict(tdict, batch_size=torch.Size([]))
        return batch
    
    def get_feature_info(self) -> Dict:
        """
        Get information about features and dimensions.
        
        Returns:
            Dictionary with feature information
        """
        return {
            'total_dims': self.total_dims,
            'boundary_dims': 538,
            'equipment_dims': 6174,
            'sequence_length': self.sequence_length,
            'time_step_offset': self.time_step_offset,
            'prediction_mask_sum': int(np.sum(self.prediction_mask)) if self.prediction_mask is not None else 0,
            'equipment_breakdown': self.processor.equipment_info.copy(),
            'topology_attention_shape': self.attention_indices.shape if self.attention_indices is not None else 'Not loaded'
        }
    
    def get_data_statistics(self) -> Dict:
        """Get dataset statistics."""
        return {
            'split': self.split,
            'num_sequences': len(self.all_sequences),
            'num_samples': len(self.sample_dirs),
            'total_dims': self.total_dims,
            'sequence_length': self.sequence_length,
            'time_step_offset': self.time_step_offset
        }
    
    
    def get_sample_by_id(self, sample_id: str) -> List[int]:
        """
        Get all sequence indices for a specific sample ID.

        Args:
            sample_id: Sample identifier

        Returns:
            List of sequence indices belonging to this sample
        """
        indices: List[int] = []

        if hasattr(self, "sequence_metadata") and self.sequence_metadata:
            for i, metadata in enumerate(self.sequence_metadata):
                if metadata.get('sample_id') == sample_id:
                    indices.append(i)
            if indices:
                return indices

        if hasattr(self, "samples") and self.samples:
            for sample in self.samples:
                if sample.get('sample_id') != sample_id:
                    continue
                start_idx = int(sample.get('start_idx', 0))
                end_idx = int(sample.get('end_idx', start_idx))
                if end_idx > start_idx:
                    indices.extend(range(start_idx, end_idx))
            if indices:
                return indices

        return indices

    def map_index_2_variable_name(self, index: int) -> Optional[str]:
        """
        Map variable index to variable name.

        Args:
            index: Variable index (0-6711)

        Returns:
            Variable name or None if index not found
        """
        return self.index_to_variable.get(index)

    def map_variable_name_2_index(self, variable_name: str) -> Optional[int]:
        """
        Map variable name to variable index.

        Args:
            variable_name: Variable name (e.g., "B_001_p_in")

        Returns:
            Variable index or None if variable name not found
        """
        return self.variable_to_index.get(variable_name)


def collate_fn(batch_list: List[TensorDict], 
               normalizer: Optional[DataNormalizer] = None,
               apply_normalization: bool = True,
               tokenizer: Optional[DataTokenizer] = None) -> Dict[str, Any]:
    """
    Custom collate function for TensorDict batches with optional normalization
    and discretized token outputs.
    
    Args:
        batch_list: List of TensorDict samples
        normalizer: Optional DataNormalizer instance for data normalization
        apply_normalization: Whether to apply normalization (False for visualization)
        tokenizer: DataTokenizer for discretization (required)
        
    Returns:
        Regular dictionary with batched tensors
        - input: [B, T, V] 
        - target: [B, T, V]
        - prediction_mask: [B, V] - 预测变量mask (0=boundary, 1=equipment)
        - attention_indices: [B, V, max_neighbors_variable] - 拓扑注意力索引
        - input_tokens: [B, T, V] token ids
        - target_tokens: [B, T, V] token ids
    Raises:
        ValueError: If prediction mask variable dimension does not match input features.
    """
    B = len(batch_list)
    # Stack dynamic tensors
    inputs = torch.stack([item['input'] for item in batch_list]).contiguous()       # [B, T, V]
    targets = torch.stack([item['target'] for item in batch_list]).contiguous()     # [B, T, V]
    prediction_masks = torch.stack([item['mask'] for item in batch_list]).contiguous()  # [B, V]
    attention_indices = torch.stack([
        item['attention_indices'] for item in batch_list
    ]).contiguous()  # [B, V, num_neighbors_variable]

    if prediction_masks.shape[1] != inputs.shape[-1]:
        raise ValueError(
            f"Prediction mask dimension mismatch: mask has {prediction_masks.shape[1]} variables "
            f"but inputs have {inputs.shape[-1]}"
        )

    token_outputs: Dict[str, torch.Tensor] = {}
    sample_has_tokens = 'input_tokens' in batch_list[0] and 'target_tokens' in batch_list[0]
    def _validate_token_ids(name: str, tokens: torch.Tensor, vocab_size: int) -> None:
        if tokens.numel() == 0:
            return
        max_id = int(tokens.max().item())
        min_id = int(tokens.min().item())
        if min_id < 0 or max_id >= vocab_size:
            invalid_mask = (tokens < 0) | (tokens >= vocab_size)
            first_invalid = torch.nonzero(invalid_mask, as_tuple=False)[0]
            b_idx, t_idx, v_idx = (int(first_invalid[dim]) for dim in range(first_invalid.numel()))
            invalid_value = int(tokens[b_idx, t_idx, v_idx].item())
            raise RuntimeError(
                f"{name} contains out-of-range token id {invalid_value} at batch={b_idx}, time={t_idx}, "
                f"variable={v_idx}; expected range [0, {vocab_size - 1}], observed min={min_id}, max={max_id}"
            )

    if sample_has_tokens:
        input_tokens = torch.stack([item['input_tokens'] for item in batch_list]).to(dtype=torch.long)
        target_tokens = torch.stack([item['target_tokens'] for item in batch_list]).to(dtype=torch.long)
        token_outputs['input_tokens'] = input_tokens
        token_outputs['target_tokens'] = target_tokens
    else:
        raise RuntimeError(
            "Batch缺少预计算的token_ids；请确保缓存使用python build_cache.py重新生成v2.1版本后再加载数据集。"
        )

    def _sanitize_negatives(name: str) -> None:
        tokens = token_outputs.get(name)
        if tokens is None or tokens.numel() == 0:
            return
        negative_mask = tokens < 0
        if negative_mask.any():
            num_replaced = int(negative_mask.sum().item())
            logger.debug(
                "%s contained %d negative token ids; replacing them with 0 as padding.",
                name,
                num_replaced
            )
            tokens = tokens.clone()
            tokens[negative_mask] = 0
            token_outputs[name] = tokens

    _sanitize_negatives('input_tokens')
    _sanitize_negatives('target_tokens')

    token_vocab_size: Optional[int] = None
    if tokenizer is not None:
        vocab_size = tokenizer.vocab_size
        token_vocab_size = vocab_size
        _validate_token_ids("input_tokens", token_outputs['input_tokens'], vocab_size)
        _validate_token_ids("target_tokens", token_outputs['target_tokens'], vocab_size)

        cached_medians = getattr(tokenizer, '_cached_token_value_medians_tensor', None)
        cached_half_widths = getattr(tokenizer, '_cached_token_half_widths_tensor', None)
        if cached_medians is None or cached_half_widths is None:
            medians_np, half_widths_np = tokenizer.get_token_value_stats()
            cached_medians = torch.from_numpy(medians_np.astype(np.float32))
            cached_half_widths = torch.from_numpy(half_widths_np.astype(np.float32))
            tokenizer._cached_token_value_medians_tensor = cached_medians
            tokenizer._cached_token_half_widths_tensor = cached_half_widths

        input_token_ids = token_outputs['input_tokens']
        target_token_ids = token_outputs['target_tokens']

        input_token_medians = cached_medians[input_token_ids]
        target_token_medians = cached_medians[target_token_ids]
        input_token_half_widths = cached_half_widths[input_token_ids]
        target_token_half_widths = cached_half_widths[target_token_ids]

        token_outputs['input_token_medians'] = input_token_medians
        token_outputs['target_token_medians'] = target_token_medians

        # Keep raw (unnormalized) values for computing offsets
        epsilon = 1e-4
        input_diff = inputs - input_token_medians
        target_diff = targets - target_token_medians
        input_near_zero = input_token_half_widths.abs() < epsilon
        target_near_zero = target_token_half_widths.abs() < epsilon
        safe_input_half = torch.where(input_near_zero, torch.ones_like(input_token_half_widths), input_token_half_widths)
        safe_target_half = torch.where(target_near_zero, torch.ones_like(target_token_half_widths), target_token_half_widths)
        input_offsets = input_diff / safe_input_half
        target_offsets = target_diff / safe_target_half
        input_offsets = torch.where(input_near_zero, torch.zeros_like(input_offsets), input_offsets)
        target_offsets = torch.where(target_near_zero, torch.zeros_like(target_offsets), target_offsets)
        token_outputs['input_token_offsets'] = input_offsets
        token_outputs['target_token_offsets'] = target_offsets
    elif sample_has_tokens:
        max_token_id = max(
            int(token_outputs['input_tokens'].max().item()),
            int(token_outputs['target_tokens'].max().item())
        )
        token_vocab_size = max_token_id + 1
    
    # Apply normalization if requested and normalizer is provided
    if apply_normalization and normalizer is not None and normalizer.fitted:
        inputs = normalizer.transform(inputs)
        targets = normalizer.transform(targets)
        logger.debug(f"Applied {normalizer.method} normalization to batch")
    
    # Collect metadata as list
    metadata = [item['metadata'] for item in batch_list]

    result = {
        'input': inputs,                    # [B, T, V]
        'target': targets,                  # [B, T, V]  
        'prediction_mask': prediction_masks, # [B, V] - 哪些变量需要预测
        'attention_indices': attention_indices,  # [B, V, 64] - 拓扑注意力索引
        'metadata': metadata,
        'normalized': apply_normalization and normalizer is not None  # 标记是否已归一化
    }

    result.update(token_outputs)
    if token_vocab_size is None:
        raise RuntimeError("Unable to determine tokenizer vocabulary size for collated batch.")
    result['token_vocab_size'] = token_vocab_size
    
    return result


def create_collate_fn(normalizer: Optional[DataNormalizer] = None,
                     apply_normalization: bool = True,
                     tokenizer: Optional[DataTokenizer] = None,
                     target_dims: Optional[int] = None):
    """
    创建带有归一化参数的collate函数。
    
    Args:
        normalizer: 数据归一化器
        apply_normalization: 是否应用归一化
        tokenizer: 离散化token管理器
        
    Returns:
        配置好的collate函数
    """
    if tokenizer is None:
        raise RuntimeError("Tokenizer must be provided when creating the collate function.")

    if target_dims is None:
        raise RuntimeError("target_dims must be provided to align tokenizer with dataset.")

    tokenizer_holder: Dict[str, DataTokenizer] = {'tokenizer': tokenizer}

    def _ensure_tokenizer_alignment(expected_dims: int) -> DataTokenizer:
        current = tokenizer_holder['tokenizer']
        if current.total_dims != expected_dims:
            raise RuntimeError(
                f"Tokenizer dims ({current.total_dims}) do not match expected dims ({expected_dims}). "
                "请提前在静态目录生成并对齐 tokenizer。"
            )
        return current

    def _collate_fn(batch_list: List[TensorDict]) -> Dict[str, Any]:
        tokenizer_aligned = _ensure_tokenizer_alignment(target_dims)
        return collate_fn(
            batch_list,
            normalizer=normalizer,
            apply_normalization=apply_normalization,
            tokenizer=tokenizer_aligned
        )
    
    return _collate_fn


def load_normalizer(static_dir: str, method: str = 'standard') -> Optional[DataNormalizer]:
    """
    加载预计算的归一化器。
    
    Args:
        static_dir: 静态目录路径
        method: 归一化方法
        
    Returns:
        已加载的归一化器，如果加载失败返回None
    """
    try:
        normalizer = DataNormalizer(static_dir, method=method)
        if normalizer.load_stats():
            logger.info(f"Successfully loaded {method} normalizer from {static_dir}")
            return normalizer
        else:
            logger.warning(f"Failed to load {method} normalizer from {static_dir}")
            return None
    except Exception as e:
        logger.error(f"Error loading normalizer from {static_dir}: {e}")
        return None


def create_fast_dataset(data_dir: str,
                       split: str = 'train',
                       sequence_length: int = 3,
                       use_cache: bool = True,
                       force_rebuild_cache: bool = False,
                       time_step_offset: int = 1) -> FluidDataset:
    """
    Create FluidDataset with optimized settings for fast loading.

    Args:
        data_dir: Path to data directory
        split: 'train', 'val', or 'test'
        sequence_length: Length of time series sequences
        use_cache: Whether to use cache for fast loading
        force_rebuild_cache: Force rebuilding cache
        time_step_offset: Number of minutes to shift target sequences

    Returns:
        FluidDataset instance
    """
    return FluidDataset(
        data_dir=data_dir,
        split=split,
        sequence_length=sequence_length,
        use_cache=use_cache,
        force_rebuild_cache=force_rebuild_cache,
        time_step_offset=time_step_offset
    )


def create_dataloader_with_normalization(dataset: FluidDataset,
                                        batch_size: int = 32,
                                        shuffle: bool = False,
                                        num_workers: int = 0,
                                        normalizer_method: str = 'standard',
                                        apply_normalization: bool = True,
                                        tokenizer: Optional[DataTokenizer] = None,
                                        tokenizer_vocab_size: Optional[int] = None,
                                        tokenizer_stats_path: Optional[str] = None) -> torch.utils.data.DataLoader:
    """
    创建带有归一化功能的DataLoader。
    
    Args:
        dataset: FluidDataset实例
        batch_size: 批次大小
        shuffle: 是否随机打乱
        num_workers: 工作进程数
        normalizer_method: 归一化方法
        apply_normalization: 是否应用归一化（可视化时设为False）
        tokenizer: 预加载的tokenizer实例（可选）
        tokenizer_vocab_size: tokenizer词表大小
        
    Returns:
        配置好的DataLoader
    """
    # 加载归一化器
    normalizer = None
    if apply_normalization:
        base_normalizer = load_normalizer(str(dataset.static_dir), normalizer_method)
        if base_normalizer is None:
            logger.warning("Normalizer not found. Consider running compute_normalization_stats.py first")
            apply_normalization = False
        else:
            target_dims = getattr(dataset, 'total_dims', base_normalizer.total_dims)
            if target_dims != base_normalizer.total_dims:
                logger.error(
                    "Normalizer dims (%s) do not match dataset dims (%s) for static_dir %s",
                    base_normalizer.total_dims,
                    target_dims,
                    dataset.static_dir,
                )
                apply_normalization = False
            else:
                normalizer = base_normalizer
    
    # 加载tokenizer（必选）
    tokenizer_instance: Optional[DataTokenizer] = tokenizer
    loaded_from_disk = False
    tokenizer_stats_dir: Optional[Path] = None
    tokenizer_base_dir: Optional[Path] = None
    if tokenizer_instance is None:
        def _resolve_tokenizer_dirs(static_directory: Path, stats_path: Optional[str]) -> Tuple[Path, Path]:
            base_dir = static_directory
            stats_dir = base_dir / "tokenizer_save"
            if stats_path:
                resolved = Path(stats_path).expanduser().resolve()
                if resolved.is_file():
                    if resolved.name.lower() != "token_stats.csv":
                        raise ValueError(
                            f"tokenizer_stats_path file must be token_stats.csv, got {resolved.name}"
                        )
                    if resolved.parent.name != "tokenizer_save":
                        raise ValueError(
                            f"tokenizer_stats_path file must reside inside tokenizer_save/: {resolved}"
                        )
                    stats_dir = resolved.parent
                    base_dir = stats_dir.parent
                elif resolved.is_dir():
                    if (resolved / "token_stats.csv").exists():
                        stats_dir = resolved
                        base_dir = stats_dir.parent
                    else:
                        raise ValueError(
                            f"tokenizer_stats_path directory {resolved} does not contain token_stats.csv"
                        )
                else:
                    raise ValueError(f"tokenizer_stats_path does not exist: {resolved}")
            elif not stats_dir.exists():
                raise ValueError(
                    f"Tokenizer stats directory not found at default location: {stats_dir}"
                )

            if not (stats_dir / "token_stats.csv").exists():
                raise ValueError(f"token_stats.csv not found in tokenizer stats directory: {stats_dir}")

            return base_dir, stats_dir

        tokenizer_base_dir, tokenizer_stats_dir = _resolve_tokenizer_dirs(Path(dataset.static_dir), tokenizer_stats_path)
        if tokenizer_vocab_size is not None:
            tokenizer_instance = load_tokenizer_from_stats(
                tokenizer_base_dir,
                vocab_size=tokenizer_vocab_size
            )
        else:
            tokenizer_instance = load_tokenizer_from_stats(tokenizer_base_dir)
        loaded_from_disk = True
    if tokenizer_instance is None:
        raise RuntimeError("Tokenizer statistics not found. Run compute_tokenizer_stats.py first.")
    else:
        if tokenizer_stats_dir is None:
            tokenizer_stats_dir = Path(dataset.static_dir) / "tokenizer_save"
            tokenizer_base_dir = Path(dataset.static_dir)
        if loaded_from_disk:
            logger.info(
                "Loaded tokenizer from %s (vocab=%s)",
                tokenizer_stats_dir,
                getattr(tokenizer_instance, 'vocab_size', 'unknown')
            )
        else:
            logger.info(
                "Reusing provided tokenizer for %s (vocab=%s)",
                tokenizer_stats_dir,
                getattr(tokenizer_instance, 'vocab_size', 'unknown')
            )

    target_dims = getattr(dataset, 'total_dims', None)
    if target_dims is None:
        raise RuntimeError("Dataset missing total_dims attribute required for tokenizer alignment.")

    if tokenizer_instance.total_dims != target_dims:
        raise RuntimeError(
            f"Tokenizer dims ({tokenizer_instance.total_dims}) do not match dataset dims ({target_dims}) "
            f"for static_dir {dataset.static_dir}."
        )

    if hasattr(dataset, "set_tokenizer"):
        dataset.set_tokenizer(tokenizer_instance)

    # 创建collate函数
    collate_func = create_collate_fn(
        normalizer,
        apply_normalization,
        tokenizer=tokenizer_instance,
        target_dims=target_dims
    )
    
    # 创建DataLoader with optimized settings
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_func,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False,  # 保持worker进程
        prefetch_factor=2 if num_workers > 0 else None  # 预读取因子
    )
    
    logger.info(
        f"Created DataLoader: batch_size={batch_size}, normalization={apply_normalization}, "
        f"method={normalizer_method if normalizer else 'none'}, topology_attention=enabled, "
        f"token_vocab={tokenizer_instance.vocab_size}"
    )
    
    return dataloader


def manage_cache(data_dir: str, action: str = 'info', sequence_length: int = 3) -> Dict:
    """
    Manage dataset cache operations.
    
    Args:
        data_dir: Path to data directory
        action: 'info' | 'build' | 'clear' | 'validate'
        sequence_length: Sequence length for cache building
        
    Returns:
        Dictionary with operation results
    """
    cache_manager = CacheManager(data_dir)
    
    if action == 'info':
        return cache_manager.get_cache_info()
    
    elif action == 'build':
        processor = DataProcessor(data_dir)
        tokenizer = load_tokenizer_from_stats(data_dir)
        if tokenizer is None or not getattr(tokenizer, 'fitted', False):
            raise RuntimeError(
                "Tokenizer statistics not found in data/tokenizer_save. Run data/compute_tokenizer_stats.py before building cache."
            )
        cache_manager.build_cache(processor, tokenizer)
        return {'status': 'success', 'message': 'Cache built successfully'}
    
    elif action == 'clear':
        cache_manager.clear_cache()
        return {'status': 'success', 'message': 'Cache cleared successfully'}
    
    elif action == 'validate':
        is_valid = cache_manager._is_cache_valid()
        return {
            'status': 'success', 
            'is_valid': is_valid,
            'message': f'Cache is {"valid" if is_valid else "invalid"}'
        }
    
    else:
        raise ValueError(f"Invalid action: {action}. Must be 'info', 'build', 'clear', or 'validate'")
