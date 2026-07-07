import torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Any
import logging
from pathlib import Path

from .dataset import FluidDataset, create_dataloader_with_normalization
from .normalizer import DataNormalizer
from .tokenizer_save import DataTokenizer, load_tokenizer as load_tokenizer_from_stats


def _resolve_tokenizer_dirs(data_dir: str, tokenizer_stats_path: Optional[str], static_dir: Optional[str]) -> Tuple[Path, Path]:
    if static_dir:
        stats_dir = Path(static_dir).expanduser().resolve() / "tokenizer_save"
        base_dir = stats_dir.parent.parent if stats_dir.parent.name == "tokenizer_save" else Path(data_dir).expanduser().resolve()
    else:
        base_dir = Path(data_dir).expanduser().resolve()
        stats_dir = base_dir / "tokenizer_save"

    if tokenizer_stats_path:
        resolved = Path(tokenizer_stats_path).expanduser().resolve()
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

logger = logging.getLogger(__name__)

# Legacy collate function removed - use unified collate_fn from dataset.py
# This ensures compatibility with the new unified architecture where boundary 
# and equipment data are combined into a single [B, T, 6712] tensor

def create_data_loaders(data_dir: str,
                       batch_size: int = 32,
                       num_workers: int = 0,
                       sequence_length: int = 3,
                       time_step_offset: int = 1,
                       normalize: bool = True,
                       normalizer_method: str = 'standard',
                       shuffle: bool = True,
                       use_cache: bool = True,
                       cache_dir: Optional[str] = None,
                       static_dir: Optional[str] = None,
                       tokenizer_vocab_size: Optional[int] = None,
                       tokenizer: Optional[DataTokenizer] = None,
                       tokenizer_stats_path: Optional[str] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders using the unified architecture.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        sequence_length: Length of time series sequences (default: 3 minutes)
        time_step_offset: Minutes to shift targets forward relative to inputs
        normalize: Whether to normalize the data
        normalizer_method: Type of normalization ('standard' or 'minmax')
        shuffle: Whether to shuffle training data
        use_cache: Whether to use precomputed cache for fast loading
        cache_dir: Custom cache directory path (default: static_dir/cache or data/cache)
        static_dir: Static graph directory (default: data/static/full)
        tokenizer_vocab_size: Vocabulary size for token discretization (derived from stats when omitted)
        tokenizer: Optional preloaded tokenizer instance to reuse
        tokenizer_stats_path: Optional path pointing to tokenizer stats CSV or tokenizer_save directory

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info(
        f"Creating unified data loaders: batch_size={batch_size}, sequence_length={sequence_length}, "
        f"time_step_offset={time_step_offset}"
    )
    if cache_dir:
        logger.info(f"Using custom cache directory: {cache_dir}")
    
    tokenizer_instance = tokenizer
    tokenizer_base_dir, tokenizer_stats_dir = _resolve_tokenizer_dirs(data_dir, tokenizer_stats_path, static_dir)
    if tokenizer_instance is None:
        tokenizer_instance = load_tokenizer_from_stats(tokenizer_base_dir)
    if tokenizer_instance is None:
        raise RuntimeError(f"Tokenizer statistics not found under {tokenizer_stats_dir}")
    if tokenizer_vocab_size is None:
        tokenizer_vocab_size = tokenizer_instance.vocab_size
    
    # Create datasets using the new unified FluidDataset
    train_dataset = FluidDataset(
        data_dir=data_dir,
        split='train',
        sequence_length=sequence_length,
        time_step_offset=time_step_offset,
        use_cache=use_cache,
        cache_dir=cache_dir,
        static_dir=static_dir
    )

    val_dataset = FluidDataset(
        data_dir=data_dir,
        split='val',
        sequence_length=sequence_length,
        time_step_offset=time_step_offset,
        use_cache=use_cache,
        cache_dir=cache_dir,
        static_dir=static_dir
    )

    test_dataset = FluidDataset(
        data_dir=data_dir,
        split='test',
        sequence_length=sequence_length,
        time_step_offset=time_step_offset,
        use_cache=use_cache,
        cache_dir=cache_dir,
        static_dir=static_dir
    )
    
    # Create data loaders with unified collate function and normalization
    train_loader = create_dataloader_with_normalization(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        normalizer_method=normalizer_method,
        apply_normalization=normalize,
        tokenizer=tokenizer_instance,
        tokenizer_vocab_size=tokenizer_vocab_size
    )
    
    val_loader = create_dataloader_with_normalization(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        normalizer_method=normalizer_method,
        apply_normalization=normalize,
        tokenizer=tokenizer_instance,
        tokenizer_vocab_size=tokenizer_vocab_size
    )
    
    test_loader = create_dataloader_with_normalization(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        normalizer_method=normalizer_method,
        apply_normalization=normalize,
        tokenizer=tokenizer_instance,
        tokenizer_vocab_size=tokenizer_vocab_size
    )
    
    # Log dataset statistics
    logger.info(f"Train dataset: {len(train_dataset)} sequences")
    logger.info(f"Validation dataset: {len(val_dataset)} sequences") 
    logger.info(f"Test dataset: {len(test_dataset)} sequences")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

def create_inference_loader(data_dir: str,
                           batch_size: int = 32,
                           num_workers: int = 0,
                           sequence_length: int = 3,
                           time_step_offset: int = 1,
                           normalizer_method: str = 'standard',
                           tokenizer_vocab_size: Optional[int] = None,
                           tokenizer: Optional[DataTokenizer] = None,
                           tokenizer_stats_path: Optional[str] = None) -> DataLoader:
    """
    Create data loader specifically for inference on test data using unified architecture.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size for inference
        num_workers: Number of worker processes
        sequence_length: Length of time series sequences
        time_step_offset: Minutes to shift targets forward relative to inputs
        normalizer_method: Normalization method to use
        tokenizer_vocab_size: Vocabulary size for discretization (derived when omitted)
        tokenizer: Optional preloaded tokenizer instance
        tokenizer_stats_path: Optional path pointing to tokenizer stats CSV or tokenizer_save directory
        
    Returns:
        Test data loader configured for inference
    """
    test_dataset = FluidDataset(
        data_dir=data_dir,
        split='test',
        sequence_length=sequence_length,
        time_step_offset=time_step_offset,
        use_cache=True
    )
    
    tokenizer_instance = tokenizer
    tokenizer_base_dir, tokenizer_stats_dir = _resolve_tokenizer_dirs(data_dir, tokenizer_stats_path)
    if tokenizer_instance is None:
        tokenizer_instance = load_tokenizer_from_stats(tokenizer_base_dir)
    if tokenizer_instance is None:
        raise RuntimeError(f"Tokenizer statistics not found under {tokenizer_stats_dir}")
    if tokenizer_vocab_size is None:
        tokenizer_vocab_size = tokenizer_instance.vocab_size

    # Create data loader with normalization
    test_loader = create_dataloader_with_normalization(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        normalizer_method=normalizer_method,
        apply_normalization=True,
        tokenizer=tokenizer_instance,
        tokenizer_vocab_size=tokenizer_vocab_size
    )
    
    logger.info(
        f"Created inference loader: {len(test_dataset)} sequences, {len(test_loader)} batches, "
        f"time_step_offset={time_step_offset}"
    )
    
    return test_loader

def get_sample_batch(data_loader: DataLoader) -> Dict[str, torch.Tensor]:
    """
    Get a sample batch from data loader for testing/debugging.
    
    Args:
        data_loader: DataLoader to sample from
        
    Returns:
        Sample batch dictionary with unified format
    """
    try:
        batch = next(iter(data_loader))
        logger.info(f"Sample batch shapes (unified format):")
        logger.info(f"  Input: {batch['input'].shape}")
        logger.info(f"  Target: {batch['target'].shape}")
        logger.info(f"  Prediction mask: {batch['prediction_mask'].shape}")
        logger.info(f"  Attention indices: {batch['attention_indices'].shape}")
        if 'input_tokens' in batch:
            logger.info(f"  Input tokens: {batch['input_tokens'].shape}")
        if 'target_tokens' in batch:
            logger.info(f"  Target tokens: {batch['target_tokens'].shape}")
        logger.info(f"  Normalized: {batch['normalized']}")
        logger.info(f"  Metadata count: {len(batch['metadata'])}")
        
        return batch
        
    except Exception as e:
        logger.error(f"Error getting sample batch: {e}")
        raise

# Legacy AutoregressiveCollator removed - autoregressive structure is now
# handled in the unified architecture where input/target pairs are automatically
# generated with configurable time_step_offset-minute shifts during dataset preprocessing
