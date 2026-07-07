"""
Training utilities for fluid dynamics models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Any
import logging
import os
import random
import numpy as np
from pathlib import Path

from transformers import set_seed
import swanlab
from models import FluidDecoder
from data.dataset import FluidDataset, create_dataloader_with_normalization
from data.normalizer import load_normalizer, DataNormalizer
from data.tokenizer_save import load_tokenizer as load_tokenizer_from_stats, DataTokenizer
from .trainer import create_fluid_trainer, FluidTrainer
from .config import TrainingConfig, create_default_training_config
from .callbacks import create_default_callbacks

logger = logging.getLogger(__name__)


def _align_normalizer_with_dataset(
    normalizer: Optional[DataNormalizer],
    dataset: Optional[FluidDataset]
) -> Optional[DataNormalizer]:
    """Ensure the provided normalizer matches the dataset dimensions."""
    if normalizer is None or dataset is None:
        return normalizer

    target_dims = getattr(dataset, "total_dims", None)
    if target_dims is None:
        return normalizer

    if target_dims != normalizer.total_dims:
        logger.error(
            "Normalizer dims (%d) do not match dataset dims (%d) for static_dir %s. Disabling normalization.",
            normalizer.total_dims,
            target_dims,
            getattr(dataset, "static_dir", "unknown"),
        )
        return None

    return normalizer


def _resolve_tokenizer_dirs(config: TrainingConfig) -> Tuple[Path, Path]:
    """
    Resolve base data directory and tokenizer stats directory.

    tokenizer_stats_path may point to:
    - tokenizer_save directory
    - token_stats.csv inside tokenizer_save
    - or fall back to data_dir/tokenizer_save
    """
    if config.static_dir:
        static_path = Path(config.static_dir).expanduser()
    else:
        static_path = Path(config.data_dir).expanduser() / "static" / "full"

    base_dir = static_path
    stats_dir = static_path / "tokenizer_save"

    if config.tokenizer_stats_path:
        stats_path = Path(config.tokenizer_stats_path).expanduser()
        if stats_path.is_file():
            if stats_path.name.lower() != "token_stats.csv":
                raise ValueError(
                    f"tokenizer_stats_path file must be token_stats.csv, got {stats_path.name}"
                )
            if stats_path.parent.name != "tokenizer_save":
                raise ValueError(
                    f"tokenizer_stats_path file must reside inside tokenizer_save/: {stats_path}"
                )
            stats_dir = stats_path.parent
            base_dir = stats_dir.parent
        elif stats_path.is_dir():
            if (stats_path / "token_stats.csv").exists():
                stats_dir = stats_path
                base_dir = stats_dir.parent
            else:
                raise ValueError(
                    f"tokenizer_stats_path directory {stats_path} does not contain token_stats.csv"
                )
        else:
            raise ValueError(f"tokenizer_stats_path does not exist: {stats_path}")
    else:
        if not stats_dir.exists():
            raise ValueError(
                f"Tokenizer stats directory not found at default location: {stats_dir}"
            )

    if not (stats_dir / "token_stats.csv").exists():
        raise ValueError(f"token_stats.csv not found in tokenizer stats directory: {stats_dir}")

    return base_dir, stats_dir


def _load_tokenizer_for_training(config: TrainingConfig) -> DataTokenizer:
    """Load tokenizer stats once and update training config with derived vocab size."""
    base_dir, stats_dir = _resolve_tokenizer_dirs(config)
    tokenizer = load_tokenizer_from_stats(base_dir)
    if tokenizer is None:
        raise RuntimeError(
            f"Failed to load tokenizer statistics from {stats_dir}"
        )

    config.tokenizer_vocab_size = int(tokenizer.vocab_size)
    config.tokenizer_stats_path = str(stats_dir)
    logger.info(
        "Loaded tokenizer stats from %s (vocab_size=%d)",
        stats_dir,
        config.tokenizer_vocab_size,
    )
    return tokenizer


def _attach_hybrid_statistics(model: Any, tokenizer: Optional[DataTokenizer]) -> None:
    """Attach tokenizer-derived statistics required for hybrid decoder mode."""

    if not isinstance(model, FluidDecoder):
        return

    projection_type = getattr(model.config, "input_projection_type", "").lower()
    if projection_type != "hybrid":
        return

    if tokenizer is None:
        raise ValueError("Hybrid projection requires a tokenizer instance to provide value statistics.")

    medians_np, half_widths_np = tokenizer.get_token_value_stats()
    medians_tensor = torch.from_numpy(medians_np.astype(np.float32))
    half_widths_tensor = torch.from_numpy(half_widths_np.astype(np.float32))
    model.set_token_value_statistics(medians_tensor, half_widths_tensor)


def setup_training_environment(config: TrainingConfig) -> None:
    """
    Setup training environment with proper logging, seeds, and device configuration.

    Args:
        config: Training configuration
    """
    # Check if CUDA_VISIBLE_DEVICES was already set (by train.py early setup)
    cuda_visible_devices_set = 'CUDA_VISIBLE_DEVICES' in os.environ

    # Only set CUDA_VISIBLE_DEVICES if not already set (should be set by train.py)
    if not cuda_visible_devices_set and hasattr(config, 'device') and config.device is not None and config.device != "auto":
        logger.warning("CUDA_VISIBLE_DEVICES should be set before importing torch. Setting it now may not have the desired effect.")
        if isinstance(config.device, str):
            if config.device.startswith('cuda:'):
                # Extract GPU ID from cuda:X format
                gpu_id = config.device.split(':')[1]
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
                cuda_visible_devices_set = True
            elif config.device.lower() == 'cpu':
                # Use CPU only - set empty CUDA_VISIBLE_DEVICES
                os.environ['CUDA_VISIBLE_DEVICES'] = ""
            elif config.device.isdigit():
                # String GPU ID
                os.environ['CUDA_VISIBLE_DEVICES'] = config.device
                cuda_visible_devices_set = True
        elif isinstance(config.device, int):
            # Integer GPU ID - special handling for negative values (CPU mode)
            if config.device < 0:
                # Use CPU only - set empty CUDA_VISIBLE_DEVICES
                os.environ['CUDA_VISIBLE_DEVICES'] = ""
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(config.device)
                cuda_visible_devices_set = True

    # Set up logging (需要在logger.info之前设置)
    log_level = logging.DEBUG if config.debug_mode else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(config.output_dir, 'training.log'))
        ]
    )

    logger.info("Setting up training environment...")

    # Log CUDA_VISIBLE_DEVICES setup if it was configured
    if cuda_visible_devices_set:
        logger.info(f"Set CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, device will be remapped to cuda:0")
    
    # Set random seeds for reproducibility
    set_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        # For deterministic behavior (may reduce performance)
        if config.debug_mode:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seeds set to {config.seed}")
    
    # Device setup - HuggingFace Trainer will handle device placement
    # We just log the configuration for debugging
    if config.device == "auto":
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                logger.info(f"Found {gpu_count} GPUs, will use all GPUs for training")
            else:
                logger.info(f"Found 1 GPU, will use single GPU training")
        else:
            logger.info("No GPUs available, will use CPU")
    else:
        # Specific device was requested
        if cuda_visible_devices_set:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"Using GPU {config.device} (remapped to cuda:0 via CUDA_VISIBLE_DEVICES)")
                logger.info(f"Available GPUs after mapping: {gpu_count}")
            else:
                logger.warning("CUDA not available, will fall back to CPU")
        else:
            logger.info(f"Using device configuration: {config.device}")

    # Log GPU information if available
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            try:
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                logger.info(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
            except (RuntimeError, IndexError) as e:
                logger.warning(f"Could not get info for GPU {i}: {e}")
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.logging_dir, exist_ok=True)
    
    # Save training config
    config.save_to_file(os.path.join(config.output_dir, "training_config.json"))
    
    # Initialize SwanLab if configured
    if config.use_swanlab:  # Keep config name for backward compatibility
        swanlab.init(
            project=config.swanlab_project,
            workspace=config.swanlab_entity,
            experiment_name=config.swanlab_run_name,
            config=config.to_dict(),
            logdir=config.output_dir
        )
        logger.info(f"Initialized SwanLab run: {config.swanlab_run_name}")
    
    logger.info("Training environment setup complete")


def create_model(config: TrainingConfig):
    """
    Create and initialize fluid model.

    Args:
        config: Training configuration

    Returns:
        Fluid model instance
    """
    logger.info("Creating fluid model...")

    # Load model config from separate file
    model_config = config.load_model_config()

    # Override with training config values where appropriate
    model_config.sequence_length = config.sequence_length
    # Keep model aware of the prediction offset for downstream losses/schedules
    model_config.time_step_offset = getattr(config, 'time_step_offset', 1)
    if getattr(config, "static_dir", None):
        model_config.static_dir = config.static_dir

    # Create model
    model_name = model_config.model_name.lower()

    if hasattr(model_config, "tokenizer_vocab_size"):
        model_config.tokenizer_vocab_size = getattr(config, "tokenizer_vocab_size", model_config.tokenizer_vocab_size)

    if model_name != "fluiddecoder":
        raise ValueError(
            f"Only FluidDecoder is supported in the open-source package, got: {model_name}"
        )
    model = FluidDecoder(model_config)

    # Count parameters for logging
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Created {model_name} with {trainable_params:,} trainable parameters")

    return model


def create_datasets(
    config: TrainingConfig,
    *,
    tokenizer: Optional[DataTokenizer] = None,
) -> Tuple[FluidDataset, Optional[FluidDataset], Optional[FluidDataset]]:
    """
    Create train, validation, and test datasets.

    Args:
        config: Training configuration

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    logger.info("Creating datasets...")

    # Get cache_dir from config if available
    cache_dir = getattr(config, 'cache_dir', None)
    if not cache_dir and config.static_dir:
        cache_dir = str(Path(config.static_dir) / "cache")
    if cache_dir:
        logger.info(f"Using cache directory: {cache_dir}")

    # Create training dataset
    train_dataset = FluidDataset(
        data_dir=config.data_dir,
        split='train',
        sequence_length=config.sequence_length,
        time_step_offset=config.time_step_offset,
        use_cache=True,
        max_sequences_per_sample=config.max_sequences_per_sample,
        cache_dir=cache_dir,
        static_dir=config.static_dir,
        predict_variable_name=config.predict_variable_name,
        tokenizer=tokenizer,
    )

    logger.info(f"Training dataset: {len(train_dataset)} sequences")

    # Create validation dataset
    val_dataset = FluidDataset(
        data_dir=config.data_dir,
        split='val',
        sequence_length=config.sequence_length,
        time_step_offset=config.time_step_offset,
        use_cache=True,
        max_sequences_per_sample=config.max_sequences_per_sample,
        cache_dir=cache_dir,
        static_dir=config.static_dir,
        predict_variable_name=config.predict_variable_name,
        tokenizer=tokenizer,
    )

    logger.info(f"Validation dataset: {len(val_dataset)} sequences")

    test_dataset = None

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset: FluidDataset,
    val_dataset: Optional[FluidDataset],
    config: TrainingConfig,
    *,
    tokenizer: Optional[DataTokenizer] = None
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create data loaders with normalization.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    logger.info("Creating data loaders...")

    if config.tokenizer_vocab_size is None:
        raise RuntimeError(
            "Tokenizer vocabulary size is not set. Ensure tokenizer stats are loaded before creating dataloaders."
        )
    
    # Create training dataloader
    train_dataloader = create_dataloader_with_normalization(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        normalizer_method=config.normalization_method,
        apply_normalization=config.apply_normalization,
        tokenizer=tokenizer,
        tokenizer_vocab_size=config.tokenizer_vocab_size
    )
    
    logger.info(f"Training dataloader: {len(train_dataloader)} batches")
    
    # Create validation dataloader
    val_dataloader = None
    if val_dataset:
        val_dataloader = create_dataloader_with_normalization(
            val_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            normalizer_method=config.normalization_method,
            apply_normalization=config.apply_normalization,
            tokenizer=tokenizer,
            tokenizer_vocab_size=config.tokenizer_vocab_size
        )
        logger.info(f"Validation dataloader: {len(val_dataloader)} batches")
    
    return train_dataloader, val_dataloader


def setup_training(config: Optional[TrainingConfig] = None, **config_kwargs) -> Dict[str, Any]:
    """
    Complete training setup including environment, model, datasets, and trainer.
    
    Args:
        config: Training configuration (if None, creates default)
        **config_kwargs: Override config parameters
        
    Returns:
        Dictionary containing all training components
    """
    # Create config if not provided
    if config is None:
        config = create_default_training_config(**config_kwargs)
    else:
        # Update config with any overrides
        for key, value in config_kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Setup environment
    setup_training_environment(config)

    # Load tokenizer stats once and derive vocab size
    tokenizer = _load_tokenizer_for_training(config)
    
    # Create model
    model = create_model(config)
    _attach_hybrid_statistics(model, tokenizer)
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(config, tokenizer=tokenizer)
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        train_dataset,
        val_dataset,
        config,
        tokenizer=tokenizer
    )
    
    # Load normalizer if needed
    normalizer = None
    if config.apply_normalization:
        static_dir = config.static_dir or str(Path(config.data_dir) / "static" / "full")
        normalizer = load_normalizer(static_dir, config.normalization_method)
        if normalizer is None:
            logger.warning("Normalizer not found. Training will proceed without normalization.")
        else:
            normalizer = _align_normalizer_with_dataset(normalizer, train_dataset)
            normalizer = _align_normalizer_with_dataset(normalizer, val_dataset)
            if normalizer is None:
                logger.warning("Normalization disabled for trainer due to dimension mismatch.")
    
    # Create trainer
    trainer = create_fluid_trainer(
        model=model,
        training_config=config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        normalizer=normalizer,
        tokenizer=tokenizer,
        callbacks=create_default_callbacks(
            log_memory_usage=True,
        )
    )

    # Attach tokenizer to trainer for downstream consumers (evaluation/inference reuse)
    setattr(trainer, "tokenizer", tokenizer)
    
    return {
        'config': config,
        'model': model,
        'trainer': trainer,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'train_dataloader': train_dataloader,
        'val_dataloader': val_dataloader,
        'normalizer': normalizer,
        'tokenizer': tokenizer
    }


def run_training(trainer: FluidTrainer, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the training process.
    
    Args:
        trainer: Configured FluidTrainer
        resume_from_checkpoint: Path to checkpoint to resume from
        
    Returns:
        Training results dictionary
    """
    logger.info("Starting training...")
    
    try:
        # Run training
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        trainer.save_model()
        
        # Final evaluation
        eval_result = {}
        if trainer.get_eval_dataloader() is not None:
            logger.info("Running final evaluation...")
            eval_result = trainer.evaluate()
        
        # Training summary
        training_summary = {
            'train_result': train_result,
            'eval_result': eval_result,
            'model_path': trainer.args.output_dir,
            'total_steps': trainer.state.global_step,
            'best_metric': trainer.state.best_metric,
        }
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {trainer.args.output_dir}")
        
        if eval_result:
            logger.info(f"Final evaluation loss: {eval_result.get('eval_loss', 'N/A')}")
        
        return training_summary
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Cleanup
        if trainer.training_config.use_swanlab:
            swanlab.finish()


def evaluate_model(
    trainer: FluidTrainer,
    test_dataset: Optional[FluidDataset] = None
) -> Dict[str, Any]:
    """
    Evaluate trained model on test dataset.

    Args:
        trainer: Trained FluidTrainer
        test_dataset: Test dataset (optional)

    Returns:
        Evaluation results
    """
    logger.info("Evaluating model...")
    
    # Use provided test dataset or trainer's eval dataset
    if test_dataset is not None:
        # Create test dataloader
        tokenizer = getattr(trainer, "tokenizer", None)
        test_dataloader = create_dataloader_with_normalization(
            test_dataset,
            batch_size=trainer.training_config.eval_batch_size,
            shuffle=False,
            num_workers=trainer.training_config.num_workers,
            normalizer_method=trainer.training_config.normalization_method,
            apply_normalization=trainer.training_config.apply_normalization,
            tokenizer=tokenizer,
            tokenizer_vocab_size=trainer.training_config.tokenizer_vocab_size,
            tokenizer_stats_path=trainer.training_config.tokenizer_stats_path
        )
        
        # Temporarily replace eval dataloader
        original_eval_dataloader = trainer._eval_dataloader
        trainer._eval_dataloader = test_dataloader
        
        try:
            eval_result = trainer.evaluate(metric_key_prefix="test")
        finally:
            # Restore original eval dataloader
            trainer._eval_dataloader = original_eval_dataloader
    else:
        eval_result = trainer.evaluate(metric_key_prefix="test")
    
    logger.info("Model evaluation completed")
    for key, value in eval_result.items():
        if key.startswith("test_"):
            logger.info(f"{key}: {value:.6f}")
    
    return eval_result


def load_model(model_folder_path: str, device: str = "cpu") -> nn.Module:
    """
    从训练输出文件夹加载模型

    Args:
        model_folder_path: 模型文件夹路径（如 "outputs/full_training_decoder_nano/"）
        device: 设备

    Returns:
        加载的模型
    """
    import json
    from pathlib import Path
    from safetensors.torch import load_file

    model_folder = Path(model_folder_path)

    # 读取训练配置
    training_config_path = model_folder / "training_config.json"
    if not training_config_path.exists():
        raise FileNotFoundError(f"Training config not found: {training_config_path}")

    with open(training_config_path, 'r') as f:
        training_config_dict = json.load(f)

    # 获取模型配置路径
    model_config_path = training_config_dict.get('model_config_path')
    if not model_config_path:
        raise ValueError("model_config_path not found in training config")

    # 如果模型配置路径是相对路径，则相对于项目根目录
    if not Path(model_config_path).is_absolute():
        project_root = Path(__file__).parent.parent
        model_config_path = project_root / model_config_path

    # 创建训练配置对象
    from .config import TrainingConfig
    config = TrainingConfig.from_dict(training_config_dict)

    # 创建模型
    model = create_model(config)

    # 查找safetensors文件
    safetensors_files = list(model_folder.glob("**/model.safetensors"))
    if not safetensors_files:
        # 如果没有找到model.safetensors，查找pytorch_model.bin
        bin_files = list(model_folder.glob("**/pytorch_model.bin"))
        if not bin_files:
            raise FileNotFoundError(f"No model weights found in {model_folder}")

        # 加载pytorch_model.bin
        load_kwargs = {"map_location": device}
        try:
            state_dict = torch.load(bin_files[0], weights_only=True, **load_kwargs)
        except TypeError:
            state_dict = torch.load(bin_files[0], **load_kwargs)
    else:
        # 加载safetensors
        state_dict = load_file(safetensors_files[0])

    # 加载权重
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    logger.info(f"Model loaded from {model_folder}")
    return model
