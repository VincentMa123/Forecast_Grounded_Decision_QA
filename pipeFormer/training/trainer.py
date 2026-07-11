"""
Custom trainer for fluid dynamics models with HuggingFace PreTrainedModel support.
"""

import torch
from typing import Dict, List, Optional
import logging
import os
import json
import datetime
import pandas as pd
import time
from pathlib import Path

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import numpy as np

from data import DataNormalizer, DataTokenizer, create_collate_fn, load_tokenizer
from .config import TrainingConfig
# Direct import already provided via data package
from models.base import BaseModel

logger = logging.getLogger(__name__)


class FluidDataCollator:
    """
    Data collator that converts our custom data format to HuggingFace format.
    """

    def __init__(self,
                 normalizer: Optional[DataNormalizer] = None,
                 apply_normalization: bool = True,
                 tokenizer=None):
        self.normalizer = normalizer
        self.apply_normalization = apply_normalization
        self.collate_fn = create_collate_fn(
            normalizer,
            apply_normalization,
            tokenizer=tokenizer,
            target_dims=tokenizer.total_dims if tokenizer is not None else None
        )

    def __call__(self, batch):
        """
        Convert batch from our format to HuggingFace format.
        """
        # Use our custom collate function
        collated = self.collate_fn(batch)

        # Convert to HuggingFace format
        result = {
            'input_ids': collated['input'],  # [B, T, V]
            'input': collated['input'],      # Keep original for backward compatibility
            'labels': collated['target'],    # [B, T, V]
            'prediction_mask': collated['prediction_mask'],  # [B, V]
            'attention_indices': collated['attention_indices'],  # [B, V, 64]
            'input_tokens': collated['input_tokens'],
            'target_tokens': collated['target_tokens'],
            'input_token_medians': collated.get('input_token_medians'),
            'target_token_medians': collated.get('target_token_medians'),
            'input_token_offsets': collated.get('input_token_offsets'),
            'target_token_offsets': collated.get('target_token_offsets'),
            'token_vocab_size': collated.get('token_vocab_size'),
            'metadata': collated.get('metadata', []),
            'normalized': collated.get('normalized', False)
        }
        return result


class FluidTrainer(Trainer):
    """
    Simplified trainer for HuggingFace PreTrainedModel fluid dynamics models.
    """

    def __init__(
        self,
        model: BaseModel,
        args: TrainingArguments,
        training_config: TrainingConfig,
        train_dataset=None,
        eval_dataset=None,
        train_dataloader=None,
        eval_dataloader=None,
        normalizer: Optional[DataNormalizer] = None,
        tokenizer: Optional[DataTokenizer] = None,
        **kwargs
    ):
        """
        Initialize FluidTrainer.

        Args:
            model: BaseModel for fluid dynamics
            args: Transformers training arguments
            training_config: Fluid-specific training configuration
            train_dataset: Training dataset (preferred)
            eval_dataset: Evaluation dataset (optional)
            train_dataloader: Training data loader (legacy support)
            eval_dataloader: Evaluation data loader (legacy support)
            normalizer: Data normalizer (optional, for compatibility)
            **kwargs: Additional arguments passed to Trainer
        """
        self.training_config = training_config
        self.normalizer = normalizer
        self._use_swanlab: bool = getattr(training_config, "use_swanlab", False)
        self._last_token_accuracy: Optional[float] = None

        static_dir = training_config.static_dir or str(Path(training_config.data_dir) / "static" / "full")
        tokenizer_instance = tokenizer
        if tokenizer_instance is None:
            tokenizer_instance = load_tokenizer(
                static_dir,
                vocab_size=getattr(training_config, "tokenizer_vocab_size", 4096)
            )
            if tokenizer_instance is None:
                raise RuntimeError(
                    "Tokenizer statistics not found. Run data/compute_tokenizer_stats.py before training."
                )
        else:
            expected_dims = getattr(training_config, "tokenizer_vocab_size", None)
            if (
                hasattr(tokenizer_instance, "vocab_size")
                and tokenizer_instance.vocab_size == 0
                and expected_dims is not None
            ):
                tokenizer_instance.vocab_size = int(expected_dims)

        alignment_dataset = train_dataset
        if alignment_dataset is None and train_dataloader is not None:
            alignment_dataset = getattr(train_dataloader, "dataset", None)

        self._attach_tokenizer_to_dataset(tokenizer_instance, alignment_dataset, "train")

        eval_alignment_dataset = eval_dataset
        if eval_alignment_dataset is None and eval_dataloader is not None:
            eval_alignment_dataset = getattr(eval_dataloader, "dataset", None)
        self._attach_tokenizer_to_dataset(tokenizer_instance, eval_alignment_dataset, "eval")

        # Initialize eval recording structures
        self.eval_step_counter = 0  # 跟踪eval步骤
        self.variable_names = None  # 变量名映射

        # Initialize parent trainer with transformers setup
        # Use datasets if provided, otherwise use empty dummy for dataloader support
        actual_train_dataset = train_dataset if train_dataset is not None else []
        actual_eval_dataset = eval_dataset if eval_dataset is not None else ([] if eval_dataloader is not None else None)

        # Create data collator
        data_collator = FluidDataCollator(
            normalizer,
            apply_normalization=True,
            tokenizer=tokenizer_instance
        )

        super().__init__(
            model=model,
            args=args,
            train_dataset=actual_train_dataset,
            eval_dataset=actual_eval_dataset,
            data_collator=data_collator,
            **kwargs
        )

        self.processing_class = tokenizer_instance
        self._tokenizer = tokenizer_instance

        # Store dataloaders
        self._train_dataloader = train_dataloader
        self._eval_dataloader = eval_dataloader

        # Add early stopping callback if configured
        if training_config.early_stopping_patience > 0:
            self.add_callback(EarlyStoppingCallback(
                early_stopping_patience=training_config.early_stopping_patience,
                early_stopping_threshold=training_config.early_stopping_threshold
            ))

        logger.info(f"FluidTrainer initialized for {model.__class__.__name__}")
        train_dataloader = self.get_train_dataloader()
        logger.info(f"Training batches: {len(train_dataloader)}")

        if eval_dataset is not None or eval_dataloader is not None:
            eval_dl = self.get_eval_dataloader()
            if eval_dl:
                logger.info(f"Evaluation batches: {len(eval_dl)}")
            else:
                logger.info(f"Evaluation batches: None")

        # Load variable names for detailed eval tracking
        self._load_variable_names()

    def _attach_tokenizer_to_dataset(self, tokenizer, dataset, dataset_label: str) -> None:
        if tokenizer is None or dataset is None:
            return

        target_dims = getattr(dataset, "total_dims", None)
        if target_dims is not None and tokenizer.total_dims != target_dims:
            raise RuntimeError(
                f"Tokenizer dims ({tokenizer.total_dims}) do not match {dataset_label} dataset dims ({target_dims})."
            )

        if hasattr(dataset, "set_tokenizer"):
            dataset.set_tokenizer(tokenizer)

    def get_train_dataloader(self):
        """Return the training dataloader."""
        if self._train_dataloader is not None:
            logger.info(f"Using custom train dataloader with batch_size: {self._train_dataloader.batch_size}")
            return self._train_dataloader
        else:
            logger.info(f"Using HuggingFace generated dataloader")
            return super().get_train_dataloader()
    
    def get_eval_dataloader(self, eval_dataset=None):
        """Return the evaluation dataloader."""
        if self._eval_dataloader is not None:
            return self._eval_dataloader
        else:
            return super().get_eval_dataloader(eval_dataset)

    def compute_loss(
        self,
        model: BaseModel,
        inputs,
        return_outputs: bool = False,
        **kwargs,
    ):
        """Override compute_loss so we can capture token accuracy without logging every step."""
        kwargs.pop("num_items_in_batch", None)
        outputs = model(**inputs)

        if isinstance(outputs, dict):
            loss = outputs.get("loss")
            token_accuracy = outputs.get("token_accuracy")
        elif hasattr(outputs, "loss"):
            loss = outputs.loss
            token_accuracy = getattr(outputs, "token_accuracy", None)
        else:
            loss = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            token_accuracy = None

        if loss is None:
            raise ValueError("Model forward pass did not return a loss tensor.")

        if model.training and token_accuracy is not None:
            if isinstance(token_accuracy, torch.Tensor):
                token_acc_tensor = token_accuracy.detach().float()
                if token_acc_tensor.dim() > 0:
                    token_acc_tensor = token_acc_tensor.mean()
                token_acc_value = token_acc_tensor.item()
            else:
                token_acc_value = float(token_accuracy)
            self._last_token_accuracy = token_acc_value
            if self._use_swanlab:
                try:
                    import swanlab

                    swanlab.log({"train/token_accuracy": token_acc_value}, step=self.state.global_step)
                except Exception as exc:
                    logger.debug("Failed to log train/token_accuracy to SwanLab: %s", exc)

        return (loss, outputs) if return_outputs else loss


    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and return metrics with detailed per-variable loss tracking.

        Args:
            eval_dataset: Ignored, we use the dataloader
            ignore_keys: Metric keys to ignore
            metric_key_prefix: Prefix for metric keys

        Returns:
            Dictionary of evaluation metrics
        """
        # Call parent evaluate method
        eval_results = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )

        # Log key metrics
        if f"{metric_key_prefix}_loss" in eval_results:
            logger.info(f"Evaluation loss: {eval_results[f'{metric_key_prefix}_loss']:.6f}")

        # Perform detailed evaluation if we have variable names
        if self.variable_names is not None:
            detail_start = time.time()
            logger.info("Starting detailed evaluation post-processing...")
            self._perform_detailed_evaluation()
            logger.info(
                "Detailed evaluation post-processing completed in %.2fs",
                time.time() - detail_start,
            )

        return eval_results

    def _perform_detailed_evaluation(self):
        """执行详细的eval，记录每个变量的详细数据"""
        start_time = time.time()
        model = self.model
        eval_dataloader = self.get_eval_dataloader()

        try:
            total_batches = len(eval_dataloader)
        except TypeError:
            total_batches = None
        if total_batches is not None:
            logger.info("Detailed evaluation: processing %d batches", total_batches)
        else:
            logger.info("Detailed evaluation: processing evaluation dataloader (batch count unknown)")

        # 存储所有batch的预测值、真值等，用于计算平均
        all_pred_norm = []
        all_pred_denorm = []
        all_labels_norm = []
        all_labels_denorm = []
        all_diff_norm = []
        all_diff_denorm = []

        token_correct_sum = None
        token_total_sum = None
        token_accuracy_values: List[float] = []
        overall_token_accuracy: Optional[float] = None

        model.eval()

        # 检查是否在分布式训练模式 - 只有当local_rank >= 0且torch.distributed已初始化时才是分布式
        import torch.distributed as dist
        is_distributed = self.args.local_rank != -1 and dist.is_initialized()

        progress_interval = 10
        if total_batches and total_batches > 0:
            progress_interval = max(1, total_batches // 10)
        processed_batches = 0
        sample_saved = False

        with torch.no_grad():
            for batch in eval_dataloader:
                processed_batches += 1
                # 从batch中提取数据
                inputs = batch.get('input', batch.get('input_ids'))  # [B, T, V]
                labels = batch['labels']  # [B, T, V]
                prediction_mask = batch.get('prediction_mask')  # [B, V] or [V]
                input_tokens = batch.get('input_tokens')
                target_tokens = batch.get('target_tokens')
                attention_indices = batch.get('attention_indices')

                # 确保数据在正确的设备上
                inputs = inputs.to(self.args.device)
                labels = labels.to(self.args.device)
                if prediction_mask is not None:
                    prediction_mask = prediction_mask.to(self.args.device)
                else:
                    # 创建默认的prediction mask（只预测后6174维）
                    prediction_mask = torch.zeros(inputs.shape[-1], device=self.args.device)
                    prediction_mask[538:] = 1.0  # 只预测equipment变量，忽略boundary条件
                if input_tokens is not None:
                    input_tokens = input_tokens.to(self.args.device)
                if target_tokens is not None:
                    target_tokens = target_tokens.to(self.args.device)
                if attention_indices is not None:
                    attention_indices = attention_indices.to(self.args.device)

                # 模型前向传播 - 传递labels和prediction_mask以便正确计算loss
                model_kwargs = {
                    'input_ids': inputs,
                    'labels': labels,
                    'prediction_mask': prediction_mask,
                    'input_tokens': input_tokens,
                    'target_tokens': target_tokens,
                }
                if attention_indices is not None:
                    model_kwargs['attention_indices'] = attention_indices

                outputs = model(**model_kwargs)
                token_logits = None
                predictions = None
                token_accuracy = None
                tokenizer_inst = getattr(self, "_tokenizer", None)

                def decode_token_predictions(logits_tensor: torch.Tensor) -> torch.Tensor:
                    token_ids = torch.argmax(logits_tensor, dim=-1)
                    if tokenizer_inst is not None and hasattr(tokenizer_inst, 'tokens_to_values'):
                        decoded = tokenizer_inst.tokens_to_values(token_ids)
                        if isinstance(decoded, torch.Tensor):
                            return decoded.to(token_ids.device)
                        return torch.as_tensor(decoded, device=token_ids.device, dtype=torch.float32)
                    return token_ids.float()

                if isinstance(outputs, torch.Tensor):
                    predictions = outputs
                elif hasattr(outputs, 'logits') and outputs.logits is not None:
                    predictions = outputs.logits
                    token_logits = getattr(outputs, 'token_logits', None)
                    token_accuracy = getattr(outputs, 'token_accuracy', None)
                elif isinstance(outputs, dict):
                    token_logits = outputs.get('token_logits')
                    token_accuracy = outputs.get('token_accuracy')
                    if 'value_predictions' in outputs and outputs['value_predictions'] is not None:
                        predictions = outputs['value_predictions']
                    elif 'logits' in outputs and outputs['logits'] is not None:
                        predictions = outputs['logits']
                    elif 'predictions' in outputs:
                        predictions = outputs['predictions']
                    elif 'last_hidden_state' in outputs:
                        predictions = outputs['last_hidden_state']
                    else:
                        tensor_values = [v for v in outputs.values() if isinstance(v, torch.Tensor)]
                        if tensor_values:
                            predictions = tensor_values[0]
                else:
                    predictions = outputs

                if predictions is None and token_logits is not None:
                    predictions = decode_token_predictions(token_logits)

                if predictions is None:
                    raise ValueError("Unable to determine prediction tensor from model outputs.")

                if predictions.dim() == 4:
                    if token_logits is None:
                        raise ValueError("Token logits required to decode token predictions.")
                    predictions = decode_token_predictions(token_logits)

                if predictions.dim() == 4:
                    raise ValueError("Decoded predictions still have unexpected dimensionality.")

                if predictions.shape != labels.shape:
                    if predictions.dim() == labels.dim() + 1 and predictions.shape[-1] == 1:
                        predictions = predictions.squeeze(-1)
                    else:
                        raise ValueError(
                            f"Prediction shape {tuple(predictions.shape)} does not match labels {tuple(labels.shape)}"
                        )

                if token_logits is not None and target_tokens is not None:
                    time_steps = token_logits.size(1)
                    loss_time_steps = getattr(self.model.config, "loss_time_steps", time_steps)
                    loss_steps = min(loss_time_steps, time_steps - 1) if time_steps > 1 else time_steps
                    if loss_steps <= 0:
                        loss_steps = time_steps

                    selected_logits = token_logits[:, -loss_steps:, :, :]
                    selected_targets = target_tokens[:, -loss_steps:, :]

                    batch_size = selected_targets.size(0)
                    effective_steps = selected_targets.size(1)
                    num_variables = selected_targets.size(2)

                    if prediction_mask.dim() == 1:
                        mask = prediction_mask.view(1, 1, num_variables).expand(batch_size, effective_steps, num_variables)
                    elif prediction_mask.dim() == 2:
                        mask = prediction_mask.unsqueeze(1).expand(batch_size, effective_steps, num_variables)
                    elif prediction_mask.dim() == 3:
                        mask = prediction_mask[:, -effective_steps:, :]
                        if mask.shape[0] != batch_size:
                            raise ValueError("prediction_mask batch dimension mismatch")
                    else:
                        raise ValueError("prediction_mask must have 1, 2 or 3 dimensions")

                    mask = mask.to(selected_targets.device) > 0
                    active = mask & (selected_targets >= 0)

                    if token_correct_sum is None:
                        acc_device = selected_targets.device
                        token_correct_sum = torch.zeros(num_variables, device=acc_device, dtype=torch.float64)
                        token_total_sum = torch.zeros(num_variables, device=acc_device, dtype=torch.float64)

                    predicted_tokens = torch.argmax(selected_logits, dim=-1)
                    correct_counts = ((predicted_tokens == selected_targets) & active).sum(dim=(0, 1)).to(dtype=torch.float64)
                    total_counts = active.sum(dim=(0, 1)).to(dtype=torch.float64)

                    token_correct_sum += correct_counts
                    token_total_sum += total_counts

                if (not sample_saved) and self.args.local_rank in [-1, 0]:
                    try:
                        self._write_sample_prediction_csv(predictions, labels)
                        sample_saved = True
                    except Exception as exc:
                        logger.warning("Failed to save sample prediction CSV: %s", exc)

                if token_accuracy is not None:
                    if isinstance(token_accuracy, torch.Tensor):
                        token_acc_tensor = token_accuracy.detach().float()
                        if token_acc_tensor.dim() > 0:
                            token_acc_tensor = token_acc_tensor.mean()
                        token_acc_value = token_acc_tensor.item()
                    else:
                        token_acc_value = float(token_accuracy)
                    token_accuracy_values.append(token_acc_value)

                # 提取当前batch的详细数据（用于累积）
                with torch.no_grad():
                    # 处理prediction_mask维度
                    if prediction_mask.dim() == 1:  # [V] -> [B, V]
                        prediction_mask = prediction_mask.unsqueeze(0).expand(predictions.shape[0], -1)

                    # 应用prediction_mask: 只对需要预测的变量计算
                    mask_expanded = prediction_mask.unsqueeze(1).expand(-1, predictions.shape[1], -1)
                    valid_counts = mask_expanded.sum(dim=(0, 1)) + 1e-8

                    # 当前batch的平均值
                    pred_norm_avg = (predictions * mask_expanded).sum(dim=(0, 1)) / valid_counts
                    labels_norm_avg = (labels * mask_expanded).sum(dim=(0, 1)) / valid_counts
                    diff_norm_avg = pred_norm_avg - labels_norm_avg

                    # 反归一化在CPU上完成，避免在GPU上重复分配大张量
                    if self.normalizer is not None:
                        predictions_cpu = predictions.detach().cpu()
                        labels_cpu = labels.detach().cpu()
                        mask_cpu = mask_expanded.detach().cpu()
                        valid_counts_cpu = valid_counts.detach().cpu()

                        pred_denorm = self.normalizer.denormalize(predictions_cpu)
                        labels_denorm = self.normalizer.denormalize(labels_cpu)
                        pred_denorm_avg = (pred_denorm * mask_cpu).sum(dim=(0, 1)) / valid_counts_cpu
                        labels_denorm_avg = (labels_denorm * mask_cpu).sum(dim=(0, 1)) / valid_counts_cpu
                        diff_denorm_avg = pred_denorm_avg - labels_denorm_avg
                    else:
                        pred_denorm_avg = pred_norm_avg.clone()
                        labels_denorm_avg = labels_norm_avg.clone()
                        diff_denorm_avg = diff_norm_avg.clone()

                    # 累积所有batch的结果
                    all_pred_norm.append(pred_norm_avg.cpu().numpy())
                    all_pred_denorm.append(pred_denorm_avg.cpu().numpy())
                    all_labels_norm.append(labels_norm_avg.cpu().numpy())
                    all_labels_denorm.append(labels_denorm_avg.cpu().numpy())
                    all_diff_norm.append(diff_norm_avg.cpu().numpy())
                    all_diff_denorm.append(diff_denorm_avg.cpu().numpy())

                

        # 对所有batch求平均
        if all_pred_norm:
            token_correct_tensor = token_correct_sum
            token_total_tensor = token_total_sum
            token_accuracy_per_var = None

            final_pred_norm = np.mean(all_pred_norm, axis=0)
            final_pred_denorm = np.mean(all_pred_denorm, axis=0)
            final_labels_norm = np.mean(all_labels_norm, axis=0)
            final_labels_denorm = np.mean(all_labels_denorm, axis=0)
            final_diff_norm = np.mean(all_diff_norm, axis=0)
            final_diff_denorm = np.mean(all_diff_denorm, axis=0)

            # 在分布式训练中，需要聚合所有GPU的结果
            if is_distributed:

                # 将numpy数组转换为tensor以便进行分布式通信
                def gather_and_average(data_array):
                    """收集所有GPU的数据并计算平均值"""
                    # 转换为tensor
                    tensor = torch.tensor(data_array, device=self.args.device)

                    # 获取world size
                    world_size = dist.get_world_size()

                    # 创建列表来存储所有GPU的数据
                    if self.args.local_rank == 0:
                        gathered_list = [torch.zeros_like(tensor) for _ in range(world_size)]
                    else:
                        gathered_list = None

                    # 收集所有GPU的数据到rank 0
                    dist.gather(tensor, gathered_list, dst=0)

                    # 在rank 0上计算平均值
                    if self.args.local_rank == 0:
                        gathered_tensor = torch.stack(gathered_list)
                        averaged = gathered_tensor.mean(dim=0)
                        return averaged.cpu().numpy()
                    else:
                        return None

                def gather_and_sum(data_tensor: torch.Tensor) -> torch.Tensor:
                    tensor = data_tensor.to(self.args.device)
                    world_size = dist.get_world_size()
                    if self.args.local_rank == 0:
                        gathered_list = [torch.zeros_like(tensor) for _ in range(world_size)]
                    else:
                        gathered_list = None
                    dist.gather(tensor, gathered_list, dst=0)
                    if self.args.local_rank == 0:
                        stacked = torch.stack(gathered_list)
                        return stacked.sum(dim=0)
                    return None

                # 聚合所有数据
                final_pred_norm = gather_and_average(final_pred_norm)
                final_pred_denorm = gather_and_average(final_pred_denorm)
                final_labels_norm = gather_and_average(final_labels_norm)
                final_labels_denorm = gather_and_average(final_labels_denorm)
                final_diff_norm = gather_and_average(final_diff_norm)
                final_diff_denorm = gather_and_average(final_diff_denorm)

                if token_correct_tensor is not None and token_total_tensor is not None:
                    token_correct_tensor = gather_and_sum(token_correct_tensor)
                    token_total_tensor = gather_and_sum(token_total_tensor)

                # 只有rank 0有有效数据
                if self.args.local_rank != 0:
                    logger.info(
                        "Detailed evaluation finished on rank %d in %.2fs (processed %d batches)",
                        self.args.local_rank,
                        time.time() - start_time,
                        processed_batches,
                    )
                    return  # 非主进程直接返回

            # 增加eval步骤计数
            self.eval_step_counter += 1

            if token_correct_tensor is not None and token_total_tensor is not None:
                token_correct_cpu = token_correct_tensor.detach().cpu().numpy()
                token_total_cpu = token_total_tensor.detach().cpu().numpy()
                with np.errstate(divide='ignore', invalid='ignore'):
                    token_accuracy_per_var = np.divide(
                        token_correct_cpu,
                        token_total_cpu,
                        out=np.full_like(token_correct_cpu, np.nan),
                        where=token_total_cpu > 0,
                    )
                total_correct = np.nansum(token_correct_cpu)
                total_count = np.nansum(token_total_cpu)
                if total_count > 0:
                    overall_token_accuracy = float(total_correct / total_count)
            elif token_accuracy_values:
                overall_token_accuracy = float(np.mean(token_accuracy_values))

            # 增量保存到CSV文件
            self._save_eval_data_to_csvs(
                final_pred_norm,
                final_pred_denorm,
                final_labels_norm,
                final_labels_denorm,
                final_diff_norm,
                final_diff_denorm,
                token_accuracy_per_var,
            )

            if overall_token_accuracy is not None:
                logger.info(
                    "Detailed evaluation token accuracy=%.4f",
                    overall_token_accuracy,
                )

            logger.info(f"详细评估步骤 {self.eval_step_counter} 完成，已保存 {len(self.variable_names)} 个变量的详细数据到CSV")
        else:
            logger.warning("Detailed evaluation: no batches processed, skipping CSV export.")

        logger.info(
            "Detailed evaluation completed in %.2fs (processed %d%s batches)",
            time.time() - start_time,
            processed_batches,
            f"/{total_batches}" if total_batches is not None else "",
        )

        model.train()
    
    def log(self, logs: Dict[str, float], start_time=None):
        """
        Custom logging with additional context.
        
        Args:
            logs: Dictionary of logs to record
            start_time: Optional start time for timing calculations
        """
        # Add training configuration context to logs
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch

        if self.state.global_step is not None:
            logs["step"] = self.state.global_step

        token_acc = getattr(self, "_last_token_accuracy", None)
        if token_acc is not None and self.model is not None and self.model.training:
            logs.setdefault("train/token_accuracy", token_acc)

        # Add learning rate
        if hasattr(self.lr_scheduler, 'get_last_lr'):
            logs["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
        
        # HuggingFace Transformers >= 4.50.0 automatically handles SwanLab integration
        # when report_to="swanlab" is set in TrainingArguments
        # No manual swanlab.log() calls needed
        
        # Call parent log method with start_time parameter
        super().log(logs, start_time)
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save model with additional metadata using torch save instead of HuggingFace.

        Args:
            output_dir: Directory to save to
            _internal_call: Whether this is an internal call
        """
        # 在分布式训练中，只让主进程保存模型，避免重复保存
        if self.args.local_rank not in [-1, 0]:
            return

        if output_dir is None:
            output_dir = self.args.output_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save model state dict using PyTorch
        model_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model state dict saved to {model_path}")

        # Save model config if available
        if hasattr(self.model, 'config'):
            config_path = os.path.join(output_dir, "config.json")
            if hasattr(self.model.config, 'to_dict'):
                config_dict = self.model.config.to_dict()
            else:
                config_dict = vars(self.model.config) if self.model.config else {}
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)

        # Save training configuration
        config_path = os.path.join(output_dir, "training_config.json")
        self.training_config.save_to_file(config_path)

        # Save additional metadata
        metadata = {
            "training_completed": self.state.epoch >= self.args.num_train_epochs if self.state.epoch else False,
            "total_steps": self.state.global_step,
            "epochs_completed": self.state.epoch,
            "best_metric": self.state.best_metric,
            "model_info": self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}
        }

        metadata_path = os.path.join(output_dir, "training_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Model and metadata saved to {output_dir}")

        # Eval data is already incrementally saved during evaluation

    # HuggingFace PreTrainedModel handles prediction_step and compute_loss automatically
    # No custom implementation needed - the base Trainer class will work with our models

    def _load_variable_names(self):
        """Load variable name mappings, prioritizing static-directory-specific mapping when available."""
        candidate_paths = []

        # Prefer mapping stored alongside the selected static directory
        static_dir = getattr(self.training_config, "static_dir", None)
        if static_dir:
            candidate_paths.append(os.path.join(static_dir, "index_variable_mapping.csv"))

        # Fallback to global mapping
        candidate_paths.append(os.path.join(self.training_config.data_dir, "static", "full", "index_variable_mapping.csv"))
        candidate_paths.append(os.path.join(self.training_config.data_dir, "index_variable_mapping.csv"))

        for path in candidate_paths:
            resolved_path = os.path.abspath(path)
            if not os.path.exists(resolved_path):
                continue

            try:
                df = pd.read_csv(resolved_path)
                if 'variable_name' not in df.columns:
                    logger.warning(f"'variable_name' column missing in {resolved_path}, skipping.")
                    continue

                if 'index' in df.columns:
                    df = df.sort_values('index')

                self.variable_names = df['variable_name'].tolist()
                logger.info(
                    "Loaded %d variable names for detailed eval tracking from %s",
                    len(self.variable_names),
                    resolved_path,
                )
                return
            except Exception as exc:
                logger.error("Failed to load variable mapping %s: %s", resolved_path, exc)

        logger.warning(
            "No variable mapping file found in candidates %s; falling back to default variable names.",
            candidate_paths,
        )
        # Create default variable names
        self.variable_names = [f"var_{i}" for i in range(6712)]

    def _write_sample_prediction_csv(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        """Save one sample's predictions/labels to CSV for quick inspection."""
        step = getattr(self.state, "global_step", 0)
        sample_dir = Path(self.args.output_dir) / "sample_predictions"
        sample_dir.mkdir(parents=True, exist_ok=True)
        csv_path = sample_dir / f"eval_{step}.csv"

        start_time = time.time()
        preds_np = predictions.detach().to("cpu").float().numpy()
        labels_np = labels.detach().to("cpu").float().numpy()

        if preds_np.ndim == 2:
            preds_np = preds_np[None, ...]
        if labels_np.ndim == 2:
            labels_np = labels_np[None, ...]

        first_pred = preds_np[0]
        first_label = labels_np[0]

        if first_pred.ndim == 1:
            first_pred = first_pred[None, :]
            first_label = first_label[None, :]

        seq_len = first_pred.shape[0]
        num_vars = first_pred.shape[1]

        if self.variable_names:
            variable_names = list(self.variable_names[:num_vars])
            if len(variable_names) < num_vars:
                variable_names.extend(f"var_{i}" for i in range(len(variable_names), num_vars))
        else:
            variable_names = [f"var_{i}" for i in range(num_vars)]

        logger.info(
            "Saving sample prediction CSV: %s (time steps=%d, variables=%d)",
            csv_path,
            seq_len,
            num_vars,
        )

        progress_interval = max(1, seq_len // 10)
        records = []
        row_labels = []
        for idx in range(seq_len):
            records.append(first_label[idx])
            row_labels.append(f"data_line_{idx + 1}_real")
            records.append(first_pred[idx])
            row_labels.append(f"data_line_{idx + 1}_predict")

            if (idx + 1) % progress_interval == 0 or idx == seq_len - 1:
                logger.info("  Sample CSV progress: %d/%d time steps", idx + 1, seq_len)

        df = pd.DataFrame(records, columns=variable_names, index=row_labels)
        df.to_csv(csv_path, float_format="%.6f")

        logger.info(
            "Sample prediction CSV saved in %.2fs to %s",
            time.time() - start_time,
            csv_path,
        )


    def _save_eval_data_to_csvs(self, pred_norm, pred_denorm, labels_norm, labels_denorm, diff_norm, diff_denorm, token_accuracy_per_var=None):
        """
        增量保存评估数据到多个CSV文件

        Args:
            pred_norm: [V] 模型预测值 (归一化)
            pred_denorm: [V] 模型预测值 (原始数据值域)
            labels_norm: [V] 真值 (归一化)
            labels_denorm: [V] 真值 (原始数据值域)
            diff_norm: [V] 差值 (归一化)
            diff_denorm: [V] 差值 (原始数据值域)
            token_accuracy_per_var: [V] Token准确率 (可选)
        """
        # 在分布式训练中，只让主进程（rank 0）保存文件，避免多进程同时写入
        if self.args.local_rank not in [-1, 0]:
            return

        eval_results_dir = os.path.join(self.args.output_dir, "eval_results")
        os.makedirs(eval_results_dir, exist_ok=True)
        logger.info("Saving detailed evaluation CSVs into %s", eval_results_dir)

        # 生成时间戳和eval步骤信息 - 增加秒数避免重复
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%Y%m%d_%H%M%S")
        # 添加global_step确保唯一性
        global_step = self.state.global_step if hasattr(self, 'state') else 0
        step_info = f"eval步骤_{self.eval_step_counter}_step{global_step}_{time_str}"

        # 准备基础结果文件的数据
        data_configs = [
            ("模型预测值_归一化.csv", pred_norm),
            ("模型预测值_原始值域.csv", pred_denorm),
            ("真值_归一化.csv", labels_norm),
            ("真值_原始值域.csv", labels_denorm),
            ("差值_归一化.csv", diff_norm),
            ("差值_原始值域.csv", diff_denorm)
        ]
        if token_accuracy_per_var is not None:
            data_configs.append(("token_accuracy.csv", token_accuracy_per_var))

        total_start = time.time()
        for filename, data in data_configs:
            file_path = os.path.join(eval_results_dir, filename)
            file_start = time.time()

            # 准备当前步骤的数据
            new_data = []
            for var_idx, value in enumerate(data):
                var_name = self.variable_names[var_idx] if self.variable_names and var_idx < len(self.variable_names) else f"var_{var_idx}"
                new_data.append({
                    'variable_name': var_name,
                    step_info: float(value)
                })

            new_df = pd.DataFrame(new_data)

            # 如果文件已存在，合并数据；否则创建新文件
            if os.path.exists(file_path):
                existing_df = pd.read_csv(file_path)

                # 检查是否有重复的列名
                if step_info in existing_df.columns:
                    logger.warning(f"列名 {step_info} 已存在于 {filename}，将覆盖旧数据")
                    # 删除旧的重复列
                    existing_df = existing_df.drop(columns=[step_info])

                # 基于variable_name合并，不使用suffixes以避免重复列错误
                merged_df = pd.merge(existing_df, new_df, on='variable_name', how='outer', suffixes=('', '_dup'))

                # 删除任何意外产生的重复列
                dup_cols = [col for col in merged_df.columns if col.endswith('_dup')]
                if dup_cols:
                    logger.warning(f"发现重复列 {dup_cols}，将删除")
                    merged_df = merged_df.drop(columns=dup_cols)
            else:
                merged_df = new_df

            # 保存文件 - 使用文件锁防止并发写入
            # 使用临时文件和原子重命名来避免损坏
            temp_file_path = file_path + f'.tmp_{os.getpid()}_{time.time()}'
            merged_df.to_csv(temp_file_path, index=False)

            # 原子性地重命名文件
            os.replace(temp_file_path, file_path)
            logger.info(
                "  Wrote %s with %d variables in %.2fs",
                filename,
                len(new_df),
                time.time() - file_start,
            )

        logger.info(
            "Finished saving detailed evaluation CSVs in %.2fs",
            time.time() - total_start,
        )

        logger.info(f"增量保存评估数据完成: eval步骤 {self.eval_step_counter}, 时间 {time_str}")




def create_fluid_trainer(
    model: BaseModel,
    training_config: TrainingConfig,
    train_dataset=None,
    eval_dataset=None,
    train_dataloader=None,
    eval_dataloader=None,
    normalizer: Optional[DataNormalizer] = None,
    tokenizer: Optional[DataTokenizer] = None,
    **trainer_kwargs
) -> FluidTrainer:
    """
    Create a FluidTrainer with BaseModel.

    Args:
        model: BaseModel for fluid dynamics
        training_config: Training configuration
        train_dataset: Training dataset (preferred)
        eval_dataset: Evaluation dataset (optional)
        train_dataloader: Training data loader (legacy support)
        eval_dataloader: Evaluation data loader (legacy support)
        normalizer: Data normalizer (optional)
        tokenizer: Preloaded tokenizer to reuse (optional)
        **trainer_kwargs: Additional trainer arguments

    Returns:
        Configured FluidTrainer instance
    """
    # Get training arguments
    training_args_dict = training_config.get_transformers_training_args()
    
    # If no eval_dataset or eval_dataloader provided, disable evaluation
    if eval_dataset is None and eval_dataloader is None:
        training_args_dict['eval_strategy'] = 'no'
        training_args_dict['save_strategy'] = 'steps'  # Keep save_strategy as is
        training_args_dict['load_best_model_at_end'] = False
        logger.warning("No evaluation dataloader provided. Disabling evaluation.")
    
    # Convert training config to transformers arguments
    training_args = TrainingArguments(**training_args_dict)

    # Setup logging level based on config
    import transformers
    log_level_map = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR
    }

    # Set transformers logging level
    transformers.logging.set_verbosity(log_level_map.get(training_config.log_level, logging.INFO))

    # Set our logger level too
    logger.setLevel(log_level_map.get(training_config.log_level, logging.INFO))

    logger.debug(f"Logging level set to: {training_config.log_level}")
    
    # Create trainer
    trainer = FluidTrainer(
        model=model,
        args=training_args,
        training_config=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        normalizer=normalizer,
        tokenizer=tokenizer,
        **trainer_kwargs
    )
    
    logger.info("FluidTrainer created successfully")

    return trainer
