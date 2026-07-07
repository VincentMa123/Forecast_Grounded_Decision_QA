"""
Base model class for fluid dynamics neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any
import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all fluid dynamics models.
    
    All models should:
    1. Inherit from this base class
    2. Implement the forward method
    3. Handle prediction masks correctly
    4. Support standard loss computation
    """
    
    def __init__(self, input_dim: int = 6712, output_dim: int = 6712, **kwargs):
        """
        Initialize base model.
        
        Args:
            input_dim: Input feature dimension (default: 6712)
            output_dim: Output feature dimension (default: 6712)
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.boundary_dims = 538  # First 538 dimensions are boundary conditions
        self.equipment_dims = 6174  # Remaining dimensions are equipment parameters

        # Allow subclasses/configs to override dimension splits
        boundary_override = kwargs.get("boundary_dims")
        equipment_override = kwargs.get("equipment_dims")

        if boundary_override is not None:
            try:
                boundary_override = int(boundary_override)
            except (TypeError, ValueError) as exc:
                raise ValueError("boundary_dims must be a positive integer") from exc
            if boundary_override <= 0:
                raise ValueError("boundary_dims must be a positive integer")
            self.boundary_dims = boundary_override

        if equipment_override is not None:
            try:
                equipment_override = int(equipment_override)
            except (TypeError, ValueError) as exc:
                raise ValueError("equipment_dims must be a positive integer") from exc
            if equipment_override <= 0:
                raise ValueError("equipment_dims must be a positive integer")
            self.equipment_dims = equipment_override

        if boundary_override is not None and equipment_override is not None:
            expected_total = self.boundary_dims + self.equipment_dims
            if expected_total != self.input_dim:
                raise ValueError(
                    f"boundary_dims ({self.boundary_dims}) + equipment_dims ({self.equipment_dims}) "
                    f"must equal input_dim ({self.input_dim})"
                )
        
        # Model metadata
        self.model_name = self.__class__.__name__
        self.model_config = dict(kwargs)

        loss_time_steps = self.model_config.get('loss_time_steps', 1)
        try:
            loss_time_steps = int(loss_time_steps)
        except (TypeError, ValueError) as exc:
            raise ValueError("loss_time_steps must be a positive integer") from exc
        if loss_time_steps <= 0:
            raise ValueError("loss_time_steps must be a positive integer")
        self.loss_time_steps = loss_time_steps
        self.model_config['loss_time_steps'] = self.loss_time_steps
        
    @abstractmethod
    def forward(self, input_ids=None, labels=None, **kwargs):
        """
        Forward pass compatible with transformers library.
        
        Args:
            input_ids: Input tensor [B, T, V=6712] or Dict containing inputs
            labels: Target tensor [B, T, V=6712] for loss computation (optional)
            **kwargs: Additional arguments
                
        Returns:
            If labels provided: Dict with 'loss' and 'logits'
            Else: predictions tensor [B, T, V=6712] or Dict with 'logits'
        """
        pass
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        prediction_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute MAE loss using the trailing `loss_time_steps` prediction steps.

        Args:
            predictions: Model predictions [B, T, V]
            labels: Target tensor [B, T, V]
            prediction_mask: Prediction mask [B, V] or [B, T, V] (optional)

        Returns:
            MAE loss tensor (scalar)
        """
        time_steps = predictions.size(1)
        loss_steps = min(self.loss_time_steps, time_steps)

        selected_predictions = predictions[:, -loss_steps:, :]  # [B, S, V]
        selected_labels = labels[:, -loss_steps:, :]  # [B, S, V]

        mae_loss = F.l1_loss(selected_predictions, selected_labels, reduction='none')  # [B, S, V]

        if prediction_mask is None:
            mask = torch.ones_like(mae_loss, dtype=mae_loss.dtype, device=mae_loss.device)
        else:
            if prediction_mask.dim() == 2:
                mask = prediction_mask.unsqueeze(1).expand(-1, loss_steps, -1)
            elif prediction_mask.dim() == 3:
                mask = prediction_mask[:, -loss_steps:, :]
            else:
                raise ValueError("prediction_mask must have 2 or 3 dimensions")
            mask = mask.to(mae_loss.device, dtype=mae_loss.dtype)

        masked_loss = mae_loss * mask

        mask_sum = mask.sum()
        if mask_sum <= 0:
            return mae_loss.new_tensor(0.0)

        total_loss = masked_loss.sum() / mask_sum

        # 如果模型是eval状态
        loss_threshold = 10
        if total_loss.item() > loss_threshold and self.training == False:
            # 输出大于loss_threshold的index
            high_loss_indices = (masked_loss > loss_threshold).nonzero()

            # 保存高损失数据到JSON
            self._save_high_loss_data(
                total_loss.item(),
                selected_predictions.detach().cpu(),
                selected_labels.detach().cpu(),
                mask.detach().cpu(),
                masked_loss.detach().cpu(),
                high_loss_indices.cpu()
            )
  
        return total_loss

    def _save_high_loss_data(self, loss_value: float, predictions: torch.Tensor,
                           labels: torch.Tensor, mask: torch.Tensor,
                           masked_loss: torch.Tensor, high_loss_indices: torch.Tensor):
        """
        保存高损失数据到JSON文件

        Args:
            loss_value: 总损失值
            predictions: 预测值 [B, S, V]
            labels: 真实标签 [B, S, V]
            mask: 预测掩码 [B, S, V]
            masked_loss: 掩码后的损失 [B, S, V]
            high_loss_indices: 高损失位置的索引 [N, 3] (batch, step, feature)
        """
        # 创建保存目录
        save_dir = "./big_loss"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "error.json")

        # 准备保存的数据
        data_to_save = {
            "timestamp": datetime.now().isoformat(),
            "total_loss": float(loss_value),
            "model_name": self.model_name,
            "batch_size": predictions.shape[0],
            "loss_time_steps": predictions.shape[1],
            "feature_dim": predictions.shape[2],
            "high_loss_indices": {
                "shape": list(high_loss_indices.shape),
                "values": high_loss_indices.numpy().tolist()
            },
            "high_loss_count": len(high_loss_indices),
        }

        if high_loss_indices.numel() > 0:
            batch_idx = high_loss_indices[:, 0]
            step_idx = high_loss_indices[:, 1]
            feature_idx = high_loss_indices[:, 2]
            data_to_save["high_loss_predictions"] = predictions[batch_idx, step_idx, feature_idx].numpy().tolist()
            data_to_save["high_loss_labels"] = labels[batch_idx, step_idx, feature_idx].numpy().tolist()
        else:
            data_to_save["high_loss_predictions"] = []
            data_to_save["high_loss_labels"] = []

        # 读取现有数据(如果文件存在)
        existing_data = []
        if os.path.exists(save_path):
            try:
                with open(save_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to read existing error data: {e}")
                existing_data = []

        # 追加新数据
        existing_data.append(data_to_save)

        # 保存到文件
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            logger.info(f"High loss data saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save high loss data: {e}")

    def predict(self, batch: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Prediction interface for inference.
        
        Args:
            batch: Input batch
            **kwargs: Additional prediction parameters
            
        Returns:
            predictions: Model predictions [B, T, V]
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(batch)
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        param_count = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'total_parameters': param_count,
            'trainable_parameters': trainable_params,
            'config': self.model_config
        }
    
    def save_checkpoint(self, filepath: str, epoch: int = 0, optimizer_state: Optional[Dict] = None, **kwargs):
        """Save model checkpoint."""
        checkpoint = {
            'model_name': self.model_name,
            'model_state_dict': self.state_dict(),
            'model_config': self.model_config,
            'epoch': epoch,
            'model_info': self.get_model_info()
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
            
        # Add any additional info
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, filepath)
        logger.info(f"Model checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str, strict: bool = True) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Load model state
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        logger.info(f"Model checkpoint loaded from {filepath}")
        return checkpoint
    
    def freeze_parameters(self, freeze_embeddings: bool = False):
        """Freeze model parameters for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = False
            
        logger.info("Model parameters frozen")
    
    def unfreeze_parameters(self):
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True
            
        logger.info("Model parameters unfrozen")


class MaskedMSELoss(nn.Module):
    """Masked MSE Loss for equipment prediction (last time step only)."""

    def __init__(self, boundary_dims: int = 538):
        super().__init__()
        self.boundary_dims = boundary_dims

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, prediction_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute masked MSE loss only for the last time step.

        Args:
            predictions: [B, T, V]
            targets: [B, T, V]
            prediction_mask: [B, V] where 1=predict, 0=ignore

        Returns:
            Scalar loss
        """
        # Only compute loss for the last time step
        last_predictions = predictions[:, -1, :]  # [B, V]
        last_targets = targets[:, -1, :]  # [B, V]

        # Compute MSE loss
        mse = F.mse_loss(last_predictions, last_targets, reduction='none')  # [B, V]

        # Apply mask and average
        masked_mse = mse * prediction_mask.float()
        loss = masked_mse.sum() / prediction_mask.float().sum().clamp(min=1e-8)

        return loss
