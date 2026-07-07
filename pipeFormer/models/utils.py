"""
Decoder-only model utilities for the PipeFormer open-source release.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from .base import BaseModel
from .decoder import FluidDecoder
from .config import ModelConfig, DecoderConfig, load_config_from_file

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(param.numel() for param in model.parameters() if param.requires_grad)
    return sum(param.numel() for param in model.parameters())


def initialize_weights(model: nn.Module, method: str = "xavier") -> None:
    def init_fn(module):
        if isinstance(module, nn.Linear):
            if method == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif method == "he":
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            elif method == "normal":
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif method == "zero":
                nn.init.zeros_(module.weight)
            else:
                raise ValueError(f"Unknown initialization method: {method}")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    model.apply(init_fn)
    logger.info("Model weights initialized using %s", method)


def create_model(
    model_type: str,
    config: Optional[Union[ModelConfig, Dict[str, Any], str]] = None,
    **kwargs,
) -> BaseModel:
    if isinstance(config, (str, Path)):
        config = load_config_from_file(str(config))
    elif isinstance(config, dict):
        if model_type.lower() != "decoder":
            raise ValueError("Only decoder models are supported in the open-source package")
        config = DecoderConfig.from_dict(config)

    if config is not None:
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    if model_type.lower() != "decoder":
        raise ValueError("Only decoder models are supported in the open-source package")

    model = FluidDecoder(config)
    logger.info("Created %s model with %d parameters", model_type, count_parameters(model))
    return model


def load_model(
    checkpoint_path: str,
    model_type: Optional[str] = None,
    config: Optional[Union[ModelConfig, Dict[str, Any]]] = None,
    strict: bool = True,
) -> BaseModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_model_type = checkpoint.get("model_name", model_type)
    saved_config = checkpoint.get("model_config", config)
    if saved_model_type is None:
        raise ValueError("Model type not found in checkpoint and not provided")

    model = create_model(saved_model_type, saved_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    logger.info("Model loaded from %s", checkpoint_path)
    return model


def save_model(
    model: BaseModel,
    save_path: str,
    epoch: int = 0,
    optimizer_state: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    model.save_checkpoint(save_path, epoch, optimizer_state, **kwargs)
