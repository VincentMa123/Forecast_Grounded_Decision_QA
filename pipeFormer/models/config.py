"""
Decoder-only configuration classes for the PipeFormer open-source release.
"""

from dataclasses import dataclass
from typing import Any, Dict
import json


@dataclass
class ModelConfig:
    """Base configuration for the released model family."""

    input_dim: int = 6712
    output_dim: int = 6712
    sequence_length: int = 3
    boundary_dims: int = 538
    equipment_dims: int = 6174
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    dropout_rate: float = 0.1
    loss_time_steps: int = 1
    model_name: str = "BaseModel"
    model_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_str: str):
        return cls.from_dict(json.loads(json_str))


@dataclass
class DecoderConfig(ModelConfig):
    """Configuration for the released PipeFormer decoder."""

    model_name: str = "FluidDecoder"
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 6
    d_ff: int = 3072
    attention_dropout: float = 0.1
    use_topology_attention: bool = True
    whether_causal: bool = True
    time_position_encoding: str = "sinusoidal"
    variable_position_encoding: str = "sinusoidal"
    max_time_positions: int = 10
    max_variable_positions: int = 6712
    projection_hidden_dim: int = 256
    tokenizer_vocab_size: int = 4096
    input_projection_type: str = "value_projection"
    use_layer_norm: bool = True
    activation: str = "gelu"

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        valid_projection_types = {"token_embedding", "value_projection"}
        if self.input_projection_type not in valid_projection_types:
            raise ValueError(
                f"input_projection_type ({self.input_projection_type}) must be one of {valid_projection_types}"
            )


def create_default_configs() -> Dict[str, ModelConfig]:
    return {"decoder": DecoderConfig()}


def load_config_from_file(filepath: str) -> ModelConfig:
    with open(filepath, "r", encoding="utf-8") as handle:
        config_dict = json.load(handle)

    model_type = config_dict.get("model_name", "").lower()
    if not model_type:
        filename = filepath.lower()
        if "decoder" not in filename:
            raise ValueError(f"Only decoder configs are supported in the open-source package: {filepath}")
        model_type = "decoder"

    if "decoder" in model_type or model_type == "fluiddecoder":
        return DecoderConfig.from_dict(config_dict)

    raise ValueError(f"Only decoder configs are supported in the open-source package: {filepath}")


def save_config_to_file(config: ModelConfig, filepath: str):
    with open(filepath, "w", encoding="utf-8") as handle:
        handle.write(config.to_json())
