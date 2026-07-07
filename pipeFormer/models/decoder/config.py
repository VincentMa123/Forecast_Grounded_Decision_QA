"""
Configuration classes for Decoder model.
"""

from dataclasses import dataclass
from ..config import ModelConfig


@dataclass
class DecoderConfig(ModelConfig):
    """Configuration for Decoder model."""
    
    model_name: str = "FluidDecoder"
    
    # Architecture parameters
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 6
    d_ff: int = 3072
    
    # Attention parameters
    attention_dropout: float = 0.1
    
    # Positional encoding
    time_position_encoding: str = "sinusoidal"  # "sinusoidal" or "learned"
    variable_position_encoding: str = "sinusoidal"  # "sinusoidal" or "learned"
    max_time_positions: int = 10
    max_variable_positions: int = 6712
    
    # Input/output projection (简化版本)
    projection_hidden_dim: int = 256
    tokenizer_vocab_size: int = 4096
    input_projection_type: str = "value_projection"
    hybrid_ce_weight: float = 1.0
    hybrid_mae_weight: float = 1.0
    hybrid_softmax_temperature: float = 1.0
    hybrid_num_frequencies: int = 10
    
    # Optimization
    use_layer_norm: bool = True
    activation: str = "gelu"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        
        valid_projection_types = {"token_embedding", "value_projection", "hybrid"}
        if self.input_projection_type not in valid_projection_types:
            raise ValueError(
                f"input_projection_type ({self.input_projection_type}) must be one of {valid_projection_types}"
            )

        if self.hybrid_ce_weight < 0 or self.hybrid_mae_weight < 0:
            raise ValueError("hybrid_ce_weight and hybrid_mae_weight must be non-negative")
        if self.hybrid_softmax_temperature <= 0:
            raise ValueError("hybrid_softmax_temperature must be positive")
        if self.hybrid_num_frequencies <= 0:
            raise ValueError("hybrid_num_frequencies must be positive")
