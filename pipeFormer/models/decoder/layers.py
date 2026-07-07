"""
Decoder layer components.
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import SimpleMultiHeadAttention


class DecoderBlock(nn.Module):
    """简化的Decoder block。"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        
        self.attention = SimpleMultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attention_indices_follow_timeline: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None, output_attentions: bool = False):
        """
        Decoder block前向传播。

        Args:
            x: 输入张量 [B, T*V, d_model]
            attention_indices_follow_timeline: 时间线注意力索引 [B, T*V, T*max_neighbors_variable]
            attention_mask: 注意力mask [B, T*V, T*max_neighbors_variable]
            output_attentions: 是否返回注意力权重

        Returns:
            如果output_attentions=True: (输出张量 [B, T*V, d_model], 注意力权重 [B, H, T*V, T*max_neighbors])
            否则: 输出张量 [B, T*V, d_model]
        """
        # Self-attention with residual connection
        attn_result = self.attention(x, attention_indices_follow_timeline, attention_mask, output_attentions)
        if output_attentions:
            attn_out, attention_weights = attn_result
        else:
            attn_out = attn_result
            attention_weights = None

        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        if output_attentions:
            return x, attention_weights
        else:
            return x