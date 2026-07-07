"""
Positional encoding modules for the decoder.
"""

import torch
import torch.nn as nn
import math


class CombinedPositionalEncoding(nn.Module):
    """组合位置编码：时间维度 + 变量维度。"""

    def __init__(self, d_model: int, max_time_positions: int = 10,
                 max_variable_positions: int = 6712,
                 time_encoding_type: str = "learnable",
                 variable_encoding_type: str = "learnable"):
        super().__init__()
        self.d_model = d_model
        self.max_time_positions = max_time_positions
        self.max_variable_positions = max_variable_positions

        # 时间位置编码 - 默认使用可学习的embedding
        if time_encoding_type == "sinusoidal":
            self.time_pe = self._create_sinusoidal_encoding(max_time_positions, d_model)
            self.register_buffer('time_pe_buffer', self.time_pe)
        else:  # learnable
            self.time_pe = nn.Embedding(max_time_positions, d_model)
            # 初始化embedding权重
            nn.init.normal_(self.time_pe.weight, mean=0, std=d_model ** -0.5)

        # 变量位置编码 - 默认使用可学习的embedding
        if variable_encoding_type == "sinusoidal":
            self.variable_pe = self._create_sinusoidal_encoding(max_variable_positions, d_model)
            self.register_buffer('variable_pe_buffer', self.variable_pe)
        else:  # learnable
            self.variable_pe = nn.Embedding(max_variable_positions, d_model)
            # 初始化embedding权重
            nn.init.normal_(self.variable_pe.weight, mean=0, std=d_model ** -0.5)

        self.time_encoding_type = time_encoding_type
        self.variable_encoding_type = variable_encoding_type
    
    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """创建正弦-余弦位置编码。"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, x: torch.Tensor, time_steps: int, num_variables: int) -> torch.Tensor:
        """
        添加组合位置编码到输入。
        
        Args:
            x: 输入张量 [B, T*V, d_model]
            time_steps: 时间步数 T
            num_variables: 变量数 V
            
        Returns:
            位置编码后的张量 [B, T*V, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 创建时间和变量索引
        time_indices = torch.arange(time_steps, device=x.device).repeat_interleave(num_variables)  # [T*V]
        variable_indices = torch.arange(num_variables, device=x.device).repeat(time_steps)  # [T*V]
        
        # 获取位置编码
        if self.time_encoding_type == "sinusoidal":
            time_encoding = self.time_pe_buffer[time_indices]  # [T*V, d_model]
        else:
            time_encoding = self.time_pe(time_indices)  # [T*V, d_model]
        
        if self.variable_encoding_type == "sinusoidal":
            variable_encoding = self.variable_pe_buffer[variable_indices]  # [T*V, d_model]
        else:
            variable_encoding = self.variable_pe(variable_indices)  # [T*V, d_model]
        
        # 组合编码并添加到输入
        combined_encoding = time_encoding + variable_encoding  # [T*V, d_model]
        combined_encoding = combined_encoding.unsqueeze(0).expand(batch_size, -1, -1)  # [B, T*V, d_model]
        
        return x + combined_encoding