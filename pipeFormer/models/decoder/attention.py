"""
Multi-head attention mechanisms for the decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SimpleMultiHeadAttention(nn.Module):
    """简化的多头注意力机制，支持稀疏attention。"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, attention_indices_follow_timeline: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None, output_attentions: bool = False):
        """
        多头稀疏attention前向传播。

        Args:
            x: 输入张量 [B, T*V, d_model]
            attention_indices_follow_timeline: 时间线注意力索引 [B, T*V, T*max_neighbors_variable]
            attention_mask: 注意力mask [B, T*V, T*max_neighbors_variable] (可选)
            output_attentions: 是否返回注意力权重 (可选)

        Returns:
            如果output_attentions=True: (输出张量 [B, T*V, d_model], 注意力权重 [B, H, T*V, T*max_neighbors])
            否则: 输出张量 [B, T*V, d_model]
        """

        if attention_indices_follow_timeline is not None:
            # 稀疏attention路径
            return self._sparse_attention(x, attention_indices_follow_timeline, attention_mask, output_attentions)
        else:
            # 传统全注意力路径
            return self._full_attention(x, attention_mask, output_attentions)
    
    def timeline_attention_indexing(self, x, attention_indices_follow_timeline):
        """
        使用高级索引 
        Args:
            x: 输入张量 [B, T*V, d_model]
            attention_indices_follow_timeline: 时间线注意力索引 [B, T*V, T*max_neighbors_variable]
        
        Returns:
            output: [B, T*V, T*max_neighbors_variable, d_model]
        """
        B, TV, d_model = x.shape
        B, TV, num_indices = attention_indices_follow_timeline.shape
        
        # 创建batch索引
        batch_indices = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, TV, num_indices)
        
        # 使用高级索引选择元素
        # x[batch_indices, attention_indices_follow_timeline] -> [B, T*V, T*max_neighbors_variable, d_model]
        selected = x[batch_indices, attention_indices_follow_timeline]
        
        return selected
    
    def _sparse_attention(self, x: torch.Tensor, attention_indices: torch.Tensor,
                         attention_mask: Optional[torch.Tensor] = None, output_attentions: bool = False):
        """
        稀疏attention实现。这里Q和K是先进行投影，然后进行索引选择。

        Args:
            x: 输入张量 [B, T*V, d_model]
            attention_indices: 注意力索引 [B, T*V, T*64]
            attention_mask: 注意力mask [B, T*V, T*max_neighbors_variable]
            output_attentions: 是否返回注意力权重

        Returns:
            如果output_attentions=True: (输出张量 [B, T*V, d_model], 注意力权重 [B, H, T*V, T*max_neighbors])
            否则: 输出张量 [B, T*V, d_model]
        """
        batch_size, seq_len, d_model = x.shape  # [B, T*V, d_model]
        max_neighbors = attention_indices.shape[-1]  # max_neighbors_variable
        
        # Q: [B, T*V, d_model] -> [B, H, T*V, d_k]
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # 选择K和V: [B, T*V, T*max_neighbors_variable, d_model]
        k = self.w_k(x)
        v = self.w_v(x)
        k_selected = self.timeline_attention_indexing(k, attention_indices)  # [B, T*V, T*max_neighbors_variable, d_model]
        v_selected = self.timeline_attention_indexing(v, attention_indices)  # [B, T*V, T*max_neighbors_variable, d_model]


        # 重塑K和V为多头格式: [B, T*V, T*max_neighbors_variable, d_model] -> [B, H, T*V, T*max_neighbors_variable, d_k]
        k_selected = k_selected.view(batch_size, seq_len, max_neighbors, self.n_heads, self.d_k).permute(0, 3, 1, 2, 4)  # [B, H, T*V, T*max_neighbors_variable, d_k]
        v_selected = v_selected.view(batch_size, seq_len, max_neighbors, self.n_heads, self.d_k).permute(0, 3, 1, 2, 4)  # [B, H, T*V, T*max_neighbors_variable, d_k]
        
        # 计算稀疏注意力分数: Q @ K^T
        # Q: [B, H, T*V, d_k], K: [B, H, T*V, T*max_neighbors_variable, d_k]
        # 需要计算 Q[i] @ K[i, :].T 对于每个i
        Q_expanded = Q.unsqueeze(3)  # [B, H, T*V, 1, d_k]
        scores = torch.matmul(Q_expanded, k_selected.transpose(-2, -1))  # [B, H, T*V, 1, T*max_neighbors_variable]
        scores = scores.squeeze(3) / math.sqrt(self.d_k)  # [B, H, T*V, T*max_neighbors_variable]
        
        # 应用attention mask（如果有）
        if attention_mask is not None:
            # attention_mask: [B, T*V, T*max_neighbors_variable] -> [B, 1, T*V, T*max_neighbors_variable]
            attention_mask = attention_mask.unsqueeze(1)  # [B, 1, T*V, T*max_neighbors_variable]
            # 使用适合fp16的mask值，避免溢出
            mask_value = -30000.0 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(attention_mask == 0, mask_value)
        
        # Softmax和dropout - 生成概率分布的注意力权重
        attention_weights = F.softmax(scores, dim=-1)  # [B, H, T*V, T*max_neighbors_variable] - Softmax后的注意力权重
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力到values: [B, H, T*V, T*max_neighbors_variable] @ [B, H, T*V, T*max_neighbors_variable, d_k] -> [B, H, T*V, d_k]
        attention_weights_expanded = attention_weights.unsqueeze(-1)  # [B, H, T*V, T*max_neighbors_variable, 1]
        output = torch.sum(attention_weights_expanded * v_selected, dim=3)  # [B, H, T*V, d_k]
        
        # 重塑和输出投影
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # [B, T*V, d_model]
        output = self.w_o(output)

        if output_attentions:
            # 返回: (输出 [B, T*V, d_model], 注意力权重 [B, H, T*V, T*max_neighbors])
            return output, attention_weights
        else:
            return output
    
    
    def _full_attention(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, output_attentions: bool = False):
        """
        传统全注意力实现。

        Args:
            x: 输入张量 [B, T*V, d_model]
            attention_mask: 注意力mask [B, T*V, T*V]
            output_attentions: 是否返回注意力权重

        Returns:
            如果output_attentions=True: (输出张量 [B, T*V, d_model], 注意力权重 [B, H, T*V, T*V])
            否则: 输出张量 [B, T*V, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 线性投影和重塑
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, T*V, d_k]
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, T*V, d_k]
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, T*V, d_k]
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, T*V, T*V]
        
        if attention_mask is not None:
            # 扩展mask到多头维度
            attention_mask = attention_mask.unsqueeze(1)  # [B, 1, T*V, T*V]
            # 使用适合fp16的mask值，避免溢出
            mask_value = -30000.0 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(attention_mask == 0, mask_value)
        
        # Softmax和dropout - 生成概率分布的注意力权重
        attention_weights = F.softmax(scores, dim=-1)  # [B, H, T*V, T*V] - Softmax后的注意力权重
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力到values
        output = torch.matmul(attention_weights, V)  # [B, H, T*V, d_k]
        
        # 重塑和输出投影
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)  # [B, T*V, d_model]
        output = self.w_o(output)

        if output_attentions:
            # 返回: (输出 [B, T*V, d_model], 注意力权重 [B, H, T*V, T*V])
            return output, attention_weights
        else:
            return output