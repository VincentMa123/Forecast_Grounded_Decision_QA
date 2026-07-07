"""
Attention mask creation utilities for the decoder.
"""

import torch


class DecoderAttentionMask:
    """生成Decoder的attention mask和索引。"""
    
    @staticmethod
    def create_decoder_mask(
        batch_size: int,
        time_steps: int,
        num_variables: int,
        attention_indices: torch.Tensor,
        device: torch.device = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        创建decoder的稀疏attention mask基于拓扑结构。
        
        Args:
            batch_size: 批次大小 B
            time_steps: 时间步数 T
            num_variables: 变量数 V (6712)
            attention_indices: 拓扑注意力索引 [B, V, max_neighbors_variable]
            device: 设备
            causal: 是否使用因果mask
            
        Returns:
            attention_mask: [B, T*V, T*max_neighbors_variable] 稀疏attention mask
        """
        if device is None:
            device = torch.device('cpu')
        
        max_neighbors = attention_indices.shape[-1]  # max_neighbors_variable

        if not causal:
            return torch.ones(
                batch_size,
                time_steps * num_variables,
                time_steps * max_neighbors,
                device=device,
                dtype=torch.float32,
            )
        
        # 初始化mask: [B, T*V, T*max_neighbors_variable]
        mask = torch.zeros(
            batch_size,
            time_steps * num_variables,
            time_steps * max_neighbors,
            device=device,
            dtype=torch.float32,
        )
        
        # 外层循环：当前时间步 T
        for current_t in range(time_steps):
            # 当前时间步的变量索引范围: [current_t*V : (current_t+1)*V]
            current_start = current_t * num_variables
            current_end = (current_t + 1) * num_variables
            
            # 内层循环：历史时间步，包括当前时间步 (0 to current_t)
            for hist_t in range(current_t + 1):  # 因果性：只能看到当前及之前的时间步
                # 历史时间步在目标维度中的起始位置: hist_t * max_neighbors_variable
                hist_target_start = hist_t * max_neighbors
                hist_target_end = (hist_t + 1) * max_neighbors
                
                # 为当前时间步的所有变量批量设置mask
                # 形状: [B, V, max_neighbors_variable]
                mask[:, current_start:current_end, hist_target_start:hist_target_end] = 1.0
        
        return mask
