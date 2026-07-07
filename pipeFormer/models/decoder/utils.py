"""
Utility functions for tensor manipulations and visualization in the decoder.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def expand_tensor_follow_timeline(x: torch.Tensor, T: int) -> torch.Tensor:
    """
    将张量从 [B, V, max_neighbors_variable] 扩展到 [B, T*V, T*max_neighbors_variable], 本来每个都是取max_neighbors_variable个，但是现在因为有T的维度在
    
    Args:
        x: 输入张量，形状为 [B, V, max_neighbors_variable]
        T: 时间步数
    
    Returns:
        扩展后的张量，形状为 [B, T*V, T*max_neighbors_variable]
    """
    B, V, dim = x.shape
    
    # 方法1: 使用 repeat 和 reshape
    # 首先在时间维度上重复  
    # [B, V, max_neighbors_variable] -> [B, T, V, max_neighbors_variable] -> [B, T*V, max_neighbors_variable] -> [B, T*V, T, max_neighbors_variable] -> [B, T*V, T*max_neighbors_variable]
    x_expanded_dim2 = x.unsqueeze(1).repeat(1, T, 1, 1).reshape(B, T * V, dim)
    x_final = x_expanded_dim2.unsqueeze(2).repeat(1, 1, T, 1)
    for i in range(1, T):
        x_final[:, :, i, :] = x_final[:, :, i, :] + V * i
    x_final = x_final.reshape(B, T * V, T * dim)
    
    return x_final


def save_attention_mask_image(attention_mask: torch.Tensor, save_dir: str, filename: str = "attention_mask_batch0.png") -> None:
    """
    将注意力掩码的第0个batch可视化并保存为PNG。

    Args:
        attention_mask: [B, N, M] 或 [N, M] 的注意力掩码张量
        save_dir: 保存目录
        filename: 保存文件名
    """
    os.makedirs(save_dir, exist_ok=True)
    if attention_mask.dim() == 3:
        mask_2d = attention_mask[0]
    else:
        mask_2d = attention_mask

    mask_np = mask_2d.float().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    im = ax.imshow(mask_np, cmap="viridis", aspect="auto")
    ax.set_title("Attention Mask (batch 0)")
    ax.set_xlabel("Key Indices")
    ax.set_ylabel("Query Indices")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    save_path = os.path.join(save_dir, filename)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
