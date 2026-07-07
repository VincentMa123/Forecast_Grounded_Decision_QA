# 放在 data/dataset.py 顶部或单独 utils 中
import numpy as np
import torch

def _to_writable_contiguous_float32(a: np.ndarray) -> torch.Tensor:
    """确保 numpy 数组是可写且 C 连续，然后再转 torch.Tensor（共享内存）"""
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)

    needs_copy = (
        not a.flags['C_CONTIGUOUS']
        or not a.flags['WRITEABLE']
        or 0 in getattr(a, 'strides', ())
    )
    if needs_copy:
        # 强制生成全新的 C 连续、可写数组
        a = np.array(a, dtype=np.float32, copy=True, order='C')
    return torch.from_numpy(a)
