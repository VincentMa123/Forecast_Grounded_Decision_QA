import json
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class DataNormalizer:
    """
    独立的数据归一化管理器，用于流体动力学数据的标准化处理。

    支持特性：
    - 多种归一化方法：standard (z-score), minmax, robust
    - 分维度统计：对6712维数据的每一维分别计算统计量
    - CSV持久化存储：保存和加载均值、标准差等统计量到CSV文件
    - 批量处理：高效处理大批量数据
    - 异常处理：处理NaN、常数列等边界情况
    """

    def __init__(self,
                 static_dir: str,
                 method: str = 'standard',
                 boundary_dims: Optional[int] = None,
                 equipment_dims: Optional[int] = None):
        """
        初始化数据归一化器。

        Args:
            static_dir: 静态目录路径（如 data/static/full 或子图目录）
            method: 归一化方法 ('standard', 'minmax', 'robust')
            boundary_dims: boundary变量维度数（可选，若为空则尝试从静态目录元数据推断）
            equipment_dims: equipment变量维度数（可选）
        """
        self.static_dir = Path(static_dir).resolve()
        self.method = method.lower()
        self.boundary_dims = boundary_dims
        self.equipment_dims = equipment_dims
        self.total_dims = int(boundary_dims or 0) + int(equipment_dims or 0)

        # 验证归一化方法
        if self.method not in ['standard', 'minmax', 'robust']:
            raise ValueError(f"Unsupported normalization method: {method}")

        # 统计量存储
        self.mean_ = None      # shape [6712]
        self.std_ = None       # shape [6712]
        self.min_ = None       # shape [6712] (for minmax)
        self.max_ = None       # shape [6712] (for minmax)
        self.q25_ = None       # shape [6712] (for robust)
        self.q75_ = None       # shape [6712] (for robust)
        self.median_ = None    # shape [6712] (for robust)

        # 统计量文件存储在静态目录下的 normalizer_save 子目录
        self.stats_dir = self.static_dir / "normalizer_save"
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        self.variable_names: Optional[List[str]] = None
        self._variable_name_to_index: Dict[str, int] = {}
        self.global_indices: Optional[np.ndarray] = None
        self.fitted = False

        # 尝试从静态目录的图元数据中推断维度配置
        self.graph_hyperparameters = self._load_graph_hyperparameters()
        variables_meta = self.graph_hyperparameters.get('variables', {}) if isinstance(self.graph_hyperparameters, dict) else {}
        if self.boundary_dims is None and isinstance(variables_meta, dict):
            boundary_value = variables_meta.get('boundary_variables')
            if boundary_value is not None:
                try:
                    self.boundary_dims = int(boundary_value)
                except Exception:
                    logger.warning("Invalid boundary_variables in graph_hyperparameters for %s: %s", self.static_dir, boundary_value)
        if self.equipment_dims is None and isinstance(variables_meta, dict):
            equipment_value = variables_meta.get('equipment_variables')
            if equipment_value is not None:
                try:
                    self.equipment_dims = int(equipment_value)
                except Exception:
                    logger.warning("Invalid equipment_variables in graph_hyperparameters for %s: %s", self.static_dir, equipment_value)
        if self.total_dims == 0:
            total_meta = None
            if isinstance(variables_meta, dict):
                total_meta = variables_meta.get('total_variables')
            if total_meta is not None:
                try:
                    self.total_dims = int(total_meta)
                except Exception:
                    logger.warning("Invalid total_variables in graph_hyperparameters for %s: %s", self.static_dir, total_meta)
            elif self.boundary_dims is not None or self.equipment_dims is not None:
                self.total_dims = int(self.boundary_dims or 0) + int(self.equipment_dims or 0)

        logger.info(
            "DataNormalizer initialized: method=%s, static_dir=%s, dims=%s",
            method,
            self.static_dir,
            self.total_dims or 'unknown',
        )

    def _assert_stats_ready(self) -> None:
        """
        Ensure stats required by the selected normalization method are available.
        """
        if not self.fitted:
            raise RuntimeError("Normalizer not loaded. Call load_stats() first.")

        if self.method == 'standard':
            if self.mean_ is None or self.std_ is None:
                raise RuntimeError("Standard normalization stats are missing mean/std values.")
        elif self.method == 'minmax':
            if self.min_ is None or self.max_ is None:
                raise RuntimeError("Minmax normalization stats are missing min/max values.")
        elif self.method == 'robust':
            if self.q25_ is None or self.q75_ is None or self.median_ is None:
                raise RuntimeError("Robust normalization stats are missing quantile values.")

    def _load_graph_hyperparameters(self) -> Dict[str, Any]:
        hyper_path = self.static_dir / "graph_hyperparameters.json"
        if not hyper_path.exists():
            return {}
        try:
            with hyper_path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, dict):
                return data
            logger.warning("Graph hyperparameters at %s are not a JSON object.", hyper_path)
            return {}
        except Exception as exc:
            logger.warning("Failed to load graph hyperparameters from %s: %s", hyper_path, exc)
            return {}

    def _resolve_mapping_path(self) -> Optional[Path]:
        candidates = [
            self.static_dir / "index_variable_mapping.csv",
            self.static_dir.parent / "index_variable_mapping.csv",
            self.static_dir.parent.parent / "index_variable_mapping.csv" if self.static_dir.parent.parent != self.static_dir.parent else None,
        ]
        for candidate in candidates:
            if candidate is not None and candidate.exists():
                return candidate
        return None

    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        对数据进行归一化变换。

        Args:
            data: 输入数据，shape [..., 6712]

        Returns:
            归一化后的数据，保持输入类型和形状
        """
        self._assert_stats_ready()

        # 处理torch.Tensor
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            device = data.device
            data_np = data.detach().cpu().numpy()
        else:
            data_np = data

        # 保存原始形状
        original_shape = data_np.shape

        # 重塑为2D用于归一化
        if data_np.ndim > 2:
            data_2d = data_np.reshape(-1, self.total_dims)
        elif data_np.ndim == 2:
            data_2d = data_np
        else:
            raise ValueError(f"Unsupported data shape: {original_shape}")

        # 验证维度
        if data_2d.shape[1] != self.total_dims:
            raise ValueError(f"Expected {self.total_dims} features, got {data_2d.shape[1]}")

        # 应用归一化
        if self.method == 'standard':
            normalized = (data_2d - self.mean_) / self.std_
        elif self.method == 'minmax':
            normalized = (data_2d - self.min_) / (self.max_ - self.min_)
            normalized = np.clip(normalized, 0.0, 1.0)
        elif self.method == 'robust':
            iqr = self.q75_ - self.q25_
            normalized = (data_2d - self.median_) / iqr

        # FP16安全裁剪：防止混合精度训练溢出
        # FP16范围约为[-65504, 65504]，留一些安全边界
        normalized = np.clip(normalized, -50000.0, 50000.0)

        # 重塑回原始形状
        normalized = normalized.reshape(original_shape)

        # 转换回torch.Tensor
        if is_tensor:
            normalized = torch.from_numpy(normalized.astype(np.float32)).to(device)

        return normalized

    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        对归一化数据进行反变换，恢复原始尺度。

        Args:
            data: 归一化后的数据，shape [..., 6712]

        Returns:
            恢复原始尺度的数据
        """
        self._assert_stats_ready()

        # 处理torch.Tensor
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            device = data.device
            data_np = data.detach().cpu().numpy()
        else:
            data_np = data

        # 保存原始形状
        original_shape = data_np.shape

        # 重塑为2D
        if data_np.ndim > 2:
            data_2d = data_np.reshape(-1, self.total_dims)
        elif data_np.ndim == 2:
            data_2d = data_np
        elif data_np.ndim == 1:
            data_2d = data_np.reshape(1, -1)
        else:
            raise ValueError(f"Unsupported data shape: {original_shape}")

        # 应用反变换
        if self.method == 'standard':
            denormalized = data_2d * self.std_ + self.mean_
        elif self.method == 'minmax':
            clipped = np.clip(data_2d, 0.0, 1.0)
            denormalized = clipped * (self.max_ - self.min_) + self.min_
        elif self.method == 'robust':
            iqr = self.q75_ - self.q25_
            denormalized = data_2d * iqr + self.median_

        # 重塑回原始形状
        denormalized = denormalized.reshape(original_shape)

        # 转换回torch.Tensor
        if is_tensor:
            denormalized = torch.from_numpy(denormalized.astype(np.float32)).to(device)

        return denormalized

    def denormalize(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        对归一化数据进行反变换，恢复原始尺度（inverse_transform的别名）。

        Args:
            data: 归一化后的数据，shape [..., 6712]

        Returns:
            恢复原始尺度的数据
        """
        return self.inverse_transform(data)

    def save_stats(self, filename: Optional[str] = None) -> str:
        """
        保存归一化统计量到CSV文件。

        Args:
            filename: 保存文件名（可选），默认为 normalization_stats_{method}.csv

        Returns:
            保存文件的完整路径
        """
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Nothing to save.")

        if filename is None:
            filename = f"normalization_stats_{self.method}.csv"

        save_path = self.stats_dir / filename

        # 读取变量名
        if self.variable_names is not None and len(self.variable_names) == self.total_dims:
            feature_names = list(self.variable_names)
        else:
            mapping_path = self._resolve_mapping_path()
            if mapping_path is not None and mapping_path.exists():
                try:
                    mapping_df = pd.read_csv(mapping_path)
                    feature_names = mapping_df['variable_name'].astype(str).tolist()
                except Exception as exc:
                    logger.warning("Failed to read variable mapping from %s: %s", mapping_path, exc)
                    feature_names = [f"feature_{i:04d}" for i in range(self.total_dims)]
            else:
                feature_names = [f"feature_{i:04d}" for i in range(self.total_dims)]
                logger.warning("Variable mapping file not found near %s; using index names", self.static_dir)

        # 构建DataFrame
        df_dict = {
            'variable_name': feature_names,
            'normalization_method': [self.method] * self.total_dims
        }

        # 根据方法添加相应统计量
        if self.method == 'standard':
            df_dict.update({
                'mean': self.mean_,
                'std': self.std_,
                'min': self.min_ if self.min_ is not None else np.full(self.total_dims, np.nan),
                'max': self.max_ if self.max_ is not None else np.full(self.total_dims, np.nan),
                'q25': self.q25_ if self.q25_ is not None else np.full(self.total_dims, np.nan),
                'median': self.median_ if self.median_ is not None else np.full(self.total_dims, np.nan),
                'q75': self.q75_ if self.q75_ is not None else np.full(self.total_dims, np.nan)
            })
        elif self.method == 'minmax':
            df_dict.update({
                'mean': self.mean_ if self.mean_ is not None else np.full(self.total_dims, np.nan),
                'std': self.std_ if self.std_ is not None else np.full(self.total_dims, np.nan),
                'min': self.min_,
                'max': self.max_,
                'q25': self.q25_ if self.q25_ is not None else np.full(self.total_dims, np.nan),
                'median': self.median_ if self.median_ is not None else np.full(self.total_dims, np.nan),
                'q75': self.q75_ if self.q75_ is not None else np.full(self.total_dims, np.nan)
            })
        elif self.method == 'robust':
            df_dict.update({
                'mean': self.mean_ if self.mean_ is not None else np.full(self.total_dims, np.nan),
                'std': self.std_ if self.std_ is not None else np.full(self.total_dims, np.nan),
                'min': self.min_ if self.min_ is not None else np.full(self.total_dims, np.nan),
                'max': self.max_ if self.max_ is not None else np.full(self.total_dims, np.nan),
                'q25': self.q25_,
                'median': self.median_,
                'q75': self.q75_
            })

        # 创建DataFrame并保存
        df = pd.DataFrame(df_dict)
        df.to_csv(save_path, index=False, float_format='%.6f')

        logger.info(f"Normalization stats saved to {save_path}")
        return str(save_path)

    def load_stats(self, filename: Optional[str] = None) -> bool:
        """
        从CSV文件加载归一化统计量。

        Args:
            filename: 加载文件名（可选），默认为 normalization_stats_{method}.csv

        Returns:
            加载是否成功
        """
        if filename is None:
            filename = f"normalization_stats_{self.method}.csv"

        load_path = self.stats_dir / filename

        if not load_path.exists():
            logger.error(f"Stats file not found: {load_path}")
            return False

        # 读取CSV文件
        df = pd.read_csv(load_path)

        # 验证列存在
        required_cols = ['variable_name', 'normalization_method']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns in {load_path}")
            return False

        # 验证维度（允许自动调整）
        if len(df) != self.total_dims:
            logger.warning(f"Stats dimension mismatch: expected {self.total_dims}, got {len(df)}; adopting stats dimension.")
            self.total_dims = len(df)
            self.boundary_dims = min(self.boundary_dims, self.total_dims) if self.boundary_dims is not None else self.boundary_dims
            self.equipment_dims = max(self.total_dims - (self.boundary_dims or 0), 0)

        self.variable_names = df['variable_name'].astype(str).tolist()
        self._variable_name_to_index = {name: idx for idx, name in enumerate(self.variable_names)}
        self.global_indices = np.arange(len(df), dtype=np.int32)

        # 加载统计量
        if self.method == 'standard':
            if 'mean' not in df.columns or 'std' not in df.columns:
                logger.error("Missing 'mean' or 'std' columns for standard normalization")
                return False
            self.mean_ = df['mean'].values
            self.std_ = df['std'].values

            # 可选加载其他统计量
            if 'min' in df.columns:
                self.min_ = df['min'].values
            if 'max' in df.columns:
                self.max_ = df['max'].values
            if 'q25' in df.columns:
                self.q25_ = df['q25'].values
            if 'median' in df.columns:
                self.median_ = df['median'].values
            if 'q75' in df.columns:
                self.q75_ = df['q75'].values

        elif self.method == 'minmax':
            if 'min' not in df.columns or 'max' not in df.columns:
                logger.error("Missing 'min' or 'max' columns for minmax normalization")
                return False
            self.min_ = df['min'].values
            self.max_ = df['max'].values

        elif self.method == 'robust':
            if 'q25' not in df.columns or 'q75' not in df.columns or 'median' not in df.columns:
                logger.error("Missing quantile columns for robust normalization")
                return False
            self.q25_ = df['q25'].values
            self.q75_ = df['q75'].values
            self.median_ = df['median'].values

        self.fitted = True
        logger.info(f"Normalization stats loaded from {load_path}")
        return True

    def get_stats_summary(self) -> Dict:
        """
        获取归一化统计量的摘要信息。

        Returns:
            统计量摘要字典
        """
        if not self.fitted:
            return {'fitted': False, 'method': self.method}

        summary = {
            'fitted': True,
            'method': self.method,
            'total_dims': self.total_dims,
            'boundary_dims': self.boundary_dims,
            'equipment_dims': self.equipment_dims
        }

        if self.method == 'standard':
            summary.update({
                'mean_range': [float(self.mean_.min()), float(self.mean_.max())],
                'std_range': [float(self.std_.min()), float(self.std_.max())],
                'zero_std_cols': int(np.sum(self.std_ == 1.0))  # 检查设置为1的常数列
            })
        elif self.method == 'minmax':
            summary.update({
                'min_range': [float(self.min_.min()), float(self.min_.max())],
                'max_range': [float(self.max_.min()), float(self.max_.max())],
                'const_cols': int(np.sum((self.max_ - self.min_) == 1.0))  # 检查设置范围为1的常数列
            })
        elif self.method == 'robust':
            iqr = self.q75_ - self.q25_
            summary.update({
                'median_range': [float(self.median_.min()), float(self.median_.max())],
                'iqr_range': [float(iqr.min()), float(iqr.max())],
                'zero_iqr_cols': int(np.sum(iqr == 1.0))  # 检查设置为1的零IQR列
            })

        return summary


def load_normalizer(static_dir: str, method: str = 'standard'):
    """
    加载已保存的归一化器。

    Args:
        static_dir: 静态目录路径
        method: 归一化方法 ('standard', 'minmax', 'robust')

    Returns:
        加载的DataNormalizer实例，如果加载失败则返回None
    """
    try:
        normalizer = DataNormalizer(static_dir=static_dir, method=method)
        if normalizer.load_stats():
            logger.info("Successfully loaded %s normalizer from %s", method, static_dir)
            return normalizer
        logger.warning("Failed to load normalizer stats from %s", static_dir)
        return None
    except Exception as e:
        logger.error("Error loading normalizer from %s: %s", static_dir, e)
        return None
