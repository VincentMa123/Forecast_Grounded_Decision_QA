"""
静态设备参数处理模块

处理 P_arguments.xlsx 和 C_arguments.xlsx 文件，为数据集添加静态设备参数。
- P_arguments.xlsx: 管道静态参数，归一化为 [num_pipes, 9] 的张量
- C_arguments.xlsx: 压缩机静态参数，PCA压缩为 [num_compressors, pca_dim] 的张量
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler
import logging
import sys



from .process_eq_argu import embed_from_excel


logger = logging.getLogger(__name__)


class StaticEquipmentProcessor:
    """
    静态设备参数处理器

    功能：
    1. 读取并处理 P_arguments.xlsx（管道参数）
    2. 读取并处理 C_arguments.xlsx（压缩机参数）
    3. 对管道参数进行归一化
    4. 对压缩机参数进行PCA降维
    5. 保存处理结果供数据集使用
    """

    def __init__(self, data_dir: str):
        """
        初始化静态设备参数处理器

        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.equipment_args_dir = self.data_dir / "equipment_arguments"

        # 管道参数的9个维度（排除上下游节点）
        self.pipe_feature_columns = [
            'pipe_length_km', 'outer_diameter_mm', 'wall_thickness_mm',
            'heat_transfer_coefficient', 'wall_roughness_mm',
            'outlet_elevation_m', 'inlet_elevation_m', 'outlet_soil_temp_c', 'inlet_soil_temp_c'
        ]

        # 处理结果存储
        self.pipe_features = None      # [num_pipes, 9]
        self.pipe_names = None         # [num_pipes]
        self.compressor_features = None  # [num_compressors, pca_dim]
        self.compressor_names = None   # [num_compressors]
        self.pipe_scaler = None        # 管道参数归一化器
        self.compressor_pca = None     # 压缩机PCA模型

        self._validate_directories()

    def _validate_directories(self):
        """验证必要的目录和文件是否存在"""
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        if not self.equipment_args_dir.exists():
            raise ValueError(f"Equipment arguments directory does not exist: {self.equipment_args_dir}")

        # 检查必要的文件
        p_file = self.equipment_args_dir / "P_arguments.xlsx"
        c_file = self.equipment_args_dir / "C_arguments.xlsx"

        if not p_file.exists():
            raise ValueError(f"P_arguments.xlsx not found: {p_file}")
        if not c_file.exists():
            raise ValueError(f"C_arguments.xlsx not found: {c_file}")

    def process_pipe_arguments(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理管道参数文件 P_arguments.xlsx

        Returns:
            pipe_features: 归一化的管道参数 [num_pipes, 9]
            pipe_names: 管道名称列表 [num_pipes]
        """
        logger.info("Processing P_arguments.xlsx...")

        # 读取管道参数文件
        p_file = self.equipment_args_dir / "P_arguments.xlsx"
        p_df = pd.read_excel(p_file)

        logger.info(f"Loaded {len(p_df)} pipes from P_arguments.xlsx")

        # 提取管道名称
        pipe_names = p_df['管道名称'].values

        # 提取9个特征维度（排除上下游节点）
        # 将原始列名映射到新的列名
        column_mapping = {
            '管长\nkm': 'pipe_length_km',
            '管道外径\nmm': 'outer_diameter_mm',
            '管壁厚度\nmm': 'wall_thickness_mm',
            '总传热系数\nW/(m2·℃)': 'heat_transfer_coefficient',
            '管壁粗糙度\nmm': 'wall_roughness_mm',
            '出口高程\nm': 'outlet_elevation_m',
            '进口高程\nm': 'inlet_elevation_m',
            '出口地温\n℃': 'outlet_soil_temp_c',
            '入口地温\n℃': 'inlet_soil_temp_c'
        }

        # 重命名列
        p_df = p_df.rename(columns=column_mapping)
        pipe_features_raw = p_df[self.pipe_feature_columns].values

        # 数据清洗：处理可能的缺失值
        if np.isnan(pipe_features_raw).any():
            logger.warning("Found NaN values in pipe features, filling with median values")
            for i in range(pipe_features_raw.shape[1]):
                col = pipe_features_raw[:, i]
                nan_mask = np.isnan(col)
                if nan_mask.any():
                    median_val = np.nanmedian(col)
                    pipe_features_raw[nan_mask, i] = median_val

        # 标准化管道参数
        self.pipe_scaler = StandardScaler()
        pipe_features_normalized = self.pipe_scaler.fit_transform(pipe_features_raw)

        self.pipe_features = pipe_features_normalized.astype(np.float32)
        self.pipe_names = pipe_names

        logger.info(f"Processed pipe features: shape={self.pipe_features.shape}")
        logger.info(f"Pipe feature columns: {self.pipe_feature_columns}")

        return self.pipe_features, self.pipe_names

    def process_compressor_arguments(self, k: int = 8, grid_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理压缩机参数文件 C_arguments.xlsx

        使用物理归一化和PCA降维来提取压缩机特征嵌入

        Args:
            k: PCA降维后的维度 (默认8维)
            grid_size: 性能曲线网格大小 (默认32)

        Returns:
            compressor_features: PCA降维后的压缩机参数 [num_compressors, k]
            compressor_names: 压缩机名称列表 [num_compressors]
        """
        logger.info("Processing C_arguments.xlsx using physics-based embedding...")

        # 使用 embed_from_excel 函数提取压缩机嵌入
        c_file = self.equipment_args_dir / "C_arguments.xlsx"

        try:
            embed_df = embed_from_excel(str(c_file), k=k, grid_size=grid_size)

            # 提取压缩机名称和特征
            compressor_names = embed_df['id'].values
            compressor_features = embed_df.drop(columns=['id']).values.astype(np.float32)

            logger.info(f"Loaded {len(compressor_names)} compressors from C_arguments.xlsx")
            logger.info(f"Processed compressor features using physics-based PCA: shape={compressor_features.shape}")

            self.compressor_features = compressor_features
            self.compressor_names = compressor_names

            return self.compressor_features, self.compressor_names

        except Exception as e:
            logger.error(f"Failed to process compressor arguments: {e}")
            logger.warning("Falling back to simple encoding...")

            # 备用方案：使用简单的设备名称编码
            c_df = pd.read_excel(c_file)
            compressor_names = c_df['设备名称'].values

            num_compressors = len(compressor_names)
            compressor_features = np.arange(
                num_compressors, dtype=np.float32
            ).reshape(num_compressors, 1)

            self.compressor_features = compressor_features
            self.compressor_names = compressor_names

            logger.info(f"Using fallback index-based encoding: shape={compressor_features.shape}")

            return self.compressor_features, self.compressor_names

    def process_all(self, compressor_k: int = 8, compressor_grid_size: int = 32) -> Dict[str, np.ndarray]:
        """
        处理所有静态设备参数

        Args:
            compressor_k: 压缩机PCA降维后的维度
            compressor_grid_size: 压缩机性能曲线网格大小

        Returns:
            包含所有处理结果的字典
        """
        logger.info("Processing all static equipment parameters...")

        # 处理管道参数
        pipe_features, pipe_names = self.process_pipe_arguments()

        # 处理压缩机参数
        compressor_features, compressor_names = self.process_compressor_arguments(
            k=compressor_k,
            grid_size=compressor_grid_size
        )

        result = {
            'pipe_features': pipe_features,
            'pipe_names': pipe_names,
            'compressor_features': compressor_features,
            'compressor_names': compressor_names
        }

        logger.info("All static equipment parameters processed successfully")
        return result

    def save_processed_data(self, output_dir: Optional[str] = None):
        """
        保存处理后的静态设备参数到CSV文件

        Args:
            output_dir: 输出目录路径，如果为None则使用默认路径
        """
        if self.pipe_features is None or self.compressor_features is None:
            raise ValueError("No processed data to save. Call process_all() first.")

        if output_dir is None:
            output_dir = self.data_dir / "process_eq_argu"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存管道参数
        pipe_df = pd.DataFrame(
            self.pipe_features,
            columns=self.pipe_feature_columns,
            index=self.pipe_names
        )
        pipe_file = output_dir / "pipe_features.csv"
        pipe_df.to_csv(pipe_file)
        logger.info(f"Saved pipe features to: {pipe_file}")

        # 保存压缩机参数
        compressor_dim = self.compressor_features.shape[1]
        compressor_columns = [f"pca_feature_{i+1}" for i in range(compressor_dim)]
        compressor_df = pd.DataFrame(
            self.compressor_features,
            columns=compressor_columns,
            index=self.compressor_names
        )
        compressor_file = output_dir / "compressor_features.csv"
        compressor_df.to_csv(compressor_file)
        logger.info(f"Saved compressor features to: {compressor_file}")

        # 保存元数据
        metadata = {
            'num_pipes': len(self.pipe_names),
            'num_compressors': len(self.compressor_names),
            'pipe_feature_dim': self.pipe_features.shape[1],
            'compressor_feature_dim': self.compressor_features.shape[1],
            'pipe_feature_columns': self.pipe_feature_columns
        }

        metadata_file = output_dir / "metadata.csv"
        metadata_df = pd.DataFrame([metadata])
        metadata_df.to_csv(metadata_file, index=False)
        logger.info(f"Saved metadata to: {metadata_file}")

        # 归一化器和PCA模型参数通过保存归一化后的数据来保存，不使用pickle

        logger.info(f"All static equipment data saved to: {output_dir}")
        logger.info(f"  - Pipes: {metadata['num_pipes']} x {metadata['pipe_feature_dim']}")
        logger.info(f"  - Compressors: {metadata['num_compressors']} x {metadata['compressor_feature_dim']}")

    def load_processed_data(self, input_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        从CSV文件加载处理后的静态设备参数

        Args:
            input_dir: 输入目录路径，如果为None则使用默认路径

        Returns:
            包含所有处理结果的字典
        """
        if input_dir is None:
            input_dir = self.data_dir / "process_eq_argu"

        input_dir = Path(input_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Processed data directory not found: {input_dir}")

        # 加载管道参数
        pipe_file = input_dir / "pipe_features.csv"
        if not pipe_file.exists():
            raise FileNotFoundError(f"Pipe features file not found: {pipe_file}")
        pipe_df = pd.read_csv(pipe_file, index_col=0)
        self.pipe_features = pipe_df.values.astype(np.float32)
        self.pipe_names = pipe_df.index.values

        # 加载压缩机参数
        compressor_file = input_dir / "compressor_features.csv"
        if not compressor_file.exists():
            raise FileNotFoundError(f"Compressor features file not found: {compressor_file}")
        compressor_df = pd.read_csv(compressor_file, index_col=0)
        self.compressor_features = compressor_df.values.astype(np.float32)
        self.compressor_names = compressor_df.index.values

        # 加载元数据
        metadata_file = input_dir / "metadata.csv"
        if metadata_file.exists():
            metadata_df = pd.read_csv(metadata_file)
            metadata = metadata_df.iloc[0].to_dict()
            self.pipe_feature_columns = eval(metadata['pipe_feature_columns'])  # 转换字符串回列表
        else:
            self.pipe_feature_columns = [
                'pipe_length_km', 'outer_diameter_mm', 'wall_thickness_mm',
                'heat_transfer_coefficient', 'wall_roughness_mm',
                'outlet_elevation_m', 'inlet_elevation_m', 'outlet_soil_temp_c', 'inlet_soil_temp_c'
            ]

        # 不需要加载归一化器和PCA模型，因为数据已经是处理后的状态
        self.pipe_scaler = None
        self.compressor_pca = None

        result = {
            'pipe_features': self.pipe_features,
            'pipe_names': self.pipe_names,
            'compressor_features': self.compressor_features,
            'compressor_names': self.compressor_names,
            'pipe_feature_columns': self.pipe_feature_columns,
            'metadata': {
                'num_pipes': len(self.pipe_names),
                'num_compressors': len(self.compressor_names),
                'pipe_feature_dim': self.pipe_features.shape[1],
                'compressor_feature_dim': self.compressor_features.shape[1]
            }
        }

        logger.info(f"Loaded static equipment data from: {input_dir}")
        logger.info(f"  - Pipes: {result['metadata']['num_pipes']} x {result['metadata']['pipe_feature_dim']}")
        logger.info(f"  - Compressors: {result['metadata']['num_compressors']} x {result['metadata']['compressor_feature_dim']}")

        return result


def create_static_equipment_data(data_dir: str,
                                force_reprocess: bool = False,
                                compressor_k: int = 8,
                                compressor_grid_size: int = 32) -> Dict[str, np.ndarray]:
    """
    创建静态设备参数数据的便捷函数

    Args:
        data_dir: 数据目录路径
        force_reprocess: 是否强制重新处理
        compressor_k: 压缩机PCA降维后的维度
        compressor_grid_size: 压缩机性能曲线网格大小

    Returns:
        包含所有处理结果的字典
    """
    processor = StaticEquipmentProcessor(data_dir)

    # 检查是否已有处理结果
    default_output_dir = Path(data_dir) / "process_eq_argu"

    if not force_reprocess and default_output_dir.exists():
        # 检查必要的文件是否存在
        required_files = [
            default_output_dir / "pipe_features.csv",
            default_output_dir / "compressor_features.csv",
            default_output_dir / "metadata.csv"
        ]
        if all(file.exists() for file in required_files):
            logger.info("Loading existing static equipment data...")
            return processor.load_processed_data()

    # 处理数据并保存
    result = processor.process_all(
        compressor_k=compressor_k,
        compressor_grid_size=compressor_grid_size
    )
    processor.save_processed_data()

    return result


if __name__ == "__main__":
    # 测试代码
    import logging

    logging.basicConfig(level=logging.INFO)

    # 处理静态设备参数
    data_dir = "data"
    result = create_static_equipment_data(data_dir, force_reprocess=True)

    print("Processing completed!")
    print(f"Pipe features shape: {result['pipe_features'].shape}")
    print(f"Compressor features shape: {result['compressor_features'].shape}")
    print(f"Pipe names: {result['pipe_names'][:5]}...")
    print(f"Compressor names: {result['compressor_names'][:5]}...")
