"""
Topology Attention Index Builder
构建基于拓扑结构的注意力索引矩阵，用于高效计算Graph Attention
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import os
import pickle
import logging
from pathlib import Path

from .graph import PipelineGraph
from .processor import DataProcessor
from tqdm import tqdm
logger = logging.getLogger(__name__)


class TopologyAttentionIndexBuilder:
    """构建基于拓扑图的注意力索引矩阵"""

    def __init__(
        self,
        data_dir: str,
        static_dir: Optional[str] = None,
        max_neighbors_variable: int = 32,
    ):
        """
        初始化拓扑注意力索引构建器
        
        Args:
            data_dir: 数据目录路径
            static_dir: 静态文件目录（默认 data/static/full）
            max_neighbors_variable: 每个变量最多关联的邻接变量数量（包括自身）
        """
        self.data_dir = Path(data_dir)
        self.static_dir = Path(static_dir).resolve() if static_dir else (self.data_dir / "static" / "full")
        self.static_dir.mkdir(parents=True, exist_ok=True)
        self.processor = DataProcessor(str(self.data_dir))

        # 判断是否为子图静态目录
        self.is_subgraph = self.static_dir.name != "full"

        # 加载或构建图网络（优先静态目录）
        self.graph: PipelineGraph = self._load_or_build_graph()

        if max_neighbors_variable <= 0:
            raise ValueError(f"max_neighbors_variable must be positive, got {max_neighbors_variable}")

        # 设定最大邻接数量 (11 + 1 自己)
        self.max_neighbors = 50
        # 设定最大邻接变量数量 (自己的所有变量要交互以及邻接设备的所有变量，按照顺序取到 max_neighbors_variable 为止)
        self.max_neighbors_variable = int(max_neighbors_variable)

        # 变量维度信息
        self.boundary_dims = 538  # 默认值，稍后根据变量更新
        self.equipment_dims = 6174
        self.total_dims = 6712
        self._full_var_to_index: Optional[Dict[str, int]] = None

        # 活跃变量映射
        self.active_variable_names, self.active_global_indices = self._load_active_variables()
        self.total_dims = len(self.active_variable_names)
        self._local_name_to_index = {
            name: idx for idx, name in enumerate(self.active_variable_names)
        }
        self.boundary_indices = [idx for idx, name in enumerate(self.active_variable_names) if self._is_boundary_variable(name)]
        self.boundary_dims = len(self.boundary_indices)
        self.equipment_dims = max(self.total_dims - self.boundary_dims, 0)
        self._equipment_to_indices = self._group_variables_by_equipment()

        # 设备类型信息
        self.equipment_info = self.processor.equipment_info

    def _load_or_build_graph(self) -> PipelineGraph:
        """加载或构建管网图"""
        primary_cache = self.static_dir / "pipeline_graph_cache.pkl"

        if primary_cache.exists():
            try:
                graph = PipelineGraph.load_graph(str(primary_cache))
                logger.info("Loaded pipeline graph from cache: %s", primary_cache)
                return graph
            except Exception as exc:
                raise RuntimeError(f"Failed to load graph cache {primary_cache}: {exc}") from exc

        if self.is_subgraph:
            raise FileNotFoundError(
                f"子图静态目录 {self.static_dir} 缺少 pipeline_graph_cache.pkl；"
                "请先运行 build_graph.py --subgraph-center ... 生成子图拓扑。"
            )

        # full graph: 尝试其他候选路径或重新构建
        fallback_paths = [
            self.data_dir / "pipeline_graph_cache.pkl",
        ]
        for cache_path in fallback_paths:
            if cache_path.exists():
                try:
                    graph = PipelineGraph.load_graph(str(cache_path))
                    logger.info("Loaded pipeline graph from fallback cache: %s", cache_path)
                    return graph
                except Exception as exc:
                    logger.warning("Failed to load fallback graph cache %s: %s", cache_path, exc)

        graph = PipelineGraph(str(self.data_dir / "equipment_arguments"))
        default_cache = primary_cache
        default_cache.parent.mkdir(parents=True, exist_ok=True)
        graph.save_graph(str(default_cache))
        logger.info("Rebuilt pipeline graph and saved to %s", default_cache)
        return graph

    def _load_active_variables(self) -> Tuple[List[str], List[int]]:
        mapping_path = self.static_dir / "index_variable_mapping.csv"
        variable_names: List[str] = []
        global_indices: List[int] = []

        if mapping_path.exists():
            try:
                mapping_df = pd.read_csv(mapping_path)
                if 'index' not in mapping_df.columns or 'variable_name' not in mapping_df.columns:
                    raise ValueError(f"{mapping_path} 缺少必要的列: index, variable_name")
                mapping_df = mapping_df.sort_values('index')
                variable_names = mapping_df['variable_name'].astype(str).tolist()
                if 'global_index' in mapping_df.columns:
                    global_indices = mapping_df['global_index'].astype(int).tolist()
                else:
                    global_map = self._get_full_variable_mapping()
                    resolved: List[int] = []
                    missing: List[str] = []
                    for idx, name in enumerate(variable_names):
                        global_idx = global_map.get(name)
                        if global_idx is None:
                            missing.append(name)
                            global_idx = idx
                        resolved.append(int(global_idx))
                    if missing:
                        logger.warning(
                            "%d variable(s) missing from full mapping when loading %s; using local indices as fallback. Examples: %s",
                            len(missing),
                            mapping_path,
                            missing[:5],
                        )
                    global_indices = resolved
                return variable_names, global_indices
            except Exception as exc:
                logger.warning("Failed to load active variable mapping from %s: %s", mapping_path, exc)

        full_map = self._get_full_variable_mapping()
        ordered = sorted(full_map.items(), key=lambda item: item[1])
        variable_names = [name for name, _ in ordered]
        global_indices = [idx for _, idx in ordered]
        return variable_names, global_indices

    def _get_full_variable_mapping(self) -> Dict[str, int]:
        if hasattr(self, '_full_var_to_index') and self._full_var_to_index is not None:
            return self._full_var_to_index
        var_to_index = self._build_variable_index_mapping()
        self._full_var_to_index = var_to_index
        return var_to_index

    def _is_boundary_variable(self, var_name: str) -> bool:
        return ':' in var_name

    def _group_variables_by_equipment(self) -> Dict[str, List[int]]:
        equipment_map: Dict[str, List[int]] = {}
        for idx, name in enumerate(self.active_variable_names):
            equipment_name, _ = self._parse_variable_name(name)
            if equipment_name is None:
                continue
            equipment_map.setdefault(equipment_name, []).append(idx)
        return equipment_map

    def _get_equipment_indices(self, equipment_name: Optional[str]) -> List[int]:
        if equipment_name is None:
            return []
        return self._equipment_to_indices.get(equipment_name, [])

    def _load_attention_file(
        self,
        base_dir: Path,
        filename: str = "attention_indices.pkl",
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        load_path = base_dir / filename

        if not load_path.exists():
            logger.warning("Attention indices file not found: %s", load_path)
            return None

        try:
            with open(load_path, 'rb') as f:
                data = pickle.load(f)

            indices = data['attention_indices']
            variable_names = data['variable_names']

            logger.info("Attention indices loaded from: %s, shape: %s", load_path, getattr(indices, 'shape', None))
            return indices, variable_names

        except Exception as exc:
            logger.error("Failed to load attention indices from %s: %s", load_path, exc)
            return None
    
    def build_attention_index_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建注意力索引矩阵 [6712, 32]
        
        每一行代表一个变量，包含该变量应该与之交互的其他变量的索引
        第一个索引总是自己的索引，后面是拓扑相关的邻接变量索引
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (注意力索引矩阵 [6712, 32], 变量名矩阵 [6712, 32])，-1表示填充位置
        """
        logger.info("Building topology attention index matrix...")
        
        # 初始化索引矩阵，用-1填充
        attention_indices = np.full((self.total_dims, self.max_neighbors_variable), -1, dtype=np.int32)
        # 初始化变量名矩阵，用空字符串填充
        variable_names = np.full((self.total_dims, self.max_neighbors_variable), '', dtype=object)

        # 为每个变量构建注意力索引
        for var_idx, var_name in enumerate(tqdm(self.active_variable_names, desc="Building attention indices", unit="var")):

            # 第一个位置总是自己
            attention_indices[var_idx, 0] = var_idx
            variable_names[var_idx, 0] = var_name

            # 如果是边界变量，只与自己交互（全部填充为自身索引，避免-1）
            if self._is_boundary_variable(var_name):
                attention_indices[var_idx, :] = var_idx
                variable_names[var_idx, :] = var_name
                continue

            # 获取设备名称和参数类型
            equipment_name, param_type = self._parse_variable_name(var_name)

            if equipment_name is None:
                continue

            # 查找邻接的目标设备
            adjacent_equipments: List[str] = []
            if self.graph is not None:
                try:
                    adjacent_equipments = self.graph.find_adjacent_target_nodes(
                        equipment_name,
                        certain_length=self.max_neighbors - 1,  # 减去自己的位置
                        max_depth=50
                    )
                except Exception as exc:
                    logger.debug("Failed to find adjacent nodes for %s: %s", equipment_name, exc)

            # 将邻接设备的所有变量加入索引
            neighbor_count = 1  # 从1开始，因为0位置是自己

            # 首先添加自己设备的其他变量 - 按索引顺序排序
            seen_indices = {var_idx}
            own_equipment_indices = self._get_equipment_indices(equipment_name)
            for other_idx in own_equipment_indices:
                if other_idx == var_idx:
                    continue
                if neighbor_count >= self.max_neighbors_variable:
                    break
                attention_indices[var_idx, neighbor_count] = other_idx
                variable_names[var_idx, neighbor_count] = self.active_variable_names[other_idx]
                seen_indices.add(other_idx)
                neighbor_count += 1

            # 然后添加邻接设备的所有变量 - 这里不能排序
            for adj_equipment in adjacent_equipments:
                if neighbor_count >= self.max_neighbors_variable:
                    break

                for neighbor_idx in self._get_equipment_indices(adj_equipment):
                    if neighbor_idx in seen_indices:
                        continue
                    if neighbor_count >= self.max_neighbors_variable:
                        break
                    attention_indices[var_idx, neighbor_count] = neighbor_idx
                    variable_names[var_idx, neighbor_count] = self.active_variable_names[neighbor_idx]
                    seen_indices.add(neighbor_idx)
                    neighbor_count += 1
        
        # 统计填充情况
        valid_count = np.sum(attention_indices >= 0)
        total_count = attention_indices.size
        fill_ratio = valid_count / total_count
        
        logger.info(f"Attention index matrix built: shape={attention_indices.shape}, "
                   f"valid_indices={valid_count}/{total_count} ({fill_ratio:.2%})")
        
        return attention_indices, variable_names

    def _build_variable_index_mapping(self) -> Dict[str, int]:
        """
        构建变量名到索引的映射关系
        
        Returns:
            Dict[str, int]: 变量名 -> 全局索引的映射
        """
        var_to_index = {}
        current_idx = 0
        
        # Boundary变量 (索引 0-537)
        boundary_file = self.data_dir / "dataset" / "train" / "第001个算例" / "Boundary.csv"
        if boundary_file.exists():
            boundary_df = pd.read_csv(boundary_file)
            boundary_cols = [col for col in boundary_df.columns if col != 'TIME']

            # 确保边界列按字典序排序，保证一致性
            for col in sorted(boundary_cols):
                var_to_index[col] = current_idx
                current_idx += 1
        else:
            # 如果文件不存在，使用设备信息构建边界变量名
            logger.warning("Boundary.csv not found, using fallback variable naming")
            for i in range(self.boundary_dims):
                var_to_index[f'boundary_var_{i}'] = current_idx
                current_idx += 1
        
        # 设备变量 (索引 538+) - 使用固定顺序确保一致性
        equipment_files = ['B.csv', 'C.csv', 'H.csv', 'N.csv', 'P.csv', 'R.csv', 'T&E.csv']

        # 按固定顺序处理设备文件
        for file_name in sorted(equipment_files):
            file_path = self.data_dir / "dataset" / "train" / "第001个算例" / file_name
            
            if file_path.exists():
                df = pd.read_csv(file_path)
                equipment_cols = [col for col in df.columns if col != 'TIME']

                # 确保列名按字典序排序，保证一致性
                for col in sorted(equipment_cols):
                    var_to_index[col] = current_idx
                    current_idx += 1
            else:
                # 使用设备信息作为备选
                equipment_type = file_name.split('.')[0]
                if equipment_type in self.equipment_info:
                    expected_dims = self.equipment_info[equipment_type]
                    for i in range(expected_dims):
                        var_to_index[f'{equipment_type}_var_{i}'] = current_idx
                        current_idx += 1
        
        if self.static_dir.name == "full":
            logger.info(f"Built variable index mapping: {len(var_to_index)} variables")
        else:
            logger.debug(
                "Built global variable index mapping (%d variables) while processing %s",
                len(var_to_index),
                self.static_dir,
            )
        return var_to_index
    
    def _parse_variable_name(self, var_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        解析变量名，提取设备名和参数类型
        
        Args:
            var_name: 变量名，如 "B_001_p_in" 或 "T_001:SNQ"
            
        Returns:
            Tuple[设备名, 参数类型] 如 ("B_001", "p_in")
        """
        # 处理边界变量格式 "T_001:SNQ"
        if ':' in var_name:
            equipment_name, param_type = var_name.split(':', 1)
            return equipment_name, param_type
        
        # 处理设备变量格式 "B_001_p_in"
        parts = var_name.split('_')
        if len(parts) >= 3:
            # 设备名: B_001, C_002, etc.
            equipment_name = '_'.join(parts[:2])  # B_001
            
            # 参数类型: p_in, t_out, etc.
            param_type = '_'.join(parts[2:])  # p_in
            
            return equipment_name, param_type
        
        return None, None
    
    def save_attention_indices(self, indices: np.ndarray, variable_names: np.ndarray, filename: str = "attention_indices.pkl") -> str:
        """
        保存注意力索引矩阵到文件
        
        Args:
            indices: 注意力索引矩阵
            variable_names: 变量名矩阵
            filename: 保存文件名
            
        Returns:
            保存文件的完整路径
        """
        save_path = self.static_dir / filename
        
        # 保存索引和变量名
        data_to_save = {
            'attention_indices': indices,
            'variable_names': variable_names
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        # 同时保存为CSV文件方便检查
        self._save_as_csv(indices, variable_names)
        self._save_index_variable_mapping()
        
        logger.info(f"Attention indices saved to: {save_path}")
        return str(save_path)
    
    def _save_as_csv(self, indices: np.ndarray, variable_names: np.ndarray):
        """
        保存为CSV文件方便检查

        Args:
            indices: 注意力索引矩阵
            variable_names: 变量名矩阵
        """
        # 保存索引CSV
        indices_df = pd.DataFrame(indices)
        indices_df.index.name = 'variable_index'
        indices_csv_path = self.static_dir / "attention_indices.csv"
        indices_df.to_csv(indices_csv_path)

        # 保存变量名CSV
        variable_names_df = pd.DataFrame(variable_names)
        variable_names_df.index.name = 'variable_index'
        var_names_csv_path = self.static_dir / "attention_variable_names.csv"
        variable_names_df.to_csv(var_names_csv_path)

        logger.info(f"CSV files saved: {indices_csv_path}, {var_names_csv_path}")

    def _save_index_variable_mapping(self):
        """
        保存简单的index-variable_name映射CSV
        """
        mapping_data = []
        for idx, name in enumerate(self.active_variable_names):
            mapping_data.append({
                'index': idx,
                'variable_name': name,
            })

        mapping_df = pd.DataFrame(mapping_data)
        mapping_csv_path = self.static_dir / "index_variable_mapping.csv"
        mapping_df.to_csv(mapping_csv_path, index=False)

        logger.info(
            "Index-variable mapping saved to: %s (variables=%d)",
            mapping_csv_path,
            len(mapping_data),
        )

    def load_attention_indices(self, filename: str = "attention_indices.pkl") -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        从静态目录加载注意力索引矩阵。
        """
        return self._load_attention_file(self.static_dir, filename)
    
    def get_or_build_attention_indices(self, force_rebuild: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取或构建注意力索引矩阵
        
        Args:
            force_rebuild: 是否强制重建
            
        Returns:
            注意力索引矩阵和变量名矩阵 [6712, 32]
        """
        if not force_rebuild:
            # 尝试加载现有的索引矩阵
            result = self.load_attention_indices()
            if result is not None:
                return result
        
        # 构建新的索引矩阵
        indices, variable_names = self.build_attention_index_matrix()
        
        # 保存到文件
        self.save_attention_indices(indices, variable_names)
        
        return indices, variable_names
    
    def visualize_attention_pattern(self, indices: np.ndarray) -> Dict:
        """
        可视化注意力模式的统计信息
        
        Args:
            indices: 注意力索引矩阵
            sample_variables: 要详细分析的变量索引列表
            
        Returns:
            统计信息字典
        """
        # 基本统计
        valid_mask = indices >= 0
        valid_counts_per_var = np.sum(valid_mask, axis=1)
        
        stats = {
            'total_variables': indices.shape[0],
            'max_neighbors': indices.shape[1],
            'avg_valid_neighbors': np.mean(valid_counts_per_var),
            'min_valid_neighbors': np.min(valid_counts_per_var),
            'max_valid_neighbors': np.max(valid_counts_per_var),
            'boundary_neighbors': valid_counts_per_var[: min(self.boundary_dims, indices.shape[0])],
            'equipment_neighbors': valid_counts_per_var[min(self.boundary_dims, indices.shape[0]):]
        }
        
        # 边界变量统计
        boundary_avg = np.mean(stats['boundary_neighbors'])
        equipment_avg = np.mean(stats['equipment_neighbors'])
        
        logger.info(f"Attention Pattern Statistics:")
        logger.info(f"  Total variables: {stats['total_variables']}")
        logger.info(f"  Average neighbors per variable: {stats['avg_valid_neighbors']:.2f}")
        logger.info(f"  Boundary variables avg neighbors: {boundary_avg:.2f}")
        logger.info(f"  Equipment variables avg neighbors: {equipment_avg:.2f}")
        
        return stats


def build_and_save_attention_indices(
    data_dir: str,
    static_dir: Optional[str] = None,
    force_rebuild: bool = False,
    max_neighbors_variable: Optional[int] = None,
) -> str:
    """
    构建并保存注意力索引矩阵
    
    Args:
        data_dir: 数据目录路径
        static_dir: 静态文件目录
        force_rebuild: 是否强制重建
        
    Returns:
        保存文件的路径
    """
    if max_neighbors_variable is not None and max_neighbors_variable <= 0:
        raise ValueError(f"max_neighbors_variable must be positive, got {max_neighbors_variable}")

    if max_neighbors_variable is not None:
        builder = TopologyAttentionIndexBuilder(
            data_dir,
            static_dir=static_dir,
            max_neighbors_variable=max_neighbors_variable,
        )
    else:
        builder = TopologyAttentionIndexBuilder(data_dir, static_dir=static_dir)
    indices, variable_names = builder.get_or_build_attention_indices(force_rebuild)
    
    # 显示统计信息
    builder.visualize_attention_pattern(indices)
    
    return str(builder.static_dir / "attention_indices.pkl")


def load_attention_indices(
    data_dir: str,
    static_dir: Optional[str] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    加载注意力索引矩阵
    
    Args:
        data_dir: 数据目录路径
        static_dir: 静态文件目录
        
    Returns:
        注意力索引矩阵和变量名矩阵，失败返回None
    """
    builder = TopologyAttentionIndexBuilder(data_dir, static_dir=static_dir)
    return builder.load_attention_indices()


if __name__ == "__main__":
    # 测试代码
    data_dir = "/home/chbds/zly/gaspipe/fluid_model/data"
    
    print("Building topology attention indices...")
    indices_path = build_and_save_attention_indices(data_dir, force_rebuild=True)
    
    print(f"Attention indices saved to: {indices_path}")
    
    # 加载测试
    result = load_attention_indices(data_dir)
    if result is not None:
        indices, variable_names = result
        print(f"Successfully loaded indices with shape: {indices.shape}")
        print(f"Sample indices[0]: {indices[0]}")
        print(f"Sample variable_names[0]: {variable_names[0]}")
        print(f"Sample indices[1000]: {indices[1000]}")
        print(f"Sample variable_names[1000]: {variable_names[1000]}")
