"""
Gas Pipeline Network Graph Builder
构建天然气管网图网络，并提供邻接节点查找功能
"""
from typing import Dict, List, Set, Optional
from collections import defaultdict, deque
import pandas as pd
import os
import pickle
import random


class PipelineGraph:
    """天然气管网图网络类"""
    
    def __init__(self, equipment_dir: str = "data/equipment_arguments", 
                 connected_nodes_file: str = "relative_PPT/connected_nodes.csv"):
        """
        初始化管网图
        
        Args:
            equipment_dir: 设备参数文件目录路径
            connected_nodes_file: T/E节点与N节点连接关系文件路径
        """
        self.equipment_dir = equipment_dir
        self.connected_nodes_file = connected_nodes_file
        # 图的邻接表表示 {node: set(connected_nodes)}
        self.graph: Dict[str, Set[str]] = defaultdict(set)

        # 目标设备类型 - 只关心这些类型的设备
        self.target_equipment_types = {'T', 'E', 'C', 'R', 'B'}
        
        self._build_graph()
    
    def _build_graph(self):
        """从设备参数文件构建图网络"""
        # 设备文件映射 - 使用固定顺序确保一致性
        equipment_files = [
            ('B', 'B_arguments.xlsx'),  # 球阀
            ('C', 'C_arguments.xlsx'),  # 压缩机
            ('H', 'H_arguments.xlsx'),  # 管段
            ('P', 'P_arguments.xlsx'),  # 管道
            ('R', 'R_arguments.xlsx')   # 调节阀
        ]

        # 按固定顺序处理设备文件
        for equip_type, filename in equipment_files:
            file_path = os.path.join(self.equipment_dir, filename)
            if os.path.exists(file_path):
                self._process_equipment_file(file_path, equip_type)

        # 处理T/E节点与N节点的连接关系
        self._process_te_n_connections()
        
    
    def _process_equipment_file(self, file_path: str, equip_type: str):
        """
        处理单个设备参数文件
        
        Args:
            file_path: 文件路径
            equip_type: 设备类型 (B, C, H, P, R)
        """
        df = pd.read_excel(file_path)
        
        # 获取设备名称列（第一列）
        device_col = df.columns[0]
        upstream_col = df.columns[1]  # 上游节点
        downstream_col = df.columns[2]  # 下游节点
        
        for _, row in df.iterrows():
            device_name = row[device_col]
            upstream_node = row[upstream_col]
            downstream_node = row[downstream_col]
            
            # 连接
            self.graph[device_name].add(upstream_node)
            self.graph[device_name].add(downstream_node)
            self.graph[upstream_node].add(device_name)
            self.graph[downstream_node].add(device_name)
    
    def _process_te_n_connections(self):
        """
        处理T/E节点与N节点的连接关系
        从connected_nodes.csv文件中读取T和E节点与N节点的连接关系
        """
        if not os.path.exists(self.connected_nodes_file):
            print(f"警告: T/E节点连接文件不存在: {self.connected_nodes_file}")
            return
        
        try:
            # 读取连接关系文件
            df = pd.read_csv(self.connected_nodes_file, header=None)
            df.columns = ['N_node', 'TE_node']
            
            for _, row in df.iterrows():
                n_node = row['N_node'].strip().strip('"')
                te_node = row['TE_node'].strip().strip('"')
                
                # 验证节点名称格式
                if (n_node.startswith('N_') and 
                    (te_node.startswith('T_') or te_node.startswith('E_'))):
                    
                    # 建立双向连接
                    self.graph[n_node].add(te_node)
                    self.graph[te_node].add(n_node)
                else:
                    print(f"警告: 跳过无效连接 {n_node} <-> {te_node}")
            
            print(f"成功处理T/E节点连接关系: {len(df)} 条记录")
            
        except Exception as e:
            print(f"处理T/E节点连接关系时出错: {e}")
            
    
    
    def find_adjacent_target_nodes(
        self,
        start_node: str,
        certain_length: int = None,
        max_depth: int = 1000,
        seed: int = 42,
        stop_nodes: Optional[Set[str]] = None,
    ) -> List[str]:
        """
        使用广度优先搜索查找邻接的所有目标节点
        使用固定随机种子保证每次运行结果一致，但不使用字典序排序以避免偏置

        Args:
            start_node: 起始节点
            certain_length: 广度优先搜索返回固定数量的节点
            max_depth: 最大搜索深度
            seed: 随机种子，用于固定邻居节点的遍历顺序
            stop_nodes: 不继续向外扩展的节点集合，搜索到时收录但不入队

        Returns:
            List[str]: 目标设备名称列表（按BFS发现顺序，距离近的在前）
        """
        if start_node not in self.graph:
            return []

        # 设置随机种子以确保确定性
        rng = random.Random(seed)

        stops = set(stop_nodes) if stop_nodes is not None else set()

        if start_node in stops:
            return []

        visited = {start_node}
        queue = deque([(start_node, 0)])  # (节点, 深度)
        result = []

        while queue:
            current_node, depth = queue.popleft()

            if current_node != start_node:
                result.append(current_node)
                if certain_length is not None and certain_length > 0 and len(result) >= certain_length:
                    break

            if depth >= max_depth:
                print(f"达到最大深度 {max_depth}，停止搜索")
                continue

            # 获取邻居节点列表并使用固定随机种子打乱顺序
            neighbors = sorted(list(self.graph[current_node]))
            rng.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    if neighbor in stops:
                        print(f"遇到停止节点 {neighbor}，停止搜索") 
                        continue
                    queue.append((neighbor, depth + 1))

        # 返回结果，保持BFS发现顺序（距离近的在前）
        return result
    
    def get_node_connections(self, node: str) -> Set[str]:
        """
        获取节点的直接连接
        
        Args:
            node: 节点名称
            
        Returns:
            直接连接的节点集合
        """
        return self.graph.get(node, set())

    def find_adjacent(self, node: str) -> List[str]:
        """
        获取指定节点的直接相邻节点（去重、已排序）。

        Args:
            node: 节点名称

        Returns:
            直接相邻节点列表（按名称排序）
        """
        return sorted(list(self.get_node_connections(node))) if node in self.graph else []

    def extract_subgraph(self, nodes: List[str]) -> 'PipelineGraph':
        """
        基于给定节点集合提取子图。

        仅保留节点集合中的节点以及它们之间的边。

        Args:
            nodes: 要保留的节点列表

        Returns:
            一个新的 PipelineGraph 实例（仅包含子图）
        """
        node_set = set(nodes)

        # 创建实例但不重新构建图
        sub = self.__class__.__new__(self.__class__)
        sub.equipment_dir = self.equipment_dir
        sub.connected_nodes_file = self.connected_nodes_file
        sub.target_equipment_types = self.target_equipment_types

        # 仅拷贝子图中的节点与边
        sub.graph = defaultdict(set)
        for n in node_set:
            if n in self.graph:
                for nb in self.graph[n]:
                    if nb in node_set:
                        sub.graph[n].add(nb)

        return sub
    
    def save_graph(self, file_path: str):
        """
        保存图网络到文件

        Args:
            file_path: 保存文件路径
        """
        graph_data = {
            'graph': dict(self.graph),  # Convert defaultdict to regular dict
            'equipment_dir': self.equipment_dir,
            'connected_nodes_file': self.connected_nodes_file,
            'target_equipment_types': self.target_equipment_types
        }
        with open(file_path, 'wb') as f:
            pickle.dump(graph_data, f)
        print(f"图网络已保存到: {file_path}")

    def save_connections_csv(self, csv_path: str = "data/save_connect_all_nodes.csv"):
        """
        保存所有图节点的连接关系到CSV文件，避免重复连接

        Args:
            csv_path: CSV文件保存路径
        """
        connections_data = []
        processed_pairs = set()

        # 按节点名称排序确保一致的处理顺序
        for node in sorted(self.graph.keys()):
            connected_nodes = self.graph[node]
            # 对连接的节点也进行排序
            for connected_node in sorted(connected_nodes):
                # 创建排序后的连接对，避免重复 (A,B) 和 (B,A)
                pair = tuple(sorted([node, connected_node]))

                if pair not in processed_pairs:
                    processed_pairs.add(pair)
                    connections_data.append({
                        'node': pair[0],
                        'connected_node': pair[1]
                    })

        # 转换为DataFrame并保存
        df_connections = pd.DataFrame(connections_data)

        # 按节点名称排序
        df_connections = df_connections.sort_values(['node', 'connected_node'])

        df_connections.to_csv(csv_path, index=False)
        print(f"图节点连接关系已保存到: {csv_path}")
        print(f"总连接数 (去重后): {len(df_connections)}")

        return csv_path
    
    @classmethod
    def load_graph(cls, file_path: str) -> 'PipelineGraph':
        """
        从文件加载图网络
        
        Args:
            file_path: 图网络文件路径
            
        Returns:
            PipelineGraph: 加载的图网络对象
        """
        with open(file_path, 'rb') as f:
            graph_data = pickle.load(f)
        
        # 创建实例但不重新构建图
        instance = cls.__new__(cls)  # Create instance without calling __init__
        instance.equipment_dir = graph_data['equipment_dir']
        instance.connected_nodes_file = graph_data.get('connected_nodes_file', 'relative_PPT/connected_nodes.csv')
        instance.target_equipment_types = graph_data['target_equipment_types']
        instance.graph = defaultdict(set, graph_data['graph'])  # Restore as defaultdict
        
        print(f"图网络已从文件加载: {file_path}")
        return instance
    

    
    
    def get_all_equipment_by_type(self, equip_type: str) -> List[str]:
        """
        获取指定类型的所有设备

        Args:
            equip_type: 设备类型 (T, E, C, R, B)

        Returns:
            List[str]: 指定类型的设备列表（已排序）
        """
        result = []
        prefix = f"{equip_type}_"
        # 确保键的处理顺序也是一致的
        for node in sorted(self.graph.keys()):
            if node.startswith(prefix):
                result.append(node)
        return result  # 已经是排序的了
    
    def print_graph_stats(self):
        """打印图网络统计信息"""
        print("=== 管网图统计信息 ===")
        print(f"总节点数: {len(self.graph)}")
        
        # 按设备类型统计
        equipment_types = ['T', 'E', 'C', 'R', 'B', 'H', 'P', 'N']
        type_counts = defaultdict(int)
        
        for node in self.graph.keys():
            # 通过节点名称前缀判断类型
            for equip_type in equipment_types:
                if node.startswith(f"{equip_type}_"):
                    type_counts[equip_type] += 1
                    break
        
        print("\n设备类型统计:")
        for equip_type in sorted(type_counts.keys()):
            count = type_counts[equip_type]
            type_name = {
                'B': '球阀', 'C': '压缩机', 'H': '管段', 
                'P': '管道', 'R': '调节阀', 'T': '气源', 'E': '分输点', 'N': '节点'
            }.get(equip_type, equip_type)
            print(f"  {equip_type}({type_name}): {count}")
        
        # 连接度统计
        degrees = [len(connections) for connections in self.graph.values()]
        if degrees:
            print(f"\n连接度统计:")
            print(f"  平均连接度: {sum(degrees) / len(degrees):.2f}")
            print(f"  最大连接度: {max(degrees)}")
            print(f"  最小连接度: {min(degrees)}")


def test_graph():
    """测试图网络构建和查找功能"""
    print("开始构建管网图...")
    
    # 构建图
    graph = PipelineGraph()
    
    # 打印统计信息
    graph.print_graph_stats()
    
    # 测试邻接节点查找
    print("\n=== 测试邻接节点查找 ===")
    
    # 测试几个节点
    test_nodes = ["N_001", "N_002", "B_001", "C_001"]
    
    for node in test_nodes:
        if node in graph.graph:
            print(f"\n节点 {node} 的邻接目标设备:")
            adjacent = graph.find_adjacent_target_nodes(node, max_depth=3)
            
            if adjacent:
                print(f"  找到 {len(adjacent)} 个邻接目标设备:")
                # 按类型分组显示
                type_groups = defaultdict(list)
                for equipment in adjacent:
                    equip_type = equipment.split('_')[0]
                    type_groups[equip_type].append(equipment)
                
                for equip_type in sorted(type_groups.keys()):
                    equipment_list = type_groups[equip_type]
                    type_name = {
                        'B': '球阀', 'C': '压缩机', 'R': '调节阀', 
                        'T': '气源', 'E': '分输点'
                    }.get(equip_type, equip_type)
                    print(f"    {equip_type}({type_name}): {equipment_list[:5]}{'...' if len(equipment_list) > 5 else ''}")
            else:
                print("  未找到邻接的目标设备")
        else:
            print(f"\n节点 {node} 不存在于图中")


if __name__ == "__main__":
    test_graph()
