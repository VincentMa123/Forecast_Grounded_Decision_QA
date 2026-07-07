"""
Build Graph for Gas Pipeline Network
构建天然气管网图的主入口文件
python build_graph.py --subgraph-center P_036 --subgraph-size 42 --subgraph-mode neighbor_count --only-subgraph
python build_graph.py --subgraph-center T_001 --subgraph-mode end_point --subgraph-end N_004 --only-subgraph
python build_graph.py --subgraph-center T_007 --prediction-mask-mode center_only --only-subgraph
"""
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
from collections import defaultdict
import importlib.util
import argparse
import pandas as pd


def _load_pipeline_graph() -> type:
    data_dir = Path(__file__).resolve().parent / "data"
    graph_path = data_dir / "graph.py"
    if not graph_path.exists():
        raise ValueError(f"缺少图定义文件: {graph_path}")
    spec = importlib.util.spec_from_file_location("pipeline_graph_module", graph_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载模块: {graph_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "PipelineGraph"):
        raise ValueError("graph.py 中缺少 PipelineGraph 类")
    return module.PipelineGraph


PipelineGraph = _load_pipeline_graph()


def _is_boundary_variable(var_name: str) -> bool:
    return ':' in var_name


def _count_graph_edges(graph: PipelineGraph) -> int:
    edges = set()
    for node, neighbors in graph.graph.items():
        for nb in neighbors:
            if node == nb:
                continue
            pair = tuple(sorted((node, nb)))
            edges.add(pair)
    return len(edges)


def _collect_node_type_counts(graph: PipelineGraph) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for node in graph.graph.keys():
        prefix = node.split('_')[0] if '_' in node else node
        counts[prefix] += 1
    return dict(sorted(counts.items()))


def _extract_equipment_name(var_name: str) -> Optional[str]:
    if ':' in var_name:
        equipment, _ = var_name.split(':', 1)
        return equipment
    parts = var_name.split('_')
    if len(parts) >= 2:
        return '_'.join(parts[:2])
    return None


def _update_graph_hyperparams(save_dir: Path, graph: PipelineGraph,
                              context: Dict[str, Any],
                              variable_names: Optional[pd.Series] = None) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    hyper_path = save_dir / "graph_hyperparameters.json"
    if hyper_path.exists():
        try:
            with open(hyper_path, "r", encoding="utf-8") as f:
                hyper = json.load(f)
        except Exception:
            hyper = {}
    else:
        hyper = {}

    graph_section = hyper.setdefault("graph", {})
    graph_section.update({
        "total_nodes": len(graph.graph),
        "total_edges": _count_graph_edges(graph),
        "node_type_counts": _collect_node_type_counts(graph),
    })
    graph_section.update(context)

    if variable_names is not None:
        variable_list = variable_names.astype(str).tolist()
        boundary_vars = [name for name in variable_list if _is_boundary_variable(name)]
        equipment_vars = [name for name in variable_list if name not in boundary_vars]
        variables_section = hyper.setdefault("variables", {})
        variables_section.update({
            "total_variables": len(variable_list),
            "boundary_variables": len(boundary_vars),
            "equipment_variables": len(equipment_vars),
        })

    hyper.setdefault("tokenizer", {})

    with open(hyper_path, "w", encoding="utf-8") as f:
        json.dump(hyper, f, ensure_ascii=False, indent=2)


def build_pipeline_graph(equipment_dir: str = "data/equipment_arguments",
                        save_path: str = None,
                        load_from_cache: bool = True,
                        save_connections_csv: bool = True) -> PipelineGraph:
    """
    构建管网图并可选择保存

    Args:
        equipment_dir: 设备参数文件目录
        save_path: 图对象保存路径（可选）
        load_from_cache: 是否尝试从缓存加载
        save_connections_csv: 是否保存连接关系CSV文件

    Returns:
        PipelineGraph: 构建完成的图对象
    """
    # 默认缓存路径
    if save_path is None:
        save_path = Path("data/static/full/pipeline_graph_cache.pkl")
    else:
        save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 尝试从缓存加载
    if load_from_cache and save_path.exists():
        try:
            print(f"正在从缓存加载图网络: {save_path}")
            graph = PipelineGraph.load_graph(str(save_path))
            graph.print_graph_stats()

            # 如果需要，保存连接关系CSV
            if save_connections_csv:
                csv_path = save_path.parent / "save_connect_all_nodes.csv"
                graph.save_connections_csv(str(csv_path))

            return graph
        except Exception as e:
            print(f"缓存加载失败: {e}")
            print("重新构建图网络...")

    print("正在构建管网图...")

    # 构建图
    graph = PipelineGraph(equipment_dir)

    # 打印统计信息
    graph.print_graph_stats()

    # 保存到缓存
    if save_path:
        graph.save_graph(str(save_path))

    # 保存连接关系CSV
    if save_connections_csv:
        csv_path = save_path.parent / "save_connect_all_nodes.csv"
        graph.save_connections_csv(str(csv_path))

    mapping_series = None
    mapping_path = save_path.parent / "index_variable_mapping.csv"
    if mapping_path.exists():
        try:
            mapping_series = pd.read_csv(mapping_path)['variable_name']
        except Exception:
            mapping_series = None

    _update_graph_hyperparams(
        save_path.parent,
        graph,
        {
            "is_subgraph": False,
            "center_node": None,
            "neighbor_count": None,
            "max_depth": None,
            "seed": None,
        },
        mapping_series,
    )

    print("管网图构建完成！")
    return graph


def _ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def build_and_save_subgraph(center_node: str,
                            neighbor_count: int = 42,
                            max_depth: int = 1000,
                            seed: int = 42,
                            static_base: str = "data/static",
                            selection_mode: str = "neighbor_count",
                            end_node: Optional[str] = None,
                            prediction_mask_mode: str = "all",
                            stop_nodes: Optional[List[str]] = None) -> str:
    """
    基于指定模式生成子图，并将相关静态文件保存到规范路径。

    保存内容（位于 data/static/{identifier}/ 下）：
    - pipeline_graph_cache.pkl（仅包含子图）
    - save_connect_all_nodes.csv（子图连接关系）
    - index_variable_mapping.csv（子图变量索引映射）
    - prediction_mask.csv（可编辑预测mask，列: variable_name,predict）

    注意：此函数仅生成静态定义文件。注意力索引（attention_indices）如需子图版本，可后续调用
    topology_attention_index 构建并保存到相同目录，数据加载时也会在缺失时回退为自注意力。

    Args:
        center_node: 子图中心节点（如 "B_001"）
        neighbor_count: 邻居数量（不含中心节点），仅在 neighbor_count 模式下生效
        max_depth: BFS最大深度
        seed: 随机种子（用于BFS邻居遍历顺序稳定）
        static_base: 静态文件基路径
        selection_mode: 子图截取模式，支持 'neighbor_count'、'end_point' 或 'stop_nodes'
        end_node: 起点-终点模式下的终点节点
        stop_nodes: stop_nodes 模式下截断搜索的节点集合
        prediction_mask_mode: 预测掩码模式，'all' 表示所有非边界变量预测，'center_only' 表示仅中心节点相关变量

    Returns:
        子图静态目录完整路径
    """
    # 1) 加载或构建完整图
    full_graph = build_pipeline_graph(load_from_cache=True, save_connections_csv=False)

    selection_mode = selection_mode or "neighbor_count"
    selection_mode = selection_mode.lower()
    prediction_mask_mode = (prediction_mask_mode or "all").lower()

    neighbors: List[str] = []
    end_node_value: Optional[str] = None
    stop_node_list: List[str] = []
    identifier: str

    if selection_mode == "neighbor_count":
        neighbors = full_graph.find_adjacent_target_nodes(
            center_node,
            certain_length=neighbor_count,
            max_depth=max_depth,
            seed=seed
        )
        if not neighbors:
            print(f"未找到邻居节点或中心节点不存在: {center_node}")
            neighbors = []
        sub_nodes = [center_node] + neighbors
        identifier = f"{center_node}_{neighbor_count}"
    elif selection_mode == "end_point":
        if not end_node:
            raise ValueError("selection_mode 为 'end_point' 时必须提供 end_node")
        neighbors = full_graph.find_adjacent_target_nodes(
            center_node,
            certain_length=None,
            max_depth=max_depth,
            seed=seed,
            stop_nodes={end_node}
        )
        if end_node != center_node and end_node not in neighbors:
            print(f"警告: 在最大深度 {max_depth} 内未搜索到终点 {end_node}，将仅包含已遍历的节点。")
        sub_nodes = [center_node] + neighbors
        identifier = f"{center_node}_{end_node}"
        end_node_value = end_node if (end_node == center_node or end_node in neighbors) else None
    elif selection_mode == "stop_nodes":
        stops = []
        if stop_nodes:
            stops = [node for node in (item.strip() for item in stop_nodes) if node]
        if not stops:
            raise ValueError("selection_mode 为 'stop_nodes' 时必须提供 stop_nodes")
        if center_node in stops:
            raise ValueError("center_node 不能包含在 stop_nodes 中")
        neighbors = full_graph.find_adjacent_target_nodes(
            center_node,
            certain_length=None,
            max_depth=max_depth,
            seed=seed,
            stop_nodes=set(stops)
        )
        sub_nodes = [center_node] + neighbors
        stop_node_list = sorted(set(stops))
        for node in stop_node_list:
            if node not in sub_nodes:
                sub_nodes.append(node)
        identifier = f"{center_node}_stops_{'_'.join(stop_node_list)}"
    else:
        raise ValueError(f"不支持的 selection_mode: {selection_mode}")

    # 3) 提取子图并保存 graph 与 connections
    sub_graph = full_graph.extract_subgraph(sub_nodes)

    base_dir = Path(static_base)
    save_dir = base_dir / identifier
    _ensure_dir(str(save_dir))

    graph_cache_path = save_dir / "pipeline_graph_cache.pkl"
    connections_csv_path = save_dir / "save_connect_all_nodes.csv"
    sub_graph.save_graph(str(graph_cache_path))
    sub_graph.save_connections_csv(str(connections_csv_path))

    # 4) 生成并保存子图变量映射（index_variable_mapping.csv）
    full_mapping_path = Path("data/index_variable_mapping.csv")
    if not full_mapping_path.exists():
        raise ValueError(f"缺少全图变量映射文件: {full_mapping_path}")
    full_mapping_df = pd.read_csv(full_mapping_path)
    if "variable_name" not in full_mapping_df.columns:
        raise ValueError(f"全图变量映射缺少 variable_name 列: {full_mapping_path}")

    sub_node_set = set(sub_nodes)
    selected_vars: List[str] = []
    for var_name in full_mapping_df["variable_name"].astype(str):
        equip_name = _extract_equipment_name(var_name)
        if equip_name and equip_name in sub_node_set:
            selected_vars.append(var_name)

    # 重新编号为 0..N-1
    mapping_rows = []
    for new_idx, var_name in enumerate(selected_vars):
        mapping_rows.append({
            'index': new_idx,
            'variable_name': var_name,
        })

    mapping_df = pd.DataFrame(mapping_rows)
    mapping_csv_path = save_dir / "index_variable_mapping.csv"
    mapping_df.to_csv(mapping_csv_path, index=False)
    print(f"子图变量映射已保存: {mapping_csv_path}，变量数: {len(mapping_rows)}")

    # 生成默认预测mask CSV，便于后续自定义预测变量
    mask_rows = []
    for row in mapping_rows:
        var_name = row['variable_name']
        if _is_boundary_variable(var_name):
            predict_flag = 0
        elif prediction_mask_mode == "center_only":
            predict_flag = int(center_node in var_name)
        elif prediction_mask_mode == "all":
            predict_flag = 1
        else:
            raise ValueError(f"不支持的 prediction_mask_mode: {prediction_mask_mode}")
        mask_rows.append({
            "variable_name": var_name,
            "predict": int(predict_flag)
        })
    mask_df = pd.DataFrame(mask_rows)
    mask_csv_path = save_dir / "prediction_mask.csv"
    mask_df.to_csv(mask_csv_path, index=False)
    print(f"子图预测掩码已保存: {mask_csv_path}，可手动编辑predict列以定制预测变量")

    graph_context = {
        "is_subgraph": True,
        "center_node": center_node,
        "neighbor_count": neighbor_count if selection_mode == "neighbor_count" else None,
        "actual_neighbors": max(0, len(sub_nodes) - 1),
        "max_depth": max_depth,
        "seed": seed,
        "selection_mode": selection_mode,
        "end_node": end_node_value,
        "path_length": len(sub_nodes) - 1 if selection_mode == "end_point" else None,
        "stop_nodes": stop_node_list if stop_node_list else None,
        "prediction_mask_mode": prediction_mask_mode,
    }
    _update_graph_hyperparams(
        save_dir,
        sub_graph,
        graph_context,
        mapping_df['variable_name'],
    )

    return str(save_dir)


def find_equipment_neighbors(graph: PipelineGraph, equipment_id: str, certain_length:int = None, max_depth: int = 3):
    """
    查找指定设备的邻接目标设备
    
    Args:
        graph: 管网图对象
        equipment_id: 设备ID
        max_depth: 最大搜索深度
    """
    print(f"\n=== 查找设备 {equipment_id} 的邻接目标设备 ===")

    adjacent = graph.find_adjacent_target_nodes(equipment_id, certain_length, max_depth)

    if adjacent:
        print(f"找到 {len(adjacent)} 个邻接目标设备:")
        
        # 按类型分组
        from collections import defaultdict
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
            
            print(f"  {equip_type}({type_name}): {len(equipment_list)}个")
            # 显示前10个
            display_list = equipment_list[:10]
            print(f"    {display_list}")
            if len(equipment_list) > 10:
                print(f"    ... 还有{len(equipment_list) - 10}个")
    else:
        print("未找到邻接的目标设备")


def demo_graph_usage():
    """演示图网络的使用"""
    # 构建图
    graph = build_pipeline_graph()
    
    # 演示查找邻接设备
    test_equipments = ["R_008","C_005", "E_103","N_302","C_018", "E_073","B_001"]
    
    for equipment_id in test_equipments:
        if equipment_id in graph.graph:
            find_equipment_neighbors(graph, equipment_id, certain_length=15,max_depth=20)
        else:
            print(f"\n设备 {equipment_id} 不存在于图中")
    
    # 显示各类型设备数量
    print("\n=== 各类型设备统计 ===")
    target_types = ['T', 'E', 'C', 'R', 'B']
    for equip_type in target_types:
        equipment_list = graph.get_all_equipment_by_type(equip_type)
        type_name = {
            'B': '球阀', 'C': '压缩机', 'R': '调节阀', 
            'T': '气源', 'E': '分输点'
        }.get(equip_type, equip_type)
        print(f"{equip_type}({type_name}): {len(equipment_list)}个")


def main_for_attention_index(
    static_dir: Optional[str] = None,
    force_rebuild: bool = False,
    max_neighbors_variable: Optional[int] = None,
):
    """用于构建注意力索引的主函数，方便debug"""
    from data.topology_attention_index import build_and_save_attention_indices

    data_dir = "data"
    print("正在构建拓扑注意力索引...")

    indices_path = build_and_save_attention_indices(
        data_dir,
        static_dir=static_dir,
        force_rebuild=force_rebuild,
        max_neighbors_variable=max_neighbors_variable,
    )
    print(f"注意力索引已保存至: {indices_path}")


def main():
    """主函数：支持构建完整图/子图与注意力索引调试。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--subgraph-center', type=str, default=None, help='子图中心节点名，如 B_001')
    parser.add_argument('--subgraph-size', type=int, default=42, help='子图邻居数量（不含中心），仅在 neighbor_count 模式下使用')
    parser.add_argument('--subgraph-mode', type=str, default='neighbor_count', choices=['neighbor_count', 'end_point', 'stop_nodes'], help='子图截取模式：neighbor_count、end_point 或 stop_nodes')
    parser.add_argument('--subgraph-end', type=str, default=None, help='end_point 模式下的终点节点')
    parser.add_argument('--subgraph-stop', action='append', default=None, help='stop_nodes 模式下的断点节点，可重复使用或以逗号分隔')
    parser.add_argument('--max-depth', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prediction-mask-mode', type=str, default='all', choices=['all', 'center_only'], help='预测掩码模式：all 为所有非边界变量，center_only 为仅中心节点变量')
    parser.add_argument('--only-subgraph', action='store_true', help='仅生成子图静态文件')
    parser.add_argument('--build-attn', action='store_true', help='构建注意力索引（默认构建到 data/）')
    parser.add_argument('--static-dir', type=str, default=None, help='静态文件目录（默认 data/static/full）')
    parser.add_argument('--force-attn-rebuild', action='store_true', help='强制重建注意力索引')
    parser.add_argument(
        '--attention-neighbors',
        type=int,
        default=None,
        help='拓扑注意力索引中每个变量的最大邻居数（包括自身），用于消融实验，例如 16/32/64/128；不指定则使用默认配置。',
    )
    args = parser.parse_args()

    if args.subgraph_center:
        stop_nodes = None
        if args.subgraph_stop:
            stop_nodes = []
            for entry in args.subgraph_stop:
                if entry is None:
                    continue
                parts = entry.split(',')
                for item in parts:
                    node = item.strip()
                    if node:
                        stop_nodes.append(node)

        out_dir = build_and_save_subgraph(
            center_node=args.subgraph_center,
            neighbor_count=args.subgraph_size,
            max_depth=args.max_depth,
            seed=args.seed,
            selection_mode=args.subgraph_mode,
            end_node=args.subgraph_end,
            prediction_mask_mode=args.prediction_mask_mode,
            stop_nodes=stop_nodes,
        )
        print(f"子图静态文件已生成: {out_dir}")
        if args.only_subgraph:
            return

    if args.build_attn:
        main_for_attention_index(
            static_dir=args.static_dir,
            force_rebuild=args.force_attn_rebuild,
            max_neighbors_variable=args.attention_neighbors,
        )
        return

    # 默认：构建完整图，并保存连接CSV
    graph_cache_path = args.static_dir if args.static_dir else "data/static/full/pipeline_graph_cache.pkl"
    graph = build_pipeline_graph(
        equipment_dir="data/equipment_arguments",
        save_path=graph_cache_path,
        load_from_cache=True,
        save_connections_csv=True
    )
    print("图网络构建完成，连接关系CSV已保存")


if __name__ == "__main__":
    main()
