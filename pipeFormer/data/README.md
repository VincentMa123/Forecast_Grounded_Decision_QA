## 数据读取顺序-大问题
_process_samples 是最后加载cache的索引顺序的，这里面似乎没有按照attention indice的顺序排序，这里一定要自己take这件事。

在 topology_attention_index.py:204 和 topology_attention_index.py:182 中，代码对列名进行了 字典序排序：

  # 第182行 - Boundary变量排序
  for col in sorted(boundary_cols):
      var_to_index[col] = current_idx

  # 第204行 - 设备变量排序  
  for col in sorted(equipment_cols):
      var_to_index[col] = current_idx

就是这段代码需要修改成按照dataset 的顺序
# 按预定义顺序添加equipment数据
equipment_order = ['B', 'C', 'H', 'N', 'P', 'R', 'T&E']
        
for equipment_type in equipment_order:
    if equipment_type in equipment_dict:
        eq_df = equipment_dict[equipment_type]
        # 验证时间长度匹配
        if len(eq_df) != len(boundary_df):
            logger.warning(f"{equipment_type} data length {len(eq_df)} != boundary length {len(boundary_df)}")
  但是在 processor.py 的 combine_all_data 函数中，添加列时使用的是 DataFrame 的原始列顺序：你需要使用map_variable_name_2_index 和 map_index_2_variable_name整理下。compute_normalization_stats这个函数也需要重构，因为这里应该使用完全排序之后的数据。

# cache里面应该保存原始数据
主要问题是normalizer可以变得非常复杂，不同的数据类型会使用不同的分布，然后如果是0,1的数据或者其他数据可能会有不同的分布拟合情况。甚至可以用别的分布去拟合。甚至这个分布还可以玩出花来。甚至讲故事都可以把这个因果性给加进去。

# 运行新的tokenizer
python data/compute_tokenizer_stats.py --data_dir ./data