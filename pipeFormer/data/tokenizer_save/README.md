Tokenizer 参数说明
====================

本目录存放 `DataTokenizer` 的实现、配置与统计产物。核心可调参数如下，运行 `data/compute_tokenizer_stats.py` 时可通过命令行覆写，也会写入 `graph_hyperparameters.json` 以便后续复现。

- `constant_freq_threshold` (`core.py`, 默认 `0.02`)
  - 判定非全常量变量时，单个取值若在该变量中出现频率 ≥ 阈值，就会被视作“常量”分量并分配独立 token。调小会保留更多高频离散值，调大则让更多值落入连续区间。
- `constant_variable_threshold` (默认 `0.9`)
  - 如果一个变量中最常见取值的覆盖率 ≥ 阈值，则整个变量被当作常量变量，仅产生一个常量 token；剩余极少数样本会被忽略。
- `quantile_step` (默认 `0.02`)
  - 连续区间切分的最小覆盖比例，相当于量化步长。较小的值会生成更多、更窄的区间 token；较大的值则使区间更粗、token 更少。
- `quantile_method` (默认 `linear`)
  - 计算分位点时传递给 `numpy.quantile` 的 `method` 参数，用于控制插值策略。
- `range_gap_epsilon` (默认 `1e-6`)
  - 为了避免相邻区间重叠/统计残缺，在写入区间上界时会减去 `epsilon`，使得上界略小于下一区间的下界。
- `round_gap` (默认 `0.01`)
  - 离散化前的取值舍入粒度；同时用于常量 token 的范围宽度（`± round_gap / 2`）。

其他文件：

- `token_stats.csv`：由 `compute_tokenizer_stats.py` 生成的离散化描述。
- `data_tokenizer_config.json`：最近一次拟合保存的配置快照，包含上述参数的取值。

调整参数后请重新运行 `data/compute_tokenizer_stats.py`，否则缓存的统计与模型行为会不一致。
