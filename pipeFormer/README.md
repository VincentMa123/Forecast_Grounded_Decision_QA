# PipeFormer: A Physics-Informed Tokenized Sparse Transformer for Natural Gas Pipeline Networks

[中文说明](README_zh.md)

PipeFormer is a paper-oriented open-source release for transient forecasting on large natural gas pipeline networks. The package focuses on the core decoder model, topology preprocessing, tokenization pipeline, training scaffold, and paper figures so that readers can connect the published method to runnable code more easily.

The original paper studies real-time short-horizon prediction for an industrial pipeline network digital twin. In that setting, the model uses the previous 5 minutes of system states to predict the next 5 minutes with a 1-minute sampling interval. The full state covers pressure, flow, temperature, and control-related variables on a network with more than 1,000 physical nodes and 6,712 modeled variables.

## Publication

- Journal page: https://www.sciencedirect.com/science/article/pii/S2667143326000417
- DOI: [`10.1016/j.jpse.2026.100472`](https://doi.org/10.1016/j.jpse.2026.100472)

## Problem Setting In The Paper

The paper reformulates natural-gas transient simulation as a multivariate forecasting problem on a sparse graph. Each time step represents one minute. A historical window of length `L = 5` is mapped to a prediction horizon of `H = 5`. The target state includes pressure, flow, and temperature variables, while auxiliary inputs include device states and operator-set boundary conditions.

This setup is difficult for three reasons:

- Real dispatching requires latency far below full numerical simulation.
- Operational data are strongly non-Gaussian, long-tailed, and often multi-modal because compressors, valves, and station set points change discretely.
- The prediction should remain compatible with fluid-dynamics laws rather than behaving like a generic black-box forecast.

## Method Walkthrough

PipeFormer combines four ideas from the paper:

- Topology graph construction: the pipeline system is treated as a graph whose vertices are physical components such as nodes, pipe segments, compressors, valves, and users.
- Topology-aware sparse attention: each variable attends mainly to physically connected neighbors found through a prioritized local search. In the paper, the standard operating point is `K = 32`, which reduces attention complexity from `O(N^2)` to `O(NK)`.
- Discrete tokenization: continuous sensor values are mapped to discrete symbols so the model predicts categories instead of directly regressing every scalar. The paper describes a 4096-level token granularity for continuous variables, while intrinsically discrete variables such as valve state can stay binary.
- Physics-driven objective: the paper augments data loss with residuals from the continuity and momentum equations so predicted pressure and flow trajectories remain closer to physically valid evolution.

This release primarily exposes the tokenized sparse decoder and preprocessing pipeline. The paper's full physics-informed formulation is documented here for interpretability and reproducibility context, while the released training entrypoint is centered on the decoder-oriented code in this package.

## Main Results Reported In The Paper

- Overall MAPE: `27.1%` on the industrial test set.
- Overall MAE: `57,711`.
- Best baseline in the paper: TLPN with `35.6%` overall MAPE, meaning PipeFormer reduces relative error by about 24% against that reference.
- Neighbor-count ablation: `K = 32` gives a strong balance between accuracy and efficiency, with `14.8 ms` single-step inference latency in the reported setup.
- Tokenization ablation: overall MAPE drops from `56.0` to `32.4` after tokenization, then to `27.1` after adding the paper's physics-informed objective.

One practical detail from the paper is that MAPE and MAE can disagree on some scenarios because variable magnitudes differ dramatically across the network. Large flow or inventory variables can dominate MAE, while many small pressure variables can dominate MAPE.

## Repository Guide

- `build_graph.py`: builds topology artifacts, subgraph assets, and editable prediction masks from static network resources.
- `build_cache.py`: prepares cached training and validation sequences aligned with the selected static graph.
- `data/topology_attention_index.py`: computes the neighbor indices used by sparse attention.
- `data/tokenizer_save/` and `data/compute_tokenizer_stats.py`: build and persist the discrete token vocabulary and token metadata.
- `models/decoder/model.py`: the main decoder-only implementation.
- `models/decoder/attention.py` and `models/decoder/masks.py`: sparse attention execution and mask construction.
- `training/trainer.py`: HuggingFace-style training loop integration for the released models.

## Visual Guide

### 1. Overall architecture and sparse attention

![PipeFormer architecture overview and topology-aware sparse attention.](readme_assets/attention_structure.png)

Read this figure from left to right. The left panel is the full industrial pipeline network, the middle panel is a local subgraph used to define the receptive field, and the right panel shows how historical embeddings are assembled for prediction. The key point is that the target variable does not attend to every component in the network; it only attends to itself, same-device variables, and physically nearby components.

### 2. Why tokenization is necessary

![Real operational variables show non-Gaussian and multi-modal distributions.](readme_assets/fig_distribution.png)

This figure explains why the paper does not treat the task as ordinary Gaussian regression. Randomly sampled variables show clear long tails, skewness, and multi-modality, and even the training and validation splits do not resemble simple normal distributions. The tokenization step is designed to make learning more stable under exactly these industrial data characteristics.

### 3. How the training target is defined in the paper

![Training combines data error with physics-driven residual constraints.](readme_assets/model_loss.png)

The paper's learning objective has two parts. One part fits observations, and the other part penalizes violations of continuity and momentum conservation through a differentiable physics module. This is the conceptual bridge between sequence modeling and gas-flow dynamics.

### 4. Attention-map interpretability

![Attention-map interpretation for a representative pipe-flow variable.](readme_assets/fig_attention_map.png)

This figure is the paper's interpretability check. For the output flow of pipe segment `P_021_q_out`, the strongest attention weight falls on the corresponding inflow `P_021_q_in`, and another strong weight falls on the upstream valve `B_015`. The result shows that the sparse attention mechanism is not just saving compute; it is also learning physically meaningful upstream dependencies.

### 5. Why `K = 32` is the default paper setting

![Ablation results used in the paper to justify design choices.](readme_assets/result_ablation.png)

The x-axis is the topological neighbor count `K`. Larger neighborhoods improve accuracy at first, but memory usage and latency also rise. The paper selects `K = 32` because it captures enough spatial context to cut the error sharply while avoiding the heavy cost of much larger neighborhoods.

### 6. Representative transient prediction behavior

![Representative transient prediction example from the paper.](readme_assets/fig_time_series.png)

This plot shows how PipeFormer tracks sharp transient changes in pressure and flow. The main visual takeaway is not only that the method is accurate on average, but that it reacts faster than baseline models when the operating regime changes suddenly, which is exactly the behavior needed in dispatching scenarios.

## Quick Start

```bash
python build_graph.py --build-attn --static-dir data/static/<your_static_dir>
python build_cache.py --data-dir data --static-dir data/static/<your_static_dir> --skip-tokens --force
python data/compute_tokenizer_stats.py --data_dir data --static-dir data/static/<your_static_dir> --cache_dir data/static/<your_static_dir>/cache --force
python data/compute_normalization_stats.py --static_dir data/static/<your_static_dir> --method standard
python train.py --config configs/quick_test_decoder.json
```

These commands assume that you have already prepared compatible sequence files and static topology artifacts. The quick-test configuration is useful for validating the pipeline wiring before reproducing larger experiments.

## Recommended Reading Order

- Start with `models/decoder/model.py`, `models/decoder/attention.py`, and `models/decoder/masks.py` to understand how sparse topology-constrained attention is executed.
- Then read `data/topology_attention_index.py`, `build_graph.py`, and `build_cache.py` to see how the graph neighborhood and sample cache are built.
- Next inspect `data/tokenizer_save/` and `data/compute_tokenizer_stats.py` to connect the paper's tokenization idea to the release implementation.
- Finish with `training/trainer.py` and `train.py` to understand how the released package trains and evaluates the decoder stack.
