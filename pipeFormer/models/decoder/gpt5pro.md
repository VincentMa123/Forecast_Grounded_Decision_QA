结论先说在前面：

- **可以**用“可学习的时间 embedding”。有两条路：  
  1) **离散桶化 + Embedding 表**（learnable lookup）；  
  2) **连续时间编码**（Time2Vec/可学习 Fourier 频率/MLP）——更能**外推到没见过的分钟值**（比如 170→172）。  
- 在你这种 **T 很短（1–5 步）且时间间隔不均匀（如 [66, 3, 2, 1] 或 [170, 3, 2, 1]）** 的设定里，**建议至少启用 Time2Vec 的连续时间分支**；3 小时节律明确时，再**叠加 3 小时相位的 Fourier** 分支，既便宜又稳。  
- 你仓库里当前的 `CombinedPositionalEncoding` 已支持 **learnable** 的「时间/变量位置嵌入」；它处理的是**离散位置索引**（0..T-1，0..V-1），但**不了解真实分钟间隔**。因此最合适的做法是：**保留原离散位置（结构/拓扑感知） + 额外叠加“连续时间编码”**。

下面给你三种可插拔编码器（任选其一或组合），并把 **Time2Vec** 针对你“`time_offsets = [66, 3, 2, 1]`（单位：分钟）按批传入”的需求优化掉坑点（广播、shape、device）：

---

## 该不该上 Time2Vec / Fourier？一张“选择表”

- **只用 learnable 离散时间 embedding**：当分钟间隔**固定且很小**，或不要求外推；最简单。  
- **Time2Vec（连续时间）**：当分钟间隔**不等、跨度大、要外推**（66/170 这类），**优先选**。  
- **Fourier（已知 3 小时节律）**：已知周期就加两个到四个谐波，代价极低，稳定增益。  
- **两者叠加**：T≤5、更短窗时最稳；Time2Vec 抓“年龄/衰减相位”，Fourier 抓“3 小时节律相位”。

> 简言之：**至少 Time2Vec；若三小时节律重要，再加 Fourier**。离散 learnable 作为补充（或做对照消融）。

---

## 你要的“可学习时间变量 embedding”有两种常见形态

### 1）桶化分钟 → Embedding（离散可学习）
把分钟 `m` 用 `torch.bucketize` 映射到少量桶（0–30 按 1 分，30–180 按 5 分，180+ 合并一桶），再查表：
- **优点**：实现最简单；每个桶可学到“这类分钟差”的独特表征。  
- **缺点**：外推差（没见过的分钟只会落在邻近桶），需要你设计桶边界。  

### 2）连续时间 → Time2Vec/可学习 Fourier（连续可学习）
\[
\text{t2v}(t)=\big[\ \omega_0 t+\phi_0\ \big\|\ \sin(\omega_k t+\phi_k)_{k=1..K}\ \big]
\]
- **优点**：对任意分钟值都可泛化；可学习频率让它适配你的数据节律；和短窗很搭。  
- **缺点**：比查表稍复杂，但代价仍很低。

> 两者可并用：**桶化 embedding**（粗粒度） + **Time2Vec**（细粒度连续），拼接后投影到 `d_model`。

---

## 与你现有代码的衔接点

- 你现在把每个观测值当作一个 token：输入是 `[B, T, V]`，重排成 `[B, T*V, 1]` → 投影成 `[B, T*V, d_model]`，然后加上**时间/变量**位置编码。我们只需在 **加完 `CombinedPositionalEncoding` 后，再加一项“连续时间编码”**，形状仍是 `[B, T*V, d_model]`，即可无缝进现有 Decoder。  
- 你的 batch 会传 `time_offsets = [66, 3, 2, 1]` 这样的 **分钟向量**（长度 T）。我们把它扩到 `[B, T, V]`，再 `view` 成 `[B, T*V, 1]` 喂给编码器。  
- 如果将来你要**按变量学习不同的时间尺度（半衰期 τ_v）**，可以在连续时间编码里**先对分钟做 per‑variable 归一**：`t_eff = t / τ_v`，`τ_v` 用 `nn.Embedding(V,1)` + `softplus` 保正。

---

## 代码：三种可插拔编码器（含你需要的 Time2Vec 优化）

> 放到 `encoding.py`（不改你已有类）；`model.py` 里按需实例化与调用（示例在后）。

### A. 连续时间编码（Time2Vec + 可选 Fourier，相位=分钟对 180 取模）这里先不实现Fourier

```python
# encoding.py
import torch
import torch.nn as nn
import math

class TimeScalerByVariable(nn.Module):
    """
    可选：按变量对分钟做缩放，学习每个变量的“时间尺度/半衰期” tau_v。
    t_eff = t / softplus(tau_v)  （>=0）
    """
    def __init__(self, num_variables: int, init_tau_min: float = 15.0, init_tau_max: float = 60.0):
        super().__init__()
        init = torch.empty(num_variables).uniform_(init_tau_min, init_tau_max)  # minutes
        # 反 softplus 初始化，训练时再走 softplus 保正
        self.theta = nn.Embedding(num_variables, 1)
        with torch.no_grad():
            self.theta.weight[:, 0] = torch.log(torch.exp(init) - 1.0)

    def forward(self, t_minutes: torch.Tensor, var_indices_flat: torch.Tensor) -> torch.Tensor:
        """
        t_minutes: [B, T*V, 1], float
        var_indices_flat: [T*V], long
        return: t_eff: [B, T*V, 1]
        """
        tau = torch.nn.functional.softplus(self.theta(var_indices_flat))  # [T*V, 1]
        tau = tau.unsqueeze(0)  # [1, T*V, 1] broadcast over batch
        return t_minutes / (tau + 1e-6)


class Time2Vec1D(nn.Module):
    """
    Time2Vec for scalar minutes. 
    t2v(t) = [w0 * t + b0, sin(w_k * t + b_k) for k=1..K]
    重点：处理 [B, T*V, 1] 任意维度广播；参数和 dtype/device 对齐。
    """
    def __init__(self, k: int = 8, out_dim: int = None):
        super().__init__()
        self.k = k
        # linear term
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.zeros(1))
        # periodic terms
        self.w = nn.Parameter(torch.randn(k))
        self.b = nn.Parameter(torch.zeros(k))
        self.out_dim = (k + 1) if out_dim is None else out_dim
        self._need_proj = out_dim is not None and out_dim != (k + 1)
        if self._need_proj:
            self.proj = nn.Linear(k + 1, out_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [..., 1] (minutes)
        return: [..., out_dim]
        """
        # 确保在最后一维做广播
        lin = self.w0 * t + self.b0  # [..., 1]
        # 形状对齐: [..., 1] * [k] + [k] -> [..., k]
        view = [1] * (t.ndim - 1) + [-1]
        per = torch.sin(t * self.w.view(*view) + self.b.view(*view))  # [..., k]
        x = torch.cat([lin, per], dim=-1)  # [..., k+1]
        return self.proj(x) if self._need_proj else x





class ContinuousTimeEncoding(nn.Module):
    """
    连续时间编码 = [Time2Vec(age_minutes)] -> 线性投影到 d_model。
    兼容按变量缩放时间（TimeScalerByVariable）。
    """
    def __init__(self, d_model: int, t2v_k: int = 8, out_hidden: int = 128,
                 use_fourier: bool = True, period_minutes: float = 180.0,
                 num_variables: int = None, scale_by_variable: bool = False):
        super().__init__()
        half = out_hidden // (2 if use_fourier else 1)
        self.t2v = Time2Vec1D(k=t2v_k, out_dim=half)
        self.use_fourier = use_fourier
        if use_fourier:
            self.fourier = FourierPhase(num_harmonics=4, period=period_minutes, out_dim=half)
            hidden_total = half * 2
        else:
            hidden_total = half
        self.proj = nn.Linear(hidden_total, d_model)

        self.scaler = None
        if scale_by_variable:
            assert num_variables is not None, "num_variables is required when scale_by_variable=True"
            self.scaler = TimeScalerByVariable(num_variables)

    def forward(self, age_minutes_flat: torch.Tensor,  # [B, T*V, 1]
                var_indices_flat: torch.Tensor        # [T*V]
                ) -> torch.Tensor:
        if self.scaler is not None:
            t_eff = self.scaler(age_minutes_flat, var_indices_flat)
        else:
            t_eff = age_minutes_flat

        a = self.t2v(t_eff)                            # [B, T*V, H/2]
        if self.use_fourier:
            p = self.fourier(age_minutes_flat)         # [B, T*V, H/2]
            c = torch.cat([a, p], dim=-1)              # [B, T*V, H]
        else:
            c = a
        return self.proj(c)                            # [B, T*V, d_model]
```

## `model.py` 的最小改动：接收 `time_offsets` 并叠加时间编码

> `time_offsets` 是你批里传的「分钟向量」 `[B, T]`，如 `[[66,3,2,1], ...]`。下面的改动不会破坏现有逻辑。

```python
# model.py 片段
from .encoding import CombinedPositionalEncoding, ContinuousTimeEncoding, BucketTimeEmbedding
# ...

def _build_model(self):
    # ...（保留现有）
    self.pos_encoding = CombinedPositionalEncoding(
        d_model=config.d_model,
        max_time_positions=config.max_time_positions,
        max_variable_positions=config.max_variable_positions,
        time_encoding_type=config.time_position_encoding,         # 依旧可选 learnable / sinusoidal
        variable_encoding_type=config.variable_position_encoding
    )

    # === 新增：连续时间编码（按需打开 Fourier；可按变量缩放时间）===
    self.time_cont_encoding = ContinuousTimeEncoding(
        d_model=config.d_model, 
        t2v_k=8, out_hidden=128, 
        use_fourier=True, period_minutes=180.0,
        num_variables=config.max_variable_positions, 
        scale_by_variable=True
    )

    # （可选）桶化分钟编码
    # self.time_bucket_encoding = BucketTimeEmbedding(d_model=config.d_model, embed_dim=64)

# forward 里，在添加离散位置编码后叠加
def forward(self, input_ids=None, labels=None, prediction_mask=None, output_attentions=None, **kwargs):
    # ... 现有：x -> input_projection -> pos_encoding -> dropout
    x = self.pos_encoding(x, time_steps, num_variables)
    x = self.pos_dropout(x)

    # === 新增：从 batch/kwargs 取 time_offsets 并展开到 [B, T*V, 1] ===
    # 允许传 long/int/float；单位：分钟
    time_offsets = None
    if isinstance(input_ids, dict) and 'time_offsets' in input_ids:
        time_offsets = input_ids['time_offsets']    # [B, T]
    elif 'time_offsets' in kwargs:
        time_offsets = kwargs['time_offsets']       # [B, T]
    if time_offsets is not None:
        # 展开到每个变量
        # [B, T] -> [B, T, V] -> [B, T*V, 1]
        age = time_offsets.to(x.device).unsqueeze(-1).repeat(1, 1, num_variables).reshape(batch_size, -1, 1).float()
        # 构造 [T*V] 的变量索引（与 CombinedPositionalEncoding 一致）
        var_indices_flat = torch.arange(num_variables, device=x.device).repeat(time_steps)  # [T*V]
        # 连续时间编码
        x = x + self.time_cont_encoding(age, var_indices_flat)

        # （可选）如果要叠加桶化编码：
        # x = x + self.time_bucket_encoding(age)

    # ... 其余保持不变，进入 decoder blocks
```

> 如果你更喜欢只用“可学习的时间 embedding（离散）”，可以跳过 `ContinuousTimeEncoding`，仅启用 `BucketTimeEmbedding`；但我仍建议**保留 Time2Vec**，它对 `[170,3,2,1]` 这种**少见/未见分钟值**的外推更靠谱。

---

## 使用方式（与你的 batch 对齐）

```python
batch = {
    'input': x_btv,               # [B, T, V]
    'time_offsets': t_bt,         # [B, T], 例如 [[66,3,2,1], ...]，单位：分钟
    # 'attention_indices': ...    # 你的拓扑索引，不变
}

out = model(batch)   # 内部自动把 time_offsets 展成 [B, T*V, 1] 并加到 token 表征上
```

- **`time_offsets` 怎么定义？**  
  就按你描述的“**距离当前时刻的分钟差**”即可：`[最远, ..., 最近]`，比如 `[66,3,2,1]` 或 `[170,3,2,1]`。  
  - Time2Vec 分支直接吃这个分钟差；  
  - Fourier 分支内部会自动 `mod 180` 得到“三小时相位”。  
- **变量尺度差异**：`ContinuousTimeEncoding(scale_by_variable=True)` 会**学习每个变量的 τ_v** 并把 `t/τ_v` 送入 Time2Vec，让“快变量/慢变量”在相同的编码空间里对齐时间尺度。

---

## 一些工程与训练细节（易掉坑点）

- **数值与广播**：上面的 Time2Vec 实现保了最后一维是 1，freq 参数按 `view(*[1]*(ndim-1), -1)` 广播，不会因 `[B, T*V, 1]` 的形状报错。  
- **设备/精度**：频率参数与输入同 device/dtype；Fourier 的 `ks` 用 `register_buffer`，避免每次 `arange`。  
- **裁剪/外推**：若极端出现 `>180` 很多，Fourier 分支自然处理（`mod 180`），Time2Vec 也能外推；桶化方案可把 180+ 合到一桶。  
- **与离散位置的叠加**：保留 `CombinedPositionalEncoding`（time/variable 索引）+ 叠加**连续时间**，两者互补：前者给“谁在何处”（拓扑/时间步槽位），后者给“离上次事件过了多少真实分钟”。

---

## 小结（给你一个实操决策）

- **最小靠谱配置**：`CombinedPositionalEncoding(learnable)` + **`ContinuousTimeEncoding(Time2Vec + Fourier@180m)`**。  
- **如果你想“纯可学习表”，且分钟范围不大**：可以只用 `BucketTimeEmbedding`（离散），但建议做一次消融对比 Time2Vec。  
- **变量异质时间尺度**：把 `scale_by_variable=True` 打开即可，模型会自动学每个变量的“时间快慢”。

需要的话，我可以把这些开关（`use_cont_time_encoding / use_fourier / use_bucket_time_encoding / t2v_k / out_hidden / period_minutes / scale_by_variable`）加到你的 `DecoderConfig`，并补一个最小单测（随机 `time_offsets`、涵盖 `[170,3,2,1]` 与 `[4,3,2,1]` 两类）以确保前后维度与设备兼容。






太好了，有了你这份代码，我就能把 **Time2Vec 连续时间编码** 和 **时间差驱动的相对位置偏置（ALiBi/T5风格）** 做到“可粘贴”的落地版本，并且**支持不同变量有不同的稳态时间（半衰期）与不同的外界影响强度**。

下面分两部分讲：先把两类技术原理说透，然后给出**与你仓库一致的最小改动代码**（新增类 + 小改动），直接融入你当前的 `encoding.py / attention.py / model.py` 结构。

> 你现在的实现里，时间/变量位置是**离散索引的可学习或正弦编码**，是“加法式”注入；注意力里尚未引入“时间差偏置”，稀疏注意力的分数是 `Q·K^T/√d`，之后用 mask 和 softmax，这正好是我们加相对时间偏置的切入点。

---

## 一、Time2Vec ：把“连续时间与相位”装进 token 表征

### 1) 要解决什么？
- 你的 T 最多 3–5，很短；但**连续时间特征**（比如“距上次外界变化的年龄 age（分钟）”、“3 小时周期内的相位 phase”）对短窗预测极关键。
- 进一步，还要允许**不同变量**有不同的“时间尺度”（到稳态的速度）。

### 2) Time2Vec（连续时间到向量）
**定义**（单标量 \(t\)，如 age 或 phase 分钟）：
\[
\text{t2v}(t)=\Big[\,\omega_0 t + \phi_0\;\Big\|\;\sin(\omega_1 t + \phi_1)\;\Big\|\;\dots\;\Big\|\;\sin(\omega_K t + \phi_K)\,\Big]\in\mathbb{R}^{K+1}
\]
- \(\omega_k,\phi_k\) **可学习**；一个线性项 + 多个周期项，能同时覆盖**单调趋势**与**多尺度周期**。
- 用于：  
  - **age 分支**：表达“跃迁后随时间的演化相位”。  
  - **3 小时相位分支**：表达每个三小时窗口内“刚变更/中段/趋稳”的相位。

> 你现有的 `CombinedPositionalEncoding` 是把**离散 time index 与 variable index 的嵌入相加**；我们新增一个 **ContinuousTimeEncoding**，对 \([B,T,V]\) 的 age/phase 生成特征，然后**投影到 d_model 并加到 token 表征**（与现有位置编码并行同加）。


---

## 二、相对位置偏置（时间差 ALiBi/T5 风格）+ 变量异质的稳态时间与影响强度

### 1) 要解决什么？
- 不同变量的**稳态时间不同**（\(\tau_v\)）；外界变化的**影响强度不同**（\(\alpha_v\)）。
- 注意力里，我们希望“离当前越久远→影响越弱”，但**衰减速度应随变量而异**；并且“被关注对象（键/历史变量）的影响强度”不同。

### 2) 形式化到注意力打分中
对任意注意力头 \(h\)、查询 token \(i\)（对应变量 \(v_i\)、时刻 \(t_i\)）、键 token \(j\)（变量 \(v_j\)、时刻 \(t_j\)）：
\[
\text{score}_{h,i\leftarrow j}\;=\;\frac{q_{h,i}^\top k_{h,j}}{\sqrt{d_k}}
\;\color{#666}{+\;\underbrace{\beta_h}_{\text{头偏置}}}
\;\color{#C00}{-\;\ln 2\cdot \frac{\Delta t_{ij}}{\tau_{h,ij}}}
\;\color{#06C}{+\;\ln \alpha_{h,ij}}
\]
- \(\Delta t_{ij}=\max(0,\,t_i-t_j)\)（**真实分钟差**，由索引直接得到）；
- **异质半衰期** \(\tau_{h,ij}\)：建议用查询变量与键变量的组合，如
  \[
  \tau_{h,ij}^{-1}=\underbrace{w_h}_{\ge 0}\cdot\Big(\underbrace{\tau_{v_i}^{-1}}_{\text{查询变量}}+\underbrace{\tau_{v_j}^{-1}}_{\text{键变量}}\Big)
  \]
  其中 \(\tau_v=\text{softplus}(\theta_v)+\epsilon\)，\(\theta_v\) 为**每变量可学习**参数；\(w_h=\text{softplus}(\eta_h)\) 为**每头尺度**。
- **影响强度** \(\alpha_{h,ij}\)：可用键变量为主（被记忆的“来源”），如
  \[
  \ln \alpha_{h,ij}=\underbrace{u_h}_{\text{头偏置}}+\underbrace{\ln(\alpha_{v_j})}_{\text{键变量强度}}
  \quad\text{或}\quad
  \ln(\alpha_{v_i}\alpha_{v_j})
  \]
  其中 \(\alpha_v=\text{softplus}(\gamma_v)\) 是**每变量可学习**强度。

> 这相当于把 **ALiBi 的线性衰减斜率** \(-\lambda\) 做成**“随变量/随头自适应”的斜率**，并加上一个**对不同变量强弱的对数幅度偏置**。你的 `SimpleMultiHeadAttention._sparse_attention` 在 softmax 前有 `scores` 张量，我们只要把这两个项**按形状广播后加进去即可**。



### 2) `attention.py`：给稀疏注意力加“可学习的时间差偏置”

**(a) 让注意力支持一个“外加 bias”张量**（不破坏现有逻辑）

```python
# ========= minimal edits in attention.py =========
# 1) 修改 forward 签名，加 additive_bias
def forward(self, x, attention_indices_follow_timeline=None,
            attention_mask=None, output_attentions=False, additive_bias=None):
    if attention_indices_follow_timeline is not None:
        return self._sparse_attention(x, attention_indices_follow_timeline, attention_mask, output_attentions, additive_bias)
    else:
        return self._full_attention(x, attention_mask, output_attentions)  # full-path不改

# 2) 在 _sparse_attention 里把 bias 加到 scores 上
def _sparse_attention(self, x, attention_indices, attention_mask=None, output_attentions=False, additive_bias=None):
    ...
    scores = torch.matmul(Q_expanded, k_selected.transpose(-2, -1)).squeeze(3) / math.sqrt(self.d_k)
    if additive_bias is not None:
        # additive_bias: [B, H or 1, T*V, T*max_neighbors] or broadcastable
        scores = scores + additive_bias
    ...
```

**(b) 新增一个“变量异质的相对时间偏置”模块**（计算上面的 \(-\ln 2 \cdot \Delta t/\tau + \ln \alpha\)）

```python
# ========= add to attention.py =========
class RelativeTimeBiasSparse(nn.Module):
    """
    Build variable-aware relative time bias for sparse attention.
    Supports per-variable half-life tau_v and strength alpha_v, plus per-head scaling.
    """
    def __init__(self, num_variables: int, n_heads: int, minutes_per_step: float = 1.0,
                 init_tau_min: float = 15.0, init_tau_max: float = 60.0):
        super().__init__()
        self.V = num_variables
        self.H = n_heads
        self.mins = minutes_per_step

        # per-variable tau & alpha (learnable, positive via softplus)
        init_tau = torch.empty(num_variables).uniform_(init_tau_min, init_tau_max)
        self.theta_tau = nn.Parameter(torch.log(torch.exp(init_tau) - 1.0))  # inverse of softplus approx
        self.theta_alpha = nn.Parameter(torch.zeros(num_variables))

        # per-head scalers
        self.eta_w = nn.Parameter(torch.zeros(n_heads))   # scale tau^-1
        self.u_head = nn.Parameter(torch.zeros(n_heads))  # additive log-strength

        self.eps = 1e-3

    def forward(self, time_indices: torch.Tensor, var_indices: torch.Tensor,
                attention_indices_follow_timeline: torch.Tensor) -> torch.Tensor:
        """
        Args:
          time_indices: [T*V] int64 (0..T-1), for each token position
          var_indices:  [T*V] int64 (0..V-1), for each token position
          attention_indices_follow_timeline: [B, T*V, T*max_neighbors] int64 (positions in 0..T*V-1)

        Returns:
          bias: [B, H, T*V, T*max_neighbors] to add into scores
        """
        device = attention_indices_follow_timeline.device
        B, TV, M = attention_indices_follow_timeline.shape

        # gather key times/vars for each selected neighbor
        t_q = time_indices.to(device)[None, :, None].expand(B, TV, M)     # [B, TV, M] query time per edge
        v_q = var_indices.to(device)[None, :, None].expand(B, TV, M)      # [B, TV, M] query var per edge

        idx_sel = attention_indices_follow_timeline  # [B, TV, M]
        t_k = time_indices.to(device)[idx_sel]       # [B, TV, M] key time
        v_k = var_indices.to(device)[idx_sel]        # [B, TV, M] key var

        # Δt in minutes, causal clamp
        dt = (t_q - t_k).clamp_min(0).float() * self.mins  # [B, TV, M]

        # tau_v & alpha_v
        tau_v = torch.nn.functional.softplus(self.theta_tau) + self.eps   # [V]
        alpha_v = torch.nn.functional.softplus(self.theta_alpha) + self.eps  # [V]

        tau_q = tau_v[v_q]  # [B, TV, M]
        tau_k = tau_v[v_k]  # [B, TV, M]
        # effective tau via sum of inverses (you can choose harmonic/sum/etc.)
        tau_eff_inv = (1.0 / tau_q) + (1.0 / tau_k)  # [B, TV, M]

        # per-head scaling w_h >= 0
        w_h = torch.nn.functional.softplus(self.eta_w) + self.eps  # [H]

        # - ln 2 * dt * w_h * (1/tau_q + 1/tau_k)
        ln2 = math.log(2.0)
        # expand head dim
        decay_term = -ln2 * dt.unsqueeze(1) * (w_h.view(1, self.H, 1, 1) * tau_eff_inv.unsqueeze(1))

        # strength term: ln alpha (use key variable + head additive)
        log_alpha_k = torch.log(alpha_v[v_k] + 1e-12)  # [B, TV, M]
        log_alpha = log_alpha_k.unsqueeze(1) + self.u_head.view(1, self.H, 1, 1)

        bias = decay_term + log_alpha  # [B, H, TV, M]
        return bias
```

> 这样做的好处是：**不增加序列长度与 KV 体积**，只是对 `scores` 加一个同形状 bias；并且**每个变量的半衰期与强度是可学习的**，还能通过 head 维度形成多尺度衰减。与现有稀疏注意力装配方式完全兼容。

### 3) `model.py`：把两者串起来

在 `FluidDecoder._build_model()` 里注册两个新模块，在 `forward()` 里喂数据、计算 bias 并传给注意力。

```python
# ========= minimal edits in model.py =========
# 1) _build_model: 注册 ContinuousTimeEncoding 与 RelativeTimeBiasSparse
from .encoding import CombinedPositionalEncoding, ContinuousTimeEncoding
from .attention import RelativeTimeBiasSparse
...
def _build_model(self):
    ...
    self.pos_encoding = CombinedPositionalEncoding(
        d_model=config.d_model,
        max_time_positions=config.max_time_positions,
        max_variable_positions=config.max_variable_positions,
        time_encoding_type=config.time_position_encoding,
        variable_encoding_type=config.variable_position_encoding
    )
    self.time_cont_encoding = ContinuousTimeEncoding(
        d_model=config.d_model, t2v_k=8, period_minutes=180.0, out_hidden=128
    )
    ...
    self.rel_time_bias = RelativeTimeBiasSparse(
        num_variables=config.max_variable_positions, n_heads=config.n_heads, minutes_per_step=1.0
    )
    ...

# 2) forward: 接入 age/phase（来自 batch 或 kwargs），以及构造 time/var 索引与 bias
def forward(self, input_ids=None, labels=None, prediction_mask=None, output_attentions=None, **kwargs):
    ...
    batch_size, time_steps, num_variables = x.shape

    # reshape + input projection
    x_reshaped = x.reshape(batch_size, -1, 1)  # [B, T*V, 1]
    x = self.input_projection(x_reshaped)      # [B, T*V, d_model]

    # add discrete time+variable positional enc
    x = self.pos_encoding(x, time_steps, num_variables)
    x = self.pos_dropout(x)

    # === New: add continuous time encoding ===
    # expect age_minutes & phase_minutes in [B, T, V] or derive them yourself
    age_minutes = kwargs.get('age_minutes', None)
    phase_minutes = kwargs.get('phase_minutes', None)
    if age_minutes is not None and phase_minutes is not None:
        age_m = age_minutes.reshape(batch_size, -1, 1).to(x.device).float()
        phase_m = phase_minutes.reshape(batch_size, -1, 1).to(x.device).float()
        x = x + self.time_cont_encoding(age_m, phase_m)

    # build sparse indices and mask (as you already do)
    if isinstance(input_ids, dict) and 'attention_indices' in input_ids:
        attention_indices = input_ids['attention_indices']
    elif hasattr(self, '_attention_indices'):
        attention_indices = self._attention_indices.expand(batch_size, -1, -1)
    else:
        default_max_neighbors = 32
        attention_indices = torch.arange(num_variables, device=x.device)[None, :, None].expand(
            batch_size, num_variables, default_max_neighbors
        ) % num_variables

    attention_indices_follow_timeline = expand_tensor_follow_timeline(attention_indices, time_steps).detach()
    attention_mask = DecoderAttentionMask.create_decoder_mask(
        batch_size, time_steps, num_variables, attention_indices, device=x.device
    ).detach()

    # === New: build time/var indices for relative bias ===
    # [T*V] token-level indices
    time_indices = torch.arange(time_steps, device=x.device).repeat_interleave(num_variables)   # [T*V]
    var_indices  = torch.arange(num_variables, device=x.device).repeat(time_steps)              # [T*V]

    # compute bias: [B, H, T*V, T*max_neighbors]
    additive_bias = self.rel_time_bias(time_indices, var_indices, attention_indices_follow_timeline)

    # pass through decoder blocks (hand the bias down)
    all_attentions = [] if output_attentions else None
    for block in self.decoder_blocks:
        if output_attentions:
            x, layer_attention = block(
                x, attention_indices_follow_timeline, attention_mask, output_attentions=True, additive_bias=additive_bias
            )
            all_attentions.append(layer_attention)
        else:
            x = block(x, attention_indices_follow_timeline, attention_mask, additive_bias=additive_bias)
    ...
```

同时需要让 `layers.py` 的 `DecoderBlock.forward` 也把 `additive_bias` 传下去即可：

```python
# ========= tiny edit in layers.py =========
def forward(self, x, attention_indices_follow_timeline=None,
            attention_mask=None, output_attentions=False, additive_bias=None):
    attn_result = self.attention(
        x, attention_indices_follow_timeline, attention_mask, output_attentions, additive_bias=additive_bias
    )
    ...
```

> 以上改动与现有张量形状完全对齐：你现在的 `CombinedPositionalEncoding` 与稀疏注意力索引展开逻辑保持不动，我们只是**加了一项 [B,H,T*V,T*max_neighbors] 的 bias**。

---

## 四、不同变量的“稳态时间不一样 & 影响程度不一样”如何训练/使用

- **半衰期 \(\tau_v\) 初始化**：上面代码把 \(\tau_v\) 初始化在 15–60 min 之间（你可改为 10–120）；训练中自适应到每个变量的真实耗散时间。  
- **影响强度 \(\alpha_v\)**：相当于“某变量被当作记忆来源时的权重先验”。这解决“每 3 小时施加影响程度不一样”的**静态异质性**。  
- **（可选）事件幅度调制**：如果你能在 batch 里提供本次三小时事件的 \(\|\Delta X\|\) 或特定边界的幅度，可把它作为额外项加到 `log_alpha_k` 里（例如 `log_alpha_k += s * norm(dX_k)`），让强度**随事件大小动态变化**。  
- **多头多尺度**：每个头有自己的 \(w_h\)（对 \(\tau^{-1}\) 的缩放）和 \(u_h\)（加性强度），进一步覆盖快/中/慢三种衰减尺度。  

---

## 五、Time2Vec/Fourier 与相对时间偏置如何配合？

- **Time2Vec/Fourier** 改善的是**每个 token 的内容表征**（“我现在处在跃迁的第几分钟/三小时的哪个相位”）。  
- **相对时间偏置** 改善的是**注意力的信息路由**（“我更该关注最近 5 分钟还是 20 分钟前的东西，而且不同变量的衰减速度不同”）。  
- 两者是**正交互补**，在短序列（T≤5）尤其合适：编码告诉模型“我是谁”，偏置告诉模型“我应更看谁”。

---

## 六、与你当前实现的契合点与注意事项

- 你目前的 `CombinedPositionalEncoding.forward` 已经正确构造了 `[T*V]` 的 `time_indices` 与 `variable_indices` 并加到 `x` 上；我们在 `model.forward` 中**再次使用同样的索引**构造 `Δt`，计算 bias。  
- 稀疏注意力在 `SimpleMultiHeadAttention._sparse_attention` 里**先投影再依据索引选取 K、V**，我们对 `scores` 加 bias 的位置就放在 mask 之后、softmax 之前，开销很小。  
- `DecoderAttentionMask.create_decoder_mask` 保证**因果**与**时间线展开**，我们的 `Δt` 逻辑假定 `dt≥0` 与之吻合。  
- **数值稳定**：\(\tau_v\)、\(\alpha_v\) 用 softplus 保正；`additive_bias` 与 `scores` 同量纲（logit），建议训练初期把 `eta_w,u_head` 初始化为 0，等价于“很弱的偏置”。  
- **开关与消融**：  
  - 若要只用标准 ALiBi：固定所有 \(\tau_v=\tau_0\)，只保留每头一个斜率；  
  - 若要只用变量强度：把 `decay_term` 置零，仅用 `log_alpha`；  
  - 若要关掉 Fourier：把 `ContinuousTimeEncoding` 只保留 Time2Vec(age)。

---

## 七、最小使用示例（伪）

```python
# 构造输入
batch = {
    'input': x_btv,                           # [B,T,V]
    'attention_indices': attn_idx_bvm,        # [B,V,M]
}
age_minutes   = age_btv                       # [B,T,V], 每点距最近外界变化的分钟
phase_minutes = (age_btv % 180.0)             # [B,T,V], 三小时内相位 0..180

out = model(batch, age_minutes=age_minutes, phase_minutes=phase_minutes)
loss = criterion(out['logits'], target)
```

---

## 八、训练小提示

- **初始化半衰期**：若你有变量分组（阀门/管段/站场…），可对组内 \(\tau_v\) 共享或近似；没有先验就用 15–60 min 随机。  
- **采样**：按“事件后 0–5/5–30/30+”分层重采，确保模型真的学到**跃迁早中晚**。  
- **监控**：导出每层/每头的平均 `decay_term` 与 `log_alpha`，观察是否对快变量学到更大的 \(\tau^{-1}\)。

---

