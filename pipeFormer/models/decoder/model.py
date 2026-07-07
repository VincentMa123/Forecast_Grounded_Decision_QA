"""
Decoder-only model for fluid dynamics time series prediction.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple
import logging

from ..base import BaseModel
from .config import DecoderConfig
from .encoding import CombinedPositionalEncoding
from .layers import DecoderBlock
from .masks import DecoderAttentionMask
from .utils import expand_tensor_follow_timeline, save_attention_mask_image

logger = logging.getLogger(__name__)


class FourierValueProjection(nn.Module):
    """Project scalar inputs via Fourier features before applying an MLP."""

    def __init__(
        self,
        *,
        output_dim: int,
        hidden_dim: int,
        num_frequencies: int = 10,
    ) -> None:
        super().__init__()
        if num_frequencies <= 0:
            raise ValueError("num_frequencies must be positive")
        self.num_frequencies = num_frequencies
        frequencies = torch.pow(2.0, torch.arange(num_frequencies, dtype=torch.float32))
        self.register_buffer("frequencies", frequencies, persistent=False)
        mlp_input_dim = 2 * num_frequencies + 1  # 原始标量 + sin + cos
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.dim() != 3 or values.size(-1) != 1:
            raise ValueError("values must have shape [B, N, 1]")
        z = values.to(dtype=self.frequencies.dtype)
        two_pi = 2.0 * math.pi
        freq = self.frequencies.view(1, 1, -1).to(device=z.device, dtype=z.dtype)
        angles = two_pi * z * freq
        sin_features = torch.sin(angles)
        cos_features = torch.cos(angles)
        features = torch.cat((z, sin_features, cos_features), dim=-1)
        return self.mlp(features)


class FluidDecoder(BaseModel):
    """
    纯Decoder模型，用于天然气管网流体动力学预测。
    
    架构：
    1. 输入重塑：[B, T, V] -> [B, T*V, d_model]
    2. 组合位置编码：时间编码 + 变量编码
    3. Decoder层堆叠
    4. 输出投影：[B, T*V, d_model] -> [B, T, V]
    """
    
    def __init__(self, config: Optional[DecoderConfig] = None, **kwargs):
        """
        初始化FluidDecoder模型。
        
        Args:
            config: DecoderConfig实例
            **kwargs: 额外参数（会覆盖config）
        """
        if config is None:
            config = DecoderConfig()
        
        # 更新config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 从config中删除input_dim和output_dim，避免重复传递
        config_dict = config.to_dict()
        config_dict.pop('input_dim', None)
        config_dict.pop('output_dim', None)
        super().__init__(input_dim=config.input_dim, output_dim=config.output_dim, **config_dict)
        
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        
        # 构建模型
        self._build_model()

        # 初始化权重
        self._initialize_weights()

        self._has_token_value_stats: bool = False

        logger.info(f"FluidDecoder initialized: {self.get_model_info()}")
    
    def _build_model(self):
        """构建decoder模型架构。"""
        config = self.config

        # 输入投影：根据配置选择token嵌入或数值投影
        if config.input_projection_type == "token_embedding":
            self.token_embedding = nn.Embedding(config.tokenizer_vocab_size, config.d_model)
            self.value_projection = None
            self.output_projection = None  # token模式下不需要数值输出头
            # token词表输出头，用于预测离散token id（与输入嵌入权重共享）
            self.token_output = nn.Linear(config.d_model, config.tokenizer_vocab_size, bias=False)
            self.token_output.weight = self.token_embedding.weight
            self.offset_output = None
        elif config.input_projection_type == "value_projection":
            self.token_embedding = None
            self.value_projection = FourierValueProjection(
                output_dim=config.d_model,
                hidden_dim=config.projection_hidden_dim,
                num_frequencies=10,
            )
            self.token_output = None
            self.output_projection = nn.Sequential(
                nn.Linear(config.d_model, config.projection_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(config.projection_hidden_dim),
                nn.Linear(config.projection_hidden_dim, 1)
            )
            self.offset_output = None
        elif config.input_projection_type == "hybrid":
            self.token_embedding = nn.Embedding(config.tokenizer_vocab_size, config.d_model)
            self.value_projection = None
            self.output_projection = None
            self.hybrid_offset_projection = FourierValueProjection(
                output_dim=config.d_model,
                hidden_dim=config.projection_hidden_dim,
                num_frequencies=getattr(config, 'hybrid_num_frequencies', 10),
            )
            self.hybrid_median_projection = nn.Sequential(
                nn.Linear(1, config.projection_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(config.projection_hidden_dim),
                nn.Linear(config.projection_hidden_dim, config.d_model)
            )
            self.hybrid_value_norm = nn.LayerNorm(config.d_model)
            self.hybrid_projection = nn.Sequential(
                nn.Linear(config.d_model * 2, config.d_model),
                nn.GELU(),
                nn.LayerNorm(config.d_model)
            )
            self.hybrid_dropout = nn.Dropout(config.dropout_rate)
            self.token_output = nn.Linear(config.d_model, config.tokenizer_vocab_size, bias=False)
            self.token_output.weight = self.token_embedding.weight
            self.offset_output = nn.Linear(config.d_model, config.tokenizer_vocab_size)
        else:
            raise ValueError(
                f"Unsupported input_projection_type: {config.input_projection_type}"
            )

        # 组合位置编码
        self.pos_encoding = CombinedPositionalEncoding(
            d_model=config.d_model,
            max_time_positions=config.max_time_positions,
            max_variable_positions=config.max_variable_positions,
            time_encoding_type=config.time_position_encoding,
            variable_encoding_type=config.variable_position_encoding
        )
        
        self.pos_dropout = nn.Dropout(config.dropout_rate)
        
        # Decoder层
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.attention_dropout,
                activation=config.activation
            ) for _ in range(config.n_layers)
        ])
        
        # 最终归一化
        if config.use_layer_norm:
            self.final_norm = nn.LayerNorm(config.d_model)
        else:
            self.final_norm = None

    def _initialize_weights(self):
        """初始化模型权重，增强FP16数值稳定性。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用较小的权重初始化以提高FP16稳定性
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # 如果使用token embedding, 共享输入和输出的词向量参数
        if getattr(self, "token_embedding", None) is not None and getattr(self, "token_output", None) is not None:
            self.token_output.weight = self.token_embedding.weight

    def forward(self, input_ids, labels=None, prediction_mask=None, output_attentions=None, **kwargs):
        """
        FluidDecoder前向传播

        Args:
            input_ids: 输入张量 [B, T, V=6712]
            labels: 目标张量 [B, T, V=6712] 用于损失计算 (可选)
            prediction_mask: 预测mask [B, V] (可选)
            output_attentions: 是否输出注意力权重 (可选)
            **kwargs: 额外参数（如attention_indices, input_tokens等）

        Returns:
            如果提供labels: {'loss': tensor, 'logits': tensor, 'attentions': List[Tensor[B,H,T*V,T*max_neighbors]] (如果output_attentions=True)}
            否则: {'logits': tensor, 'attentions': List[Tensor[B,H,T*V,T*max_neighbors]] (如果output_attentions=True)} 或 tensor
        """
        # 处理输入格式，统一使用显式参数，其余从kwargs中读取
        x = input_ids

        tokens = kwargs.get('input_tokens')
        target_tokens = kwargs.get('target_tokens')

        if labels is None:
            labels = kwargs.get('labels', kwargs.get('target'))
        if prediction_mask is None:
            prediction_mask = kwargs.get('prediction_mask')
        if output_attentions is None:
            output_attentions = kwargs.get('output_attentions', False)

        # 默认不输出注意力权重
        if output_attentions is None:
            output_attentions = False
        
        batch_size, time_steps, num_variables = x.shape
        
        projection_type = self.config.input_projection_type
        if projection_type == "token_embedding" or projection_type == "hybrid":
            if tokens is None:
                raise ValueError("input_tokens must be provided when using token_embedding-based projections.")
            token_ids = tokens.reshape(batch_size, -1).long()
            vocab_size = self.token_embedding.num_embeddings
            if token_ids.numel() > 0:
                max_id = int(token_ids.max().item())
                min_id = int(token_ids.min().item())
                if min_id < 0 or max_id >= vocab_size:
                    invalid_mask = (token_ids < 0) | (token_ids >= vocab_size)
                    first_invalid = torch.nonzero(invalid_mask, as_tuple=False)[0]
                    batch_idx = int(first_invalid[0].item())
                    flat_idx = int(first_invalid[1].item())
                    time_idx = flat_idx // num_variables
                    variable_idx = flat_idx % num_variables
                    offending = int(token_ids[batch_idx, flat_idx].item())
                    raise RuntimeError(
                        f"token_ids out of range for embedding (vocab_size={vocab_size}): "
                        f"value={offending}, batch={batch_idx}, time={time_idx}, variable={variable_idx}, "
                        f"min_id={min_id}, max_id={max_id}"
                    )
            token_embeddings = self.token_embedding(token_ids)  # [B, T*V, d_model]
            if projection_type == "token_embedding":
                x = token_embeddings
            else:
                input_token_medians = kwargs.get('input_token_medians')
                input_token_offsets = kwargs.get('input_token_offsets')
                if input_token_medians is None:
                    raise ValueError("input_token_medians must be provided for hybrid projection.")
                if input_token_offsets is None:
                    raise ValueError("input_token_offsets must be provided for hybrid projection.")
                median_features = input_token_medians.reshape(batch_size, -1, 1).to(token_embeddings.dtype)
                offset_features = input_token_offsets.reshape(batch_size, -1, 1).to(token_embeddings.dtype)
                value_offset_emb = self.hybrid_offset_projection(offset_features)
                value_median_emb = self.hybrid_median_projection(median_features)
                value_embedding = self.hybrid_value_norm(value_offset_emb + value_median_emb)
                hybrid_input = torch.cat((token_embeddings, value_embedding), dim=-1)
                x = self.hybrid_projection(hybrid_input)
                x = self.hybrid_dropout(x)
        elif projection_type == "value_projection":
            value_reshaped = x.reshape(batch_size, -1, 1)
            x = self.value_projection(value_reshaped)
        else:
            raise ValueError(f"Unsupported input_projection_type during forward: {projection_type}")
        
        # 添加位置编码
        x = self.pos_encoding(x, time_steps, num_variables)
        x = self.pos_dropout(x)
        
        # 根据配置选择稀疏拓扑注意力或全注意力
        use_sparse = getattr(self.config, 'use_topology_attention', True)
        attention_indices = kwargs.get('attention_indices')
        attention_indices_follow_timeline = None
        attention_mask = None
        use_causal_mask = getattr(self.config, 'whether_causal', True)

        if use_sparse:
            # 稀疏拓扑注意力需要外部提供attention indices
            if attention_indices is None:
                raise ValueError("attention_indices not provided for sparse topology attention.")

            attention_indices_follow_timeline = expand_tensor_follow_timeline(attention_indices, time_steps)
            # no grad
            attention_indices_follow_timeline = attention_indices_follow_timeline.detach()

            # 创建attention mask（稀疏掩码）
            attention_mask = DecoderAttentionMask.create_decoder_mask(
                batch_size, time_steps, num_variables, 
                attention_indices, device=x.device, causal=use_causal_mask
            )  # [B, T*V, T*max_neighbors_variable]
            attention_mask = attention_mask.detach()
            if False:
                save_attention_mask_image(
                    attention_mask,
                    save_dir="models/decoder/vis",
                    filename="attention_mask_batch0.png"
                )

        # 通过decoder层，如果需要保存注意力权重
        all_attentions = [] if output_attentions else None  # List[Tensor[B, H, T*V, T*max_neighbors]]
        for block in self.decoder_blocks:
            if output_attentions:
                # 获取层输出和注意力权重 [B, H, T*V, T*max_neighbors]
                x, layer_attention = block(x, attention_indices_follow_timeline, attention_mask, output_attentions=True)
                all_attentions.append(layer_attention)  # 收集每层的注意力权重
            else:
                x = block(x, attention_indices_follow_timeline, attention_mask)
        
        # 最终归一化
        if self.final_norm is not None:
            x = self.final_norm(x)
        
        value_predictions: Optional[torch.Tensor] = None
        hybrid_probabilities: Optional[torch.Tensor] = None

        if self.output_projection is not None:
            projected = self.output_projection(x)
            value_predictions = projected.squeeze(-1).view(batch_size, time_steps, num_variables)

        token_logits = None
        vocab_size = self.config.tokenizer_vocab_size
        if self.token_output is not None:
            token_logits = self.token_output(x).view(batch_size, time_steps, num_variables, vocab_size)

        offset_logits: Optional[torch.Tensor] = None
        offset_values: Optional[torch.Tensor] = None
        if getattr(self, "offset_output", None) is not None:
            offset_logits = self.offset_output(x).view(batch_size, time_steps, num_variables, vocab_size)
            offset_values = torch.tanh(offset_logits)
            if projection_type == "hybrid":
                value_predictions, hybrid_probabilities = self._compute_hybrid_value_prediction(
                    token_logits, offset_values
                )

        # 准备返回结果
        result: Dict[str, torch.Tensor] = {}
        if token_logits is not None:
            result['token_logits'] = token_logits
        if offset_logits is not None:
            result['offset_logits'] = offset_logits
        if offset_values is not None:
            result['token_offsets'] = offset_values
        if hybrid_probabilities is not None:
            result['token_probabilities'] = hybrid_probabilities
        if value_predictions is not None:
            result['value_predictions'] = value_predictions

        result['logits'] = token_logits if token_logits is not None else value_predictions

        # 如果需要返回注意力权重
        if output_attentions:
            result['attentions'] = all_attentions  # List[Tensor[B, H, T*V, T*max_neighbors]] - 每层的注意力权重

            if attention_indices is None:
                # 全注意力模式：为可视化生成基础索引映射 [1, V, V]
                base_indices = torch.arange(num_variables, device=x.device, dtype=torch.long)
                attention_indices_storage = base_indices.unsqueeze(0).repeat(num_variables, 1).unsqueeze(0)  # [1, V, V]
                attention_indices_follow_storage = None
            else:
                attention_indices_storage = attention_indices
                attention_indices_follow_storage = attention_indices_follow_timeline

            result['attention_indices'] = attention_indices_storage
            result['attention_indices_follow_timeline'] = attention_indices_follow_storage

        # 返回格式兼容transformers
        projection_type = self.config.input_projection_type
        if projection_type == "token_embedding":
            if token_logits is None:
                raise RuntimeError("token_logits unavailable; token-based loss requires token_embedding projection.")
            if target_tokens is not None:
                loss, token_accuracy = self.compute_token_loss(token_logits, target_tokens, prediction_mask)
                result['loss'] = loss
                result['token_accuracy'] = token_accuracy.detach()
        elif projection_type == "value_projection":
            if labels is None:
                logger.debug("Labels not provided; skipping scalar loss for value_projection mode.")
            elif value_predictions is None:
                raise RuntimeError("value_projection mode requires value head predictions when labels are provided.")
            else:
                loss = super().compute_loss(value_predictions, labels, prediction_mask)
                result['loss'] = loss
        elif projection_type == "hybrid":
            if token_logits is None:
                raise RuntimeError("token_logits unavailable; hybrid mode requires token logits for CE loss.")
            total_loss: Optional[torch.Tensor] = None
            if target_tokens is not None:
                ce_loss, token_accuracy = self.compute_token_loss(token_logits, target_tokens, prediction_mask)
                result['token_accuracy'] = token_accuracy.detach()
                result['cross_entropy_loss'] = ce_loss.detach()
                total_loss = self.config.hybrid_ce_weight * ce_loss
            if labels is not None:
                if value_predictions is None:
                    raise RuntimeError("hybrid mode requires value predictions when labels are provided.")
                mae_loss = super().compute_loss(value_predictions, labels, prediction_mask)
                result['mae_loss'] = mae_loss.detach()
                weighted_mae = self.config.hybrid_mae_weight * mae_loss
                if total_loss is None:
                    total_loss = weighted_mae
                else:
                    total_loss = total_loss + weighted_mae
            if total_loss is not None:
                result['loss'] = total_loss
        else:
            raise ValueError(f"Unsupported input_projection_type during loss computation: {projection_type}")

        return result

    def compute_token_loss(
        self,
        token_logits: torch.Tensor,
        target_tokens: torch.Tensor,
        prediction_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """针对token分类的交叉熵损失和准确率，支持尾部多个时间步。"""

        if target_tokens.dtype != torch.long:
            target_tokens = target_tokens.long()
            
        time_steps = token_logits.size(1)
        loss_steps = min(self.loss_time_steps, time_steps - 1)

        selected_logits = token_logits[:, -loss_steps:, :, :]  # [B, S, V, vocab]
        selected_targets = target_tokens[:, -loss_steps:, :]  # [B, S, V]

        if prediction_mask is None:
            mask = torch.ones_like(selected_targets, dtype=torch.bool, device=selected_targets.device)
        else:
            if prediction_mask.dim() == 2:
                mask = prediction_mask.unsqueeze(1).expand_as(selected_targets)
            else:
                raise ValueError("prediction_mask must have 1, 2 or 3 dimensions")
            mask = mask.to(selected_targets.device) > 0

        active = mask & (selected_targets >= 0)
        if not torch.any(active):
            # 无有效样本时返回零损失，爆warning之后返回,避免NaN
            zero = selected_logits.new_tensor(0.0)
            return zero, zero

        logits_flat = selected_logits[active]
        targets_flat = selected_targets[active]
        loss = F.cross_entropy(logits_flat, targets_flat, reduction='mean')
        if logits_flat.numel() == 0:
            accuracy = loss.new_tensor(0.0)
        else:
            predictions_flat = torch.argmax(logits_flat, dim=-1)
            correct = (predictions_flat == targets_flat).float()
            accuracy = correct.mean()
        return loss, accuracy

    def set_token_value_statistics(self, medians: torch.Tensor, half_widths: torch.Tensor) -> None:
        """Attach token median and half-width lookup tables required for hybrid decoding."""

        if medians.dim() != 1 or half_widths.dim() != 1:
            raise ValueError("Token statistic tensors must be 1D.")
        if medians.shape != half_widths.shape:
            raise ValueError("Token median and half-width tensors must share the same shape.")
        if (half_widths <= 0).any():
            raise ValueError("All token half widths must be positive.")

        expected_vocab = int(self.config.tokenizer_vocab_size)
        if medians.numel() != expected_vocab:
            raise ValueError(
                f"Token statistics length {medians.numel()} does not match tokenizer_vocab_size={expected_vocab}"
            )

        ref_param = next(self.parameters())
        device = ref_param.device
        dtype = ref_param.dtype

        medians_tensor = medians.to(device=device, dtype=dtype)
        half_widths_tensor = half_widths.to(device=device, dtype=dtype)

        if 'token_value_medians' in self._buffers:
            self._buffers['token_value_medians'] = medians_tensor
        else:
            self.register_buffer('token_value_medians', medians_tensor, persistent=False)

        if 'token_value_half_widths' in self._buffers:
            self._buffers['token_value_half_widths'] = half_widths_tensor
        else:
            self.register_buffer('token_value_half_widths', half_widths_tensor, persistent=False)

        self._has_token_value_stats = True

    def _compute_hybrid_value_prediction(
        self,
        token_logits: torch.Tensor,
        offset_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute value predictions via distribution expectation for hybrid mode."""

        if not getattr(self, "_has_token_value_stats", False):
            raise RuntimeError("Hybrid mode requires token value statistics. Call set_token_value_statistics() first.")

        medians = getattr(self, 'token_value_medians', None)
        half_widths = getattr(self, 'token_value_half_widths', None)
        if medians is None or half_widths is None:
            raise RuntimeError("Token value statistics buffers are missing; ensure they are registered before use.")

        temperature = float(getattr(self.config, 'hybrid_softmax_temperature', 1.0))
        if temperature <= 0:
            raise ValueError("hybrid_softmax_temperature must be positive.")

        logits = token_logits / temperature
        probabilities = torch.softmax(logits, dim=-1)

        medians_lookup = medians.to(device=token_logits.device, dtype=token_logits.dtype).view(1, 1, 1, -1)
        half_width_lookup = half_widths.to(device=token_logits.device, dtype=token_logits.dtype).view(1, 1, 1, -1)

        refined_values = medians_lookup + half_width_lookup * offset_values
        predicted = torch.sum(probabilities * refined_values, dim=-1)
        return predicted, probabilities

    @staticmethod
    def decode_tokens_to_values(
        tokenizer: Any,
        token_ids: torch.Tensor,
        *,
        use_median: bool = False,
    ) -> torch.Tensor:
        """将token id转换为浮点数值。

        Args:
            tokenizer: 具备 ``tokens_to_values`` 接口的tokenizer实例
            token_ids: Token id张量 [B, T, V]
            use_median: 是否使用中位数进行逆变换

        Returns:
            与输入形状一致的float张量
        """

        if tokenizer is None or not hasattr(tokenizer, "tokens_to_values"):
            raise ValueError("Tokenizer with tokens_to_values method is required for decoding token ids.")

        return tokenizer.tokens_to_values(token_ids, use_median=use_median)

    def sample_token_ids(
        self,
        token_logits: torch.Tensor,
        *,
        deterministic: bool = True,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """从token logits中采样token id。"""

        if deterministic:
            return torch.argmax(token_logits, dim=-1)

        if temperature <= 0:
            raise ValueError("temperature must be positive when sampling stochastically")

        logits = token_logits / temperature
        if top_k is not None:
            if top_k <= 0:
                raise ValueError("top_k must be positive when provided")
            top_k = min(top_k, logits.size(-1))
            top_values, _ = torch.topk(logits, top_k, dim=-1)
            kth = top_values[..., -1, None]
            logits = torch.where(logits < kth, torch.full_like(logits, float('-inf')), logits)

        probs = F.softmax(logits, dim=-1)
        cat = torch.distributions.Categorical(probs=probs)
        sampled = cat.sample()
        return sampled

    def sample_token_values(
        self,
        token_logits: torch.Tensor,
        tokenizer: Any,
        *,
        deterministic: bool = True,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_median: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从token logits采样并解码回浮点数值。"""

        token_ids = self.sample_token_ids(
            token_logits,
            deterministic=deterministic,
            temperature=temperature,
            top_k=top_k,
        )
        values = self.decode_tokens_to_values(tokenizer, token_ids, use_median=use_median)
        if isinstance(values, torch.Tensor) and values.device != token_logits.device:
            values = values.to(token_logits.device)
        return values, token_ids
    
    def get_model_info(self) -> Dict:
        """获取详细的模型信息。"""
        base_info = super().get_model_info()
        
        decoder_info = {
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff': self.config.d_ff,
            'time_position_encoding': self.config.time_position_encoding,
            'variable_position_encoding': self.config.variable_position_encoding,
            'tokenizer_vocab_size': self.config.tokenizer_vocab_size,
            'input_projection_type': self.config.input_projection_type
        }
        
        base_info.update(decoder_info)
        return base_info
