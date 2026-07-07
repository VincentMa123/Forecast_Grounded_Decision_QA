# FluidDecoder 模型文档

## 概述

FluidDecoder 现已完全兼容 HuggingFace Transformers 库的 Trainer 和 TrainingArguments。

## 主要特性

### 1. Transformers 兼容性

模型的 `forward` 方法现在支持 transformers 标准格式：

```python
# Transformers 格式
output = model(input_ids=input_tensor, labels=labels)
# 返回: {'loss': tensor, 'logits': tensor}

# 仅推理
output = model(input_ids=input_tensor) 
# 返回: {'logits': tensor}

# 兼容原有格式
batch = {'input': input_tensor, 'target': labels}
output = model(input_ids=batch)
# 返回: {'loss': tensor, 'logits': tensor}
```

### 2. 损失函数

- 使用 MSE 损失
- 支持 prediction_mask 进行选择性预测
- 自动处理标量损失返回

### 3. 模型架构

- 纯 Decoder 架构
- 输入维度: [B, T, V] -> [B, T*V, d_model]
- 组合位置编码 (时间 + 变量)
- 多头自注意力机制
- 残差连接和层归一化

## 使用示例

### 基本使用

```python
from models.decoder import FluidDecoder
from models.config import DecoderConfig

# 创建配置
config = DecoderConfig(
    d_model=768,
    n_heads=12,
    n_layers=6
)

# 创建模型
model = FluidDecoder(config)

# 准备数据
input_tensor = torch.randn(batch_size, time_steps, 6712)
labels = torch.randn(batch_size, time_steps, 6712)

# 训练模式
output = model(input_ids=input_tensor, labels=labels)
loss = output['loss']
predictions = output['logits']

# 推理模式
output = model(input_ids=input_tensor)
predictions = output['logits']
```

### 与 Transformers Trainer 使用

```python
from transformers import Trainer, TrainingArguments

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./fluid_decoder_output',
    num_train_epochs=10,
    per_device_train_batch_size=8,
    learning_rate=1e-4,
    weight_decay=0.01,
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 开始训练
trainer.train()
```

## 配置参数

### DecoderConfig 参数

```python
@dataclass
class DecoderConfig:
    # 架构参数
    d_model: int = 768              # 隐藏维度
    n_heads: int = 12               # 注意力头数
    n_layers: int = 6               # 层数
    d_ff: int = 3072               # 前馈网络维度
    
    # 训练参数
    learning_rate: float = 1e-4     # 学习率
    dropout_rate: float = 0.1       # Dropout 率
    attention_dropout: float = 0.1  # 注意力 Dropout
    
    # 位置编码
    time_position_encoding: str = "sinusoidal"      # 时间编码类型
    variable_position_encoding: str = "sinusoidal"  # 变量编码类型
    max_time_positions: int = 10                    # 最大时间位置
    max_variable_positions: int = 6712              # 最大变量位置
    
    # 投影层
    projection_hidden_dim: int = 256  # 投影层隐藏维度
    
    # 其他
    use_layer_norm: bool = True     # 使用层归一化
    activation: str = "gelu"        # 激活函数
```

## 测试

运行测试文件验证功能：

```bash
# 基本兼容性测试
python simple_test.py

# 完整测试
python final_test.py

# Transformers 集成演示
python transformers_demo.py
```

## 注意事项

1. **输入格式**: 支持多种输入格式以保持向后兼容性
2. **损失计算**: 自动处理有无标签的情况
3. **内存效率**: 大型模型可能需要调整批次大小
4. **依赖项**: 完整的 Transformers 集成可能需要安装额外依赖

## 模型状态

✅ Transformers 兼容性已实现
✅ MSE 损失集成完成
✅ 前向传播格式标准化
✅ 测试验证通过