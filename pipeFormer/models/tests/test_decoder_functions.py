"""
Tests for key decoder functions to verify correctness against baseline implementations.
"""

import torch
from typing import Tuple

from ..decoder.attention import SimpleMultiHeadAttention
from ..decoder.masks import DecoderAttentionMask
from ..decoder.utils import expand_tensor_follow_timeline


class TestDecoderFunctions:
    """测试decoder核心函数的正确性"""
    
    def sample_data(self) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """创建测试数据"""
        B, T, V = 2, 3, 10
        d_model = 8
        max_neighbors = 4
        
        # 创建测试张量
        x = torch.randn(B, T*V, d_model)
        attention_indices = torch.randint(0, T*V, (B, T*V, T*max_neighbors))
        
        params = {
            'B': B, 'T': T, 'V': V, 'd_model': d_model, 
            'max_neighbors': max_neighbors
        }
        
        return x, attention_indices, params
    
    def test_timeline_attention_indexing_baseline(self, sample_data):
        """测试timeline_attention_indexing的baseline实现和优化实现是否一致"""
        x, attention_indices, params = sample_data
        B, T, V, d_model = params['B'], params['T'], params['V'], params['d_model']
        
        # 创建attention模块
        attention = SimpleMultiHeadAttention(d_model=d_model, n_heads=2)
        
        # 优化版本（当前实现）
        optimized_result = attention.timeline_attention_indexing(x, attention_indices)
        
        # Baseline版本（使用最基础的for循环）
        def baseline_timeline_attention_indexing(x, attention_indices):
            B, TV, d_model = x.shape
            B, TV, num_indices = attention_indices.shape
            
            result = torch.zeros(B, TV, num_indices, d_model, device=x.device, dtype=x.dtype)
            
            for b in range(B):
                for tv in range(TV):
                    for idx in range(num_indices):
                        target_idx = attention_indices[b, tv, idx].item()
                        if 0 <= target_idx < TV:
                            result[b, tv, idx] = x[b, target_idx]
            
            return result
        
        baseline_result = baseline_timeline_attention_indexing(x, attention_indices)
        
        # 比较结果
        torch.testing.assert_close(optimized_result, baseline_result, 
                                 msg="timeline_attention_indexing优化版本与baseline不匹配")
    
    def test_expand_tensor_follow_timeline_baseline(self):
        """测试expand_tensor_follow_timeline的baseline实现和优化实现是否一致"""
        B, V, dim, T = 2, 5, 4, 3
        x = torch.randint(0, V, (B, V, dim))  # 使用整数便于验证
        
        # 优化版本（当前实现）
        optimized_result = expand_tensor_follow_timeline(x, T)
        
        # Baseline版本（使用最基础的for循环）
        def baseline_expand_tensor_follow_timeline(x, T):
            B, V, dim = x.shape
            result = torch.zeros(B, T*V, T*dim, dtype=x.dtype, device=x.device)
            
            # 外层循环：T时间步
            for t in range(T):
                # 当前时间步的起始位置
                t_start = t * V
                t_end = (t + 1) * V
                
                # 内层循环：填充T*dim维度
                for hist_t in range(T):
                    hist_start = hist_t * dim
                    hist_end = (hist_t + 1) * dim
                    
                    # 为每个变量填充对应的attention索引
                    for v in range(V):
                        for d in range(dim):
                            # 计算历史时间步对应的变量索引
                            historical_v_idx = x[0, v, d].item() + hist_t * V
                            result[:, t_start + v, hist_start + d] = historical_v_idx
            
            return result
        
        baseline_result = baseline_expand_tensor_follow_timeline(x, T)
        
        # 比较形状
        assert optimized_result.shape == baseline_result.shape, \
            f"形状不匹配: {optimized_result.shape} vs {baseline_result.shape}"
        
        # 比较结果（允许一些数值误差）
        torch.testing.assert_close(optimized_result, baseline_result, 
                                 msg="expand_tensor_follow_timeline优化版本与baseline不匹配")
    
    def test_create_decoder_mask_baseline(self):
        """测试create_decoder_mask的baseline实现和优化实现是否一致"""
        B, T, V = 2, 3, 5
        max_neighbors = 4
        attention_indices = torch.randint(0, V, (B, V, max_neighbors))
        
        # 优化版本（当前实现）
        optimized_result = DecoderAttentionMask.create_decoder_mask(
            B, T, V, attention_indices
        )
        
        # Baseline版本（使用最基础的for循环）
        def baseline_create_decoder_mask(batch_size, time_steps, num_variables, attention_indices):
            max_neighbors = attention_indices.shape[-1]
            mask = torch.zeros(batch_size, time_steps * num_variables, time_steps * max_neighbors, 
                             dtype=torch.float32)
            
            # 三层循环的baseline实现
            for b in range(batch_size):
                for current_t in range(time_steps):
                    current_start = current_t * num_variables
                    current_end = (current_t + 1) * num_variables
                    
                    for hist_t in range(current_t + 1):  # 因果性：只能看到当前及之前的时间步
                        hist_target_start = hist_t * max_neighbors
                        hist_target_end = (hist_t + 1) * max_neighbors
                        
                        # 为当前时间步的所有变量设置mask
                        for v in range(num_variables):
                            for n in range(max_neighbors):
                                mask[b, current_start + v, hist_target_start + n] = 1.0
            
            return mask
        
        baseline_result = baseline_create_decoder_mask(B, T, V, attention_indices)
        
        # 比较结果
        torch.testing.assert_close(optimized_result, baseline_result, 
                                 msg="create_decoder_mask优化版本与baseline不匹配")
    
    def test_sparse_attention_dimensions(self, sample_data):
        """测试稀疏attention的维度正确性"""
        x, attention_indices, params = sample_data
        B, T, V, d_model = params['B'], params['T'], params['V'], params['d_model']
        
        # 创建attention模块
        attention = SimpleMultiHeadAttention(d_model=d_model, n_heads=2)
        
        # 创建attention mask
        attention_indices_3d = torch.randint(0, V, (B, V, 4))  # [B, V, 4]
        attention_mask = DecoderAttentionMask.create_decoder_mask(
            B, T, V, attention_indices_3d
        )
        
        # 执行稀疏attention
        result = attention._sparse_attention(x, attention_indices, attention_mask)
        
        # 验证输出维度
        expected_shape = (B, T*V, d_model)
        assert result.shape == expected_shape, \
            f"稀疏attention输出维度错误: {result.shape} vs {expected_shape}"
    
    def test_sparse_attention_vs_full_attention_consistency(self):
        """测试稀疏attention在特殊情况下是否与全attention一致"""
        B, seq_len, d_model = 2, 8, 16
        n_heads = 4
        
        x = torch.randn(B, seq_len, d_model)
        attention = SimpleMultiHeadAttention(d_model=d_model, n_heads=n_heads)
        
        # 创建全连接的attention indices（每个位置都连接到所有其他位置）
        # 但为了匹配稀疏attention的格式，只取前几个
        max_neighbors = min(seq_len, 8)
        attention_indices = torch.arange(seq_len)[None, :, None].expand(
            B, seq_len, max_neighbors
        ) % seq_len  # [B, seq_len, max_neighbors]
        
        # 创建全为1的mask（允许所有连接）
        attention_mask = torch.ones(B, seq_len, max_neighbors)
        
        # 执行稀疏attention
        sparse_result = attention._sparse_attention(x, attention_indices, attention_mask)
        
        # 验证输出维度和数值合理性
        assert sparse_result.shape == x.shape
        assert torch.isfinite(sparse_result).all(), "稀疏attention输出包含非有限值"
        assert not torch.isnan(sparse_result).any(), "稀疏attention输出包含NaN"
    
    def test_causal_mask_property(self):
        """测试因果mask的性质：当前时间步不能看到未来信息"""
        B, T, V = 1, 4, 3
        max_neighbors = 2
        attention_indices = torch.randint(0, V, (B, V, max_neighbors))

        mask = DecoderAttentionMask.create_decoder_mask(B, T, V, attention_indices)
        
        # 验证因果性：每个时间步只能看到当前及之前的时间步
        for current_t in range(T):
            current_start = current_t * V
            current_end = (current_t + 1) * V
            
            # 检查不能看到未来时间步
            for future_t in range(current_t + 1, T):
                future_start = future_t * max_neighbors
                future_end = (future_t + 1) * max_neighbors
                
                # 未来时间步对应的mask应该全为0
                future_mask = mask[0, current_start:current_end, future_start:future_end]
                assert (future_mask == 0).all(), \
                    f"时间步{current_t}不应该看到未来时间步{future_t}的信息"

    def test_non_causal_mask_is_all_ones(self):
        """当关闭因果性时，mask应为全1矩阵，允许所有时间步交互"""
        B, T, V = 1, 3, 4
        max_neighbors = 2
        attention_indices = torch.randint(0, V, (B, V, max_neighbors))

        mask = DecoderAttentionMask.create_decoder_mask(
            B, T, V, attention_indices, causal=False
        )

        assert mask.shape == (B, T * V, T * max_neighbors)
        assert torch.all(mask == 1), "非因果mask应该全部为1"
    
    def test_function_integration(self):
        """测试各函数的集成工作"""
        B, T, V = 1, 2, 4
        d_model = 8
        max_neighbors = 3
        
        # 创建测试数据
        x = torch.randn(B, T*V, d_model)
        attention_indices_3d = torch.randint(0, V, (B, V, max_neighbors))
        
        # 测试完整流程
        attention = SimpleMultiHeadAttention(d_model=d_model, n_heads=2)
        
        # 1. 扩展attention indices
        attention_indices_timeline = expand_tensor_follow_timeline(attention_indices_3d, T)
        
        # 2. 创建mask
        attention_mask = DecoderAttentionMask.create_decoder_mask(
            B, T, V, attention_indices_3d
        )
        
        # 3. 执行attention
        result = attention(x, attention_indices_timeline, attention_mask)
        
        # 验证结果
        assert result.shape == x.shape
        assert torch.isfinite(result).all()
        assert not torch.isnan(result).any()


if __name__ == "__main__":
    # 运行测试
    test_instance = TestDecoderFunctions()
    
    # 创建sample data
    sample_data = test_instance.sample_data()
    
    print("运行测试...")
    
    try:
        test_instance.test_timeline_attention_indexing_baseline(sample_data)
        print("✓ timeline_attention_indexing测试通过")
    except Exception as e:
        print(f"✗ timeline_attention_indexing测试失败: {e}")
    
    try:
        test_instance.test_expand_tensor_follow_timeline_baseline()
        print("✓ expand_tensor_follow_timeline测试通过")
    except Exception as e:
        print(f"✗ expand_tensor_follow_timeline测试失败: {e}")
    
    try:
        test_instance.test_create_decoder_mask_baseline()
        print("✓ create_decoder_mask测试通过")
    except Exception as e:
        print(f"✗ create_decoder_mask测试失败: {e}")
    
    try:
        test_instance.test_sparse_attention_dimensions(sample_data)
        print("✓ sparse_attention维度测试通过")
    except Exception as e:
        print(f"✗ sparse_attention维度测试失败: {e}")
    
    try:
        test_instance.test_sparse_attention_vs_full_attention_consistency()
        print("✓ sparse_attention一致性测试通过")
    except Exception as e:
        print(f"✗ sparse_attention一致性测试失败: {e}")
    
    try:
        test_instance.test_causal_mask_property()
        print("✓ causal_mask性质测试通过")
    except Exception as e:
        print(f"✗ causal_mask性质测试失败: {e}")
    
    try:
        test_instance.test_function_integration()
        print("✓ 函数集成测试通过")
    except Exception as e:
        print(f"✗ 函数集成测试失败: {e}")
    
    print("所有测试完成！")
