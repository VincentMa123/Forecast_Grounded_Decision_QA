#!/usr/bin/env python3
"""
缓存构建脚本 - 用于首次构建或重建数据缓存 (v2.1格式)

## 快速使用
python build_cache.py                           # 构建100%数据集
python build_cache.py --percentage 0.15        # 构建15%数据集
python build_cache.py --force                  # 强制重建缓存
python build_cache.py --info-only              # 仅查看缓存信息
python build_cache.py --read-test              # 测试缓存读取性能

## v2.1格式优化说明
- 存储原始样本数据 [1439, 6712] 而非预创建序列
- 显著减少存储空间：从117GB降至15-20GB (节省约85%)
- 在Dataset中动态创建序列，支持任意sequence_length
- 无需预设序列长度：训练时可灵活指定任意序列长度
- 预计算token ids到独立的Parquet文件，避免训练阶段的tokenizer开销

## 参数说明
--percentage, -p: 使用的数据百分比，范围(0.0, 1.0]，默认1.0即100%
--cache-dir: 自定义缓存目录路径，默认自动根据百分比命名
--force: 强制重建缓存，即使现有缓存有效
--info-only: 仅显示缓存信息，不进行构建
--strict-nan-check: 发现NaN值时终止构建（默认仅警告）
--read-test: 测试缓存读取性能
--data-dir: 指定数据目录路径（默认为./data）

## 使用示例
# 构建完整数据集
python build_cache.py

# 构建15%数据集并指定缓存目录
python build_cache.py --percentage 0.15 --cache-dir data/cache_small

# 强制重建现有缓存
python build_cache.py --force

# 查看现有缓存信息
python build_cache.py --info-only

# 测试缓存读取性能
python build_cache.py --read-test

## v2.1格式特点
- 原始样本存储: 每个样本存储为 [1439, 6712] 的完整数据
- 动态序列创建: Dataset根据需要的sequence_length动态创建序列
- 版本检测: 自动检测缓存格式版本，旧格式需重建
- 存储优化: 使用Parquet格式，大幅减少磁盘占用
- 灵活性: 无需在构建时指定序列长度

## NaN检查功能
默认情况下，构建缓存时会检查数据中的NaN值并记录警告，但继续构建。
使用 --strict-nan-check 参数可以在发现NaN时终止构建过程。
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.cache_manager import CacheManager
from data.processor import DataProcessor
from data.tokenizer_save import load_tokenizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('build_cache.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='构建fluid模型数据缓存 (v2.1格式)')
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重建缓存'
    )
    parser.add_argument(
        '--info-only',
        action='store_true',
        help='仅显示缓存信息'
    )
    parser.add_argument(
        '--read-test',
        action='store_true',
        help='测试缓存读取性能'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='数据目录路径 (默认: ./data)'
    )
    parser.add_argument(
        '--static-dir',
        type=str,
        default=None,
        help='静态拓扑/变量目录，默认 data/static/full'
    )
    parser.add_argument(
        '--strict-nan-check',
        action='store_true',
        help='发现NaN值时终止构建（默认仅警告）'
    )
    parser.add_argument(
        '--percentage', '-p',
        type=float,
        default=1.0,
        help='使用的数据百分比，范围(0.0, 1.0]，默认1.0即100%%'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='自定义缓存目录路径（默认根据 static-dir 设置在 static/<graph>/cache[*]）'
    )
    parser.add_argument(
        '--skip-tokens',
        action='store_true',
        help='仅生成原始样本缓存，不输出token缓存文件。'
    )
    parser.add_argument(
        '--skip-sequences',
        action='store_true',
        help='跳过序列Parquet写入，仅生成token/元数据。'
    )

    args = parser.parse_args()

    try:
        data_dir = Path(args.data_dir)
        static_dir = Path(args.static_dir) if args.static_dir else data_dir / "static" / "full"

        if args.cache_dir:
            cache_dir = Path(args.cache_dir)
        else:
            if args.percentage < 1.0:
                percentage_str = f"cache_{int(args.percentage * 100)}pct"
            else:
                percentage_str = "cache"
            cache_dir = static_dir / percentage_str
        # 初始化缓存管理器
        logger.info(f"初始化缓存管理器 (v2.1格式)")
        logger.info(f"  数据目录: {data_dir}")
        logger.info(f"  缓存目录: {cache_dir}")
        logger.info(f"  静态目录: {static_dir}")
        logger.info(f"  数据百分比: {args.percentage * 100:.1f}%")

        cache_manager = CacheManager(data_dir=data_dir, cache_dir=cache_dir, static_dir=static_dir)

        # 如果只是查看信息
        if args.info_only:
            cache_info = cache_manager.get_cache_info()
            print("\n=== 缓存信息 ===")
            for key, value in cache_info.items():
                print(f"{key}: {value}")
            return

        # 如果是测试缓���读取
        if args.read_test:
            cache_info = cache_manager.get_cache_info()
            if not cache_info.get('cache_valid', False):
                logger.error("缓存不存在或无效，请先构建缓存")
                print("错误：缓存不存在或无效，请先运行构建缓存")
                return

            print("=== 缓存基本信息 ===")
            for key, value in cache_info.items():
                print(f"{key}: {value}")

            # 测试缓存读取性能
            print("\n=== 缓存读取性能测试 ===")
            test_iterations = 5
            read_times = []

            for i in range(test_iterations):
                start_time = time.time()
                cached_data = cache_manager.load_cached_data("train")
                end_time = time.time()
                read_time = end_time - start_time
                read_times.append(read_time)

                print(f"第{i+1}次读取: {read_time:.4f}秒")

                # 输出数据基本信息（只在第一次迭代时输出）
                if i == 0 and cached_data:
                    print(f"\n=== 缓存数据信息 (v2.1格式) ===")
                    if isinstance(cached_data, dict):
                        print(f"缓存键数量: {len(cached_data.keys())}")
                        print(f"缓存键: {list(cached_data.keys())}")

                        for key, value in cached_data.items():
                            if hasattr(value, 'shape'):
                                print(f"{key} shape: {value.shape}")
                            elif hasattr(value, '__len__'):
                                print(f"{key} length: {len(value)}")
                            else:
                                print(f"{key} type: {type(value)}")

                        # 检查是否是v2.1格式的样本数据
                        if 'samples' in cached_data and hasattr(cached_data['samples'], '__len__'):
                            print(f"样本数量: {len(cached_data['samples'])}")
                            if len(cached_data['samples']) > 0:
                                sample_shape = cached_data['samples'][0].get('data', np.array([])).shape
                                print(f"样本数据形状: {sample_shape}")
                    else:
                        print(f"缓存数据类型: {type(cached_data)}")

            # 计算统计���息
            avg_time = sum(read_times) / len(read_times)
            min_time = min(read_times)
            max_time = max(read_times)

            print(f"\n=== 性能统计 ===")
            print(f"平均读取时间: {avg_time:.4f}秒")
            print(f"最快读取时间: {min_time:.4f}秒")
            print(f"最慢读取时间: {max_time:.4f}秒")

            logger.info("缓存读取测试完成！")
            return

        # 检查是否需要构建缓存
        cache_info = cache_manager.get_cache_info()
        cache_valid = cache_info.get('cache_valid', False)

        if cache_valid and not args.force:
            logger.info("缓存已存在且有效，跳过构建")
            print("\n=== 现有缓存信息 ===")
            print(f"缓存目录: {cache_dir}")
            for key, value in cache_info.items():
                print(f"{key}: {value}")
            print("\n使用 --force 参数可以强制重建缓存")
            return

        if args.force:
            logger.info("强制重建缓存...")
            existing_sequences = cache_manager.train_cache_path.exists() or cache_manager.val_cache_path.exists()
            existing_tokens = cache_manager.train_tokens_path.exists() or cache_manager.val_tokens_path.exists()
            preserve_sequences = existing_sequences and (args.skip_sequences or not args.skip_tokens)
            preserve_tokens = existing_tokens and args.skip_tokens
            if preserve_sequences:
                logger.info("保留现有序列缓存文件")
            if preserve_tokens:
                logger.info("保留现有token缓存文件")
            cache_manager.clear_cache(preserve_sequences=preserve_sequences, preserve_tokens=preserve_tokens)

        # 初始化数据处理器
        logger.info("初始化数据处理器...")
        processor = DataProcessor(data_dir=data_dir)

        # 构建缓存
        logger.info(f"开始构建缓存 (v2.1格式)")
        logger.info(f"  数据百分比: {args.percentage * 100:.1f}%")
        logger.info(f"  缓存目录: {cache_dir}")
        logger.info(f"  存储格式: 原始样本数据 [1439, 6712]")
        logger.info(f"  预期存储节省: ~85% (117GB → 15-20GB)")

        tokenizer = None
        if args.skip_tokens:
            logger.info("跳过token缓存生成，仅保存原始样本数据。")
        else:
            if args.skip_sequences:
                logger.info("跳过序列Parquet写入，仅维护token/元数据。现有序列文件将被保留。")
                missing_sequences = not cache_manager.train_cache_path.exists() or not cache_manager.val_cache_path.exists()
                if missing_sequences:
                    raise ValueError(
                        "无法跳过序列写入：未检测到现有的 train/val 序列 Parquet。"
                        " 请先在目标静态目录运行一次不带 --skip-sequences 的构建以生成序列缓存。"
                    )

            logger.info("加载tokenizer统计信息用于token缓存...")
            search_dirs = [static_dir, data_dir]
            for candidate in search_dirs:
                tokenizer = load_tokenizer(candidate)
                if tokenizer is not None and getattr(tokenizer, 'fitted', False):
                    logger.info("使用tokenizer统计目录: %s", candidate / 'tokenizer_save')
                    break
            if tokenizer is None or not getattr(tokenizer, 'fitted', False):
                raise RuntimeError(
                    "Tokenizer statistics not found. 请先运行 data/compute_tokenizer_stats.py (可使用 --static-dir) 构建tokenizer。"
                )

        # 记录构建时间
        build_start_time = time.time()
        cache_manager.build_cache(
            processor,
            tokenizer,
            strict_nan_check=args.strict_nan_check,
            sample_percentage=args.percentage,
            save_sequences=not args.skip_sequences
        )
        build_end_time = time.time()
        build_time = build_end_time - build_start_time

        # 显示构建结果
        cache_info = cache_manager.get_cache_info()
        print("\n=== 缓存构建完成 ===")
        print(f"构建时间: {build_time:.4f}秒")
        for key, value in cache_info.items():
            print(f"{key}: {value}")

        logger.info(f"缓存构建完成！耗时: {build_time:.4f}秒")

    except Exception as e:
        logger.error(f"缓存构建失败: {e}")
        sys.exit(1)

def main_for_read_cache():
    """读取缓存并测试性能"""
    parser = argparse.ArgumentParser(description='读取并测试fluid模型数据缓存')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='数据目录路径 (默认: ./data)'
    )
    parser.add_argument(
        '--static-dir',
        type=str,
        default=None,
        help='静态拓扑/变量目录，默认 data/static/full'
    )
    parser.add_argument(
        '--test-iterations',
        type=int,
        default=3,
        help='测试迭代次数 (默认: 3)'
    )

    args = parser.parse_args()

    try:
        # 初始化缓存管理器
        logger.info(f"初始化缓存管理器，数据目录: {args.data_dir}")
        static_dir = Path(args.static_dir) if args.static_dir else Path(args.data_dir) / "static" / "full"
        cache_manager = CacheManager(data_dir=args.data_dir, static_dir=static_dir)

        # 检查缓存是否存在
        cache_info = cache_manager.get_cache_info()
        if not cache_info.get('cache_valid', False):
            logger.error("缓存不存在或无效，请先运行构建缓存")
            print("错误：缓存不存在或无效，请先运行: python build_cache.py")
            return

        print("=== 缓存基本信息 ===")
        for key, value in cache_info.items():
            print(f"{key}: {value}")

        # 测试缓存读取性能
        print(f"\n=== 缓存读取性能测试 (迭代{args.test_iterations}次) ===")

        read_times = []
        for i in range(args.test_iterations):
            start_time = time.time()

            # 读取缓存数据
            cached_data = cache_manager.load_cached_data("test")

            end_time = time.time()
            read_time = end_time - start_time
            read_times.append(read_time)

            print(f"第{i+1}次读取: {read_time:.4f}秒")

            # 输出数据基本信息（只在第一次迭代时输出）
            if i == 0 and cached_data:
                print(f"\n=== 缓存数据信息 (v2.1格式) ===")
                if isinstance(cached_data, dict):
                    print(f"缓存键数量: {len(cached_data.keys())}")
                    print(f"缓存键: {list(cached_data.keys())}")

                    for key, value in cached_data.items():
                        if hasattr(value, 'shape'):
                            print(f"{key} shape: {value.shape}")
                        elif hasattr(value, '__len__'):
                            print(f"{key} length: {len(value)}")
                        else:
                            print(f"{key} type: {type(value)}")

                    # 检查是否是v2.1格式的样本数据
                    if 'samples' in cached_data and hasattr(cached_data['samples'], '__len__'):
                        print(f"样本数量: {len(cached_data['samples'])}")
                        if len(cached_data['samples']) > 0:
                            sample_shape = cached_data['samples'][0].get('data', np.array([])).shape
                            print(f"样本数据形状: {sample_shape}")
                else:
                    print(f"缓存数据类型: {type(cached_data)}")

        # 计算统计信息
        avg_time = sum(read_times) / len(read_times)
        min_time = min(read_times)
        max_time = max(read_times)

        print(f"\n=== 性能统计 ===")
        print(f"平均读取时间: {avg_time:.4f}秒")
        print(f"最快读取时间: {min_time:.4f}秒")
        print(f"最慢读取时间: {max_time:.4f}秒")

        logger.info("缓存读取测试完成！")

    except Exception as e:
        logger.error(f"缓存读取测试失败: {e}")
        raise

if __name__ == "__main__":
    main()
