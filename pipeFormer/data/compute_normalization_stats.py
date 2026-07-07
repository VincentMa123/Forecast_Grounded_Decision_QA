
"""
归一化统计量计算工具 - 从静态目录的 cache 中读取数据并计算归一化统计量，保存到CSV文件。

使用方法:
    python data/compute_normalization_stats.py --static_dir data/static/full --method standard 

按照CLAUDE.md的要求：
1. 从cache文件夹读取parquet文件
2. 将list[T, 6712]转换成[case_length*T, 6712]
3. 计算Z-score标准化所需的统计量：mean, std
4. 同时计算其他统计量：min, max, q25, q75, median（方便扩展）
5. 保存到normalization_stats_{method}.csv，行是变量名

参数:
    --static_dir: 静态目录
    --method: 归一化方法 (standard, minmax, robust)
    --max_samples: 最大样本数（用于控制内存使用）
    --output_name: 输出文件名（可选）
    --validate: 验证保存的统计量
    --force: 强制重新计算即使文件已存在
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import logging
import argparse
import sys
from typing import List, Optional, Union
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.normalizer import DataNormalizer




def compute_statistics_streaming(parquet_paths: Union[Path, List[Path]], static_dir: Optional[Path] = None) -> dict:
    """
    使用流式计算从一个或多个parquet文件计算所有统计量，避免内存溢出。

    Args:
        parquet_paths: parquet文件路径或路径列表
        static_dir: 静态目录路径（用于加载变量名）

    Returns:
        统计量字典，包含mean, std, min, max, q25, q75, median
    """
    if isinstance(parquet_paths, (list, tuple)):
        paths = [Path(p) for p in parquet_paths]
    else:
        paths = [Path(parquet_paths)]

    logger.info(
        "Computing statistics from %d parquet file(s) using streaming...",
        len(paths),
    )
    logger.info("Target parquet files: %s", ", ".join(str(p) for p in paths))

    n_total = 0
    total_dims: Optional[int] = None
    sum_x: Optional[np.ndarray] = None
    sum_x2: Optional[np.ndarray] = None
    global_min: Optional[np.ndarray] = None
    global_max: Optional[np.ndarray] = None

    max_samples_for_quantiles = 100000
    samples_for_quantiles: List[np.ndarray] = []

    logger.info("Phase 1: Computing mean, std, min, max from raw sample data...")

    for parquet_path in paths:
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        logger.info("Reading parquet file: %s", parquet_path)
        pf = pq.ParquetFile(parquet_path)
        num_row_groups = pf.num_row_groups

        for rg_idx in range(num_row_groups):
            rg = pf.read_row_group(rg_idx)
            df = rg.to_pandas()

            if not {'shape_0', 'shape_1', 'data'}.issubset(df.columns):
                raise ValueError(
                    "Expected cache format with 'data', 'shape_0', 'shape_1' columns, "
                    f"found columns: {list(df.columns)}"
                )

            for _, row in tqdm(
                df.iterrows(),
                desc=f"[{parquet_path.name}] Processing RG {rg_idx + 1}/{num_row_groups}",
                leave=False,
                total=len(df),
            ):
                data_bytes = row['data']
                shape_0 = int(row['shape_0'])
                shape_1 = int(row['shape_1'])
                data = np.frombuffer(data_bytes, dtype=np.float32).reshape((shape_0, shape_1))

                if total_dims is None:
                    total_dims = shape_1
                    sum_x = np.zeros(total_dims, dtype=np.float64)
                    sum_x2 = np.zeros(total_dims, dtype=np.float64)
                    global_min = np.full(total_dims, np.inf, dtype=np.float64)
                    global_max = np.full(total_dims, -np.inf, dtype=np.float64)
                elif shape_1 != total_dims:
                    raise ValueError(
                        f"Inconsistent feature dimension detected. Expected {total_dims}, got {shape_1}" \
                        f" (file={parquet_path}, row_group={rg_idx})."
                    )

                assert sum_x is not None and sum_x2 is not None
                assert global_min is not None and global_max is not None

                sum_x += np.sum(data, axis=0, dtype=np.float64)
                sum_x2 += np.sum(data ** 2, axis=0, dtype=np.float64)
                n_total += data.shape[0]

                batch_min = np.min(data, axis=0)
                batch_max = np.max(data, axis=0)
                global_min = np.minimum(global_min, batch_min)
                global_max = np.maximum(global_max, batch_max)

                if len(samples_for_quantiles) < max_samples_for_quantiles:
                    n_timesteps = data.shape[0]
                    if n_timesteps > 100:
                        sample_indices = np.random.choice(n_timesteps, 100, replace=False)
                        samples_for_quantiles.append(data[sample_indices])
                    else:
                        samples_for_quantiles.append(data)

    if total_dims is None or sum_x is None or sum_x2 is None or global_min is None or global_max is None:
        raise RuntimeError("No data found in the provided parquet files; normalization stats cannot be computed.")

    mean = sum_x / n_total
    variance = (sum_x2 / n_total) - (mean ** 2)
    variance[variance < 0] = 0
    std = np.sqrt(variance)

    if static_dir is not None:
        feature_names = load_variable_names(static_dir)
    else:
        feature_names = [f"feature_{i:04d}" for i in range(total_dims)]

    nan_std_mask = np.isnan(std)
    if np.any(nan_std_mask):
        nan_std_indices = np.where(nan_std_mask)[0]
        nan_std_names = [feature_names[i] if i < len(feature_names) else f"feature_{i:04d}" for i in nan_std_indices]
        logger.warning(
            "Found %d columns with NaN std: %s%s",
            len(nan_std_indices),
            nan_std_names[:10],
            '...' if len(nan_std_indices) > 10 else '',
        )
        std[nan_std_mask] = 1.0

    zero_std_mask = (std == 0) & (~nan_std_mask)
    if np.any(zero_std_mask):
        zero_std_indices = np.where(zero_std_mask)[0]
        zero_std_names = [feature_names[i] if i < len(feature_names) else f"feature_{i:04d}" for i in zero_std_indices]
        logger.warning(
            "Found %d constant columns (std=0): %s%s",
            len(zero_std_indices),
            zero_std_names[:10],
            '...' if len(zero_std_indices) > 10 else '',
        )
        std[zero_std_mask] = 1.0

    const_mask = global_min == global_max
    if np.any(const_mask):
        const_indices = np.where(const_mask)[0]
        const_names = [feature_names[i] if i < len(feature_names) else f"feature_{i:04d}" for i in const_indices]
        logger.warning(
            "Found %d constant columns for min/max: %s%s",
            len(const_indices),
            const_names[:10],
            '...' if len(const_indices) > 10 else '',
        )
        global_max[const_mask] = global_min[const_mask] + 1.0

    logger.info("Processed %d total timesteps from all samples", n_total)
    logger.info("Mean range: [%.4f, %.4f]", mean.min(), mean.max())
    logger.info("Std range: [%.4f, %.4f]", std.min(), std.max())

    if not samples_for_quantiles:
        raise RuntimeError("No samples collected for quantiles; ensure cache contains data.")

    logger.info("Phase 2: Computing quantiles from %d sampled time steps...", len(samples_for_quantiles))
    combined_samples = np.vstack(samples_for_quantiles)
    logger.info("Combined samples shape for quantiles: %s", combined_samples.shape)

    median = np.median(combined_samples, axis=0)
    q25 = np.percentile(combined_samples, 25, axis=0)
    q75 = np.percentile(combined_samples, 75, axis=0)

    iqr = q75 - q25
    zero_iqr_mask = iqr == 0
    if np.any(zero_iqr_mask):
        logger.warning("Found %d columns with zero IQR", int(np.sum(zero_iqr_mask)))
        q75[zero_iqr_mask] = q25[zero_iqr_mask] + 1.0

    logger.info("Median range: [%.4f, %.4f]", median.min(), median.max())
    logger.info("IQR range: [%.4f, %.4f]", iqr.min(), iqr.max())

    return {
        'mean': mean,
        'std': std,
        'min': global_min,
        'max': global_max,
        'q25': q25,
        'q75': q75,
        'median': median,
        'n_samples': n_total
    }


def load_variable_names(static_dir: Path) -> List[str]:
    """
    从index_variable_mapping.csv文件加载真实的变量名。

    Args:
        static_dir: 静态目录路径

    Returns:
        变量名列表，长度6712
    """
    candidates = [
        static_dir / "index_variable_mapping.csv",
        static_dir.parent / "index_variable_mapping.csv",
        static_dir.parent.parent / "index_variable_mapping.csv" if static_dir.parent.parent != static_dir.parent else None,
    ]

    for mapping_path in candidates:
        if mapping_path is not None and mapping_path.exists():
            logger.info("Loading variable names from %s", mapping_path)
            mapping_df = pd.read_csv(mapping_path)
            feature_names = mapping_df['variable_name'].astype(str).tolist()
            logger.info("Loaded %d variable names", len(feature_names))
            return feature_names

    logger.warning("Variable mapping file not found near %s; using index names", static_dir)
    return []


def save_stats_to_csv(stats: dict, static_dir: Path, method: str = 'standard', output_name: Optional[str] = None) -> Path:
    """
    将统计量保存到CSV文件，行是真实的变量名，列是统计量类型。

    Args:
        stats: 统计量字典
        static_dir: 静态目录路径
        method: 归一化方法
        output_name: 输出文件名（可选）

    Returns:
        输出文件路径
    """
    output_dir = (static_dir / "normalizer_save").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        output_name = f"normalization_stats_{method}.csv"
    output_path = output_dir / output_name

    logger.info("Saving statistics to %s...", output_path)

    feature_names = load_variable_names(static_dir)
    n_features = len(stats['mean'])

    if len(feature_names) != n_features:
        logger.warning(
            "Variable names count (%d) doesn't match features count (%d)",
            len(feature_names),
            n_features,
        )
        if len(feature_names) > n_features:
            feature_names = feature_names[:n_features]
        else:
            feature_names.extend([f"feature_{i:04d}" for i in range(len(feature_names), n_features)])

    df_dict = {
        'variable_name': feature_names,
        'normalization_method': [method] * n_features,
        'mean': stats['mean'],
        'std': stats['std'],
        'min': stats['min'],
        'max': stats['max'],
        'q25': stats['q25'],
        'median': stats['median'],
        'q75': stats['q75'],
    }

    df = pd.DataFrame(df_dict)
    df.to_csv(output_path, index=False, float_format='%.6f')

    logger.info("Statistics saved successfully!")
    logger.info("CSV shape: %s", df.shape)
    logger.info("Columns: %s", list(df.columns))
    if feature_names:
        logger.info("Sample variable names: %s", feature_names[:10])
        logger.info("Last variable names: %s", feature_names[-10:])

    return output_path


def validate_stats(static_dir: str, method: str, stats_file: str = None) -> bool:
    """
    验证保存的统计量是否正确。

    Args:
        static_dir: 静态目录路径
        method: 归一化方法
        stats_file: 统计量文件名

    Returns:
        验证是否通过
    """
    logger.info("Validating saved normalization stats...")

    try:
        # 加载统计量
        normalizer = DataNormalizer(static_dir, method=method)
        if not normalizer.load_stats(stats_file):
            logger.error("Failed to load stats file")
            return False

        # 检查统计量
        stats_summary = normalizer.get_stats_summary()
        logger.info(f"Loaded stats summary: {stats_summary}")

        # 基本验证
        if not hasattr(normalizer, 'mean_') or normalizer.mean_ is None:
            logger.error("Normalizer not properly loaded")
            return False

        # 测试变换
        test_data = np.random.randn(10, normalizer.total_dims).astype(np.float32)

        # 正向变换
        normalized = normalizer.transform(test_data)
        if normalized.shape != test_data.shape:
            logger.error(f"Transform shape mismatch: expected {test_data.shape}, got {normalized.shape}")
            return False

        # 反向变换
        denormalized = normalizer.inverse_transform(normalized)
        if not np.allclose(test_data, denormalized, rtol=1e-5, atol=1e-6):
            logger.error("Inverse transform failed - data not recovered correctly")
            return False

        logger.info("Stats validation passed!")
        return True

    except Exception as e:
        logger.error(f"Stats validation failed: {e}")
        return False


def compute_and_save_normalizer_stats(static_dir: str,
                                      method: str = 'standard',
                                      output_name: Optional[str] = None) -> str:
    """
    计算并保存归一化统计量。

    Args:
        static_dir: 静态目录
        method: 归一化方法
        output_name: 输出文件名

    Returns:
        保存文件的路径
    """
    logger.info(f"Computing normalization stats using {method} method...")

    static_path = Path(static_dir).resolve()
    cache_dir = static_path / "cache"
    train_parquet = cache_dir / "train_sequences.parquet"
    val_parquet = cache_dir / "val_sequences.parquet"

    # 检查文件是否存在
    if not train_parquet.exists():
        raise FileNotFoundError(f"Train sequences not found: {train_parquet}")

    parquet_paths: List[Path] = [train_parquet]
    if val_parquet.exists():
        logger.info(f"Including validation sequences for stats: {val_parquet}")
        parquet_paths.append(val_parquet)
    else:
        logger.info("Validation sequences not found; using training data only.")

    stats = compute_statistics_streaming(parquet_paths, static_path)

    output_path = save_stats_to_csv(stats, static_path, method, output_name)

    logger.info("=" * 60)
    logger.info(f"Summary:")
    logger.info(f"  Total samples processed: {stats['n_samples']:,}")
    logger.info(f"  Output file: {output_path}")
    logger.info("=" * 60)

    return str(output_path)


def main():
    """主函数：从cache计算统计量并保存到CSV"""
    parser = argparse.ArgumentParser(description='Compute normalization statistics from cached data')

    parser.add_argument('--static_dir', type=str, default=str(Path(__file__).parent / "static" / "full"),
                       help='Path to static directory containing cache and mappings (default: data/static/full).')
    parser.add_argument('--method', type=str, default='standard',
                       choices=['standard', 'minmax', 'robust'],
                       help='Normalization method (default: standard)')
    parser.add_argument('--output_name', type=str, default=None,
                       help='Output filename (optional)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate saved stats after computation')
    parser.add_argument('--force', action='store_true',
                       help='Force recomputation even if stats file exists')

    args = parser.parse_args()

    try:
        # 检查数据目录
        static_dir = Path(args.static_dir)
        if not static_dir.exists():
            logger.error(f"Static directory not found: {static_dir}")
            return 1

        # 确定输出文件名
        if args.output_name is None:
            stats_filename = f"normalization_stats_{args.method}.csv"
        else:
            stats_filename = args.output_name

        stats_path = static_dir / "normalizer_save" / stats_filename

        # 检查是否已存在统计量文件
        if stats_path.exists() and not args.force:
            logger.info(f"Stats file already exists: {stats_path}")

            if args.validate:
                if validate_stats(str(static_dir), args.method, args.output_name):
                    logger.info("Existing stats file is valid")
                    return 0
                else:
                    logger.error("Existing stats file is invalid")
                    return 1
            else:
                logger.info("Use --force to recompute, or --validate to check existing file")
                return 0

        # 计算统计量
        save_path = compute_and_save_normalizer_stats(
            static_dir=str(static_dir),
            method=args.method,
            output_name=args.output_name
        )

        logger.info(f"Normalization stats saved to: {save_path}")

        # 可选验证
        if args.validate:
            if validate_stats(str(static_dir), args.method, args.output_name):
                logger.info("Stats validation successful!")
            else:
                logger.error("Stats validation failed!")
                return 1

        logger.info("Stats computation completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error during stats computation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
