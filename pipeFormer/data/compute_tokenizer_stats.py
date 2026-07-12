"""
离散化 tokenizer 统计量计算工具。

使用说明:
    python data/compute_tokenizer_stats.py --data_dir data
python data/compute_tokenizer_stats.py --data_dir data --variable_fraction 0.1 --output_name token_stats_fraction0.1.csv
流程:
1. 从 cache 目录读取原始样本数据（默认包含 train + val/test）
2. 将所有样本的 [T, V] 数据送入新的 DataTokenizer
3. 识别常值/二值变量，余量按 1% 分位范围生成共享词表
4. 保存 token 元数据 CSV（列即 token_metadata 中字段）

注意:
    - 运行前请确保 cache 已构建，可通过 data/manage_cache.py 或 CacheManager 构建
    - 若只想使用训练集，可添加 --train_only
"""

import argparse
import logging
import secrets
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

# 将项目根目录加入路径，方便脚本直接运行
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from data.cache_manager import CacheManager
from data.tokenizer_save import DataTokenizer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _load_hyperparams(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_hyperparams(path: Path, updates: Dict[str, Any]) -> None:
    data = _load_hyperparams(path)
    for section, values in updates.items():
        if isinstance(values, dict):
            target = data.setdefault(section, {})
            target.update(values)
        else:
            data[section] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    static_dir: Optional[Path] = Path(args.static_dir).resolve() if args.static_dir else None
    if static_dir is not None and not static_dir.exists():
        raise FileNotFoundError(f"Static directory not found: {static_dir}")

    hyper_path = static_dir / "graph_hyperparameters.json" if static_dir else None
    existing_hyper = _load_hyperparams(hyper_path) if hyper_path else {}
    tokenizer_defaults = existing_hyper.get("tokenizer", {}) if existing_hyper else {}

    for arg_name, key in [
        ("constant_freq_threshold", "constant_freq_threshold"),
        ("constant_variable_threshold", "constant_variable_threshold"),
        ("quantile_step", "quantile_step"),
        ("quantile_method", "quantile_method"),
        ("range_gap_epsilon", "range_gap_epsilon"),
        ("round_gap", "round_gap"),
    ]:
        if getattr(args, arg_name) is None and key in tokenizer_defaults:
            setattr(args, arg_name, tokenizer_defaults[key])

    stats_filename = args.output_name or "token_stats.csv"
    stats_dir = static_dir / "tokenizer_save" if static_dir else data_dir / "tokenizer_save"
    stats_path = stats_dir / stats_filename

    if not args.force and stats_path.exists():
        raise FileExistsError(
            f"Output file already exists ({stats_path}). Use --force to overwrite."
        )

    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None
    if cache_dir is None and static_dir is not None:
        cache_dir = static_dir / "cache"
    cache_manager = CacheManager(str(data_dir), cache_dir=cache_dir, static_dir=str(static_dir) if static_dir else None)

    splits = ['train']
    if not args.train_only:
        splits.append('val')

    logger.info(f"Computing tokenizer stats from cache splits: {splits}")

    cached_arrays: List[np.ndarray] = []
    total_samples = 0

    for split in splits:
        try:
            samples_data, _, _ = cache_manager.load_cached_data(split, include_tokens=False)
        except Exception as exc:
            logger.warning(f"Failed to load cached data for split '{split}': {exc}")
            continue

        valid_arrays = [sample['data'] for sample in samples_data if sample.get('data') is not None]
        if not valid_arrays:
            continue
        cached_arrays.extend(valid_arrays)
        total_samples += len(valid_arrays)

    if not cached_arrays:
        raise RuntimeError("No samples available for tokenizer fitting.")

    logger.info(f"Loaded {total_samples} samples into memory for tokenizer fitting.")

    flat_arrays = [arr.reshape(-1, arr.shape[-1]).astype(np.float32, copy=False) for arr in cached_arrays]
    data_matrix = np.concatenate(flat_arrays, axis=0)
    del flat_arrays
    logger.info(f"Data matrix shape: {data_matrix.shape}, dtype={data_matrix.dtype}, "
                f"approx {data_matrix.nbytes / (1024 ** 3):.2f} GB")
    cached_arrays.clear()
    del cached_arrays

    total_variables = data_matrix.shape[1]
    fraction = args.variable_fraction
    if not (0 < fraction <= 1.0):
        raise ValueError("--variable_fraction must be in the range (0, 1].")

    seed = args.random_seed if args.random_seed is not None else secrets.randbits(64)
    rng = np.random.default_rng(seed)
    logger.info("Variable sampling seed: %s", seed)

    all_indices = np.arange(total_variables)
    if fraction < 1.0 or args.shuffle_all:
        rng.shuffle(all_indices)

    subset_size = total_variables if fraction >= 1.0 else max(1, int(np.ceil(total_variables * fraction)))
    selected_indices = all_indices[:subset_size]

    if static_dir:
        mapping_path = static_dir / "index_variable_mapping.csv"
    else:
        mapping_path = data_dir / "static" / "full" / "index_variable_mapping.csv"
        if not mapping_path.exists():
            mapping_path = data_dir / "index_variable_mapping.csv"
    variable_names: Optional[List[str]] = None
    all_variable_names: Optional[List[str]] = None
    if mapping_path.exists():
        try:
            mapping_df = pd.read_csv(mapping_path)
            all_variable_names = mapping_df.get("variable_name", pd.Series(dtype=str)).astype(str).tolist()
            if len(all_variable_names) != total_variables:
                logger.warning(
                    "index_variable_mapping.csv length mismatch (%d vs %d); falling back to generic names.",
                    len(all_variable_names),
                    total_variables,
                )
                all_variable_names = None
        except Exception as exc:
            logger.warning("Failed to read %s: %s. Falling back to generic names.", mapping_path, exc)

    boundary_mask = None
    if all_variable_names is not None:
        boundary_mask = pd.Series([1 if ':' in name else 0 for name in all_variable_names], dtype=int).values
        variable_names = [all_variable_names[int(idx)] for idx in selected_indices]
    else:
        variable_names = [f"feature_{int(idx):04d}" for idx in selected_indices]

    data_matrix = data_matrix[:, selected_indices]
    if boundary_mask is not None and len(boundary_mask) == total_variables:
        subset_boundary = int(boundary_mask[selected_indices].sum())
    else:
        subset_boundary = int(np.sum(selected_indices < args.boundary_dims))
    subset_equipment = subset_size - subset_boundary
    logger.info(
        "Selected %d variables (boundary=%d, equipment=%d) out of %d (≈%.1f%%).",
        subset_size,
        subset_boundary,
        subset_equipment,
        total_variables,
        (subset_size / total_variables) * 100.0,
    )

    tokenizer = DataTokenizer(
        data_dir=str(data_dir),
        boundary_dims=subset_boundary,
        equipment_dims=subset_equipment,
    )
    tokenizer.set_stats_directory(stats_dir, load_config=True)
    if args.constant_freq_threshold is not None:
        tokenizer.constant_freq_threshold = args.constant_freq_threshold
    if args.constant_variable_threshold is not None:
        tokenizer.constant_variable_threshold = args.constant_variable_threshold
    if args.quantile_step is not None:
        tokenizer.quantile_step = args.quantile_step
    if args.quantile_method:
        tokenizer.quantile_method = args.quantile_method
    if args.range_gap_epsilon is not None:
        tokenizer.range_gap_epsilon = args.range_gap_epsilon
    if args.round_gap is not None:
        tokenizer.set_round_gap(args.round_gap)

    tokenizer.fit(data_matrix, variable_names=variable_names, progress=not args.quiet)

    stats_path = tokenizer.save_stats(
        stats_filename=stats_filename,
    )

    summary = tokenizer.get_stats_summary()
    logger.info(f"Tokenizer summary: {summary}")
    if tokenizer.constant_variables:
        logger.info("Detected %d constant variables. Examples: %s",
                    len(tokenizer.constant_variables),
                    tokenizer.constant_variables[:5])
    if tokenizer.binary_variables:
        logger.info("Detected %d binary variables. Examples: %s",
                    len(tokenizer.binary_variables),
                    tokenizer.binary_variables[:5])
    if tokenizer.identical_pairs:
        logger.info("Detected %d high-match variable pairs.", len(tokenizer.identical_pairs))

    logger.info(f"Saved token stats: {stats_path}")

    if static_dir:
        hyper_path = static_dir / "graph_hyperparameters.json"
        tokenizer_section = {
            "vocab_size": int(tokenizer.vocab_size),
            "boundary_dims": subset_boundary,
            "equipment_dims": subset_equipment,
            "constant_freq_threshold": tokenizer.constant_freq_threshold,
            "constant_variable_threshold": tokenizer.constant_variable_threshold,
            "quantile_step": tokenizer.quantile_step,
            "quantile_method": tokenizer.quantile_method,
            "range_gap_epsilon": tokenizer.range_gap_epsilon,
            "round_gap": getattr(tokenizer, "round_gap", None),
            "stats_dir": stats_dir.relative_to(static_dir).as_posix(),
            "stats_file": stats_filename,
        }
        _write_hyperparams(hyper_path, {"tokenizer": tokenizer_section})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute tokenizer discretization statistics.")
    parser.add_argument("--data_dir", type=str, required=True, help="Root data directory.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Optional cache directory.")
    parser.add_argument("--train_only", action="store_true", help="Use only the training split.")
    parser.add_argument("--boundary_dims", type=int, default=538, help="Boundary feature dimension.")
    parser.add_argument("--equipment_dims", type=int, default=6174, help="Equipment feature dimension.")
    parser.add_argument("--static-dir", type=str, default=None, help="Static directory for specific graph/subgraph.")
    parser.add_argument("--output_name", type=str, default=None, help="Custom output CSV filename.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing stats if they exist.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress bar during fitting.")
    parser.add_argument(
        "--variable_fraction",
        type=float,
        default=1.0,
        help="Fraction (0,1] of variables to include. Values <1 randomly sample variables.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Optional random seed for variable sampling.",
    )
    parser.add_argument(
        "--shuffle_all",
        action="store_true",
        help="Shuffle variable order even when using all variables (fraction==1).",
    )
    parser.add_argument("--constant-freq-threshold", type=float, default=None, help="Override constant frequency threshold.")
    parser.add_argument("--constant-variable-threshold", type=float, default=None, help="Override constant variable threshold.")
    parser.add_argument("--quantile-step", type=float, default=None, help="Override quantile step.")
    parser.add_argument("--quantile-method", type=str, default=None, help="Override quantile method.")
    parser.add_argument("--range-gap-epsilon", type=float, default=None, help="Override range gap epsilon.")
    parser.add_argument("--round-gap", type=float, default=None, help="Override rounding gap.")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
