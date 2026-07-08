from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .schemas import ForecastRow


SOURCE_FILES = {
    "B": "B.csv",
    "C": "C.csv",
    "N": "N.csv",
    "P": "P.csv",
    "R": "R.csv",
    "TE": "T&E.csv",
}


def load_variable_mapping(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        mapping = {}
        for row in reader:
            name = row.get("variable_name")
            if not name:
                continue
            mapping[name] = {
                "index": int(row["index"]),
                "global_index": int(row["global_index"]),
            }
    return mapping


def load_variable_names(path: Path) -> List[str]:
    mapping = load_variable_mapping(path)
    return [name for name, _ in sorted(mapping.items(), key=lambda item: item[1]["index"])]


def find_default_forecast_csv(repo_root: Path) -> Path:
    sample_dir = repo_root / "pipeFormer" / "outputs" / "mock_tiny_decoder" / "sample_predictions"
    preferred = sample_dir / "eval_16.csv"
    if preferred.exists():
        return preferred
    candidates = sorted(sample_dir.glob("eval_*.csv"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No mock PipeFormer sample prediction CSV found under {sample_dir}")
    return candidates[0]


def find_default_checkpoint_dir(repo_root: Path) -> Path:
    preferred = repo_root / "pipeFormer" / "outputs" / "mock_tiny_decoder" / "checkpoint-16"
    if preferred.exists():
        return preferred
    output_dir = repo_root / "pipeFormer" / "outputs" / "mock_tiny_decoder"
    checkpoints = sorted(
        [path for path in output_dir.glob("checkpoint-*") if path.is_dir()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No PipeFormer checkpoint directory found under {output_dir}")
    return checkpoints[0]


def load_forecast_rows(path: Path) -> List[ForecastRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        label_key = reader.fieldnames[0] if reader.fieldnames else ""
        rows: List[ForecastRow] = []
        for row in reader:
            label = str(row.get(label_key) or "").strip()
            values: Dict[str, float] = {}
            for key, value in row.items():
                if key == label_key or value in (None, ""):
                    continue
                try:
                    parsed = float(value)
                except ValueError:
                    continue
                if math.isfinite(parsed):
                    values[key] = parsed
            rows.append(ForecastRow(label=label, values=values))
    if not rows:
        raise ValueError(f"No forecast rows found in {path}")
    return rows


def split_real_predict_rows(rows: List[ForecastRow]) -> Tuple[List[ForecastRow], List[ForecastRow]]:
    real_rows = [row for row in rows if row.label.endswith("_real")]
    predict_rows = [row for row in rows if row.label.endswith("_predict")]
    if not predict_rows:
        raise ValueError("Forecast CSV does not contain *_predict rows.")
    return real_rows, predict_rows


def load_sample_csv_forecast_context(
    parsed_task: Dict[str, Any],
    forecast_csv: Path,
    mapping_path: Path,
) -> Dict[str, Any]:
    variable_mapping = load_variable_mapping(mapping_path)
    changed_variable = parsed_task["changed_variable"]
    if changed_variable not in variable_mapping:
        raise ValueError(f"Parsed variable {changed_variable} is not in mock PipeFormer mapping {mapping_path}")

    rows = load_forecast_rows(forecast_csv)
    real_rows, predict_rows = split_real_predict_rows(rows)
    return {
        "mode": "read_existing_mock_forecast_csv",
        "forecast_csv": forecast_csv.as_posix(),
        "mapping_csv": mapping_path.as_posix(),
        "changed_variable_mapping": variable_mapping[changed_variable],
        "real_rows": real_rows,
        "predict_rows": predict_rows,
    }


def load_mock_forecast_context(
    parsed_task: Dict[str, Any],
    forecast_csv: Path,
    mapping_path: Path,
) -> Dict[str, Any]:
    return load_sample_csv_forecast_context(parsed_task, forecast_csv, mapping_path)


def resolve_relative(path_value: Optional[str], base_dir: Path) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_training_config(checkpoint_dir: Path, pipeformer_root: Path) -> Dict[str, Any]:
    candidates = [checkpoint_dir / "training_config.json", checkpoint_dir.parent / "training_config.json"]
    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as fh:
                config = json.load(fh)
            config["_training_config_path"] = candidate.as_posix()
            return config
    raise FileNotFoundError(f"training_config.json not found near {checkpoint_dir} or {pipeformer_root}")


def add_pipeformer_import_paths(pipeformer_root: Path) -> None:
    for path in (pipeformer_root, pipeformer_root / "data"):
        path_str = str(path.resolve())
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def ensure_optional_matplotlib() -> None:
    import importlib.util
    import types

    if importlib.util.find_spec("matplotlib") is not None:
        return
    matplotlib_stub = types.ModuleType("matplotlib")
    matplotlib_stub.use = lambda *args, **kwargs: None
    pyplot_stub = types.ModuleType("matplotlib.pyplot")
    pyplot_stub.subplots = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("matplotlib is not installed; attention-mask plotting is unavailable.")
    )
    pyplot_stub.close = lambda *args, **kwargs: None
    sys.modules.setdefault("matplotlib", matplotlib_stub)
    sys.modules.setdefault("matplotlib.pyplot", pyplot_stub)

def load_pipeformer_model(checkpoint_dir: Path, pipeformer_root: Path, device: str):
    add_pipeformer_import_paths(pipeformer_root)
    ensure_optional_matplotlib()
    import torch
    from dataclasses import fields
    from models.decoder import DecoderConfig, FluidDecoder

    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        config_path = checkpoint_dir.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"PipeFormer model config not found near {checkpoint_dir}")

    with config_path.open("r", encoding="utf-8") as fh:
        model_config = json.load(fh)

    config_fields = {field.name for field in fields(DecoderConfig)}
    filtered_config = {key: value for key, value in model_config.items() if key in config_fields}
    model = FluidDecoder(DecoderConfig.from_dict(filtered_config))
    weights_path = checkpoint_dir / "pytorch_model.bin"
    if not weights_path.exists():
        weights_path = checkpoint_dir.parent / "pytorch_model.bin"
    if not weights_path.exists():
        raise FileNotFoundError(f"PipeFormer weights not found near {checkpoint_dir}")

    load_kwargs = {"map_location": device}
    try:
        state_dict = torch.load(weights_path, weights_only=True, **load_kwargs)
    except TypeError:
        state_dict = torch.load(weights_path, **load_kwargs)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, model_config, weights_path, config_path


def load_tokenizer(static_dir: Path, vocab_size: Optional[int] = None):
    # Import tokenizer_save as a top-level package from pipeFormer/data to avoid importing data.__init__,
    # which pulls optional training dependencies such as tensordict.
    from tokenizer_save import load_tokenizer as load_tokenizer_from_stats

    tokenizer = load_tokenizer_from_stats(static_dir, vocab_size=vocab_size)
    if tokenizer is None:
        raise RuntimeError(f"Tokenizer statistics not found under {static_dir / 'tokenizer_save'}")
    return tokenizer


def source_file_for_variable(variable_name: str) -> str:
    if variable_name.startswith("T_") and ":BC" in variable_name:
        return "Boundary.csv"
    prefix = variable_name.split("_", 1)[0]
    if prefix in SOURCE_FILES:
        return SOURCE_FILES[prefix]
    raise ValueError(f"No mock CSV source mapping for variable {variable_name}")


def candidate_case_dirs(data_dir: Path, parsed_task: Dict[str, Any]) -> Iterable[Path]:
    case_id = parsed_task.get("case_id") or "mock_test_001"
    digits = "".join(ch for ch in case_id if ch.isdigit()) or "001"
    case_name = f"case_{int(digits):03d}"
    cn_name = f"第{int(digits):03d}个算例"
    dataset_dir = data_dir / "dataset"
    yield dataset_dir / "train" / case_name
    yield dataset_dir / "train" / cn_name
    yield dataset_dir / "test" / case_name
    yield dataset_dir / "test" / cn_name


def resolve_case_dir(data_dir: Path, parsed_task: Dict[str, Any]) -> Path:
    for candidate in candidate_case_dirs(data_dir, parsed_task):
        if candidate.exists():
            return candidate
    candidates = ", ".join(path.as_posix() for path in candidate_case_dirs(data_dir, parsed_task))
    raise FileNotFoundError(f"Could not find mock PipeFormer case directory. Tried: {candidates}")


def load_case_matrix(case_dir: Path, variable_names: List[str]):
    import numpy as np
    import pandas as pd

    source_names = sorted({source_file_for_variable(name) for name in variable_names})
    frames: Dict[str, Any] = {}
    for source_name in source_names:
        source_path = case_dir / source_name
        if not source_path.exists():
            raise FileNotFoundError(f"Required mock case CSV not found: {source_path}")
        frame = pd.read_csv(source_path)
        if "TIME" not in frame.columns:
            raise ValueError(f"CSV missing TIME column: {source_path}")
        frame["TIME"] = pd.to_datetime(frame["TIME"])
        frames[source_name] = frame.set_index("TIME")

    master_source = max(frames.values(), key=lambda frame: len(frame.index))
    master_index = master_source.index
    matrix = np.zeros((len(master_index), len(variable_names)), dtype=np.float32)

    for variable_idx, variable_name in enumerate(variable_names):
        source_name = source_file_for_variable(variable_name)
        source_frame = frames[source_name]
        if variable_name not in source_frame.columns:
            raise ValueError(f"Variable {variable_name} not found in {case_dir / source_name}")
        series = source_frame[variable_name].astype(float)
        if not series.index.equals(master_index):
            series = series.reindex(series.index.union(master_index)).sort_index().interpolate(method="time").reindex(master_index)
            series = series.ffill().bfill()
        matrix[:, variable_idx] = series.to_numpy(dtype=np.float32)

    return matrix, [str(item) for item in master_index]


def apply_condition_to_matrix(matrix, parsed_task: Dict[str, Any], variable_mapping: Dict[str, Dict[str, Any]]):
    import numpy as np

    changed_variable = parsed_task["changed_variable"]
    if changed_variable not in variable_mapping:
        raise ValueError(f"Parsed variable {changed_variable} is not in mock PipeFormer mapping")

    scenario_matrix = np.array(matrix, copy=True)
    percent = parsed_task.get("change_percent")
    if percent is None:
        return scenario_matrix

    direction = parsed_task.get("change_direction")
    factor = 1.0 + float(percent) / 100.0 if direction == "up" else 1.0 - float(percent) / 100.0
    variable_idx = variable_mapping[changed_variable]["index"]
    scenario_matrix[:, variable_idx] *= factor
    return scenario_matrix


def load_attention_indices(static_dir: Path):
    import numpy as np

    csv_path = static_dir / "attention_indices.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"attention_indices.csv not found: {csv_path}")
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header or header[0] != "variable_index":
            raise ValueError(f"Unexpected attention_indices.csv header: {header}")
        for row in reader:
            rows.append([int(value) for value in row[1:] if value != ""])
    return np.asarray(rows, dtype=np.int64)


def load_prediction_mask(static_dir: Path, variable_names: List[str]):
    import numpy as np

    mask_path = static_dir / "prediction_mask.csv"
    if not mask_path.exists():
        cache_mask = static_dir / "cache" / "prediction_mask.npy"
        if cache_mask.exists():
            return np.load(cache_mask).astype(np.int32)
        raise FileNotFoundError(f"prediction_mask.csv not found: {mask_path}")

    mask_by_name = {}
    with mask_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            mask_by_name[str(row["variable_name"])] = int(row["predict"])
    return np.asarray([mask_by_name.get(name, 0) for name in variable_names], dtype=np.int32)


def rows_from_arrays(label_prefix: str, values, variable_names: List[str]) -> List[ForecastRow]:
    rows = []
    for idx, row_values in enumerate(values):
        rows.append(
            ForecastRow(
                label=f"data_line_{idx + 1}_{label_prefix}",
                values={name: round(float(row_values[var_idx]), 6) for var_idx, name in enumerate(variable_names)},
            )
        )
    return rows


def run_checkpoint_inference(
    parsed_task: Dict[str, Any],
    checkpoint_dir: Path,
    pipeformer_root: Path,
    data_dir: Optional[Path],
    static_dir: Optional[Path],
    mapping_path: Path,
    device: str = "cpu",
) -> Dict[str, Any]:
    import numpy as np
    import torch

    checkpoint_dir = checkpoint_dir.resolve()
    pipeformer_root = pipeformer_root.resolve()
    training_config = load_training_config(checkpoint_dir, pipeformer_root)
    data_dir = (data_dir or resolve_relative(training_config.get("data_dir"), pipeformer_root))
    static_dir = (static_dir or resolve_relative(training_config.get("static_dir"), pipeformer_root))
    if data_dir is None or static_dir is None:
        raise ValueError("Could not resolve PipeFormer data_dir/static_dir for checkpoint inference.")

    add_pipeformer_import_paths(pipeformer_root)
    variable_mapping = load_variable_mapping(mapping_path)
    variable_names = load_variable_names(mapping_path)
    changed_variable = parsed_task["changed_variable"]
    if changed_variable not in variable_mapping:
        raise ValueError(f"Parsed variable {changed_variable} is not in mock PipeFormer mapping {mapping_path}")

    sequence_length = int(training_config.get("sequence_length", 3))
    time_step_offset = int(training_config.get("time_step_offset", 1))
    case_dir = resolve_case_dir(data_dir, parsed_task)
    base_matrix, time_labels = load_case_matrix(case_dir, variable_names)
    scenario_matrix = apply_condition_to_matrix(base_matrix, parsed_task, variable_mapping)
    if len(base_matrix) < sequence_length + time_step_offset:
        raise ValueError(f"Case {case_dir} is too short for sequence_length={sequence_length}, offset={time_step_offset}")

    input_values = scenario_matrix[:sequence_length]
    target_values = base_matrix[time_step_offset:time_step_offset + sequence_length]

    tokenizer = load_tokenizer(static_dir, vocab_size=training_config.get("tokenizer_vocab_size"))
    input_tokens = tokenizer.transform_to_tokens(input_values)
    input_tokens = np.asarray(input_tokens, dtype=np.int64)

    model, model_config, weights_path, model_config_path = load_pipeformer_model(checkpoint_dir, pipeformer_root, device)
    attention_indices = load_attention_indices(static_dir)
    prediction_mask = load_prediction_mask(static_dir, variable_names)

    with torch.no_grad():
        input_tensor = torch.as_tensor(input_values, dtype=torch.float32, device=device).unsqueeze(0)
        token_tensor = torch.as_tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(0)
        attention_tensor = torch.as_tensor(attention_indices, dtype=torch.long, device=device).unsqueeze(0)
        mask_tensor = torch.as_tensor(prediction_mask, dtype=torch.float32, device=device).unsqueeze(0)
        outputs = model(
            input_ids=input_tensor,
            input_tokens=token_tensor,
            prediction_mask=mask_tensor,
            attention_indices=attention_tensor,
        )
        token_logits = outputs.get("token_logits") if isinstance(outputs, dict) else None
        if token_logits is None:
            raise RuntimeError("PipeFormer checkpoint inference did not return token_logits.")
        predicted_tokens = torch.argmax(token_logits, dim=-1)
        decoded = tokenizer.tokens_to_values(predicted_tokens)
        if isinstance(decoded, torch.Tensor):
            predictions = decoded.squeeze(0).detach().cpu().numpy()
        else:
            predictions = np.asarray(decoded, dtype=np.float32).squeeze(0)

    return {
        "mode": "checkpoint_inference",
        "checkpoint_dir": checkpoint_dir.as_posix(),
        "weights_path": weights_path.as_posix(),
        "model_config_path": model_config_path.as_posix(),
        "training_config_path": training_config.get("_training_config_path"),
        "data_dir": data_dir.as_posix(),
        "static_dir": static_dir.as_posix(),
        "mapping_csv": mapping_path.as_posix(),
        "data_case_dir": case_dir.as_posix(),
        "time_labels": time_labels[:sequence_length + time_step_offset],
        "sequence_length": sequence_length,
        "time_step_offset": time_step_offset,
        "device": device,
        "model_input_projection_type": model_config.get("input_projection_type"),
        "changed_variable_mapping": variable_mapping[changed_variable],
        "real_rows": rows_from_arrays("real", target_values, variable_names),
        "predict_rows": rows_from_arrays("predict", predictions, variable_names),
    }


def load_pipeformer_forecast_context(
    parsed_task: Dict[str, Any],
    forecast_csv: Path,
    mapping_path: Path,
    *,
    checkpoint_dir: Optional[Path] = None,
    pipeformer_root: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    static_dir: Optional[Path] = None,
    device: str = "cpu",
    use_sample_csv: bool = False,
) -> Dict[str, Any]:
    if use_sample_csv:
        return load_sample_csv_forecast_context(parsed_task, forecast_csv, mapping_path)
    if checkpoint_dir is None or pipeformer_root is None:
        raise ValueError("checkpoint_dir and pipeformer_root are required unless use_sample_csv=True.")
    return run_checkpoint_inference(
        parsed_task=parsed_task,
        checkpoint_dir=checkpoint_dir,
        pipeformer_root=pipeformer_root,
        data_dir=data_dir,
        static_dir=static_dir,
        mapping_path=mapping_path,
        device=device,
    )