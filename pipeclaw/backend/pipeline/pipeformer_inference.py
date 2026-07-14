from __future__ import annotations

import csv
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .schemas import ForecastRow
from .variable_registry import load_variable_registry


logger = logging.getLogger(__name__)


SOURCE_FILES = {
    "B": "B.csv",
    "C": "C.csv",
    "H": "H.csv",
    "N": "N.csv",
    "P": "P.csv",
    "R": "R.csv",
    "TE": "T&E.csv",
}

EQUIPMENT_MISSING_FILL = -1.0


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


def find_default_checkpoint_dir(repo_root: Path) -> Path:
    output_dirs = [
        repo_root / "pipeFormer" / "outputs" / "mock_decoder",
    ]
    for output_dir in output_dirs:
        checkpoints = sorted(
            [path for path in output_dir.glob("checkpoint-*") if path.is_dir()],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if checkpoints:
            return checkpoints[0]
    raise FileNotFoundError(f"No PipeFormer checkpoint directory found under: {output_dirs}")


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
            logger.info("Loading PipeFormer training config: %s", candidate)
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
    loaded_models = sys.modules.get("models")
    loaded_models_path = Path(getattr(loaded_models, "__file__", "") or "")
    if loaded_models is not None and loaded_models_path.name == "models.py":
        sys.modules.pop("models", None)


def ensure_optional_matplotlib() -> None:
    import importlib.util
    import types

    if "matplotlib" in sys.modules:
        return
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
    logger.info("Loading PipeFormer checkpoint: checkpoint_dir=%s device=%s", checkpoint_dir, device)
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

    logger.info("Loading PipeFormer model config: %s", config_path)
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

    logger.info("Loading PipeFormer weights: %s", weights_path)
    load_kwargs = {"map_location": device}
    try:
        state_dict = torch.load(weights_path, weights_only=True, **load_kwargs)
    except TypeError:
        state_dict = torch.load(weights_path, **load_kwargs)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    logger.info("PipeFormer checkpoint loaded successfully")
    return model, model_config, weights_path, config_path


def load_tokenizer(static_dir: Path, vocab_size: Optional[int] = None):
    # Import tokenizer_save as a top-level package from pipeFormer/data to avoid importing data.__init__,
    # which pulls optional training dependencies such as tensordict.
    from tokenizer_save import load_tokenizer as load_tokenizer_from_stats

    logger.info("Loading PipeFormer tokenizer: static_dir=%s vocab_size=%s", static_dir, vocab_size)
    tokenizer = load_tokenizer_from_stats(static_dir, vocab_size=vocab_size)
    if tokenizer is None:
        raise RuntimeError(f"Tokenizer statistics not found under {static_dir / 'tokenizer_save'}")
    return tokenizer


def source_file_for_variable(variable_name: str) -> str:
    if ":" in variable_name:
        return "Boundary.csv"
    prefix = variable_name.split("_", 1)[0]
    if prefix in SOURCE_FILES:
        return SOURCE_FILES[prefix]
    raise ValueError(f"No mock CSV source mapping for variable {variable_name}")


def candidate_case_dirs(data_dir: Path, parsed_task: Dict[str, Any]) -> Iterable[Path]:
    case_id = parsed_task.get("case_id") or "mock_test_001"
    case_digits = "".join(ch for ch in case_id if ch.isdigit()) or "001"
    operating_condition_number = parsed_task.get("current_operating_condition_number")
    condition_number = int(operating_condition_number) if operating_condition_number is not None else int(case_digits)
    if condition_number < 1:
        raise ValueError("current_operating_condition_number must be a positive integer.")
    case_name = f"case_{condition_number:03d}"
    cn_name = f"\u7b2c{condition_number:03d}\u4e2a\u7b97\u4f8b"
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


def align_frame_to_master_index(frame, master_index, source_name: str):
    if frame.index.equals(master_index):
        return frame

    if source_name == "Boundary.csv":
        # Boundary controls are sparse operator setpoints. Match PipeFormer preprocessing by
        # holding the latest control value until the next boundary update arrives.
        return frame.reindex(frame.index.union(master_index)).sort_index().ffill().bfill().reindex(master_index)

    # Equipment/state files are expected on the model timeline. Missing measurements are
    # treated as unavailable, matching PipeFormer's sentinel-fill behavior.
    return frame.reindex(master_index).fillna(EQUIPMENT_MISSING_FILL)


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

    master_source_name, master_source = max(frames.items(), key=lambda item: len(item[1].index))
    master_index = master_source.index
    logger.info("PipeFormer case master timeline selected: source=%s rows=%d", master_source_name, len(master_index))

    aligned_frames: Dict[str, Any] = {}
    for source_name, source_frame in frames.items():
        aligned_frame = align_frame_to_master_index(source_frame, master_index, source_name)
        if not source_frame.index.equals(master_index):
            strategy = "forward_fill_boundary_controls" if source_name == "Boundary.csv" else "sentinel_fill_equipment_state"
            logger.info(
                "Aligned %s to master timeline: source_rows=%d target_rows=%d strategy=%s",
                source_name,
                len(source_frame.index),
                len(master_index),
                strategy,
            )
        aligned_frames[source_name] = aligned_frame

    matrix = np.zeros((len(master_index), len(variable_names)), dtype=np.float32)

    for variable_idx, variable_name in enumerate(variable_names):
        source_name = source_file_for_variable(variable_name)
        source_frame = aligned_frames[source_name]
        if variable_name not in source_frame.columns:
            raise ValueError(f"Variable {variable_name} not found in {case_dir / source_name}")
        matrix[:, variable_idx] = source_frame[variable_name].astype(float).to_numpy(dtype=np.float32)

    return matrix, [str(item) for item in master_index]


def resolve_boundary_adjustments(
    parsed_task: Dict[str, Any],
    variable_mapping: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    boundary_conditions = dict(parsed_task.get("boundary_conditions") or {})
    adjustments: Dict[str, Dict[str, Any]] = {}

    disturbance_variable = parsed_task.get("disturbance_variable")
    percent = parsed_task.get("disturbance_magnitude_percent")
    direction = parsed_task.get("disturbance_direction")
    if disturbance_variable and percent is not None:
        signed_percent = abs(float(percent)) if direction == "up" else -abs(float(percent))
        adjustments[disturbance_variable] = {
            "variable": disturbance_variable,
            "mode": "percent_change",
            "value": signed_percent,
            "source": "disturbance",
        }

    percentage_changes = boundary_conditions.get("percentage_changes") or {}
    for variable, value in dict(percentage_changes).items():
        adjustments[str(variable)] = {
            "variable": str(variable),
            "mode": "percent_change",
            "value": float(value),
            "source": "boundary_conditions.percentage_changes",
        }

    setpoints = boundary_conditions.get("setpoints") or {}
    for variable, value in dict(setpoints).items():
        adjustments[str(variable)] = {
            "variable": str(variable),
            "mode": "setpoint",
            "value": float(value),
            "source": "boundary_conditions.setpoints",
        }

    unknown = sorted(variable for variable in adjustments if variable not in variable_mapping)
    if unknown:
        raise ValueError(f"Boundary-condition variables are not in the PipeFormer mapping: {unknown}")

    keep_other = bool(boundary_conditions.get("keep_other_boundary_controls", True))
    if not keep_other:
        boundary_variables = {name for name in variable_mapping if ":" in name}
        missing = sorted(boundary_variables - set(adjustments))
        if missing:
            raise ValueError(
                "keep_other_boundary_controls=false requires explicit setpoints or percentage changes "
                f"for every boundary control; missing: {missing}"
            )
    return list(adjustments.values())


def apply_condition_to_matrix(
    matrix,
    parsed_task: Dict[str, Any],
    variable_mapping: Dict[str, Dict[str, Any]],
    adjustments: Optional[List[Dict[str, Any]]] = None,
):
    import numpy as np

    scenario_matrix = np.array(matrix, copy=True)
    for adjustment in adjustments or resolve_boundary_adjustments(parsed_task, variable_mapping):
        variable_idx = variable_mapping[adjustment["variable"]]["index"]
        if adjustment["mode"] == "setpoint":
            scenario_matrix[:, variable_idx] = float(adjustment["value"])
        else:
            scenario_matrix[:, variable_idx] *= 1.0 + float(adjustment["value"]) / 100.0
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


def parse_time_label(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def infer_time_step_minutes(time_labels: List[str]) -> Optional[float]:
    if len(time_labels) < 2:
        return None
    first = parse_time_label(time_labels[0])
    second = parse_time_label(time_labels[1])
    if first is None or second is None:
        return None
    minutes = (second - first).total_seconds() / 60.0
    return minutes if minutes > 0 else None


def requested_forecast_steps(
    parsed_task: Dict[str, Any],
    time_step_minutes: Optional[float],
    default_steps: int,
) -> int:
    horizon_minutes = parsed_task.get("forecast_horizon_minutes")
    if horizon_minutes is None or time_step_minutes is None:
        return max(1, default_steps)
    return max(1, int(math.ceil(float(horizon_minutes) / float(time_step_minutes))))


def future_time_labels(
    time_labels: List[str],
    start_index: int,
    steps: int,
    time_step_minutes: Optional[float],
) -> List[str]:
    labels = list(time_labels[start_index:start_index + steps])
    if len(labels) >= steps:
        return labels

    anchor_label = labels[-1] if labels else (time_labels[-1] if time_labels else "")
    anchor_time = parse_time_label(anchor_label)
    if anchor_time is None or time_step_minutes is None:
        labels.extend([f"future_step_{idx + 1}" for idx in range(len(labels), steps)])
        return labels

    while len(labels) < steps:
        anchor_time = anchor_time + timedelta(minutes=float(time_step_minutes))
        labels.append(anchor_time.isoformat(sep=" "))
    return labels


def rows_from_arrays(
    label_prefix: str,
    values,
    variable_names: List[str],
    row_labels: Optional[List[str]] = None,
) -> List[ForecastRow]:
    rows = []
    for idx, row_values in enumerate(values):
        if row_labels and idx < len(row_labels):
            label = f"{row_labels[idx]}_{label_prefix}"
        else:
            label = f"data_line_{idx + 1}_{label_prefix}"
        rows.append(
            ForecastRow(
                label=label,
                values={name: round(float(row_values[var_idx]), 6) for var_idx, name in enumerate(variable_names)},
            )
        )
    return rows


@dataclass(frozen=True)
class PipeFormerInferenceConfig:
    checkpoint_dir: Path
    pipeformer_root: Path
    data_dir: Optional[Path] = None
    static_dir: Optional[Path] = None
    mapping_path: Optional[Path] = None
    device: str = "cpu"


class PipeFormerInferenceEngine:
    """Run checkpoint inference using one resolved PipeFormer environment."""

    def __init__(self, config: PipeFormerInferenceConfig) -> None:
        self.config = config

    def forecast(self, parsed_task: Dict[str, Any]) -> Dict[str, Any]:
        return _run_checkpoint_inference(parsed_task=parsed_task, config=self.config)


def run_checkpoint_inference(
    parsed_task: Dict[str, Any],
    checkpoint_dir: Path,
    pipeformer_root: Path,
    data_dir: Optional[Path],
    static_dir: Optional[Path],
    mapping_path: Optional[Path],
    device: str = "cpu",
) -> Dict[str, Any]:
    """Compatibility wrapper for the configured inference engine."""
    return PipeFormerInferenceEngine(
        PipeFormerInferenceConfig(
            checkpoint_dir=checkpoint_dir,
            pipeformer_root=pipeformer_root,
            data_dir=data_dir,
            static_dir=static_dir,
            mapping_path=mapping_path,
            device=device,
        )
    ).forecast(parsed_task)


def _run_checkpoint_inference(
    *,
    parsed_task: Dict[str, Any],
    config: PipeFormerInferenceConfig,
) -> Dict[str, Any]:
    checkpoint_dir = config.checkpoint_dir
    pipeformer_root = config.pipeformer_root
    data_dir = config.data_dir
    static_dir = config.static_dir
    mapping_path = config.mapping_path
    device = config.device
    started_at = time.perf_counter()
    logger.info("PipeFormer checkpoint inference started: checkpoint=%s device=%s", checkpoint_dir, device)
    import numpy as np
    import torch

    checkpoint_dir = checkpoint_dir.resolve()
    pipeformer_root = pipeformer_root.resolve()
    training_config = load_training_config(checkpoint_dir, pipeformer_root)
    data_dir = (data_dir or resolve_relative(training_config.get("data_dir"), pipeformer_root))
    static_dir = (static_dir or resolve_relative(training_config.get("static_dir"), pipeformer_root))
    if data_dir is None or static_dir is None:
        raise ValueError("Could not resolve PipeFormer data_dir/static_dir for checkpoint inference.")
    mapping_path = (mapping_path or static_dir / "index_variable_mapping.csv").resolve()
    logger.info("PipeFormer data paths resolved: data_dir=%s static_dir=%s mapping=%s", data_dir, static_dir, mapping_path)

    add_pipeformer_import_paths(pipeformer_root)
    variable_mapping = load_variable_mapping(mapping_path)
    variable_names = [
        name
        for name, _ in sorted(
            variable_mapping.items(),
            key=lambda item: item[1]["index"],
        )
    ]
    variable_registry = load_variable_registry(
        static_dir / "variable_registry.json",
        variable_names,
    )
    disturbance_variable = parsed_task["disturbance_variable"]
    logger.info(
        "PipeFormer variable mapping loaded: variables=%d disturbance_variable=%s",
        len(variable_mapping),
        disturbance_variable,
    )
    if disturbance_variable not in variable_mapping:
        raise ValueError(f"Parsed variable {disturbance_variable} is not in PipeFormer mapping {mapping_path}")

    sequence_length = int(training_config.get("sequence_length", 3))
    time_step_offset = int(training_config.get("time_step_offset", 1))
    case_dir = resolve_case_dir(data_dir, parsed_task)
    logger.info("Loading PipeFormer case CSVs: case_dir=%s", case_dir)
    base_matrix, time_labels = load_case_matrix(case_dir, variable_names)
    logger.info("Loaded PipeFormer case matrix: shape=%s time_steps=%d", base_matrix.shape, len(time_labels))
    boundary_adjustments = resolve_boundary_adjustments(parsed_task, variable_mapping)
    scenario_matrix = apply_condition_to_matrix(base_matrix, parsed_task, variable_mapping, boundary_adjustments)
    logger.info(
        "Applied parsed condition: variable=%s direction=%s percent=%s",
        disturbance_variable,
        parsed_task.get("disturbance_direction"),
        parsed_task.get("disturbance_magnitude_percent"),
    )
    if len(base_matrix) < sequence_length + time_step_offset:
        raise ValueError(f"Case {case_dir} is too short for sequence_length={sequence_length}, offset={time_step_offset}")

    input_values = scenario_matrix[:sequence_length]
    time_step_minutes = infer_time_step_minutes(time_labels)
    steps_requested = requested_forecast_steps(parsed_task, time_step_minutes, sequence_length)

    tokenizer = load_tokenizer(static_dir, vocab_size=training_config.get("tokenizer_vocab_size"))
    model, model_config, weights_path, model_config_path = load_pipeformer_model(checkpoint_dir, pipeformer_root, device)
    attention_indices = load_attention_indices(static_dir)
    prediction_mask = load_prediction_mask(static_dir, variable_names)
    logger.info(
        "PipeFormer tensors prepared: input_values=%s attention=%s prediction_mask=%s requested_steps=%d time_step_minutes=%s",
        input_values.shape,
        attention_indices.shape,
        prediction_mask.shape,
        steps_requested,
        time_step_minutes,
    )

    future_rows_per_pass = max(1, min(time_step_offset, sequence_length))
    generated_chunks = []
    generated_steps = 0
    rollout_window = np.array(input_values, copy=True)
    logger.info(
        "PipeFormer autoregressive rollout started: requested_horizon_minutes=%s requested_steps=%d rows_per_pass=%d",
        parsed_task.get("forecast_horizon_minutes"),
        steps_requested,
        future_rows_per_pass,
    )

    while generated_steps < steps_requested:
        input_tokens = tokenizer.transform_to_tokens(rollout_window)
        input_tokens = np.asarray(input_tokens, dtype=np.int64)
        logger.info(
            "PipeFormer forward pass started: rollout_start_step=%d input_values=%s input_tokens=%s",
            generated_steps,
            rollout_window.shape,
            input_tokens.shape,
        )
        with torch.no_grad():
            input_tensor = torch.as_tensor(rollout_window, dtype=torch.float32, device=device).unsqueeze(0)
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
                window_predictions = decoded.squeeze(0).detach().cpu().numpy()
            else:
                window_predictions = np.asarray(decoded, dtype=np.float32).squeeze(0)

        new_predictions = window_predictions[-future_rows_per_pass:]
        remaining_steps = steps_requested - generated_steps
        new_predictions = new_predictions[:remaining_steps]
        control_columns = np.asarray(prediction_mask) <= 0
        if np.any(control_columns):
            # Boundary controls are exogenous model inputs. Keep their applied
            # values fixed during rollout instead of treating decoded logits as
            # forecasts for variables the prediction mask excludes.
            new_predictions[:, control_columns] = rollout_window[-1, control_columns]
        generated_chunks.append(new_predictions)
        generated_steps += int(new_predictions.shape[0])
        rollout_window = np.concatenate([rollout_window[new_predictions.shape[0]:], new_predictions], axis=0)
        logger.info("PipeFormer forward pass finished: generated_steps=%d/%d", generated_steps, steps_requested)

    predictions = np.concatenate(generated_chunks, axis=0)
    target_values = base_matrix[sequence_length:sequence_length + predictions.shape[0]]
    forecast_time_labels = future_time_labels(time_labels, sequence_length, int(predictions.shape[0]), time_step_minutes)
    observed_future_labels = forecast_time_labels[:len(target_values)]
    actual_horizon_minutes = None
    if time_step_minutes is not None:
        actual_horizon_minutes = round(float(time_step_minutes) * int(predictions.shape[0]), 6)
    logger.info(
        "PipeFormer autoregressive rollout finished: predictions_shape=%s actual_horizon_minutes=%s elapsed_s=%.3f",
        predictions.shape,
        actual_horizon_minutes,
        time.perf_counter() - started_at,
    )
    data_provenance = {
        "registry_schema_version": variable_registry.get("schema_version"),
        "synthetic": bool(variable_registry.get("synthetic")),
        "physical_validation_status": variable_registry.get("physical_validation_status"),
    }
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
        "time_labels": time_labels[:sequence_length] + forecast_time_labels,
        "sequence_length": sequence_length,
        "time_step_offset": time_step_offset,
        "requested_forecast_horizon_minutes": parsed_task.get("forecast_horizon_minutes"),
        "time_step_minutes": time_step_minutes,
        "requested_forecast_steps": steps_requested,
        "actual_forecast_steps": int(predictions.shape[0]),
        "actual_forecast_horizon_minutes": actual_horizon_minutes,
        "actual_forecast_horizon_source": "autoregressive_rollout_from_checkpoint_window",
        "forecast_time_labels": forecast_time_labels,
        "device": device,
        "model_input_projection_type": model_config.get("input_projection_type"),
        "data_provenance": data_provenance,
        "variable_registry": list(variable_registry.get("variables") or []),
        "disturbance_variable_mapping": variable_mapping[disturbance_variable],
        "operating_condition_number_used": parsed_task.get("current_operating_condition_number"),
        "applied_boundary_conditions": boundary_adjustments,
        "real_rows": rows_from_arrays("real", target_values, variable_names, observed_future_labels),
        "predict_rows": rows_from_arrays("predict", predictions, variable_names, forecast_time_labels),
    }
