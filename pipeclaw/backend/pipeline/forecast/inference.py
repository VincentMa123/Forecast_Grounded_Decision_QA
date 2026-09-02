from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..schemas import ForecastRow
from ..registry.variable_registry import VariableRegistry, normalize_task_variables


logger = logging.getLogger(__name__)


def build_hybrid_token_features(
    tokenizer: Any,
    values: np.ndarray,
    token_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the raw token-median and within-bin offset inputs used by hybrid mode."""
    import numpy as np

    medians, half_widths = tokenizer.get_token_value_stats()
    token_medians = np.asarray(medians, dtype=np.float32)[token_ids]
    token_half_widths = np.asarray(half_widths, dtype=np.float32)[token_ids]
    safe_half_widths = np.where(
        np.abs(token_half_widths) < 1e-4, 1.0, token_half_widths
    )
    offsets = (np.asarray(values, dtype=np.float32) - token_medians) / safe_half_widths
    offsets = np.where(np.abs(token_half_widths) < 1e-4, 0.0, offsets)
    return token_medians.astype(np.float32), offsets.astype(np.float32)


def attach_hybrid_token_statistics(
    model: Any, model_config: dict[str, Any], tokenizer: Any
) -> None:
    import torch

    if str(model_config.get("input_projection_type", "")).lower() != "hybrid":
        return
    medians, half_widths = tokenizer.get_token_value_stats()
    model.set_token_value_statistics(
        torch.as_tensor(medians, dtype=torch.float32),
        torch.as_tensor(half_widths, dtype=torch.float32),
    )


def decode_model_predictions(
    outputs: Any, tokenizer: Any, projection_type: str
) -> np.ndarray:
    """Prefer hybrid continuous values; retain argmax decoding for token-only checkpoints."""
    import numpy as np
    import torch

    if isinstance(outputs, dict) and projection_type.lower() == "hybrid":
        value_predictions = outputs.get("value_predictions")
        if value_predictions is not None:
            return value_predictions.detach().cpu().numpy()
    token_logits = outputs.get("token_logits") if isinstance(outputs, dict) else None
    if token_logits is None:
        raise RuntimeError(
            "PipeFormer checkpoint inference did not return decodable predictions."
        )
    predicted_tokens = torch.argmax(token_logits, dim=-1)
    decoded = tokenizer.tokens_to_values(predicted_tokens)
    if isinstance(decoded, torch.Tensor):
        return decoded.detach().cpu().numpy()
    return np.asarray(decoded, dtype=np.float32)


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
DISTURBANCE_TIMING_LEGACY = "legacy_observation_window"
DISTURBANCE_TIMING_CURRENT_STEP = "current_step"


def resolve_disturbance_timing_mode(
    parsed_task: Dict[str, Any],
    configured_mode: Optional[str] = None,
) -> str:
    """Resolve when a new boundary condition enters the observation window.

    Legacy is intentionally the default because the released teacher traces
    applied the condition to every row in the input window. Real-data runs can
    opt into ``current_step`` without changing the behavior used to evaluate
    students trained on those archived traces.
    """
    boundary_conditions = dict(parsed_task.get("boundary_conditions") or {})
    requested = (
        parsed_task.get("disturbance_timing_mode")
        or boundary_conditions.get("disturbance_timing_mode")
        or configured_mode
        or os.getenv("PIPEFORMER_DISTURBANCE_TIMING_MODE")
        or DISTURBANCE_TIMING_LEGACY
    )
    aliases = {
        "legacy": DISTURBANCE_TIMING_LEGACY,
        DISTURBANCE_TIMING_LEGACY: DISTURBANCE_TIMING_LEGACY,
        "forecast_origin": DISTURBANCE_TIMING_CURRENT_STEP,
        DISTURBANCE_TIMING_CURRENT_STEP: DISTURBANCE_TIMING_CURRENT_STEP,
    }
    mode = aliases.get(str(requested).strip().lower())
    if mode is None:
        raise ValueError(
            "disturbance_timing_mode must be one of: "
            f"{DISTURBANCE_TIMING_LEGACY}, {DISTURBANCE_TIMING_CURRENT_STEP}."
        )
    return mode


def load_variable_mapping(path: Path) -> Dict[str, Dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PipeFormer variable mapping not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        expected_header = ["index", "variable_name", "global_index"]
        if reader.fieldnames != expected_header:
            raise ValueError(
                f"Unexpected PipeFormer mapping header: {reader.fieldnames}; "
                f"expected {expected_header}."
            )
        mapping: Dict[str, Dict[str, Any]] = {}
        mapping_indices: Dict[int, str] = {}
        global_indices: Dict[int, str] = {}
        for line_number, row in enumerate(reader, start=2):
            if row.get(None):
                raise ValueError(f"Invalid PipeFormer mapping row {line_number}: {row}")
            raw_name = row.get("variable_name")
            name = str(raw_name or "").strip()
            if not name or raw_name != name:
                raise ValueError(
                    f"Invalid PipeFormer mapping variable name at row {line_number}."
                )
            if name in mapping:
                raise ValueError(f"Duplicate PipeFormer mapping variable name: {name}")
            try:
                index = int(str(row.get("index")))
                global_index = int(str(row.get("global_index")))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"PipeFormer mapping indices must be integers at row {line_number}."
                ) from exc
            if index < 0 or global_index < 0:
                raise ValueError(
                    f"PipeFormer mapping indices must be non-negative at row {line_number}."
                )
            if index in mapping_indices:
                raise ValueError(f"Duplicate PipeFormer mapping index: {index}")
            if global_index in global_indices:
                raise ValueError(
                    f"Duplicate PipeFormer mapping global_index: {global_index}"
                )
            mapping[name] = {
                "index": index,
                "global_index": global_index,
            }
            mapping_indices[index] = name
            global_indices[global_index] = name
    expected_indices = list(range(len(mapping)))
    actual_indices = list(mapping_indices)
    if not mapping or actual_indices != expected_indices:
        raise ValueError(
            "PipeFormer mapping indices must be contiguous and in model input order "
            f"{expected_indices}; got {actual_indices}."
        )
    return mapping


def find_default_checkpoint_dir(repo_root: Path) -> Path:
    outputs_root = repo_root / "pipeFormer" / "outputs"
    active_manifest = outputs_root / "mock_decoder_active.json"
    if active_manifest.exists() or active_manifest.is_symlink():
        try:
            active = json.loads(active_manifest.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid PipeFormer active manifest: {active_manifest}"
            ) from exc
        if not isinstance(active, dict):
            raise ValueError(
                f"PipeFormer active manifest must contain a JSON object: {active_manifest}"
            )
        if active.get("accepted") is not True:
            raise ValueError(
                f"PipeFormer active manifest is not accepted: {active_manifest}"
            )
        checkpoint_value = active.get("checkpoint_dir")
        if not isinstance(checkpoint_value, str) or not checkpoint_value.strip():
            raise ValueError(
                f"PipeFormer active manifest is missing checkpoint_dir: {active_manifest}"
            )
        relative_checkpoint = Path(checkpoint_value)
        if relative_checkpoint.is_absolute():
            raise ValueError(
                f"PipeFormer active manifest checkpoint_dir must be relative: {checkpoint_value}"
            )
        candidate = (outputs_root / relative_checkpoint).resolve()
        try:
            candidate.relative_to(outputs_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"PipeFormer active manifest checkpoint_dir escapes outputs: {checkpoint_value}"
            ) from exc
        if not candidate.is_dir():
            raise FileNotFoundError(
                f"PipeFormer active manifest checkpoint directory not found: {candidate}"
            )
        return candidate
    output_dirs = [
        outputs_root / "mock_decoder",
    ]
    for output_dir in output_dirs:
        checkpoints = sorted(
            [path for path in output_dir.glob("checkpoint-*") if path.is_dir()],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if checkpoints:
            return checkpoints[0]
    raise FileNotFoundError(
        f"No PipeFormer checkpoint directory found under: {output_dirs}"
    )


def resolve_relative(path_value: Optional[str], base_dir: Path) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_training_config(checkpoint_dir: Path, pipeformer_root: Path) -> Dict[str, Any]:
    candidates = [
        checkpoint_dir / "training_config.json",
        checkpoint_dir.parent / "training_config.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            logger.info("Loading PipeFormer training config: %s", candidate)
            with candidate.open("r", encoding="utf-8") as fh:
                config = json.load(fh)
            config["_training_config_path"] = candidate.as_posix()
            return config
    raise FileNotFoundError(
        f"training_config.json not found near {checkpoint_dir} or {pipeformer_root}"
    )


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
        RuntimeError(
            "matplotlib is not installed; attention-mask plotting is unavailable."
        )
    )
    pyplot_stub.close = lambda *args, **kwargs: None
    sys.modules.setdefault("matplotlib", matplotlib_stub)
    sys.modules.setdefault("matplotlib.pyplot", pyplot_stub)


def load_pipeformer_model(checkpoint_dir: Path, pipeformer_root: Path, device: str):
    logger.info(
        "Loading PipeFormer checkpoint: checkpoint_dir=%s device=%s",
        checkpoint_dir,
        device,
    )
    add_pipeformer_import_paths(pipeformer_root)
    ensure_optional_matplotlib()
    import torch
    from dataclasses import fields
    from models.decoder import DecoderConfig, FluidDecoder

    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        config_path = checkpoint_dir.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"PipeFormer model config not found near {checkpoint_dir}"
        )

    logger.info("Loading PipeFormer model config: %s", config_path)
    with config_path.open("r", encoding="utf-8") as fh:
        model_config = json.load(fh)

    config_fields = {field.name for field in fields(DecoderConfig)}
    filtered_config = {
        key: value for key, value in model_config.items() if key in config_fields
    }
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
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    logger.info("PipeFormer checkpoint loaded successfully")
    return model, model_config, weights_path, config_path


def load_tokenizer(static_dir: Path, vocab_size: Optional[int] = None):
    # Import tokenizer_save as a top-level package from pipeFormer/data to avoid importing data.__init__,
    # which pulls optional training dependencies such as tensordict.
    from tokenizer_save import load_tokenizer as load_tokenizer_from_stats

    logger.info(
        "Loading PipeFormer tokenizer: static_dir=%s vocab_size=%s",
        static_dir,
        vocab_size,
    )
    tokenizer = load_tokenizer_from_stats(static_dir, vocab_size=vocab_size)
    if tokenizer is None:
        raise RuntimeError(
            f"Tokenizer statistics not found under {static_dir / 'tokenizer_save'}"
        )
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
    condition_number = (
        int(operating_condition_number)
        if operating_condition_number is not None
        else int(case_digits)
    )
    if condition_number < 1:
        raise ValueError(
            "current_operating_condition_number must be a positive integer."
        )
    case_name = f"case_{condition_number:03d}"
    cn_name = f"第{condition_number:03d}个算例"
    dataset_dir = data_dir / "dataset"
    yield dataset_dir / "train" / case_name
    yield dataset_dir / "train" / cn_name
    yield dataset_dir / "test" / case_name
    yield dataset_dir / "test" / cn_name


def resolve_case_dir(data_dir: Path, parsed_task: Dict[str, Any]) -> Path:
    for candidate in candidate_case_dirs(data_dir, parsed_task):
        if candidate.exists():
            return candidate
    candidates = ", ".join(
        path.as_posix() for path in candidate_case_dirs(data_dir, parsed_task)
    )
    raise FileNotFoundError(
        f"Could not find mock PipeFormer case directory. Tried: {candidates}"
    )


def align_frame_to_master_index(frame, master_index, source_name: str):
    if frame.index.equals(master_index):
        return frame

    if source_name == "Boundary.csv":
        # Boundary controls are sparse operator setpoints. Match PipeFormer preprocessing by
        # holding the latest control value until the next boundary update arrives.
        return (
            frame.reindex(frame.index.union(master_index))
            .sort_index()
            .ffill()
            .bfill()
            .reindex(master_index)
        )

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

    master_source_name, master_source = max(
        frames.items(), key=lambda item: len(item[1].index)
    )
    master_index = master_source.index
    logger.info(
        "PipeFormer case master timeline selected: source=%s rows=%d",
        master_source_name,
        len(master_index),
    )

    aligned_frames: Dict[str, Any] = {}
    for source_name, source_frame in frames.items():
        aligned_frame = align_frame_to_master_index(
            source_frame, master_index, source_name
        )
        if not source_frame.index.equals(master_index):
            strategy = (
                "forward_fill_boundary_controls"
                if source_name == "Boundary.csv"
                else "sentinel_fill_equipment_state"
            )
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
            raise ValueError(
                f"Variable {variable_name} not found in {case_dir / source_name}"
            )
        matrix[:, variable_idx] = (
            source_frame[variable_name].astype(float).to_numpy(dtype=np.float32)
        )

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
        if str(disturbance_variable).endswith(":ST"):
            raise ValueError(
                f"Binary status variable {disturbance_variable} cannot use a percentage disturbance; "
                "provide boundary_conditions.setpoints with 0 or 1."
            )
        signed_percent = (
            abs(float(percent)) if direction == "up" else -abs(float(percent))
        )
        adjustments[disturbance_variable] = {
            "variable": disturbance_variable,
            "mode": "percent_change",
            "value": signed_percent,
            "source": "disturbance",
        }

    percentage_changes = boundary_conditions.get("percentage_changes") or {}
    for variable, value in dict(percentage_changes).items():
        if str(variable).endswith(":ST"):
            raise ValueError(
                f"Binary status variable {variable} cannot use a percentage change; use setpoint 0 or 1."
            )
        adjustments[str(variable)] = {
            "variable": str(variable),
            "mode": "percent_change",
            "value": float(value),
            "source": "boundary_conditions.percentage_changes",
        }

    setpoints = boundary_conditions.get("setpoints") or {}
    for variable, value in dict(setpoints).items():
        numeric_value = float(value)
        if str(variable).endswith(":ST") and numeric_value not in {0.0, 1.0}:
            raise ValueError(
                f"Binary status setpoint {variable} must be exactly 0 or 1, got {value}."
            )
        adjustments[str(variable)] = {
            "variable": str(variable),
            "mode": "setpoint",
            "value": numeric_value,
            "source": "boundary_conditions.setpoints",
        }

    unknown = sorted(
        variable for variable in adjustments if variable not in variable_mapping
    )
    if unknown:
        raise ValueError(
            f"Boundary-condition variables are not in the PipeFormer mapping: {unknown}"
        )

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
    timing_mode: str = DISTURBANCE_TIMING_LEGACY,
):
    import numpy as np

    scenario_matrix = np.array(matrix, copy=True)
    resolved_adjustments = adjustments or resolve_boundary_adjustments(
        parsed_task, variable_mapping
    )
    row_selection = (
        slice(None) if timing_mode == DISTURBANCE_TIMING_LEGACY else slice(-1, None)
    )
    if timing_mode not in {DISTURBANCE_TIMING_LEGACY, DISTURBANCE_TIMING_CURRENT_STEP}:
        raise ValueError(f"Unsupported disturbance timing mode: {timing_mode}")
    for adjustment in resolved_adjustments:
        variable_idx = variable_mapping[adjustment["variable"]]["index"]
        if adjustment["mode"] == "setpoint":
            scenario_matrix[row_selection, variable_idx] = float(adjustment["value"])
        else:
            scenario_matrix[row_selection, variable_idx] *= (
                1.0 + float(adjustment["value"]) / 100.0
            )
    return scenario_matrix


def boundary_application_evidence(
    base_matrix,
    adjusted_matrix,
    variable_mapping: Dict[str, Dict[str, Any]],
    adjustments: List[Dict[str, Any]],
    timing_mode: str,
) -> List[Dict[str, Any]]:
    """Record the actual boundary values supplied to the model input window."""
    import numpy as np

    row_indices = (
        list(range(len(adjusted_matrix)))
        if timing_mode == DISTURBANCE_TIMING_LEGACY
        else [len(adjusted_matrix) - 1]
    )
    evidence = []
    for adjustment in adjustments:
        variable = str(adjustment["variable"])
        variable_idx = int(variable_mapping[variable]["index"])
        before = [float(base_matrix[index, variable_idx]) for index in row_indices]
        applied = [float(adjusted_matrix[index, variable_idx]) for index in row_indices]
        requested = float(adjustment["value"])
        if adjustment["mode"] == "setpoint":
            verified = all(np.isclose(value, requested, atol=1e-8) for value in applied)
        else:
            expected = [value * (1.0 + requested / 100.0) for value in before]
            verified = all(
                np.isclose(actual, target, rtol=1e-7, atol=1e-8)
                for actual, target in zip(applied, expected)
            )
        evidence.append(
            {
                "variable": variable,
                "mode": adjustment["mode"],
                "requested_value": requested,
                "source": adjustment.get("source"),
                "applied_step_indices": row_indices,
                "input_values_before": before,
                "input_values_applied": applied,
                "verified": bool(verified),
            }
        )
    return evidence


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
        raise FileNotFoundError(f"prediction_mask.csv not found: {mask_path}")

    expected_names = set(variable_names)
    if len(expected_names) != len(variable_names):
        raise ValueError("Expected PipeFormer variable names contain duplicates.")
    mask_by_name: Dict[str, int] = {}
    with mask_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header != ["variable_name", "predict"]:
            raise ValueError(f"Unexpected prediction_mask.csv header: {header}")
        for line_number, row in enumerate(reader, start=2):
            if len(row) != 2 or not row[0]:
                raise ValueError(
                    f"Invalid prediction_mask.csv row {line_number}: {row}"
                )
            variable_name, raw_mask = row
            if variable_name in mask_by_name:
                raise ValueError(f"Duplicate prediction mask variable: {variable_name}")
            try:
                mask_value = int(raw_mask)
            except ValueError as exc:
                raise ValueError(
                    f"Prediction mask for {variable_name} must be an integer: {raw_mask}"
                ) from exc
            if mask_value not in {0, 1}:
                raise ValueError(
                    f"Prediction mask for {variable_name} must be 0 or 1: {mask_value}"
                )
            mask_by_name[variable_name] = mask_value
    actual_names = set(mask_by_name)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            f"Prediction mask variables do not match mapping; missing={missing}, extra={extra}"
        )
    return np.asarray([mask_by_name[name] for name in variable_names], dtype=np.int32)


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
    labels = list(time_labels[start_index : start_index + steps])
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
                values={
                    name: round(float(row_values[var_idx]), 6)
                    for var_idx, name in enumerate(variable_names)
                },
            )
        )
    return rows


@dataclass(frozen=True)
class PipeFormerInferenceConfig:
    checkpoint_dir: Optional[Path | str] = None
    pipeformer_root: Optional[Path | str] = None
    data_dir: Optional[Path | str] = None
    static_dir: Optional[Path | str] = None
    mapping_path: Optional[Path | str] = None
    device: Optional[str] = None
    disturbance_timing_mode: Optional[str] = None
    backend_root: Optional[Path | str] = None


@dataclass(frozen=True)
class ResolvedPipeFormerEnvironment:
    """One validated PipeFormer artifact view shared by runtime and inference."""

    pipeformer_root: Path
    checkpoint_dir: Path
    training_config: Dict[str, Any]
    data_dir: Path
    static_dir: Path
    mapping_path: Path
    variable_mapping: Dict[str, Dict[str, Any]]
    variable_names: tuple[str, ...]
    registry: VariableRegistry
    device: str
    disturbance_timing_mode: Optional[str] = None

    @property
    def registry_document(self) -> Dict[str, Any]:
        return self.registry.document


def _configured_path(value: Optional[Path | str]) -> Optional[Path]:
    return Path(value).expanduser().resolve() if value else None


def _repo_root_for_config(config: PipeFormerInferenceConfig) -> Optional[Path]:
    backend_root = _configured_path(config.backend_root)
    if backend_root is None:
        return None
    try:
        return backend_root.parents[1]
    except IndexError as exc:
        raise ValueError(
            f"Could not derive repository root from backend root: {backend_root}"
        ) from exc


def resolve_pipeformer_environment(
    config: PipeFormerInferenceConfig,
) -> ResolvedPipeFormerEnvironment:
    """Resolve and validate PipeFormer files once before authorization or rollout."""
    repo_root = _repo_root_for_config(config)
    pipeformer_root = _configured_path(config.pipeformer_root) or _configured_path(
        os.getenv("PIPEFORMER_ROOT")
    )
    if pipeformer_root is None:
        if repo_root is None:
            raise ValueError(
                "PipeFormer root is required when backend_root is not supplied."
            )
        pipeformer_root = (repo_root / "pipeFormer").resolve()
    if not pipeformer_root.is_dir():
        raise FileNotFoundError(
            f"PipeFormer root directory not found: {pipeformer_root}"
        )

    checkpoint_dir = _configured_path(config.checkpoint_dir) or _configured_path(
        os.getenv("PIPEFORMER_CHECKPOINT_DIR")
    )
    if checkpoint_dir is None:
        if repo_root is None:
            raise ValueError(
                "PipeFormer checkpoint is required when backend_root is not supplied."
            )
        checkpoint_dir = find_default_checkpoint_dir(repo_root).resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"PipeFormer checkpoint directory not found: {checkpoint_dir}"
        )

    training_config = load_training_config(checkpoint_dir, pipeformer_root)
    if not isinstance(training_config, dict):
        raise ValueError("PipeFormer training_config.json must contain an object.")
    data_dir = (
        _configured_path(config.data_dir)
        or _configured_path(os.getenv("PIPEFORMER_DATA_DIR"))
        or resolve_relative(training_config.get("data_dir"), pipeformer_root)
    )
    static_dir = (
        _configured_path(config.static_dir)
        or _configured_path(os.getenv("PIPEFORMER_STATIC_DIR"))
        or resolve_relative(training_config.get("static_dir"), pipeformer_root)
    )
    if data_dir is None or static_dir is None:
        raise ValueError(
            "Could not resolve PipeFormer data_dir/static_dir for checkpoint inference."
        )
    if not data_dir.is_dir() or not static_dir.is_dir():
        raise FileNotFoundError(
            f"PipeFormer data/static directories not found: data_dir={data_dir}, static_dir={static_dir}"
        )
    mapping_path = (
        _configured_path(config.mapping_path)
        or _configured_path(os.getenv("PIPEFORMER_MAPPING_CSV"))
        or (static_dir / "index_variable_mapping.csv").resolve()
    )
    variable_mapping = load_variable_mapping(mapping_path)
    variable_names = tuple(
        name
        for name, _ in sorted(
            variable_mapping.items(), key=lambda item: item[1]["index"]
        )
    )
    registry = VariableRegistry.read(static_dir / "variable_registry.json")
    registry.require(variable_names)
    device = str(config.device or os.getenv("PIPEFORMER_DEVICE", "cpu")).strip()
    if not device:
        raise ValueError("PipeFormer device must not be empty.")
    return ResolvedPipeFormerEnvironment(
        pipeformer_root=pipeformer_root,
        checkpoint_dir=checkpoint_dir,
        training_config=training_config,
        data_dir=data_dir,
        static_dir=static_dir,
        mapping_path=mapping_path,
        variable_mapping=variable_mapping,
        variable_names=variable_names,
        registry=registry,
        device=device,
        disturbance_timing_mode=config.disturbance_timing_mode,
    )


class PipeFormerInferenceEngine:
    """Run checkpoint inference using one resolved PipeFormer environment."""

    def __init__(
        self,
        environment: ResolvedPipeFormerEnvironment,
    ) -> None:
        self.environment = environment

    def forecast(self, parsed_task: Dict[str, Any]) -> Dict[str, Any]:
        return _run_checkpoint_inference(
            parsed_task=parsed_task, environment=self.environment
        )


def run_checkpoint_inference(
    parsed_task: Dict[str, Any],
    checkpoint_dir: Path,
    pipeformer_root: Path,
    data_dir: Optional[Path],
    static_dir: Optional[Path],
    mapping_path: Optional[Path],
    device: str = "cpu",
    disturbance_timing_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Compatibility wrapper for the configured inference engine."""
    return PipeFormerInferenceEngine(
        resolve_pipeformer_environment(
            PipeFormerInferenceConfig(
                checkpoint_dir=checkpoint_dir,
                pipeformer_root=pipeformer_root,
                data_dir=data_dir,
                static_dir=static_dir,
                mapping_path=mapping_path,
                device=device,
                disturbance_timing_mode=disturbance_timing_mode,
            )
        )
    ).forecast(parsed_task)


def _run_checkpoint_inference(
    *,
    parsed_task: Dict[str, Any],
    environment: ResolvedPipeFormerEnvironment,
) -> Dict[str, Any]:
    checkpoint_dir = environment.checkpoint_dir
    pipeformer_root = environment.pipeformer_root
    data_dir = environment.data_dir
    static_dir = environment.static_dir
    mapping_path = environment.mapping_path
    device = environment.device
    disturbance_timing_mode = resolve_disturbance_timing_mode(
        parsed_task,
        environment.disturbance_timing_mode,
    )
    started_at = time.perf_counter()
    logger.info(
        "PipeFormer checkpoint inference started: checkpoint=%s device=%s",
        checkpoint_dir,
        device,
    )
    import numpy as np
    import torch

    training_config = environment.training_config
    logger.info(
        "PipeFormer data paths resolved: data_dir=%s static_dir=%s mapping=%s",
        data_dir,
        static_dir,
        mapping_path,
    )

    add_pipeformer_import_paths(pipeformer_root)
    variable_mapping = environment.variable_mapping
    variable_names = list(environment.variable_names)
    variable_registry = environment.registry_document
    registry = environment.registry
    normalized_task = normalize_task_variables(
        parsed_task,
        registry,
    )
    parsed_task.clear()
    parsed_task.update(normalized_task)
    boundary = dict(parsed_task.get("boundary_conditions") or {})
    adjusted_variables = list(
        dict.fromkeys(
            [parsed_task.get("disturbance_variable")]
            + list(dict(boundary.get("percentage_changes") or {}))
            + list(dict(boundary.get("setpoints") or {}))
        )
    )
    registry.require_controllable_inputs(
        variable for variable in adjusted_variables if variable
    )
    disturbance_variable = parsed_task["disturbance_variable"]
    logger.info(
        "PipeFormer variable mapping loaded: variables=%d disturbance_variable=%s",
        len(variable_mapping),
        disturbance_variable,
    )
    if disturbance_variable not in variable_mapping:
        raise ValueError(
            f"Parsed variable {disturbance_variable} is not in PipeFormer mapping {mapping_path}"
        )

    sequence_length = int(training_config.get("sequence_length", 3))
    time_step_offset = int(training_config.get("time_step_offset", 1))
    case_dir = resolve_case_dir(data_dir, parsed_task)
    logger.info("Loading PipeFormer case CSVs: case_dir=%s", case_dir)
    base_matrix, time_labels = load_case_matrix(case_dir, variable_names)
    logger.info(
        "Loaded PipeFormer case matrix: shape=%s time_steps=%d",
        base_matrix.shape,
        len(time_labels),
    )
    boundary_adjustments = resolve_boundary_adjustments(parsed_task, variable_mapping)
    if len(base_matrix) < sequence_length + time_step_offset:
        raise ValueError(
            f"Case {case_dir} is too short for sequence_length={sequence_length}, offset={time_step_offset}"
        )
    input_values = apply_condition_to_matrix(
        base_matrix[:sequence_length],
        parsed_task,
        variable_mapping,
        boundary_adjustments,
        timing_mode=disturbance_timing_mode,
    )
    application_evidence = boundary_application_evidence(
        base_matrix[:sequence_length],
        input_values,
        variable_mapping,
        boundary_adjustments,
        disturbance_timing_mode,
    )
    if any(not item["verified"] for item in application_evidence):
        raise RuntimeError(
            "One or more boundary controls were not applied to the model input as requested."
        )
    logger.info(
        "Applied parsed condition: variable=%s direction=%s percent=%s timing=%s",
        disturbance_variable,
        parsed_task.get("disturbance_direction"),
        parsed_task.get("disturbance_magnitude_percent"),
        disturbance_timing_mode,
    )
    time_step_minutes = infer_time_step_minutes(time_labels)
    steps_requested = requested_forecast_steps(
        parsed_task, time_step_minutes, sequence_length
    )

    tokenizer = load_tokenizer(
        static_dir, vocab_size=training_config.get("tokenizer_vocab_size")
    )
    model, model_config, weights_path, model_config_path = load_pipeformer_model(
        checkpoint_dir, pipeformer_root, device
    )
    attach_hybrid_token_statistics(model, model_config, tokenizer)
    projection_type = str(model_config.get("input_projection_type", "token_embedding"))
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
        model_inputs: dict[str, Any] = {}
        if projection_type.lower() == "hybrid":
            token_medians, token_offsets = build_hybrid_token_features(
                tokenizer, rollout_window, input_tokens
            )
            model_inputs["input_token_medians"] = torch.as_tensor(
                token_medians, dtype=torch.float32, device=device
            ).unsqueeze(0)
            model_inputs["input_token_offsets"] = torch.as_tensor(
                token_offsets, dtype=torch.float32, device=device
            ).unsqueeze(0)
        logger.info(
            "PipeFormer forward pass started: rollout_start_step=%d input_values=%s input_tokens=%s",
            generated_steps,
            rollout_window.shape,
            input_tokens.shape,
        )
        with torch.no_grad():
            input_tensor = torch.as_tensor(
                rollout_window, dtype=torch.float32, device=device
            ).unsqueeze(0)
            token_tensor = torch.as_tensor(
                input_tokens, dtype=torch.long, device=device
            ).unsqueeze(0)
            attention_tensor = torch.as_tensor(
                attention_indices, dtype=torch.long, device=device
            ).unsqueeze(0)
            mask_tensor = torch.as_tensor(
                prediction_mask, dtype=torch.float32, device=device
            ).unsqueeze(0)
            outputs = model(
                input_ids=input_tensor,
                input_tokens=token_tensor,
                prediction_mask=mask_tensor,
                attention_indices=attention_tensor,
                **model_inputs,
            )
            window_predictions = decode_model_predictions(
                outputs, tokenizer, projection_type
            ).squeeze(0)

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
        rollout_window = np.concatenate(
            [rollout_window[new_predictions.shape[0] :], new_predictions], axis=0
        )
        logger.info(
            "PipeFormer forward pass finished: generated_steps=%d/%d",
            generated_steps,
            steps_requested,
        )

    predictions = np.concatenate(generated_chunks, axis=0)
    target_values = base_matrix[
        sequence_length : sequence_length + predictions.shape[0]
    ]
    forecast_time_labels = future_time_labels(
        time_labels, sequence_length, int(predictions.shape[0]), time_step_minutes
    )
    observed_future_labels = forecast_time_labels[: len(target_values)]
    actual_horizon_minutes = None
    if time_step_minutes is not None:
        actual_horizon_minutes = round(
            float(time_step_minutes) * int(predictions.shape[0]), 6
        )
    logger.info(
        "PipeFormer autoregressive rollout finished: predictions_shape=%s actual_horizon_minutes=%s elapsed_s=%.3f",
        predictions.shape,
        actual_horizon_minutes,
        time.perf_counter() - started_at,
    )
    data_provenance = {
        "registry_schema_version": variable_registry.get("schema_version"),
        "synthetic": bool(variable_registry.get("synthetic")),
        "physical_validation_status": variable_registry.get(
            "physical_validation_status"
        ),
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
        "requested_forecast_horizon_minutes": parsed_task.get(
            "forecast_horizon_minutes"
        ),
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
        "operating_condition_number_used": parsed_task.get(
            "current_operating_condition_number"
        ),
        "disturbance_timing_mode": disturbance_timing_mode,
        "adjusted_input_step_count": (
            sequence_length
            if boundary_adjustments
            and disturbance_timing_mode == DISTURBANCE_TIMING_LEGACY
            else 1
            if boundary_adjustments
            else 0
        ),
        "applied_boundary_conditions": boundary_adjustments,
        "boundary_application_evidence": application_evidence,
        "real_rows": rows_from_arrays(
            "real", target_values, variable_names, observed_future_labels
        ),
        "predict_rows": rows_from_arrays(
            "predict", predictions, variable_names, forecast_time_labels
        ),
    }
