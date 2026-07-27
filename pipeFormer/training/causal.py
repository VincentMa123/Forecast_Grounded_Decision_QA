"""Mock-specific causal supervision layered on top of the standard trainer.

This module deliberately leaves PipeFormer's dataset and decoder implementations
unchanged. It only repeats intervention windows and adds a focused value loss for
registry-declared affected outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .trainer import FluidTrainer


def load_intervention_manifest(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Intervention manifest must contain an object keyed by sample ID.")
    return {str(key): dict(value) for key, value in payload.items() if isinstance(value, dict)}


class CausalWindowDataset(Dataset):
    """Repeat windows near known interventions while retaining the full dataset."""

    def __init__(
        self,
        base_dataset: Dataset,
        manifest: Mapping[str, Mapping[str, Any]],
        *,
        window_repeat: int = 4,
        window_before: int = 4,
        window_after: int = 12,
    ) -> None:
        if window_repeat < 1:
            raise ValueError("window_repeat must be at least 1")
        self.base_dataset = base_dataset
        self.manifest = manifest
        self.index_map = list(range(len(base_dataset)))
        causal_indices = self._causal_indices(window_before, window_after)
        self.index_map.extend(causal_indices * (window_repeat - 1))

    def _causal_indices(self, window_before: int, window_after: int) -> list[int]:
        selected: list[int] = []
        sequence_length = int(getattr(self.base_dataset, "sequence_length", 1))
        time_step_offset = int(getattr(self.base_dataset, "time_step_offset", 1))
        for sample in getattr(self.base_dataset, "samples", ()):
            sample_id = str(sample.get("sample_id", ""))
            intervention = self.manifest.get(sample_id)
            if not intervention:
                continue
            step_index = int(intervention.get("step_index", 0))
            band_start = step_index - max(0, int(window_before))
            band_end = step_index + max(0, int(window_after))
            start_index = int(sample["start_idx"])
            for dataset_index in range(start_index, int(sample["end_idx"])):
                offset = dataset_index - start_index
                target_start = offset + time_step_offset
                target_end = target_start + sequence_length - 1
                if target_end >= band_start and target_start <= band_end:
                    selected.append(dataset_index)
        return selected

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> Any:
        return self.base_dataset[self.index_map[index]]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_dataset, name)


def build_causal_value_mask(
    metadata: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Mapping[str, Any]],
    variable_to_index: Mapping[str, int],
    *,
    time_steps: int,
    variable_count: int,
    device: torch.device,
    post_intervention_steps: int | None = None,
) -> torch.Tensor:
    """Select declared affected outputs at and after each intervention step."""

    mask = torch.zeros((len(metadata), time_steps, variable_count), device=device)
    for batch_index, item in enumerate(metadata):
        intervention = manifest.get(str(item.get("sample_id", "")))
        if not intervention:
            continue
        step_index = int(intervention.get("step_index", 0))
        sequence_offset = int(item.get("sequence_offset", 0))
        time_step_offset = int(item.get("time_step_offset", 1))
        target_indices = [
            int(variable_to_index[name])
            for name in intervention.get("effect_targets", ())
            if name in variable_to_index
        ]
        if not target_indices:
            continue
        for time_index in range(time_steps):
            absolute_step = sequence_offset + time_step_offset + time_index
            if absolute_step < step_index:
                continue
            if post_intervention_steps is not None and absolute_step > step_index + post_intervention_steps:
                continue
            mask[batch_index, time_index, target_indices] = 1.0
    return mask


def causal_auxiliary_mae(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    selected = mask.to(dtype=predictions.dtype)
    denominator = selected.sum()
    if denominator.item() == 0:
        return predictions.sum() * 0.0
    return (torch.abs(predictions - labels) * selected).sum() / denominator


class CausalFluidTrainer(FluidTrainer):
    """Standard FluidTrainer plus an affected-output value loss."""

    def __init__(
        self,
        *args: Any,
        intervention_manifest: Mapping[str, Mapping[str, Any]],
        variable_to_index: Mapping[str, int],
        causal_auxiliary_loss_weight: float = 4.0,
        causal_post_intervention_steps: int | None = 30,
        **kwargs: Any,
    ) -> None:
        self.intervention_manifest = intervention_manifest
        self.variable_to_index = variable_to_index
        self.causal_auxiliary_loss_weight = float(causal_auxiliary_loss_weight)
        self.causal_post_intervention_steps = causal_post_intervention_steps
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, **kwargs
        )
        value_predictions = (
            outputs.get("value_predictions")
            if isinstance(outputs, dict)
            else getattr(outputs, "value_predictions", None)
        )
        labels = inputs.get("labels")
        metadata = inputs.get("metadata") or []
        if value_predictions is not None and labels is not None and metadata:
            mask = build_causal_value_mask(
                metadata,
                self.intervention_manifest,
                self.variable_to_index,
                time_steps=int(value_predictions.shape[1]),
                variable_count=int(value_predictions.shape[2]),
                device=value_predictions.device,
                post_intervention_steps=self.causal_post_intervention_steps,
            )
            prediction_mask = inputs.get("prediction_mask")
            if prediction_mask is not None:
                mask = mask * prediction_mask.to(mask.dtype).unsqueeze(1)
            auxiliary_loss = causal_auxiliary_mae(value_predictions, labels, mask)
            loss = loss + self.causal_auxiliary_loss_weight * auxiliary_loss
            if isinstance(outputs, dict):
                outputs["causal_auxiliary_loss"] = auxiliary_loss.detach()
                outputs["loss"] = loss
        return (loss, outputs) if return_outputs else loss
