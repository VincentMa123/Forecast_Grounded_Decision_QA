from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .schemas import ForecastRow


def summarize_variables(
    real_rows: List[ForecastRow], predict_rows: List[ForecastRow]
) -> Dict[str, Dict[str, Any]]:
    variables = sorted({name for row in predict_rows for name in row.values})
    summaries: Dict[str, Dict[str, Any]] = {}
    for variable in variables:
        predicted_rows = [row for row in predict_rows if variable in row.values]
        predicted = [row.values[variable] for row in predicted_rows]
        observed = [row.values[variable] for row in real_rows if variable in row.values]
        deltas = [pred - obs for pred, obs in zip(predicted, observed)]
        step_changes = [
            predicted[index] - predicted[index - 1]
            for index in range(1, len(predicted))
        ]
        minimum = min(predicted) if predicted else None
        maximum = max(predicted) if predicted else None
        peak_index = (
            max(range(len(predicted)), key=lambda index: abs(predicted[index]))
            if predicted
            else None
        )
        change_peak_index = (
            max(range(len(step_changes)), key=lambda index: abs(step_changes[index]))
            + 1
            if step_changes
            else None
        )
        decline_indices = [
            index for index, change in enumerate(step_changes) if change < 0
        ]
        decline_peak_index = (
            min(decline_indices, key=lambda index: step_changes[index]) + 1
            if decline_indices
            else None
        )
        summaries[variable] = {
            "predicted_values": [round(value, 6) for value in predicted],
            "prediction_labels": [row.label for row in predicted_rows],
            "observed_values": [round(value, 6) for value in observed],
            "mean_prediction": round(sum(predicted) / len(predicted), 6)
            if predicted
            else None,
            "max_abs_prediction": round(
                max((abs(value) for value in predicted), default=0.0), 6
            ),
            "minimum_prediction": round(minimum, 6) if minimum is not None else None,
            "minimum_step_index": predicted.index(minimum)
            if minimum is not None
            else None,
            "maximum_prediction": round(maximum, 6) if maximum is not None else None,
            "maximum_step_index": predicted.index(maximum)
            if maximum is not None
            else None,
            "peak_value": round(predicted[peak_index], 6)
            if peak_index is not None
            else None,
            "peak_step_index": peak_index,
            "prediction_change": round(predicted[-1] - predicted[0], 6)
            if predicted
            else None,
            "max_abs_step_change": round(
                max((abs(value) for value in step_changes), default=0.0), 6
            ),
            "max_abs_step_change_index": change_peak_index,
            "max_step_decline": round(
                max((max(0.0, -value) for value in step_changes), default=0.0), 6
            ),
            "max_step_decline_index": decline_peak_index,
            "max_decline_from_start": round(max(0.0, predicted[0] - minimum), 6)
            if predicted
            else None,
            "recovery_from_minimum": round(predicted[-1] - minimum, 6)
            if predicted
            else None,
            "mean_delta_vs_observed": round(sum(deltas) / len(deltas), 6)
            if deltas
            else None,
            "mean_abs_delta_vs_observed": round(
                sum(abs(value) for value in deltas) / len(deltas), 6
            )
            if deltas
            else None,
        }
    return summaries


def top_variables(
    summaries: Dict[str, Dict[str, Any]],
    limit: int = 3,
    preferred_variables: Optional[Iterable[str]] = None,
    priority_variables: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    preferred = set(preferred_variables or [])
    priority = list(dict.fromkeys(priority_variables or []))
    priority_rank = {
        variable: len(priority) - index for index, variable in enumerate(priority)
    }
    ranked = sorted(
        summaries.items(),
        key=lambda item: (
            priority_rank.get(item[0], 0),
            item[0] in preferred,
            item[1].get("mean_abs_delta_vs_observed") is not None,
            item[1].get("mean_abs_delta_vs_observed")
            or item[1].get("max_abs_prediction")
            or 0.0,
        ),
        reverse=True,
    )
    return [
        {
            "variable": variable,
            "mean_prediction": summary.get("mean_prediction"),
            "mean_abs_delta_vs_observed": summary.get("mean_abs_delta_vs_observed"),
        }
        for variable, summary in ranked[:limit]
    ]
