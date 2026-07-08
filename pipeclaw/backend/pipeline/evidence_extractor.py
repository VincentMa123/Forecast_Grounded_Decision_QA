from __future__ import annotations

from typing import Any, Dict, List

from .schemas import ForecastRow


def summarize_variables(real_rows: List[ForecastRow], predict_rows: List[ForecastRow]) -> Dict[str, Dict[str, Any]]:
    variables = sorted({name for row in predict_rows for name in row.values})
    summaries: Dict[str, Dict[str, Any]] = {}
    for variable in variables:
        predicted = [row.values[variable] for row in predict_rows if variable in row.values]
        observed = [row.values[variable] for row in real_rows if variable in row.values]
        deltas = [pred - obs for pred, obs in zip(predicted, observed)]
        summaries[variable] = {
            "predicted_values": [round(value, 6) for value in predicted],
            "observed_values": [round(value, 6) for value in observed],
            "mean_prediction": round(sum(predicted) / len(predicted), 6) if predicted else None,
            "max_abs_prediction": round(max((abs(value) for value in predicted), default=0.0), 6),
            "mean_delta_vs_observed": round(sum(deltas) / len(deltas), 6) if deltas else None,
            "mean_abs_delta_vs_observed": round(sum(abs(value) for value in deltas) / len(deltas), 6) if deltas else None,
        }
    return summaries


def top_variables(summaries: Dict[str, Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    ranked = sorted(
        summaries.items(),
        key=lambda item: (
            item[1].get("mean_abs_delta_vs_observed") is not None,
            item[1].get("mean_abs_delta_vs_observed") or item[1].get("max_abs_prediction") or 0.0,
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