from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


PIPELINE_COLUMNS = ("管道划分", "管线", "pipeline")


def filter_rows_by_named_pipeline(
    rows: Iterable[Dict[str, str]],
    scope_text: str,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """Filter rows when authoritative text explicitly names a pipeline value."""
    materialized = list(rows)
    folded_scope = scope_text.casefold()
    available = list(
        dict.fromkeys(
            value
            for row in materialized
            for column in PIPELINE_COLUMNS
            if (value := str(row.get(column) or "").strip())
        )
    )
    selected = [value for value in available if value.casefold() in folded_scope]
    if not selected:
        return materialized, []
    selected_set = set(selected)
    return [
        row
        for row in materialized
        if any(
            str(row.get(column) or "").strip() in selected_set
            for column in PIPELINE_COLUMNS
        )
    ], selected
