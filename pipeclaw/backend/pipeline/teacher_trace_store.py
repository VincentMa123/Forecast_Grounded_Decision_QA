from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluator.scorer import load_records
from pipeline.io_utils import write_json, write_jsonl


@dataclass(frozen=True)
class TeacherTracePaths:
    output_jsonl: Path
    output_json: Path
    session_output_jsonl: Path


class TeacherTraceStore:
    """Read and merge durable teacher-trace and session records."""

    def __init__(self, paths: Optional[TeacherTracePaths] = None) -> None:
        self.paths = paths

    @classmethod
    def from_args(cls, args: Namespace) -> "TeacherTraceStore":
        return cls(
            TeacherTracePaths(
                output_jsonl=Path(args.output_jsonl),
                output_json=Path(args.output_json),
                session_output_jsonl=Path(args.session_output_jsonl),
            )
        )

    def load_master(self) -> List[Dict[str, Any]]:
        paths = self._require_paths()
        source = (
            paths.output_jsonl
            if paths.output_jsonl.is_file() and paths.output_jsonl.stat().st_size > 0
            else paths.output_json
        )
        return self.load(source)

    def load_sessions(self) -> List[Dict[str, Any]]:
        return self.load(self._require_paths().session_output_jsonl)

    def write_master(self, records: List[Dict[str, Any]]) -> None:
        paths = self._require_paths()
        write_jsonl(paths.output_jsonl, records, force=True)
        write_json(
            paths.output_json,
            records[0] if len(records) == 1 else records,
            force=True,
        )

    def write_sessions(self, records: List[Dict[str, Any]]) -> None:
        write_jsonl(self._require_paths().session_output_jsonl, records, force=True)

    def _require_paths(self) -> TeacherTracePaths:
        if self.paths is None:
            raise RuntimeError("TeacherTraceStore output paths were not configured.")
        return self.paths

    @staticmethod
    def load(path: Path) -> List[Dict[str, Any]]:
        path = Path(path)
        if not path.is_file() or path.stat().st_size == 0:
            return []
        return load_records(path)

    @staticmethod
    def merge_records(
        existing: List[Dict[str, Any]],
        generated: List[Dict[str, Any]],
        *,
        id_field: str,
    ) -> tuple[List[Dict[str, Any]], int]:
        merged = list(existing)
        seen = {str(item[id_field]) for item in existing if item.get(id_field)}
        duplicate_count = 0
        for item in generated:
            record_id = str(item.get(id_field) or "")
            if not record_id:
                raise ValueError(f"Generated record is missing required id field {id_field!r}.")
            if record_id in seen:
                duplicate_count += 1
                continue
            seen.add(record_id)
            merged.append(item)
        return merged, duplicate_count

    @staticmethod
    def merge_sessions(
        existing: List[Dict[str, Any]],
        generated: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], int]:
        merged = [dict(item) for item in existing]
        positions = {
            str(item["session_record_id"]): index
            for index, item in enumerate(merged)
            if item.get("session_record_id")
        }
        updated_count = 0
        for item in generated:
            record_id = str(item.get("session_record_id") or "")
            if not record_id:
                raise ValueError("Generated session record is missing session_record_id.")
            if record_id not in positions:
                positions[record_id] = len(merged)
                merged.append(item)
                continue

            updated_count += 1
            index = positions[record_id]
            prior = merged[index]
            turns = {
                int(turn["turn_id"]): turn
                for turn in prior.get("turns") or []
                if turn.get("turn_id") is not None
            }
            for turn in item.get("turns") or []:
                if turn.get("turn_id") is not None:
                    turns.setdefault(int(turn["turn_id"]), turn)
            combined = {**prior, **item}
            combined["turns"] = [turns[key] for key in sorted(turns)]
            combined["complete"] = bool(prior.get("complete") or item.get("complete"))
            combined["errors"] = (
                []
                if combined["complete"]
                else list(item.get("errors") or prior.get("errors") or [])
            )
            merged[index] = combined
        return merged, updated_count

    @staticmethod
    def contains_scenario(
        records: List[Dict[str, Any]], *, dataset_source: str, scenario_id: str
    ) -> bool:
        return any(
            str(item.get("dataset_source") or "") == dataset_source
            and str(item.get("source_scenario_id") or item.get("scenario_id") or "") == scenario_id
            for item in records
        )
    @staticmethod
    def replace_scenario(
        existing: List[Dict[str, Any]],
        generated: List[Dict[str, Any]],
        *,
        dataset_source: str,
        scenario_id: str,
        id_field: str,
    ) -> tuple[List[Dict[str, Any]], int]:
        """Replace one fully generated source/scenario pair without touching collisions."""
        def matches(item: Dict[str, Any]) -> bool:
            item_scenario = item.get("source_scenario_id") or item.get("scenario_id")
            return (
                str(item.get("dataset_source") or "") == dataset_source
                and str(item_scenario or "") == scenario_id
            )

        target_positions = [index for index, item in enumerate(existing) if matches(item)]
        if not target_positions:
            raise ValueError(
                f"No existing records match dataset {dataset_source!r} and scenario {scenario_id!r}."
            )
        if not generated or any(not matches(item) for item in generated):
            raise ValueError("Generated replacement contains a foreign or missing scenario record.")

        generated_ids = [str(item.get(id_field) or "") for item in generated]
        if any(not value for value in generated_ids) or len(generated_ids) != len(set(generated_ids)):
            raise ValueError(f"Generated replacement has missing or duplicate {id_field} values.")
        retained = [item for item in existing if not matches(item)]
        retained_ids = {str(item.get(id_field) or "") for item in retained}
        collisions = retained_ids & set(generated_ids)
        if collisions:
            raise ValueError(
                f"Generated replacement collides with retained {id_field} values: {sorted(collisions)}"
            )

        first_target = target_positions[0]
        before_count = sum(not matches(item) for item in existing[:first_target])
        return (
            retained[:before_count] + list(generated) + retained[before_count:],
            len(target_positions),
        )

    @staticmethod
    def validate_splits(records: List[Dict[str, Any]]) -> None:
        scenario_splits: Dict[str, str] = {}
        for record in records:
            scenario_id = str(record.get("scenario_id") or "")
            split = str(record.get("split") or "")
            if not scenario_id or not split:
                continue
            prior = scenario_splits.setdefault(scenario_id, split)
            if prior != split:
                raise ValueError(
                    f"Scenario {scenario_id!r} appears in both {prior!r} and {split!r}; "
                    "use the same --split-seed or regenerate with --force."
                )

    @staticmethod
    def sample_ids(scenario: Dict[str, Any]) -> List[str]:
        source = str(scenario.get("dataset_source") or "unknown_source")
        scenario_id = str(scenario.get("scenario_id") or "unknown_scenario")
        result = []
        for session_index, session in enumerate(scenario.get("sessions") or [], start=1):
            session_id = str(session.get("session_id") or f"{scenario_id}_session_{session_index:03d}")
            for fallback_turn_id, turn in enumerate(session.get("dialogue") or [], start=1):
                if not str(turn.get("user_input") or "").strip():
                    continue
                turn_id = int(turn.get("turn_id") or fallback_turn_id)
                result.append(f"{source}:{session_id}::turn_{turn_id:03d}")
        return result

    @staticmethod
    def session_ids(scenario: Dict[str, Any]) -> List[str]:
        source = str(scenario.get("dataset_source") or "unknown_source")
        scenario_id = str(scenario.get("scenario_id") or "unknown_scenario")
        return [
            f"{source}:{str(session.get('session_id') or f'{scenario_id}_session_{index:03d}')}"
            for index, session in enumerate(scenario.get("sessions") or [], start=1)
        ]


# Compatibility functions keep existing imports stable while callers migrate to the class.
load_existing_records = TeacherTraceStore.load
merge_records = TeacherTraceStore.merge_records
merge_session_records = TeacherTraceStore.merge_sessions
validate_combined_splits = TeacherTraceStore.validate_splits
scenario_sample_ids = TeacherTraceStore.sample_ids
scenario_session_record_ids = TeacherTraceStore.session_ids
