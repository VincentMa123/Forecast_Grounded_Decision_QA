from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .io_utils import load_json


def load_scenarios(path: Path) -> List[Dict[str, Any]]:
    scenarios = load_json(path)
    if not isinstance(scenarios, list):
        raise TypeError(f"Scenario file must contain a JSON list: {path}")
    return scenarios


def find_scenario(scenarios: List[Dict[str, Any]], scenario_id: str) -> Dict[str, Any]:
    scenario = next((item for item in scenarios if item.get("scenario_id") == scenario_id), None)
    if scenario is None:
        raise ValueError(f"Scenario id not found: {scenario_id}")
    return scenario


def first_user_input(scenario: Dict[str, Any]) -> str:
    sessions = scenario.get("sessions") or []
    if not sessions:
        raise ValueError(f"Scenario has no sessions: {scenario.get('scenario_id')}")
    dialogue = sessions[0].get("dialogue") or []
    if not dialogue:
        raise ValueError(f"Scenario session has no dialogue: {scenario.get('scenario_id')}")
    return str(dialogue[0].get("user_input") or "")