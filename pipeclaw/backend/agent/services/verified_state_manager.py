from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict

from pipeclaw.backend.grounding.decision_trace_state import VerifiedDecisionState


_SAFE_SESSION = re.compile(r"[^A-Za-z0-9_.-]+")


class VerifiedStateManager:
    """Persist verified state snapshots and replayable state events."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)
        self.state_root = self.workspace_root / "memory" / "state"
        self.state_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_session_id(session_id: str) -> str:
        value = _SAFE_SESSION.sub("_", str(session_id)).strip("._")
        if not value:
            raise ValueError("session_id must contain at least one safe character")
        return value

    def snapshot_path(self, session_id: str) -> Path:
        return self.state_root / f"{self._safe_session_id(session_id)}.json"

    def event_path(self, session_id: str) -> Path:
        return self.state_root / f"{self._safe_session_id(session_id)}.events.jsonl"

    def commit(
        self,
        session_id: str,
        state: VerifiedDecisionState,
    ) -> Dict[str, Any]:
        payload = state.to_dict()
        event_path = self.event_path(session_id)
        event_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path = self.snapshot_path(session_id)
        temporary = snapshot_path.with_suffix(".json.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, snapshot_path)
        except Exception:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise

        with event_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    {"event": "state_snapshot", "state": payload},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
        return {
            "state_snapshot_path": snapshot_path.as_posix(),
            "state_event_path": event_path.as_posix(),
        }

    def load(self, session_id: str) -> VerifiedDecisionState:
        snapshot = self.snapshot_path(session_id)
        if snapshot.is_file():
            try:
                return VerifiedDecisionState.from_dict(
                    json.loads(snapshot.read_text(encoding="utf-8-sig"))
                )
            except (ValueError, json.JSONDecodeError):
                pass

        event_path = self.event_path(session_id)
        latest: Dict[str, Any] | None = None
        if event_path.is_file():
            for line in event_path.read_bytes().splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line.decode("utf-8-sig"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if (
                    event.get("event") == "state_snapshot"
                    and isinstance(event.get("state"), dict)
                ):
                    latest = dict(event["state"])
        return (
            VerifiedDecisionState.from_dict(latest)
            if latest is not None
            else VerifiedDecisionState()
        )
