from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _resolve_datetime(event_time: Optional[str] = None) -> datetime:
    if not event_time:
        return datetime.now()
    try:
        return datetime.fromisoformat(event_time)
    except ValueError:
        return datetime.now()


class MemoryManager:
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = Path(workspace_dir)
        self.memory_dir = self.workspace_dir / "memory"
        self.timeline_dir = self.memory_dir / "timeline"
        self.timeline_dir.mkdir(parents=True, exist_ok=True)

    def list_timeline_files(self) -> List[Path]:
        return sorted(
            [path for path in self.timeline_dir.glob("*.md") if path.name.upper() != "README.md"],
            key=lambda item: item.as_posix(),
        )

    def build_full_memory_text(self) -> str:
        blocks = []
        for path in self.list_timeline_files():
            blocks.append(f"## {path.name}\n{path.read_text(encoding='utf-8')}")
        return "\n\n".join(blocks)

    def commit_turn(
        self,
        *,
        session_id: str,
        user_message: str,
        assistant_message: str,
        memory_updates: Optional[List[str]] = None,
        artifacts: Optional[List[str]] = None,
        event_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_dt = _resolve_datetime(event_time)
        today = resolved_dt.strftime("%Y-%m-%d")
        path = self.timeline_dir / f"{today}.md"
        existing = path.read_text(encoding="utf-8") if path.exists() else f"# {today}\n"
        lines = [
            "",
            f"## {resolved_dt.strftime('%H:%M:%S')} | session {session_id}",
            f"- User: {user_message}",
            f"- Assistant: {assistant_message}",
        ]
        if memory_updates:
            lines.append("- Memory commits:")
            lines.extend(f"  - {item}" for item in memory_updates if item)
        if artifacts:
            lines.append("- Artifacts:")
            lines.extend(f"  - {item}" for item in artifacts if item)
        path.write_text(existing.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
        return {"memory_commit_path": path.as_posix()}

    def get_summary(self) -> Dict[str, Any]:
        files = self.list_timeline_files()
        latest_path = files[-1] if files else None
        excerpt = latest_path.read_text(encoding="utf-8")[:1200] if latest_path else ""
        return {
            "workspace_root": self.workspace_dir.as_posix(),
            "timeline_files": [path.relative_to(self.workspace_dir).as_posix() for path in files],
            "latest_timeline_path": latest_path.relative_to(self.workspace_dir).as_posix() if latest_path else None,
            "latest_timeline_excerpt": excerpt,
        }
