from __future__ import annotations

from pathlib import Path
from typing import Dict

from .workspace_template_manager import WorkspaceTemplateManager


class AgentWorkspaceManager:
    def __init__(self, backend_root: Path, agent_id: str, workspace_root_base: Path | None = None):
        self.backend_root = Path(backend_root)
        self.agent_id = agent_id or "default"
        self.template_manager = WorkspaceTemplateManager(self.backend_root, workspace_root_base=workspace_root_base)
        self.workspace_root = self.template_manager.ensure_agent_workspace(self.agent_id)
        self.memory_root = self.workspace_root / "memory"
        self.assets_root = self.workspace_root / "assets"
        self.trace_root = self.workspace_root / "context_trace"
        self.temporary_dir = self.workspace_root / "temporary_dir"
        self.reports_dir = self.workspace_root / "reports"
        self.plan_path = self.workspace_root / "plan.md"
        self.ensure_structure()

    def ensure_structure(self) -> None:
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.memory_root.mkdir(parents=True, exist_ok=True)
        (self.memory_root / "timeline").mkdir(parents=True, exist_ok=True)
        self.assets_root.mkdir(parents=True, exist_ok=True)
        self.trace_root.mkdir(parents=True, exist_ok=True)
        self.temporary_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def get_trace_path(self, session_id: str) -> Path:
        return self.trace_root / f"{session_id}.json"

    def as_dict(self) -> Dict[str, str]:
        return {
            "workspace_root": self.workspace_root.as_posix(),
            "memory_root": self.memory_root.as_posix(),
            "assets_root": self.assets_root.as_posix(),
            "trace_root": self.trace_root.as_posix(),
            "temporary_dir": self.temporary_dir.as_posix(),
            "reports_dir": self.reports_dir.as_posix(),
            "plan_path": self.plan_path.as_posix(),
        }
