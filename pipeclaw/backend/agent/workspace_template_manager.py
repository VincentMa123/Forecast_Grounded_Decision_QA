from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable


CONTROL_FILES = [
    "AGENTS.md",
    "SOUL.md",
    "IDENTITY.md",
    "USER.md",
    "TOOLS.md",
    "BOOTSTRAP.md",
    "MEMORY.md",
]


class WorkspaceTemplateManager:
    def __init__(self, backend_root: Path, workspace_root_base: Path | None = None):
        self.backend_root = Path(backend_root)
        self.templates_root = self.backend_root / "workspace-templates"
        self.workspace_root_base = Path(workspace_root_base) if workspace_root_base is not None else (self.backend_root / ".openclaw")
        self.default_workspace = self.workspace_root_base / "workspace-default"

    def ensure_default_workspace(self) -> Path:
        self._ensure_templates_exist()
        self._copy_template_tree(self.default_workspace)
        return self.default_workspace

    def ensure_agent_workspace(self, agent_id: str) -> Path:
        self.ensure_default_workspace()
        workspace_root = self.workspace_root_base / f"workspace-{agent_id}"
        self._copy_template_tree(workspace_root)
        return workspace_root

    def _ensure_templates_exist(self) -> None:
        self.templates_root.mkdir(parents=True, exist_ok=True)
        self.workspace_root_base.mkdir(parents=True, exist_ok=True)
        for relative in CONTROL_FILES:
            target = self.templates_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_text(f"# {target.stem}\n", encoding="utf-8")
        for relative_dir in [
            self.templates_root / "memory" / "timeline",
            self.templates_root / "assets",
            self.templates_root / "context_trace",
            self.templates_root / "temporary_dir",
            self.templates_root / "reports",
        ]:
            relative_dir.mkdir(parents=True, exist_ok=True)

    def _copy_template_tree(self, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        for source in self._iter_template_items():
            relative = source.relative_to(self.templates_root)
            target = destination / relative
            if source.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)

    def _iter_template_items(self) -> Iterable[Path]:
        yield self.templates_root
        for path in sorted(self.templates_root.rglob("*"), key=lambda item: item.as_posix()):
            yield path
