from __future__ import annotations

from pathlib import Path
from typing import Dict, List


CONTROL_FILES = [
    "AGENTS.md",
    "SOUL.md",
    "IDENTITY.md",
    "USER.md",
    "TOOLS.md",
    "BOOTSTRAP.md",
    "MEMORY.md",
]


class MemoryAssembler:
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)
        self.memory_root = self.workspace_root / "memory"
        self.assets_root = self.workspace_root / "assets"
        self.trace_root = self.workspace_root / "context_trace"

    def build(self, session_id: str) -> Dict[str, List[Dict[str, str]]]:
        control_files = []
        for name in CONTROL_FILES:
            path = self.workspace_root / name
            if path.exists():
                control_files.append({
                    "name": name,
                    "path": path.as_posix(),
                    "content": path.read_text(encoding="utf-8"),
                })

        memory_files = []
        memory_blocks = []
        for path in sorted(self.memory_root.rglob("*.md"), key=lambda item: item.as_posix()):
            content = path.read_text(encoding="utf-8")
            entry = {
                "name": path.relative_to(self.workspace_root).as_posix(),
                "path": path.as_posix(),
                "content": content,
            }
            memory_files.append({"path": entry["path"], "name": entry["name"]})
            memory_blocks.append(entry)

        assets = []
        for path in sorted(self.assets_root.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_file():
                assets.append({"path": path.relative_to(self.workspace_root).as_posix()})

        trace_path = self.trace_root / f"{session_id}.json"
        trace_meta = {
            "trace_path": trace_path.as_posix(),
            "memory_file_count": str(len(memory_blocks)),
            "asset_file_count": str(len(assets)),
        }

        return {
            "control_files": control_files,
            "memory_files": memory_files,
            "memory_content_blocks": memory_blocks,
            "assets": assets,
            "trace_meta": trace_meta,
        }
