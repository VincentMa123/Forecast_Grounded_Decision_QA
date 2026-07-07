"""
Skill Manager - lightweight skill discovery without external YAML dependency.
"""
from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)
MAX_DESCRIPTION_CHARS = 300


@dataclass
class SkillMeta:
    name: str
    description: str
    skill_md_path: str
    skill_dir: str


class SkillManager:
    def __init__(self, skill_root: Path):
        self.skill_root = Path(skill_root).resolve()
        self.skills: Dict[str, SkillMeta] = {}
        self._discover_skills()

    def _discover_skills(self) -> None:
        if not self.skill_root.exists():
            return
        for skill_dir in sorted(self.skill_root.iterdir(), key=lambda item: item.as_posix()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            meta = self._parse_skill_md(skill_md, skill_dir)
            if meta:
                self.skills[meta.name] = meta

    def _parse_skill_md(self, skill_md: Path, skill_dir: Path) -> Optional[SkillMeta]:
        try:
            content = skill_md.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("读取 skill 失败: %s", exc)
            return None
        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        name = skill_dir.name
        description = ""
        if frontmatter_match:
            for line in frontmatter_match.group(1).splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key == "name" and value:
                    name = value
                elif key == "description" and value:
                    description = value
        if not description:
            body = content.split("---", 2)[-1].strip()
            description = body.splitlines()[0].strip() if body else skill_dir.name
        description = " ".join(description.split())
        return SkillMeta(name=name, description=description, skill_md_path=str(skill_md), skill_dir=str(skill_dir))

    def render_skills_section(self) -> str:
        if not self.skills:
            return ""
        lines = ["## Skills", "When a task matches a skill, read its SKILL.md first before using it."]
        for skill in sorted(self.skills.values(), key=lambda item: item.name):
            desc = skill.description
            if len(desc) > MAX_DESCRIPTION_CHARS:
                desc = desc[: MAX_DESCRIPTION_CHARS - 3] + "..."
            lines.append(f"- {skill.name}: {desc}")
            lines.append(f"  - file: {skill.skill_md_path}")
        return "\n".join(lines)

    def clear_temporary_scripts(self) -> None:
        if not self.skill_root.exists():
            return
        for skill_dir in self.skill_root.iterdir():
            if not skill_dir.is_dir():
                continue
            temp_dir = skill_dir / "temporary_scripts"
            temp_dir.mkdir(parents=True, exist_ok=True)
            for target in temp_dir.iterdir():
                try:
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                except Exception as exc:
                    logger.warning("清理临时脚本失败 %s: %s", target, exc)
