from __future__ import annotations

from typing import List, Optional

from .memory_manager import MemoryManager


class ContextAssembler:
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager

    def build(
        self,
        base_context: str,
        *,
        extra_sections: Optional[List[str]] = None,
    ) -> str:
        sections: List[str] = []
        memory_digest = self.memory_manager.build_memory_digest()
        if memory_digest:
            sections.append("Long-term memory digest:\n" + memory_digest)
        if extra_sections:
            sections.extend([section for section in extra_sections if section])
        sections.append(base_context)
        return "\n\n".join([section.strip() for section in sections if section and section.strip()])
