"""Helpers for managing tokenizer metadata rows."""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from .types import MetadataRow


@dataclass
class TokenMetadataStore:
    """In-memory index of metadata rows keyed by token and variable."""

    rows: List[MetadataRow] = field(default_factory=list)
    by_token_id: Dict[int, List[MetadataRow]] = field(default_factory=dict)
    by_variable: Dict[int, List[MetadataRow]] = field(default_factory=dict)

    def reset(self) -> None:
        """Clear all metadata rows and indexes."""

        self.rows.clear()
        self.by_token_id.clear()
        self.by_variable.clear()

    def register(self, row: MetadataRow) -> None:
        """Register a single metadata row."""

        normalised = dict(row)
        normalised.setdefault("duplicate_of", None)
        token_id = int(normalised["token_id"])
        variable_index = int(normalised["variable_index"])
        self.rows.append(normalised)
        self.by_token_id.setdefault(token_id, []).append(normalised)
        self.by_variable.setdefault(variable_index, []).append(normalised)

    def register_many(self, rows: Iterable[MetadataRow]) -> None:
        for row in rows:
            self.register(row)

    def alias_rows(
        self,
        canonical_rows: List[MetadataRow],
        alias_index: int,
        alias_name: str,
        canonical_name: str,
    ) -> List[MetadataRow]:
        """Clone metadata rows for a duplicated variable."""

        alias_rows: List[MetadataRow] = []
        for row in canonical_rows:
            clone = dict(row)
            clone["variable_index"] = alias_index
            clone["variable_name"] = alias_name
            clone["duplicate_of"] = canonical_name
            alias_rows.append(clone)
        return alias_rows

    def rows_for_variable(self, variable_index: int) -> List[MetadataRow]:
        return self.by_variable.get(int(variable_index), [])

    def find_token(self, token_id: int, variable_index: int) -> MetadataRow:
        for row in self.by_token_id.get(int(token_id), []):
            if int(row["variable_index"]) == int(variable_index):
                return row
        raise KeyError(
            f"Missing metadata for token_id={token_id} and variable_index={variable_index}"
        )


__all__ = ["TokenMetadataStore"]
