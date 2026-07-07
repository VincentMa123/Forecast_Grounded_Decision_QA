"""Tokenizer package split from the legacy monolithic module."""

import logging
from pathlib import Path
from typing import Optional, Union

from .core import DataTokenizer


logger = logging.getLogger(__name__)


def load_tokenizer(
    data_dir: Union[str, Path],
    vocab_size: Optional[int] = None,
) -> Optional[DataTokenizer]:
    tokenizer = DataTokenizer(
        data_dir=data_dir,
        vocab_size=vocab_size if vocab_size is not None else 0,
    )
    if tokenizer.load_stats():
        logger.info("Loaded tokenizer from %s", data_dir)
        return tokenizer
    logger.warning("Failed to load tokenizer stats from %s", data_dir)
    return None


__all__ = ["DataTokenizer", "load_tokenizer"]
