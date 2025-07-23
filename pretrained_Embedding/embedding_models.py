"""
embedding_models.py
~~~~~~~~~~~~~~~~~~~
Utility wrappers around the OpenAI embeddings endpoint (or any future provider).

Main public callable
--------------------
get_question_embeddings(df, id_col="question_id", text_col="question_text", ...)

Returns
-------
Dict[Any, torch.Tensor]  # {question_id: embedding_vector}
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Dict, Hashable, Iterable

import openai
import pandas as pd
import torch
from tqdm.auto import tqdm

_logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------#
# Configuration
# -----------------------------------------------------------------------------#
DEFAULT_MODEL = "text-embedding-3-small"
openai.api_key = os.getenv("OPENAI_API_KEY")  # must be supplied by caller


# -----------------------------------------------------------------------------#
# Low-level embedding helper
# -----------------------------------------------------------------------------#
@lru_cache(maxsize=8_192)
def _get_embedding(text: str, *, model: str = DEFAULT_MODEL) -> torch.Tensor:
    """Embed *single* piece of text, memoising so repeated calls are free."""
    if not openai.api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found—set it in your environment before calling "
            "get_question_embeddings()."
        )

    resp = openai.embeddings.create(input=[text], model=model)
    vec = resp.data[0].embedding  # List[float]
    return torch.tensor(vec, dtype=torch.float32)


# -----------------------------------------------------------------------------#
# Public API
# -----------------------------------------------------------------------------#
def get_question_embeddings(
    df: pd.DataFrame,
    *,
    id_col: str = "question_id",
    text_col: str = "question_text",
    model: str = DEFAULT_MODEL,
    show_progress: bool = True,
) -> Dict[Hashable, torch.Tensor]:
    """
    Generate an embedding vector for every row in *df* and return an ID-to-tensor
    dict suitable for downstream use in DKT/qDKT.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least the columns given by *id_col* and *text_col*.
    id_col, text_col : str
        Column names for the question identifier and the raw text.
    model : str
        OpenAI embedding model name.
    show_progress : bool
        If True, wrap iteration in `tqdm` progress bar.

    Notes
    -----
    *Duplicate IDs* are aggregated by **averaging** their embeddings.
    """
    if id_col not in df or text_col not in df:
        raise KeyError(
            f"DataFrame must contain '{id_col}' and '{text_col}' columns."
        )

    iterator: Iterable = tqdm(df.itertuples(index=False), desc="Embedding questions", disable=not show_progress)  # type: ignore[arg-type]

    embed_map: Dict[Hashable, list[torch.Tensor]] = {}

    for row in iterator:  # type: ignore[assignment]
        qid = getattr(row, id_col)
        qtext = getattr(row, text_col)
        embed_map.setdefault(qid, []).append(_get_embedding(qtext, model=model))

    # Aggregate duplicates (rare but safer to guard against)
    averaged: Dict[Hashable, torch.Tensor] = {
        qid: torch.stack(vectors).mean(dim=0) for qid, vectors in embed_map.items()
    }

    _logger.info("Generated %d unique question embeddings.", len(averaged))
    return averaged
