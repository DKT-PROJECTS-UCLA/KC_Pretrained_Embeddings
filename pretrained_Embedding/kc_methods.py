"""
kc_methods.py
~~~~~~~~~~~~~
Strategies for **Knowledge-Component (KC)** level embeddings.

Each function must take:
    - `kc_id`          : str | int
    - `question_ids`   : Sequence[Hashable]
    - `question_embeddings` : Dict[Hashable, torch.Tensor]

and return a **torch.Tensor** embedding for *that* KC.

Feel free to add keyword arguments / helper objects as you implement.
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Dict, Hashable, Sequence

import torch


import logging

_logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------#
# Enum + factory registry
# -----------------------------------------------------------------------------#
class KCEmbeddingStrategy(str, Enum):
    AVERAGE = "average"
    STUFF_WINDOW = "stuff_window"
    KC_NAME_TEXT = "kc_name_text"
    SAMPLE_SINGLE_QUESTION = "sample_single_question"


KC_FUNCTION_REGISTRY = {}


def _kc_register(kind: KCEmbeddingStrategy):
    def _decorator(fn):
        KC_FUNCTION_REGISTRY[kind] = fn
        return fn

    return _decorator


# -----------------------------------------------------------------------------#
# Strategy implementations (fill these in later)
# -----------------------------------------------------------------------------#
@_kc_register(KCEmbeddingStrategy.AVERAGE)
def kc_average(
    kc_id: Hashable,
    question_ids: Sequence[Hashable],
    question_embeddings: Dict[Hashable, torch.Tensor],
) -> torch.Tensor:
    """
    Average the embeddings of *all* questions tagged with this KC.
    
    Questions that belong to multiple KCs contribute to the multiple KC embeddings.
    
    Parameters
    ----------
    kc_id : Hashable
        The knowledge component identifier
    question_ids : Sequence[Hashable]
        List of question IDs that belong to this KC
    question_embeddings : Dict[Hashable, torch.Tensor]
        Mapping from question ID to its embedding vector
        
    Returns
    -------
    torch.Tensor
        Average embedding representing this KC
        
    Raises
    ------
    ValueError
        If no valid question embeddings are found for this KC
    """
    valid_embeddings = []
    missing_questions = []
    
    # Collect all valid embeddings for this KC
    for qid in question_ids:
        if qid in question_embeddings:
            valid_embeddings.append(question_embeddings[qid])
        else:
            missing_questions.append(qid)
    
    # Handle edge cases
    if not valid_embeddings:
        raise ValueError(
            f"KC '{kc_id}' has no valid question embeddings. "
            f"Missing questions: {missing_questions}"
        )
    
    # Log warnings for missing questions (but continue)
    if missing_questions:
        _logger.warning(
            f"KC '{kc_id}': {len(missing_questions)} questions missing embeddings "
            f"(using {len(valid_embeddings)} valid questions)"
        )
    
    # Stack all embeddings and compute mean
    # This handles different embedding dimensions gracefully
    stacked_embeddings = torch.stack(valid_embeddings)  # Shape: [num_questions, embed_dim]
    kc_embedding = torch.mean(stacked_embeddings, dim=0)  # Shape: [embed_dim]
    
    _logger.debug(
        f"KC '{kc_id}': averaged {len(valid_embeddings)} question embeddings "
        f"to create KC embedding of shape {kc_embedding.shape}"
    )
    
    return kc_embedding


@_kc_register(KCEmbeddingStrategy.STUFF_WINDOW)
def kc_stuff_window(
    kc_id: Hashable,
    question_ids: Sequence[Hashable],
    question_embeddings: Dict[Hashable, torch.Tensor],
    *,
    window_size: int = 10,
) -> torch.Tensor:
    """
    Pack (stuff) questions into a context window until *window_size* is reached,
    then derive a single embedding (e.g. by averaging).  Exactly how you derive
    that final vector is up to you—modify as needed.
    """
    raise NotImplementedError("Implement the 'stuff_window' KC embedding strategy.")


@_kc_register(KCEmbeddingStrategy.KC_NAME_TEXT)
def kc_name_as_text(
    kc_id: Hashable,
    question_ids: Sequence[Hashable],
    question_embeddings: Dict[Hashable, torch.Tensor],
    *,
    text_to_embedding_fn,  # inject dependency to avoid circular import
) -> torch.Tensor:
    """
    Treat the KC *name* itself as text and embed it (via OpenAI or other model).
    Pass in `text_to_embedding_fn` so this module stays provider-agnostic.
    """
    raise NotImplementedError("Implement the 'kc_name_text' embedding strategy.")


@_kc_register(KCEmbeddingStrategy.SAMPLE_SINGLE_QUESTION)
def kc_sample_question(
    kc_id: Hashable,
    question_ids: Sequence[Hashable],
    question_embeddings: Dict[Hashable, torch.Tensor],
    *,
    rng: random.Random | None = None,
) -> torch.Tensor:
    """
    Sample **one** question at random from the KC’s pool and use its embedding
    directly.  Good if you want low-variance vectors at runtime.
    """
    raise NotImplementedError("Implement the 'sample_single_question' strategy.")
