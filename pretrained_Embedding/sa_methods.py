"""
sa_methods.py
~~~~~~~~~~~~~
Strategies for **Student-Action (SA)** embeddings.  SA embeddings depend on the
question embedding *and* the student’s response correctness.

For *binary* responses we need **two** vectors per question (or per KC in classic
DKT):  one for “correct”, one for “incorrect”.

Each callable must take:
    - `base_embedding` : torch.Tensor      # the original question/KC vector
    - `is_correct`     : bool              # student response
and return a **torch.Tensor** of identical dimensionality.

Implementations are left as TODOs—you’ll fill them in later.
"""

from __future__ import annotations

from enum import Enum

import torch

# -----------------------------------------------------------------------------#
# Enum + registry
# -----------------------------------------------------------------------------#
class SAEmbeddingStrategy(str, Enum):
    MIRROR = "mirror"
    ADD_WORDS = "add_words"
    STACK_ANTONYM = "stack_antonym"


SA_FUNCTION_REGISTRY = {}


def _sa_register(kind: SAEmbeddingStrategy):
    def _decorator(fn):
        SA_FUNCTION_REGISTRY[kind] = fn
        return fn

    return _decorator


# -----------------------------------------------------------------------------#
# Strategy stubs
# -----------------------------------------------------------------------------#
@_sa_register(SAEmbeddingStrategy.MIRROR)
def sa_mirror(base_embedding: torch.Tensor, is_correct: bool) -> torch.Tensor:
    """
    Mirror-about-origin: 
    - INCORRECT responses use the original embedding: e
    - CORRECT responses use the reflected embedding: -e
    
    This preserves semantic relations while clearly distinguishing response correctness.
    Mirrored embeddings for the same question maintain their relative distances and 
    semantic structure in the embedding space.
    
    Parameters
    ----------
    base_embedding : torch.Tensor
        The original question or KC embedding vector
    is_correct : bool
        True if student answered correctly, False if incorrect
        
    Returns
    -------
    torch.Tensor
        - Original embedding if is_correct=False (incorrect)
        - Negated embedding if is_correct=True (correct)
    """
    if is_correct:
        # CORRECT response: reflect about origin (-e)
        return -base_embedding
    else:
        # INCORRECT response: use original embedding (e)
        return base_embedding.clone()  # Clone to avoid modifying original



@_sa_register(SAEmbeddingStrategy.ADD_WORDS)
def sa_add_words(base_embedding: torch.Tensor, is_correct: bool, *, extra_vector: torch.Tensor | None = None) -> torch.Tensor:
    """
    Add an “explanatory” vector (e.g. embedding of the word 'correct' or
    'incorrect') to the base embedding.  Caller injects *extra_vector* so we
    avoid hard-coding provider logic in this module.
    """
    raise NotImplementedError("Implement the 'add_words' SA strategy.")


@_sa_register(SAEmbeddingStrategy.STACK_ANTONYM)
def sa_stack_antonym(base_embedding: torch.Tensor, is_correct: bool, *, antonym_vector: torch.Tensor | None = None) -> torch.Tensor:
    """
    Concatenate the base embedding with an antonym word vector and project back
    to original dimension (e.g. via linear layer)—implementation is your call.
    """
    raise NotImplementedError("Implement the 'stack_antonym' SA strategy.")
