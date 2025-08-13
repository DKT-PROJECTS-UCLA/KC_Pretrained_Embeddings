"""
sa_methods.py
~~~~~~~~~~~~~
Strategies for **Student-Action (SA)** embeddings.

CORRECTED: add_words strategy is handled directly in pipeline.py since it works
on question texts before embedding, not on embeddings after.

Only mirror and stack_antonym strategies are implemented here as they work
on already-generated embeddings.
"""

from __future__ import annotations

from enum import Enum

import torch

# -----------------------------------------------------------------------------#
# Enum + registry
# -----------------------------------------------------------------------------#
class SAEmbeddingStrategy(str, Enum):
    MIRROR = "mirror"
    ADD_WORDS = "add_words"  # Handled in pipeline.py, not here
    STACK_ANTONYM = "stack_antonym"


SA_FUNCTION_REGISTRY = {}


def _sa_register(kind: SAEmbeddingStrategy):
    def _decorator(fn):
        SA_FUNCTION_REGISTRY[kind] = fn
        return fn

    return _decorator


# -----------------------------------------------------------------------------#
# Strategy implementations (only for embedding-based strategies)
# -----------------------------------------------------------------------------#
@_sa_register(SAEmbeddingStrategy.MIRROR)
def sa_mirror(base_embedding: torch.Tensor, is_correct: bool) -> torch.Tensor:
    """
    Mirror-about-origin: 
    - INCORRECT responses use the original embedding: e
    - CORRECT responses use the reflected embedding: -e
    
    This preserves semantic relations while clearly distinguishing response correctness.
    Mirrored embeddings for the same KC maintain their relative distances and 
    semantic structure in the embedding space.
    
    Parameters
    ----------
    base_embedding : torch.Tensor
        The original KC embedding vector
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


# @_sa_register(SAEmbeddingStrategy.STACK_ANTONYM)
# def sa_stack_antonym(
#     base_embedding: torch.Tensor, 
#     is_correct: bool, 
#     *, 
#     text_to_embedding_fn,
#     correct_word: str = "CORRECT",
#     incorrect_word: str = "INCORRECT",
#     antonym_embeddings: Dict[str, torch.Tensor] | None = None,
# ) -> torch.Tensor:
#     """
#     Stack (concatenate) the base KC embedding with separately generated antonym embeddings.
    
#     This method generates embeddings for "CORRECT" and "INCORRECT" words separately 
#     from the KC embedding, then stacks them to create embeddings of twice the original size.
    
#     Parameters
#     ----------
#     base_embedding : torch.Tensor
#         The original KC embedding vector
#     is_correct : bool
#         True if student answered correctly, False if incorrect
#     text_to_embedding_fn : callable
#         Function to embed the antonym words (if not pre-computed)
#     correct_word : str
#         Word for correct responses (default: "CORRECT")
#     incorrect_word : str
#         Word for incorrect responses (default: "INCORRECT") 
#     antonym_embeddings : Dict[str, torch.Tensor], optional
#         Pre-computed embeddings for antonym words to avoid re-computation
#         Format: {"CORRECT": tensor, "INCORRECT": tensor}
        
#     Returns
#     -------
#     torch.Tensor
#         Concatenated embedding: [base_embedding; antonym_embedding]
#         Resulting dimension: 2 * original_dimension
        
#     Examples
#     --------
#     If base_embedding has shape [1536], result will have shape [3072]:
#     - For correct: [kc_embedding; correct_word_embedding]  
#     - For incorrect: [kc_embedding; incorrect_word_embedding]
#     """
#     if text_to_embedding_fn is None and antonym_embeddings is None:
#         raise ValueError(
#             "Either text_to_embedding_fn or antonym_embeddings must be provided"
#         )
    
#     # Choose the appropriate antonym word
#     antonym_word = correct_word if is_correct else incorrect_word
    
#     # Get antonym embedding
#     if antonym_embeddings and antonym_word in antonym_embeddings:
#         # Use pre-computed antonym embedding
#         antonym_embedding = antonym_embeddings[antonym_word]
#     else:
#         # Generate antonym embedding on-the-fly
#         if text_to_embedding_fn is None:
#             raise ValueError(f"No pre-computed embedding for '{antonym_word}' and no embedding function provided")
        
#         try:
#             antonym_embedding = text_to_embedding_fn(antonym_word)
#             if not isinstance(antonym_embedding, torch.Tensor):
#                 antonym_embedding = torch.tensor(antonym_embedding, dtype=torch.float32)
#         except Exception as e:
#             raise RuntimeError(f"Failed to embed antonym word '{antonym_word}': {e}") from e
    
#     # Ensure both embeddings are the same dtype
#     base_embedding = base_embedding.to(dtype=torch.float32)
#     antonym_embedding = antonym_embedding.to(dtype=torch.float32)
    
#     # Stack (concatenate) the embeddings
#     stacked_embedding = torch.cat([base_embedding, antonym_embedding], dim=0)
    
#     return stacked_embedding


@_sa_register(SAEmbeddingStrategy.STACK_ANTONYM)
def sa_stack_antonym(
    base_embedding: torch.Tensor, 
    is_correct: bool, 
    *, 
    text_to_embedding_fn=None,
    correct_word: str = "CORRECT",
    incorrect_word: str = "INCORRECT",
    antonym_embeddings: Dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Stack (concatenate) the base KC embedding with separately generated antonym embeddings."""
    
    # Choose the appropriate antonym word
    antonym_word = correct_word if is_correct else incorrect_word
    
    # Get antonym embedding - prioritize pre-computed embeddings
    if antonym_embeddings and antonym_word in antonym_embeddings:
        # Use pre-computed antonym embedding
        antonym_embedding = antonym_embeddings[antonym_word]
    elif text_to_embedding_fn is not None:
        # Generate antonym embedding on-the-fly
        try:
            antonym_embedding = text_to_embedding_fn(antonym_word)
            if not isinstance(antonym_embedding, torch.Tensor):
                antonym_embedding = torch.tensor(antonym_embedding, dtype=torch.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to embed antonym word '{antonym_word}': {e}") from e
    else:
        # FIXED: Fallback to global embedding function
        import embedding_models as em
        try:
            antonym_embedding = em._get_embedding(antonym_word, model=em._current_model, provider=em._current_provider)
            if not isinstance(antonym_embedding, torch.Tensor):
                antonym_embedding = torch.tensor(antonym_embedding, dtype=torch.float32)
        except Exception as e:
            # Last resort: create dummy embedding
            print(f"⚠️  Warning: No antonym embedding available for '{antonym_word}', using dummy")
            antonym_embedding = torch.randn(base_embedding.shape[0], dtype=torch.float32)
    
    # Ensure both embeddings are the same dtype
    base_embedding = base_embedding.to(dtype=torch.float32)
    antonym_embedding = antonym_embedding.to(dtype=torch.float32)
    
    # Stack (concatenate) the embeddings
    stacked_embedding = torch.cat([base_embedding, antonym_embedding], dim=0)
    
    return stacked_embedding


# NOTE: add_words strategy is handled directly in pipeline.py
# It doesn't need a function here because it works on question texts
# before embedding, not on embeddings themselves.

# Add a dummy function to prevent registry errors
@_sa_register(SAEmbeddingStrategy.ADD_WORDS)
def sa_add_words_placeholder(*args, **kwargs):
    """
    Placeholder for add_words strategy.
    
    The actual add_words implementation is handled directly in pipeline.py
    because it works on question texts before embedding, not on embeddings.
    """
    raise NotImplementedError(
        "add_words strategy is handled directly in pipeline.py, not here."
    )