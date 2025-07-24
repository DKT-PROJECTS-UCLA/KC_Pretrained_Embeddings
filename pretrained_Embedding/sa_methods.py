"""
sa_methods.py
~~~~~~~~~~~~~
Strategies for **Student-Action (SA)** embeddings.  SA embeddings depend on the
question embedding *and* the student's response correctness.

For *binary* responses we need **two** vectors per question (or per KC in classic
DKT):  one for "correct", one for "incorrect".

Each callable must take:
    - `base_embedding` : torch.Tensor      # the original question/KC vector
    - `is_correct`     : bool              # student response
and return a **torch.Tensor** of identical dimensionality.

Implementations are left as TODOs—you'll fill them in later.
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
# Strategy implementations
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
def sa_add_words(
    base_embedding: torch.Tensor, 
    is_correct: bool, 
    *, 
    question_text: str,
    text_to_embedding_fn,
    correct_word: str = "CORRECT",
    incorrect_word: str = "INCORRECT", 
    repetitions: int = 1,
) -> torch.Tensor:
    """
    Add "CORRECT" or "INCORRECT" words to the question text and re-embed.
    
    This method appends correctness words to the text before embedding.
    Since contextualized word embeddings get averaged in a sentence, there's 
    a hyperparameter (repetitions) that creates a tradeoff between capturing 
    text information and capturing student action information.
    
    Parameters
    ----------
    base_embedding : torch.Tensor
        The original question or KC embedding (not used in this strategy)
    is_correct : bool
        True if student answered correctly, False if incorrect
    question_text : str
        Original question text to modify
    text_to_embedding_fn : callable
        Function to embed the modified text
    correct_word : str
        Word to prepend for correct responses (default: "CORRECT")
    incorrect_word : str  
        Word to prepend for incorrect responses (default: "INCORRECT")
    repetitions : int
        Number of times to repeat the correctness word (default: 1)
        Higher values emphasize correctness signal vs. semantic content
        
    Returns
    -------
    torch.Tensor
        Embedding of the modified text with correctness words prepended
        
    Examples
    --------
    Input: "What is (-5) x 1 ?", is_correct=False, repetitions=1
    Output: Embedding of "INCORRECT What is (-5) x 1 ?"
    
    Input: "What is (-5) x 1 ?", is_correct=True, repetitions=2  
    Output: Embedding of "CORRECT CORRECT What is (-5) x 1 ?"
    """
    if text_to_embedding_fn is None:
        raise ValueError("text_to_embedding_fn is required for add_words strategy")
    
    # Choose the appropriate word based on correctness
    correctness_word = correct_word if is_correct else incorrect_word
    
    # Create repeated prefix
    prefix_words = [correctness_word] * repetitions
    prefix = " ".join(prefix_words)
    
    # Combine prefix with question text
    modified_text = f"{prefix} {question_text.strip()}"
    
    # Embed the modified text
    try:
        sa_embedding = text_to_embedding_fn(modified_text)
        if not isinstance(sa_embedding, torch.Tensor):
            sa_embedding = torch.tensor(sa_embedding, dtype=torch.float32)
        return sa_embedding
    except Exception as e:
        raise RuntimeError(f"Failed to embed modified text: {e}") from e


@_sa_register(SAEmbeddingStrategy.STACK_ANTONYM)
def sa_stack_antonym(
    base_embedding: torch.Tensor, 
    is_correct: bool, 
    *, 
    text_to_embedding_fn,
    correct_word: str = "CORRECT",
    incorrect_word: str = "INCORRECT",
    antonym_embeddings: Dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    Stack (concatenate) the base embedding with separately generated antonym embeddings.
    
    This method generates embeddings for "CORRECT" and "INCORRECT" words separately 
    from the question text, then stacks them with question/KC embeddings to create 
    embeddings of twice the original size.
    
    Parameters
    ----------
    base_embedding : torch.Tensor
        The original question or KC embedding vector
    is_correct : bool
        True if student answered correctly, False if incorrect
    text_to_embedding_fn : callable
        Function to embed the antonym words (if not pre-computed)
    correct_word : str
        Word for correct responses (default: "CORRECT")
    incorrect_word : str
        Word for incorrect responses (default: "INCORRECT") 
    antonym_embeddings : Dict[str, torch.Tensor], optional
        Pre-computed embeddings for antonym words to avoid re-computation
        Format: {"CORRECT": tensor, "INCORRECT": tensor}
        
    Returns
    -------
    torch.Tensor
        Concatenated embedding: [base_embedding; antonym_embedding]
        Resulting dimension: 2 * original_dimension
        
    Examples
    --------
    If base_embedding has shape [768], result will have shape [1536]:
    - For correct: [question_embedding; correct_word_embedding]  
    - For incorrect: [question_embedding; incorrect_word_embedding]
    """
    if text_to_embedding_fn is None and antonym_embeddings is None:
        raise ValueError(
            "Either text_to_embedding_fn or antonym_embeddings must be provided"
        )
    
    # Choose the appropriate antonym word
    antonym_word = correct_word if is_correct else incorrect_word
    
    # Get antonym embedding
    if antonym_embeddings and antonym_word in antonym_embeddings:
        # Use pre-computed antonym embedding
        antonym_embedding = antonym_embeddings[antonym_word]
    else:
        # Generate antonym embedding on-the-fly
        if text_to_embedding_fn is None:
            raise ValueError(f"No pre-computed embedding for '{antonym_word}' and no embedding function provided")
        
        try:
            antonym_embedding = text_to_embedding_fn(antonym_word)
            if not isinstance(antonym_embedding, torch.Tensor):
                antonym_embedding = torch.tensor(antonym_embedding, dtype=torch.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to embed antonym word '{antonym_word}': {e}") from e
    
    # Ensure both embeddings are the same dtype
    base_embedding = base_embedding.to(dtype=torch.float32)
    antonym_embedding = antonym_embedding.to(dtype=torch.float32)
    
    # Stack (concatenate) the embeddings
    stacked_embedding = torch.cat([base_embedding, antonym_embedding], dim=0)
    
    return stacked_embedding