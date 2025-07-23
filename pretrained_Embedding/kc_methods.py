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

import logging
import random
from enum import Enum
from typing import Dict, Hashable, Sequence

import pandas as pd
import torch

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
# Strategy implementations
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
    window_size: int = 8192,  # Default to OpenAI context window
    question_df: pd.DataFrame | None = None,
    text_col: str = "question_text",
    id_col: str = "question_id",
    text_to_embedding_fn=None,  # Function to embed the concatenated text
    separator: str = " [SEP] ",  # Separator between questions
    rng: random.Random | None = None,
) -> torch.Tensor:
    """
    Pack (stuff) questions into a context window by concatenating question texts,
    then embed the concatenated text as a single KC representation.
    
    The goal is to capture as many words belonging to a KC as possible to 
    represent the KC more accurately by leveraging the full vocabulary.
    
    Parameters
    ----------
    kc_id : Hashable
        The knowledge component identifier
    question_ids : Sequence[Hashable]
        List of question IDs that belong to this KC
    question_embeddings : Dict[Hashable, torch.Tensor]
        Not used in this strategy (we re-embed the concatenated text)
    window_size : int
        Maximum context window size in characters (default: 8192 for OpenAI)
    question_df : pd.DataFrame
        DataFrame containing question texts (required for this strategy)
    text_col : str
        Column name containing question text
    id_col : str
        Column name containing question IDs
    text_to_embedding_fn : callable
        Function to embed the concatenated text string
    separator : str
        String to separate individual questions in concatenation
    rng : random.Random
        Random number generator for reproducible sampling
        
    Returns
    -------
    torch.Tensor
        Embedding of the concatenated question texts for this KC
        
    Raises
    ------
    ValueError
        If required parameters are missing or no questions can be processed
    """
    if question_df is None:
        raise ValueError("question_df is required for stuff_window strategy")
    if text_to_embedding_fn is None:
        raise ValueError("text_to_embedding_fn is required for stuff_window strategy")
    
    if rng is None:
        rng = random.Random(hash(kc_id))  # Deterministic based on KC ID
    
    # Get question texts for this KC
    question_texts = {}
    for _, row in question_df.iterrows():
        qid = row[id_col]
        if qid in question_ids:
            question_texts[qid] = row[text_col]
    
    if not question_texts:
        raise ValueError(f"KC '{kc_id}' has no valid question texts available")
    
    # Randomly shuffle questions for sampling
    available_qids = list(question_texts.keys())
    rng.shuffle(available_qids)
    
    # Pack questions into context window
    concatenated_parts = []
    total_length = 0
    used_questions = 0
    
    for qid in available_qids:
        question_text = question_texts[qid].strip()
        
        # Calculate length including separator
        additional_length = len(question_text)
        if concatenated_parts:  # Add separator length if not first question
            additional_length += len(separator)
        
        # Check if adding this question would exceed window
        if total_length + additional_length > window_size:
            break
            
        # Add question to concatenation
        concatenated_parts.append(question_text)
        total_length += additional_length
        used_questions += 1
    
    if not concatenated_parts:
        raise ValueError(f"KC '{kc_id}': No questions fit within window_size={window_size}")
    
    # Create concatenated text
    concatenated_text = separator.join(concatenated_parts)
    
    _logger.debug(
        f"KC '{kc_id}': stuffed {used_questions}/{len(available_qids)} questions "
        f"into {len(concatenated_text)} characters (limit: {window_size})"
    )
    
    # Embed the concatenated text
    try:
        kc_embedding = text_to_embedding_fn(concatenated_text)
        if not isinstance(kc_embedding, torch.Tensor):
            kc_embedding = torch.tensor(kc_embedding, dtype=torch.float32)
        return kc_embedding
    except Exception as e:
        raise RuntimeError(f"Failed to embed concatenated text for KC '{kc_id}': {e}") from e


@_kc_register(KCEmbeddingStrategy.KC_NAME_TEXT)
def kc_name_as_text(
    kc_id: Hashable,
    question_ids: Sequence[Hashable],
    question_embeddings: Dict[Hashable, torch.Tensor],
    *,
    text_to_embedding_fn,  # inject dependency to avoid circular import
    kc_definitions: Dict[Hashable, str] | None = None,
    fallback_to_id: bool = True,
) -> torch.Tensor:
    """
    Treat the KC name/definition as text and embed it directly.
    
    When each KC is manually defined, a language-based definition is provided.
    Although a short description, the few words can still be utilized to create 
    embeddings that capture semantic relations between KCs.
    
    Parameters
    ----------
    kc_id : Hashable
        The knowledge component identifier
    question_ids : Sequence[Hashable]
        List of question IDs (not used in this strategy)
    question_embeddings : Dict[Hashable, torch.Tensor]
        Question embeddings (not used in this strategy)
    text_to_embedding_fn : callable
        Function to embed the KC definition text
    kc_definitions : Dict[Hashable, str]
        Mapping from KC ID to its textual definition/description
    fallback_to_id : bool
        If True, use KC ID as text when definition is missing
        
    Returns
    -------
    torch.Tensor
        Embedding of the KC definition text
        
    Raises
    ------
    ValueError
        If no text can be found for the KC and fallback is disabled
    """
    if text_to_embedding_fn is None:
        raise ValueError("text_to_embedding_fn is required for kc_name_text strategy")
    
    # Get KC definition text
    kc_text = None
    
    if kc_definitions and kc_id in kc_definitions:
        kc_text = kc_definitions[kc_id].strip()
    
    # Fallback to KC ID if no definition available
    if not kc_text and fallback_to_id:
        # Convert KC ID to readable text (handle underscores, etc.)
        kc_text = str(kc_id).replace('_', ' ').replace('-', ' ').strip()
        _logger.debug(f"KC '{kc_id}': No definition found, using ID as text: '{kc_text}'")
    
    if not kc_text:
        raise ValueError(
            f"KC '{kc_id}': No definition found and fallback_to_id=False"
        )
    
    _logger.debug(f"KC '{kc_id}': Embedding definition text: '{kc_text}'")
    
    # Embed the KC definition text
    try:
        kc_embedding = text_to_embedding_fn(kc_text)
        if not isinstance(kc_embedding, torch.Tensor):
            kc_embedding = torch.tensor(kc_embedding, dtype=torch.float32)
        return kc_embedding
    except Exception as e:
        raise RuntimeError(f"Failed to embed KC definition for '{kc_id}': {e}") from e


@_kc_register(KCEmbeddingStrategy.SAMPLE_SINGLE_QUESTION)
def kc_sample_question(
    kc_id: Hashable,
    question_ids: Sequence[Hashable],
    question_embeddings: Dict[Hashable, torch.Tensor],
    *,
    rng: random.Random | None = None,
) -> torch.Tensor:
    """
    Sample **one** question at random from the KC's pool and use its embedding
    directly as the KC representation.
    
    If two KCs are related semantically, then questions from the KCs should 
    also have semantic relation. This method leverages that property by using
    a single representative question embedding.
    
    Parameters
    ----------
    kc_id : Hashable
        The knowledge component identifier
    question_ids : Sequence[Hashable]
        List of question IDs that belong to this KC
    question_embeddings : Dict[Hashable, torch.Tensor]
        Mapping from question ID to its embedding vector
    rng : random.Random
        Random number generator for reproducible sampling
        If None, creates deterministic generator based on KC ID
        
    Returns
    -------
    torch.Tensor
        Embedding of the randomly sampled question representing this KC
        
    Raises
    ------
    ValueError
        If no valid question embeddings are found for this KC
    """
    # Find questions that have embeddings
    valid_questions = []
    for qid in question_ids:
        if qid in question_embeddings:
            valid_questions.append(qid)
    
    if not valid_questions:
        raise ValueError(
            f"KC '{kc_id}' has no valid question embeddings available "
            f"from {len(question_ids)} total questions"
        )
    
    # Create deterministic random generator if none provided
    if rng is None:
        rng = random.Random(hash(kc_id))  # Deterministic based on KC ID
    
    # Sample one question randomly
    sampled_qid = rng.choice(valid_questions)
    sampled_embedding = question_embeddings[sampled_qid].clone()
    
    _logger.debug(
        f"KC '{kc_id}': sampled question '{sampled_qid}' from "
        f"{len(valid_questions)} available questions"
    )
    
    return sampled_embedding

