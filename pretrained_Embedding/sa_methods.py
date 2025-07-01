# sa_methods.py --------------------------------------------------------------
"""
Student-Action (SA) embedding strategies.
Each must return two tensors: (correct_vecs, incorrect_vecs)
so that DKT can look up the right row per interaction.
"""
import numpy as np

# --------------------------------------------------------------------------- #
def sa_mirror(kc_vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Mirror each KC vector w.r.t. the origin for incorrect answers.
    """
    correct = kc_vecs
    incorrect = -kc_vecs
    return correct, incorrect


def sa_add_words(question_texts: list,
                 embed_fn) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenate the question embedding with the embedding of
    the words 'CORRECT' or 'INCORRECT'.  Placeholder only.
    """
    # TODO: implement
    raise NotImplementedError


def sa_stack_antonyms(kc_vecs: np.ndarray,
                      antonym_vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Stack KC vec with an antonym vector (or zero-vec) to
    create 2×dim representations.  Placeholder only.
    """
    # TODO: implement
    raise NotImplementedError
