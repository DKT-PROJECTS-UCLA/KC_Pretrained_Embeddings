# kc_methods.py --------------------------------------------------------------
"""
All Knowledge-Component (KC) embedding strategies.
Each function signature is fixed; fill in the body later.
"""
import numpy as np

# --------------------------------------------------------------------------- #
def kc_avg(question_embs: np.ndarray,
           question_kc_ids: np.ndarray) -> np.ndarray:
    """
    Average all question vectors that belong to the same KC.

    Returns
    -------
    kc_matrix : np.ndarray  # shape = (num_kcs, dim)
    """
    # TODO: implement
    raise NotImplementedError


def kc_window_stuff(question_embs: np.ndarray,
                    question_kc_ids: np.ndarray,
                    window_size: int = 5) -> np.ndarray:
    """
    Slide a fixed-size window over questions with the same KC,
    then average or pool.  Placeholder version here.
    """
    # TODO: implement
    raise NotImplementedError


def kc_name_as_text(kc_names: list,
                    embed_fn) -> np.ndarray:
    """
    Treat the KC name itself as text; embed with `embed_fn`.
    """
    # TODO: implement
    raise NotImplementedError


def kc_sample_one(question_embs: np.ndarray,
                  question_kc_ids: np.ndarray,
                  rng=None) -> np.ndarray:
    """
    Sample ONE question vector at random to represent each KC.
    """
    # TODO: implement
    raise NotImplementedError
