# pipeline.py ----------------------------------------------------------------
"""
Core orchestrator.  You specify the operation order and
the concrete methods; the pipeline returns a ready-to-save tensor.
"""
from typing import Sequence, Callable
import numpy as np
import torch

import embedding_models as E
import kc_methods as KC
import sa_methods as SA

# --------------------------------------------------------------------------- #
def build_embeddings(df_questions,
                     op_order: Sequence[str],
                     *,
                     kc_method: str,
                     sa_method: str,
                     text_embedder: Callable[[list], np.ndarray]
                     ) -> torch.Tensor:
    """
    Parameters
    ----------
    df_questions : pd.DataFrame
        Must contain ['question_id', 'question_text', 'kc_id', 'kc_name'].
    op_order : Sequence[str]
        Something like ('Q', 'KC', 'SA') or any permutation thereof.
    kc_method : str
        One of ['avg', 'window', 'name', 'sample'].
    sa_method : str
        One of ['mirror', 'add_words', 'stack_antonyms'].
    text_embedder : Callable
        Function that maps List[str] -> np.ndarray

    Returns
    -------
    torch.Tensor
        final_embedding_tensor  # shape = (2 * num_questions, final_dim)
        First half rows  : correct embeddings
        Second half rows : incorrect embeddings
    """
    # --------------------------------------------------------------------- #
    # 0) Unpack columns
    q_texts   = df_questions['question_text'].tolist()
    q_ids     = df_questions['question_id'].to_numpy()
    kc_ids    = df_questions['kc_id'].to_numpy()
    kc_names  = df_questions['kc_name'].tolist()

    # --------------------------------------------------------------------- #
    # Temporary storage
    question_vecs = None
    kc_vecs       = None
    correct_vecs  = None
    incorrect_vecs = None

    # --------------------------------------------------------------------- #
    # Function dispatch maps
    kc_dispatch = {
        'avg'    : KC.kc_avg,
        'window' : KC.kc_window_stuff,
        'name'   : KC.kc_name_as_text,
        'sample' : KC.kc_sample_one,
    }
    sa_dispatch = {
        'mirror'        : SA.sa_mirror,
        'add_words'     : SA.sa_add_words,
        'stack_antonyms': SA.sa_stack_antonyms,
    }

    # --------------------------------------------------------------------- #
    for step in op_order:
        if step == 'Q':
            if question_vecs is None:
                question_vecs = text_embedder(q_texts)          # (N, d)
        elif step == 'KC':
            if kc_vecs is None:
                if kc_method == 'name':
                    kc_vecs = kc_dispatch[kc_method](kc_names, text_embedder)
                else:
                    kc_vecs = kc_dispatch[kc_method](question_vecs, kc_ids)
        elif step == 'SA':
            if correct_vecs is None:
                correct_vecs, incorrect_vecs = sa_dispatch[sa_method](
                    kc_vecs if kc_vecs is not None else question_vecs,
                    # Additional args if the SA method needs them
                )
        else:
            raise ValueError(f"Unknown step {step}")

    # --------------------------------------------------------------------- #
    # Assemble final tensor in DKT's [correct; incorrect] row layout
    if correct_vecs is None or incorrect_vecs is None:
        raise RuntimeError("SA step never produced embeddings.")

    final = np.concatenate([correct_vecs, incorrect_vecs], axis=0)  # (2N, dim)
    return torch.tensor(final, dtype=torch.float32)
