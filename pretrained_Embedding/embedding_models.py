# embedding_models.py --------------------------------------------------------
"""
Unified wrapper around text-embedding providers.
Currently supports OpenAI; add more in the future.
"""

from typing import List
import openai  # pip install openai
import numpy as np

# --------------------------------------------------------------------------- #
def embed_openai(texts: List[str],
                 model_name: str = "text-embedding-ada-002",
                 batch_size: int = 1000) -> np.ndarray:
    """
    Embed a list of strings with the OpenAI embeddings endpoint.

    Parameters
    ----------
    texts : List[str]
        Raw text strings.
    model_name : str
        OpenAI embedding model to use.
    batch_size : int
        How many strings to send per request.

    Returns
    -------
    np.ndarray  # shape = (len(texts), dim)
    """
    # -- IMPLEMENT YOUR RATE-LIMIT FRIENDLY LOOP HERE ------------------------ #
    # For now, stub a random matrix so the rest of the pipeline is runnable.
    dim = 768  # <- will equal the OpenAI model dim when implemented
    return np.random.randn(len(texts), dim).astype("float32")
