"""
pipeline.py
~~~~~~~~~~~
End-to-end orchestration for producing pre-trained embeddings to feed directly
into DKT/qDKT’s RNN input layer.

Typical usage
-------------
    cfg = EmbeddingPipelineConfig(
        question_embedding_model="text-embedding-3-small",
        kc_strategy="average",
        sa_strategy="mirror",
        execution_order=("question", "kc", "sa"),  # default
    )
    pipeline = EmbeddingPipeline(cfg)
    q_vecs, kc_vecs, sa_vecs = pipeline.run(question_df, qid_to_kc)

Where
-----
    question_df : pd.DataFrame with ['question_id', 'question_text', ...]
    qid_to_kc   : Dict[question_id, kc_id]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Hashable, Sequence, Tuple

import pandas as pd
import torch

import embedding_models as em
import kc_methods as km
import sa_methods as sm

_logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------#
# Configuration
# -----------------------------------------------------------------------------#
@dataclass
class EmbeddingPipelineConfig:
    # Question embeddings
    question_embedding_model: str = em.DEFAULT_MODEL

    # KC + SA strategy names (validated at init)
    kc_strategy: str = km.KCEmbeddingStrategy.AVERAGE.value
    sa_strategy: str = sm.SAEmbeddingStrategy.MIRROR.value

    # Order in which to execute stages
    # e.g. ("sa", "question", "kc") would generate SA first, then questions, etc.
    execution_order: Tuple[str, str, str] = ("question", "kc", "sa")

    # Misc.   (extend as needed)
    kc_kwargs: dict = field(default_factory=dict)
    sa_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        # Validate strategy names up-front
        if self.kc_strategy not in km.KC_FUNCTION_REGISTRY:
            raise ValueError(f"Unknown KC strategy '{self.kc_strategy}'.")
        if self.sa_strategy not in sm.SA_FUNCTION_REGISTRY:
            raise ValueError(f"Unknown SA strategy '{self.sa_strategy}'.")
        for stage in self.execution_order:
            if stage not in {"question", "kc", "sa"}:
                raise ValueError("execution_order may contain only "
                                 "'question', 'kc', or 'sa'.")


# -----------------------------------------------------------------------------#
# Pipeline
# -----------------------------------------------------------------------------#
class EmbeddingPipeline:
    """Composable pipeline that wires together the chosen strategies."""

    def __init__(self, cfg: EmbeddingPipelineConfig) -> None:
        self.cfg = cfg

        # Resolve strategy callables
        self._kc_fn = km.KC_FUNCTION_REGISTRY[km.KCEmbeddingStrategy(cfg.kc_strategy)]
        self._sa_fn = sm.SA_FUNCTION_REGISTRY[sm.SAEmbeddingStrategy(cfg.sa_strategy)]

    # ---------------------------------------------------------------------#
    # Public entry point
    # ---------------------------------------------------------------------#
    def run(
        self,
        question_df: pd.DataFrame,
        qid_to_kc: Dict[Hashable, Hashable],
    ) -> Tuple[Dict[Hashable, torch.Tensor], Dict[Hashable, torch.Tensor], Dict[Tuple[Hashable, bool], torch.Tensor]]:
        """
        Execute the pipeline in the configured order and return three dicts:

            question_embeddings : {question_id -> torch.Tensor}
            kc_embeddings       : {kc_id       -> torch.Tensor}
            sa_embeddings       : {(qid, bool) -> torch.Tensor}

        Notes
        -----
        *If* SA is generated before KC or Question embeddings you must ensure
        the strategy functions can operate with whatever inputs exist at that
        point.  The default order ("question", "kc", "sa") avoids headaches.
        """
        stage_outputs = {}

        for stage in self.cfg.execution_order:
            _logger.info("=== Generating %s embeddings ===", stage.upper())

            if stage == "question":
                stage_outputs["question"] = em.get_question_embeddings(
                    question_df,
                    model=self.cfg.question_embedding_model,
                )

            elif stage == "kc":
                if "question" not in stage_outputs:
                    raise RuntimeError(
                        "KC stage expects question embeddings to exist "
                        "but 'question' stage has not run yet."
                    )
                q_embs = stage_outputs["question"]

                # Build {kc_id: [qid1, qid2, ...]}
                kc_to_qids: Dict[Hashable, list] = {}
                for qid, kc in qid_to_kc.items():
                    kc_to_qids.setdefault(kc, []).append(qid)

                kc_vecs = {}
                for kc_id, qids in kc_to_qids.items():
                    kc_vecs[kc_id] = self._kc_fn(
                        kc_id,
                        qids,
                        q_embs,
                        **self.cfg.kc_kwargs,
                    )
                stage_outputs["kc"] = kc_vecs

            # elif stage == "sa":
            #     # Decide whether we base SA on questions or KCs
            #     base_vectors = (
            #         stage_outputs.get("question")
            #         if "question" in stage_outputs
            #         else stage_outputs.get("kc")
            #     )
            #     if base_vectors is None:
            #         raise RuntimeError(
            #             "SA stage could not find base embeddings (question/kc)."
            #         )

            #     sa_vecs = {}
            #     for qid, base_vec in base_vectors.items():
            #         sa_vecs[(qid, True)] = self._sa_fn(base_vec, True, **self.cfg.sa_kwargs)
            #         sa_vecs[(qid, False)] = self._sa_fn(base_vec, False, **self.cfg.sa_kwargs)
            #     stage_outputs["sa"] = sa_vecs
            
            elif stage == "sa":
                # SA embeddings are always created for questions, not KCs
                # So we need to determine which base embeddings to use
                if "question" in stage_outputs:
                    # Use question embeddings as base
                    base_vectors = stage_outputs["question"]
                    sa_vecs = {}
                    for qid, base_vec in base_vectors.items():
                        sa_vecs[(qid, True)] = self._sa_fn(base_vec, True, **self.cfg.sa_kwargs)
                        sa_vecs[(qid, False)] = self._sa_fn(base_vec, False, **self.cfg.sa_kwargs)
                elif "kc" in stage_outputs:
                    # Use KC embeddings as base, but still create SA embeddings for each question
                    kc_vectors = stage_outputs["kc"]
                    sa_vecs = {}
                    for qid, kc_id in qid_to_kc.items():
                        if kc_id in kc_vectors:
                            base_vec = kc_vectors[kc_id]
                            sa_vecs[(qid, True)] = self._sa_fn(base_vec, True, **self.cfg.sa_kwargs)
                            sa_vecs[(qid, False)] = self._sa_fn(base_vec, False, **self.cfg.sa_kwargs)
                    
                else:
                    raise RuntimeError(
                        "SA stage could not find base embeddings (question/kc)."
                    )
                
                stage_outputs["sa"] = sa_vecs
            else:  # pragma: no cover
                raise AssertionError(f"Unknown stage '{stage}' encountered.")

        # Hand back everything we generated (missing ones will default to {})
        return (
            stage_outputs.get("question", {}),
            stage_outputs.get("kc", {}),
            stage_outputs.get("sa", {}),
        )
