"""
pipeline.py
~~~~~~~~~~~
End-to-end orchestration for producing pre-trained embeddings to feed directly
into DKT/qDKT's RNN input layer.

CORRECTED: Properly handles add_words strategy at question text level with
KC-level output aggregation. SA embeddings are generated at KC level for all strategies.

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
    
Output
------
    sa_vecs : Dict[(kc_id, bool), torch.Tensor] - 2*N_kc embeddings
"""

from __future__ import annotations

import logging
import random
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
        if cfg.sa_strategy != "add_words":  # add_words handled directly in pipeline
            self._sa_fn = sm.SA_FUNCTION_REGISTRY[sm.SAEmbeddingStrategy(cfg.sa_strategy)]
        else:
            self._sa_fn = None

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
            sa_embeddings       : {(kc_id, bool) -> torch.Tensor}  # KC-level SA

        Notes
        -----
        SA embeddings are generated at KC level, producing 2*N_kc total embeddings
        where N_kc is the number of unique knowledge components.
        """
        stage_outputs = {}

        for stage in self.cfg.execution_order:
            _logger.info("=== Generating %s embeddings ===", stage.upper())

            if stage == "question":
                stage_outputs["question"] = self._handle_question_stage(question_df, stage_outputs)

            elif stage == "kc":
                stage_outputs["kc"] = self._handle_kc_stage(question_df, qid_to_kc, stage_outputs)

            elif stage == "sa":
                stage_outputs["sa"] = self._handle_sa_stage(question_df, qid_to_kc, stage_outputs)

            else:  # pragma: no cover
                raise AssertionError(f"Unknown stage '{stage}' encountered.")

        # Hand back everything we generated (missing ones will default to {})
        return (
            stage_outputs.get("question", {}),
            stage_outputs.get("kc", {}),
            stage_outputs.get("sa", {}),
        )

    # ---------------------------------------------------------------------#
    # Question Stage Handling
    # ---------------------------------------------------------------------#
    def _handle_question_stage(self, question_df: pd.DataFrame, stage_outputs: dict) -> dict:
        """Handle question embedding generation."""
        
        if "sa" in stage_outputs and isinstance(stage_outputs["sa"], dict) and \
           self._is_modified_questions_dict(stage_outputs["sa"]):
            # SA stage ran first with add_words and created modified question texts
            modified_questions = stage_outputs["sa"]
            
            # Generate embeddings for all modified questions
            question_embeddings = {}
            
            # Create temporary DataFrame with modified texts
            modified_data = []
            for (qid, is_correct), modified_text in modified_questions.items():
                modified_data.append({
                    'question_id': f"{qid}_{'correct' if is_correct else 'incorrect'}",
                    'question_text': modified_text,
                    'original_qid': qid,
                    'is_correct': is_correct
                })
            
            modified_df = pd.DataFrame(modified_data)
            
            # Generate embeddings for modified questions
            temp_embeddings = em.get_question_embeddings(
                modified_df,
                id_col='question_id',
                text_col='question_text', 
                model=self.cfg.question_embedding_model,
            )
            
            # Reorganize embeddings by (original_qid, is_correct)
            for _, row in modified_df.iterrows():
                temp_qid = row['question_id']
                original_qid = row['original_qid']
                is_correct = row['is_correct']
                
                if temp_qid in temp_embeddings:
                    question_embeddings[(original_qid, is_correct)] = temp_embeddings[temp_qid]
            
            _logger.info(f"Generated embeddings for {len(question_embeddings)} modified questions")
            return question_embeddings
        
        else:
            # Normal question embedding generation
            embeddings = em.get_question_embeddings(
                question_df,
                model=self.cfg.question_embedding_model,
            )
            return embeddings

    # ---------------------------------------------------------------------#
    # KC Stage Handling
    # ---------------------------------------------------------------------#
    def _handle_kc_stage(self, question_df: pd.DataFrame, qid_to_kc: Dict[Hashable, Hashable], stage_outputs: dict) -> dict:
        """Handle KC embedding generation."""
        
        if "question" in stage_outputs and \
           self._is_modified_questions_dict(stage_outputs["question"]):
            # Working with modified question embeddings from add_words
            return self._handle_kc_with_modified_questions(qid_to_kc, stage_outputs["question"])
            
        elif self.cfg.execution_order[0] == "kc" and self.cfg.kc_strategy == "stuff_window":
            # KC comes first with stuff_window strategy
            return self._handle_kc_stuff_window_first(question_df, qid_to_kc, stage_outputs)
            
        elif self.cfg.execution_order[0] == "sa" and self.cfg.kc_strategy == "stuff_window":
            # SA (add_words) comes first, then KC (stuff_window) - Pipeline 9
            return self._handle_kc_stuff_window_after_sa(qid_to_kc, stage_outputs)
            
        else:
            # Standard KC processing after questions
            return self._handle_kc_standard(qid_to_kc, stage_outputs)

    def _handle_kc_with_modified_questions(self, qid_to_kc: Dict[Hashable, Hashable], modified_q_embs: dict) -> dict:
        """Handle KC generation when working with modified question embeddings."""
        
        # Separate correct and incorrect embeddings
        correct_q_embs = {qid: emb for (qid, is_correct), emb in modified_q_embs.items() if is_correct}
        incorrect_q_embs = {qid: emb for (qid, is_correct), emb in modified_q_embs.items() if not is_correct}
        
        # Build KC mappings
        kc_to_qids: Dict[Hashable, list] = {}
        for qid, kc in qid_to_kc.items():
            kc_to_qids.setdefault(kc, []).append(qid)
        
        kc_vecs = {}
        
        # Process each KC for both correct and incorrect versions
        for kc_id, qids in kc_to_qids.items():
            
            # CORRECT KC embedding
            if self.cfg.kc_strategy == "average":
                valid_correct = [correct_q_embs[qid] for qid in qids if qid in correct_q_embs]
                if valid_correct:
                    kc_vecs[(kc_id, True)] = torch.stack(valid_correct).mean(dim=0)
                    
            elif self.cfg.kc_strategy == "sample_single_question":
                valid_correct_qids = [qid for qid in qids if qid in correct_q_embs]
                if valid_correct_qids:
                    rng = random.Random(hash((kc_id, True)))
                    sampled_qid = rng.choice(valid_correct_qids)
                    kc_vecs[(kc_id, True)] = correct_q_embs[sampled_qid].clone()
            
            # INCORRECT KC embedding  
            if self.cfg.kc_strategy == "average":
                valid_incorrect = [incorrect_q_embs[qid] for qid in qids if qid in incorrect_q_embs]
                if valid_incorrect:
                    kc_vecs[(kc_id, False)] = torch.stack(valid_incorrect).mean(dim=0)
                    
            elif self.cfg.kc_strategy == "sample_single_question":
                valid_incorrect_qids = [qid for qid in qids if qid in incorrect_q_embs]
                if valid_incorrect_qids:
                    rng = random.Random(hash((kc_id, False)))
                    sampled_qid = rng.choice(valid_incorrect_qids)
                    kc_vecs[(kc_id, False)] = incorrect_q_embs[sampled_qid].clone()
        
        _logger.info(f"Generated KC embeddings for {len(kc_to_qids)} KCs (correct + incorrect versions)")
        return kc_vecs

    def _handle_kc_stuff_window_first(self, question_df: pd.DataFrame, qid_to_kc: Dict[Hashable, Hashable], stage_outputs: dict) -> dict:
        """Handle stuff_window when KC comes first (normal stuff_window, not add_words)."""
        
        if self.cfg.kc_strategy != "stuff_window":
            raise RuntimeError(
                f"KC strategy '{self.cfg.kc_strategy}' cannot run before questions. "
                "Use 'stuff_window' or change execution order."
            )
        
        # Build {kc_id: [qid1, qid2, ...]}
        kc_to_qids: Dict[Hashable, list] = {}
        for qid, kc in qid_to_kc.items():
            kc_to_qids.setdefault(kc, []).append(qid)

        kc_vecs = {}
        for kc_id, qids in kc_to_qids.items():
            kc_vecs[kc_id] = self._kc_fn(
                kc_id,
                qids,
                {},  # Empty dict since stuff_window doesn't use question embeddings
                **self.cfg.kc_kwargs,
            )
        
        return kc_vecs

    # def _handle_kc_stuff_window_after_sa(self, qid_to_kc: Dict[Hashable, Hashable], stage_outputs: dict) -> dict:
    #     """Handle stuff_window when SA (add_words) comes first - Pipeline 9."""
        
    #     if "sa" not in stage_outputs or not isinstance(stage_outputs["sa"], dict):
    #         raise RuntimeError("Expected SA stage to have created modified question texts")
        
    #     # SA created modified question texts
    #     modified_questions = stage_outputs["sa"]
        
    #     # Separate correct and incorrect question texts
    #     correct_texts = {qid: text for (qid, is_correct), text in modified_questions.items() if is_correct}
    #     incorrect_texts = {qid: text for (qid, is_correct), text in modified_questions.items() if not is_correct}
        
    #     # Build KC mappings
    #     kc_to_qids: Dict[Hashable, list] = {}
    #     for qid, kc in qid_to_kc.items():
    #         kc_to_qids.setdefault(kc, []).append(qid)
        
    #     kc_vecs = {}
        
    #     # Process each KC for both correct and incorrect versions using stuff_window
    #     for kc_id, qids in kc_to_qids.items():
            
    #         # CORRECT KC embedding using stuff_window on correct texts
    #         correct_qids_for_kc = [qid for qid in qids if qid in correct_texts]
    #         if correct_qids_for_kc:
    #             # Create temporary DataFrame for correct texts
    #             correct_data = [{'question_id': qid, 'question_text': correct_texts[qid]} 
    #                           for qid in correct_qids_for_kc]
    #             correct_df = pd.DataFrame(correct_data)
                
    #             kc_vecs[(kc_id, True)] = self._kc_fn(
    #                 f"{kc_id}_correct",
    #                 correct_qids_for_kc,
    #                 {},  # empty question_embeddings
    #                 question_df=correct_df,
    #                 **self.cfg.kc_kwargs,
    #             )
            
    #         # INCORRECT KC embedding using stuff_window on incorrect texts
    #         incorrect_qids_for_kc = [qid for qid in qids if qid in incorrect_texts]
    #         if incorrect_qids_for_kc:
    #             # Create temporary DataFrame for incorrect texts
    #             incorrect_data = [{'question_id': qid, 'question_text': incorrect_texts[qid]} 
    #                             for qid in incorrect_qids_for_kc]
    #             incorrect_df = pd.DataFrame(incorrect_data)
                
    #             kc_vecs[(kc_id, False)] = self._kc_fn(
    #                 f"{kc_id}_incorrect", 
    #                 incorrect_qids_for_kc,
    #                 {},  # empty question_embeddings
    #                 question_df=incorrect_df,
    #                 **self.cfg.kc_kwargs,
    #             )
        
    #     _logger.info(f"Generated stuff_window KC embeddings for {len(kc_to_qids)} KCs (correct + incorrect versions)")
    #     return kc_vecs

    def _handle_kc_stuff_window_after_sa(self, qid_to_kc: Dict[Hashable, Hashable], stage_outputs: dict) -> dict:
        """Handle stuff_window when SA (add_words) comes first - Pipeline 9."""
        
        if "sa" not in stage_outputs or not isinstance(stage_outputs["sa"], dict):
            raise RuntimeError("Expected SA stage to have created modified question texts")
        
        # SA created modified question texts
        modified_questions = stage_outputs["sa"]
        
        # Separate correct and incorrect question texts
        correct_texts = {qid: text for (qid, is_correct), text in modified_questions.items() if is_correct}
        incorrect_texts = {qid: text for (qid, is_correct), text in modified_questions.items() if not is_correct}
        
        # Build KC mappings
        kc_to_qids: Dict[Hashable, list] = {}
        for qid, kc in qid_to_kc.items():
            kc_to_qids.setdefault(kc, []).append(qid)
        
        kc_vecs = {}
        
        # Process each KC for both correct and incorrect versions using stuff_window
        for kc_id, qids in kc_to_qids.items():
            
            # CORRECT KC embedding using stuff_window on correct texts
            correct_qids_for_kc = [qid for qid in qids if qid in correct_texts]
            if correct_qids_for_kc:
                # Create temporary DataFrame for correct texts
                correct_data = [{'question_id': qid, 'question_text': correct_texts[qid]} 
                            for qid in correct_qids_for_kc]
                correct_df = pd.DataFrame(correct_data)
                
                # FIXED: Pass only the required parameters
                kc_vecs[(kc_id, True)] = self._kc_fn(
                    f"{kc_id}_correct",
                    correct_qids_for_kc,
                    {},  # empty question_embeddings
                    question_df=correct_df,
                    window_size=self.cfg.kc_kwargs.get('window_size', 2048)
                )
            
            # INCORRECT KC embedding using stuff_window on incorrect texts
            incorrect_qids_for_kc = [qid for qid in qids if qid in incorrect_texts]
            if incorrect_qids_for_kc:
                # Create temporary DataFrame for incorrect texts
                incorrect_data = [{'question_id': qid, 'question_text': incorrect_texts[qid]} 
                                for qid in incorrect_qids_for_kc]
                incorrect_df = pd.DataFrame(incorrect_data)
                
                # FIXED: Pass only the required parameters
                kc_vecs[(kc_id, False)] = self._kc_fn(
                    f"{kc_id}_incorrect", 
                    incorrect_qids_for_kc,
                    {},  # empty question_embeddings
                    question_df=incorrect_df,
                    window_size=self.cfg.kc_kwargs.get('window_size', 2048)
                )
        
        _logger.info(f"Generated stuff_window KC embeddings for {len(kc_to_qids)} KCs (correct + incorrect versions)")
        return kc_vecs



    def _handle_kc_standard(self, qid_to_kc: Dict[Hashable, Hashable], stage_outputs: dict) -> dict:
        """Handle standard KC processing after questions."""
        
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
        
        return kc_vecs

    # ---------------------------------------------------------------------#
    # SA Stage Handling
    # ---------------------------------------------------------------------#
    def _handle_sa_stage(self, question_df: pd.DataFrame, qid_to_kc: Dict[Hashable, Hashable], stage_outputs: dict) -> dict:
        """Handle SA embedding generation."""
        
        if self.cfg.sa_strategy == "add_words":
            # SPECIAL HANDLING: add_words works on question texts, not embeddings
            return self._handle_sa_add_words(question_df)
        else:
            # NORMAL SA HANDLING: mirror/stack_antonym work on embeddings
            return self._handle_sa_standard(qid_to_kc, stage_outputs)

    def _handle_sa_add_words(self, question_df: pd.DataFrame) -> dict:
        """Handle add_words SA strategy by creating modified question texts."""
        
        # Create modified question texts with CORRECT/INCORRECT prefixes
        modified_questions = {}  # {(qid, bool): modified_text}
        
        for _, row in question_df.iterrows():
            qid = row.get('question_id')  # Adjust column name as needed
            qtext = row.get('question_text')  # Adjust column name as needed
            
            if qid is None or qtext is None:
                continue
                
            # Create correct and incorrect versions
            correct_word = self.cfg.sa_kwargs.get('correct_word', 'CORRECT')
            incorrect_word = self.cfg.sa_kwargs.get('incorrect_word', 'INCORRECT') 
            repetitions = self.cfg.sa_kwargs.get('repetitions', 1)
            
            # Add words to question text
            correct_prefix = " ".join([correct_word] * repetitions)
            incorrect_prefix = " ".join([incorrect_word] * repetitions)
            
            modified_questions[(qid, True)] = f"{correct_prefix} {qtext.strip()}"
            modified_questions[(qid, False)] = f"{incorrect_prefix} {qtext.strip()}"
        
        _logger.info(f"Generated {len(modified_questions)} modified question texts")
        return modified_questions

    def _handle_sa_standard(self, qid_to_kc: Dict[Hashable, Hashable], stage_outputs: dict) -> dict:
        """Handle standard SA processing (mirror/stack_antonym)."""
        
        # Determine base embeddings to use
        if "kc" in stage_outputs:
            # Use KC embeddings as base (preferred)
            base_vectors = stage_outputs["kc"]
            base_type = "KC"
        elif "question" in stage_outputs:
            # Fallback: derive KC-level SA from question embeddings
            # First need to create temporary KC representations
            q_embs = stage_outputs["question"]
            
            # Build temporary KC embeddings by averaging questions
            kc_to_qids: Dict[Hashable, list] = {}
            for qid, kc in qid_to_kc.items():
                kc_to_qids.setdefault(kc, []).append(qid)
            
            temp_kc_vecs = {}
            for kc_id, qids in kc_to_qids.items():
                valid_embeddings = []
                for qid in qids:
                    if qid in q_embs:
                        valid_embeddings.append(q_embs[qid])
                
                if valid_embeddings:
                    temp_kc_vecs[kc_id] = torch.stack(valid_embeddings).mean(dim=0)
            
            base_vectors = temp_kc_vecs
            base_type = "derived KC"
        else:
            raise RuntimeError(
                "SA stage could not find base embeddings (question/kc)."
            )

        # Generate SA embeddings for each KC
        sa_vecs = {}
        for kc_id, base_vec in base_vectors.items():
            # Create correct and incorrect embeddings for this KC
            sa_vecs[(kc_id, True)] = self._sa_fn(base_vec, True, **self.cfg.sa_kwargs)
            sa_vecs[(kc_id, False)] = self._sa_fn(base_vec, False, **self.cfg.sa_kwargs)
        
        _logger.info(f"Generated SA embeddings for {len(base_vectors)} KCs using {base_type} as base")
        return sa_vecs

    # ---------------------------------------------------------------------#
    # Helper Methods
    # ---------------------------------------------------------------------#
    def _is_modified_questions_dict(self, data: dict) -> bool:
        """Check if dict contains modified questions from add_words strategy."""
        if not data:
            return False
        
        # Check if keys are (qid, bool) tuples and values are strings (for SA stage)
        # or values are tensors (for question stage after SA)
        sample_key = next(iter(data.keys()))
        sample_value = next(iter(data.values()))
        
        return (isinstance(sample_key, tuple) and 
                len(sample_key) == 2 and 
                isinstance(sample_key[1], bool) and
                (isinstance(sample_value, str) or isinstance(sample_value, torch.Tensor)))