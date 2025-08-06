#!/usr/bin/env python3
"""
generate_embeddings.py
~~~~~~~~~~~~~~~~~~~~~~
Complete script to generate embeddings using your 9 pipeline configurations.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple 

import pandas as pd
import torch
from tqdm import tqdm

# Import your pipeline modules
from pipeline import EmbeddingPipelineConfig, EmbeddingPipeline
import embedding_models as em

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# STEP 1: SETUP - Prepare Your Data
# =============================================================================

# def load_your_data():
#     """
#     Replace this function with your actual data loading logic.
#     """
#     # EXAMPLE - Replace with your actual data loading
#     question_df = pd.DataFrame()
#     qid_to_kc = pd.DataFrame()
#     return question_df, qid_to_kc

def load_your_data(data_dir: str = "./data", 
                      format: str = "pickle") -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load question_df and qid_to_kc from files.
    
    Args:
        data_dir: Directory containing the saved files
        format: "pickle", "json", or "combined"
        
    Returns:
        question_df, qid_to_kc
    """
    # Load question_df from CSV
    csv_path = os.path.join(data_dir, "question_df.csv")
    if os.path.exists(csv_path):
        question_df = pd.read_csv(csv_path)
        print(f"Loaded question_df from {csv_path}")
    else:
        raise FileNotFoundError(f"Cannot find {csv_path}")
    
    # Load qid_to_kc based on format
    if format == "pickle":
        pickle_path = os.path.join(data_dir, "qid_to_kc.pkl")
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                qid_to_kc = pickle.load(f)
            print(f"Loaded qid_to_kc from {pickle_path}")
        else:
            raise FileNotFoundError(f"Cannot find {pickle_path}")
            
    elif format == "json":
        json_path = os.path.join(data_dir, "qid_to_kc.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                qid_to_kc = json.load(f)
            print(f"Loaded qid_to_kc from {json_path}")
        else:
            raise FileNotFoundError(f"Cannot find {json_path}")
            
    elif format == "combined":
        combined_path = os.path.join(data_dir, "question_data_combined.pkl")
        if os.path.exists(combined_path):
            with open(combined_path, 'rb') as f:
                data = pickle.load(f)
            question_df = data['question_df']
            qid_to_kc = data['qid_to_kc']
            print(f"Loaded combined data from {combined_path}")
        else:
            raise FileNotFoundError(f"Cannot find {combined_path}")
    
    print(f"Loaded {len(question_df)} questions and {len(qid_to_kc)} KC mappings")
    return question_df, qid_to_kc


def check_requirements():
    """Verify all requirements are met before running."""
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "❌ OPENAI_API_KEY not found!\n"
            "Set it with: export OPENAI_API_KEY='your-key-here'"
        )
    
    print("✅ OpenAI API key found")
    
    # Test OpenAI connection
    try:
        test_embedding = em._get_embedding("test", model="text-embedding-3-small")
        print(f"✅ OpenAI API working (test embedding shape: {test_embedding.shape})")
    except Exception as e:
        raise RuntimeError(f"❌ OpenAI API test failed: {e}")

# =============================================================================
# STEP 2: DEFINE ALL 9 PIPELINE CONFIGURATIONS
# =============================================================================

def get_pipeline_configs(question_df):
    """Return all 9 valid pipeline configurations."""
    
    # Helper function for embedding text
    def embed_text(text):
        return em._get_embedding(text, model="text-embedding-3-small")
    
    # Pre-compute antonym embeddings for efficiency
    antonym_embeddings = {
        "CORRECT": embed_text("CORRECT"),
        "INCORRECT": embed_text("INCORRECT")
    }
    
    configs = {
        # Group 1: ("question", "kc", "sa") - 4 pipelines
        "pipeline_1": EmbeddingPipelineConfig(
            kc_strategy="average",
            sa_strategy="mirror",
            execution_order=("question", "kc", "sa")
        ),
        
        "pipeline_2": EmbeddingPipelineConfig(
            kc_strategy="average", 
            sa_strategy="stack_antonym",
            execution_order=("question", "kc", "sa"),
            sa_kwargs={
                "antonym_embeddings": antonym_embeddings,
                "text_to_embedding_fn": embed_text
            }
        ),
        
        "pipeline_3": EmbeddingPipelineConfig(
            kc_strategy="sample_single_question",
            sa_strategy="mirror", 
            execution_order=("question", "kc", "sa")
        ),
        
        "pipeline_4": EmbeddingPipelineConfig(
            kc_strategy="sample_single_question",
            sa_strategy="stack_antonym",
            execution_order=("question", "kc", "sa"),
            sa_kwargs={
                "antonym_embeddings": antonym_embeddings,
                "text_to_embedding_fn": embed_text
            }
        ),
        
        # Group 2: ("kc", "question", "sa") - 2 pipelines  
        "pipeline_5": EmbeddingPipelineConfig(
            kc_strategy="stuff_window",
            sa_strategy="mirror",
            execution_order=("kc", "question", "sa"),
            kc_kwargs={
                "window_size": 8192,
                "question_df": question_df,
                "text_to_embedding_fn": embed_text
            }
        ),
        
        "pipeline_6": EmbeddingPipelineConfig(
            kc_strategy="stuff_window",
            sa_strategy="stack_antonym", 
            execution_order=("kc", "question", "sa"),
            kc_kwargs={
                "window_size": 8192,
                "question_df": question_df,
                "text_to_embedding_fn": embed_text
            },
            sa_kwargs={
                "antonym_embeddings": antonym_embeddings,
                "text_to_embedding_fn": embed_text
            }
        ),
        
        # Group 3: ("sa", "question", "kc") - 2 pipelines
        "pipeline_7": EmbeddingPipelineConfig(
            kc_strategy="average",
            sa_strategy="add_words",
            execution_order=("sa", "question", "kc"),
            sa_kwargs={
                "repetitions": 2,
                "text_to_embedding_fn": embed_text
            }
        ),
        
        "pipeline_8": EmbeddingPipelineConfig(
            kc_strategy="sample_single_question", 
            sa_strategy="add_words",
            execution_order=("sa", "question", "kc"),
            sa_kwargs={
                "repetitions": 2,
                "text_to_embedding_fn": embed_text
            }
        ),
        
        # Group 4: ("sa", "kc", "question") - 1 pipeline
        "pipeline_9": EmbeddingPipelineConfig(
            kc_strategy="stuff_window",
            sa_strategy="add_words",
            execution_order=("sa", "kc", "question"),
            kc_kwargs={
                "window_size": 8192,
                "question_df": question_df,
                "text_to_embedding_fn": embed_text
            },
            sa_kwargs={
                "repetitions": 2,
                "text_to_embedding_fn": embed_text
            }
        )
    }
    
    return configs

# =============================================================================
# STEP 3: GENERATE EMBEDDINGS FOR ALL PIPELINES
# =============================================================================

def generate_embeddings_for_pipeline(pipeline_name, config, question_df, qid_to_kc, output_dir):
    """Generate embeddings for a single pipeline configuration."""
    
    print(f"\n🚀 Generating embeddings for {pipeline_name}")
    print(f"   Strategy: {config.kc_strategy} + {config.sa_strategy}")
    print(f"   Order: {config.execution_order}")
    
    try:
        # Create pipeline and run
        pipeline = EmbeddingPipeline(config)
        q_vecs, kc_vecs, sa_vecs = pipeline.run(question_df, qid_to_kc)
        
        # Print results
        print(f"   ✅ Generated:")
        print(f"      - {len(q_vecs)} question embeddings") 
        print(f"      - {len(kc_vecs)} KC embeddings")
        print(f"      - {len(sa_vecs)} SA embeddings")
        
        # Get embedding dimensions
        if q_vecs:
            q_dim = next(iter(q_vecs.values())).shape[0]
            print(f"      - Question embedding dim: {q_dim}")
        if kc_vecs:
            kc_dim = next(iter(kc_vecs.values())).shape[0] 
            print(f"      - KC embedding dim: {kc_dim}")
        if sa_vecs:
            sa_dim = next(iter(sa_vecs.values())).shape[0]
            print(f"      - SA embedding dim: {sa_dim}")
            total_sa_dims = len(sa_vecs) * sa_dim
            print(f"      - Total SA dimensions: {total_sa_dims}")
        
        # Save embeddings
        output_path = output_dir / f"{pipeline_name}_embeddings.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump({
                'question_embeddings': q_vecs,
                'kc_embeddings': kc_vecs, 
                'sa_embeddings': sa_vecs,
                'config': config
            }, f)
        
        print(f"   💾 Saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def generate_all_embeddings():
    """Main function to generate embeddings for all pipelines."""
    
    print("🎯 Starting embedding generation for all 9 pipelines")
    
    # Step 1: Check requirements
    check_requirements()
    
    # Step 2: Load data
    print("\n📊 Loading your data...")
    question_df, qid_to_kc = load_your_data(data_dir="./my_data")
    print(f"   - Loaded {len(question_df)} questions")
    print(f"   - Loaded {len(set(qid_to_kc.values()))} unique KCs")
    
    # Step 3: Get pipeline configurations
    print("\n⚙️ Setting up pipeline configurations...")
    configs = get_pipeline_configs(question_df)
    print(f"   - Configured {len(configs)} pipelines")
    
    # Step 4: Create output directory
    output_dir = Path("generated_embeddings")
    output_dir.mkdir(exist_ok=True)
    print(f"   - Output directory: {output_dir}")
    
    # Step 5: Generate embeddings for each pipeline
    results = {}
    
    for pipeline_name, config in configs.items():
        success = generate_embeddings_for_pipeline(
            pipeline_name, config, question_df, qid_to_kc, output_dir
        )
        results[pipeline_name] = success
    
    # Step 6: Summary
    print(f"\n📈 SUMMARY:")
    successful = sum(results.values())
    total = len(results)
    print(f"   - Successful: {successful}/{total} pipelines")
    
    for pipeline_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {pipeline_name}")
    
    if successful > 0:
        print(f"\n🎉 Generated embeddings saved in: {output_dir}/")
        print("   You can now use these embeddings to train your DKT model!")

# =============================================================================
# STEP 4: EXAMPLE USAGE OF GENERATED EMBEDDINGS  
# =============================================================================

def load_and_use_embeddings(pipeline_name="pipeline_1"):
    """Example of how to load and use generated embeddings."""
    
    embedding_path = Path("generated_embeddings") / f"{pipeline_name}_embeddings.pkl"
    
    if not embedding_path.exists():
        print(f"❌ Embeddings not found: {embedding_path}")
        return
    
    # Load embeddings
    with open(embedding_path, 'rb') as f:
        data = pickle.load(f)
    
    q_vecs = data['question_embeddings']
    kc_vecs = data['kc_embeddings'] 
    sa_vecs = data['sa_embeddings']
    config = data['config']
    
    print(f"📥 Loaded {pipeline_name} embeddings:")
    print(f"   - Questions: {len(q_vecs)}")
    print(f"   - KCs: {len(kc_vecs)}")
    print(f"   - SA pairs: {len(sa_vecs)}")
    
    # Example: Get embedding for specific student interaction
    example_qid = list(q_vecs.keys())[0]
    correct_embedding = sa_vecs[(example_qid, True)]
    incorrect_embedding = sa_vecs[(example_qid, False)]
    
    print(f"\n📋 Example embeddings for question '{example_qid}':")
    print(f"   - Correct response: shape {correct_embedding.shape}")
    print(f"   - Incorrect response: shape {incorrect_embedding.shape}")
    
    return q_vecs, kc_vecs, sa_vecs

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Generate all embeddings
    generate_all_embeddings()
    
    # Example of loading and using embeddings
    print("\n" + "="*50)
    print("EXAMPLE: Loading Pipeline 1 embeddings")
    print("="*50)
    load_and_use_embeddings("pipeline_1")