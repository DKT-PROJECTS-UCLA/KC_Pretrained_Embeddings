#!/usr/bin/env python3
"""
generate_embeddings.py
~~~~~~~~~~~~~~~~~~~~~~
Complete script to generate embeddings using your 9 pipeline configurations.
UPDATED: Now supports OpenAI, Cohere, and BERT/SentenceTransformers providers.
"""

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
# STEP 1: SETUP - Now with Provider Selection
# =============================================================================

def load_your_data(data_dir: str = "./my_data", 
                      format: str = "pickle") -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load question_df and qid_to_kc from files."""
    # Load question_df from CSV
    csv_path = Path(data_dir) / "question_df.csv"
    if csv_path.exists():
        question_df = pd.read_csv(csv_path)
        print(f"✅ Loaded question_df from {csv_path}")
    else:
        raise FileNotFoundError(f"Cannot find {csv_path}")
    
    # Load qid_to_kc based on format
    if format == "pickle":
        pickle_path = Path(data_dir) / "qid_to_kc.pkl"
        if pickle_path.exists():
            with open(pickle_path, 'rb') as f:
                qid_to_kc = pickle.load(f)
            print(f"✅ Loaded qid_to_kc from {pickle_path}")
        else:
            raise FileNotFoundError(f"Cannot find {pickle_path}")
            
    elif format == "json":
        import json
        json_path = Path(data_dir) / "qid_to_kc.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                qid_to_kc = json.load(f)
            print(f"✅ Loaded qid_to_kc from {json_path}")
        else:
            raise FileNotFoundError(f"Cannot find {json_path}")
            
    elif format == "combined":
        combined_path = Path(data_dir) / "question_data_combined.pkl"
        if combined_path.exists():
            with open(combined_path, 'rb') as f:
                data = pickle.load(f)
            question_df = data['question_df']
            qid_to_kc = data['qid_to_kc']
            print(f"✅ Loaded combined data from {combined_path}")
        else:
            raise FileNotFoundError(f"Cannot find {combined_path}")
    
    print(f"📊 Dataset: {len(question_df)} questions, {len(set(qid_to_kc.values()))} unique KCs")
    return question_df, qid_to_kc

def check_requirements():
    """Setup and validate the embedding provider."""
    
    print("🔧 Setting up embedding provider...")
    
    try:
        # This will prompt user for provider selection
        provider, model = em.setup_embedding_system()
        
        # Test the embedding system
        print("🧪 Testing embedding system...")
        test_embedding = em._get_embedding("test", model=model, provider=provider)
        embedding_dim = em.get_embedding_dimension(model)
        
        print(f"✅ Embedding system working!")
        print(f"   Provider: {provider.upper()}")
        print(f"   Model: {model}")
        print(f"   Dimensions: {embedding_dim}")
        print(f"   Test shape: {test_embedding.shape}")
        
        return provider, model, embedding_dim
        
    except Exception as e:
        raise RuntimeError(f"❌ Embedding system setup failed: {e}")

# =============================================================================
# STEP 2: DYNAMIC PIPELINE CONFIGURATIONS
# =============================================================================

# def get_pipeline_configs(question_df, embedding_model, provider):
#     """Return all 9 valid pipeline configurations with dynamic setup."""
    
#     # Helper function for embedding text using current provider
#     def embed_text(text):
#         return em._get_embedding(text, model=embedding_model, provider=provider)
    
#     # Get auto-detected window size
#     from kc_methods import get_window_size_for_model
#     auto_window_size = get_window_size_for_model(embedding_model)
    
#     print(f"🔧 Configuration:")
#     print(f"   Model: {embedding_model}")
#     print(f"   Window size: {auto_window_size} chars")
#     print(f"   Dimensions: {em.get_embedding_dimension(embedding_model)}")
    
#     # Pre-compute antonym embeddings for efficiency
#     print("🔄 Pre-computing antonym embeddings...")
#     antonym_embeddings = {
#         "CORRECT": embed_text("CORRECT"),
#         "INCORRECT": embed_text("INCORRECT")
#     }
#     print(f"✅ Antonym embeddings ready")
    
#     configs = {
#         # Group 1: ("question", "kc", "sa") - 4 pipelines
#         "pipeline_1": EmbeddingPipelineConfig(
#             question_embedding_model=embedding_model,
#             kc_strategy="average",
#             sa_strategy="mirror",
#             execution_order=("question", "kc", "sa")
#         ),
        
#         "pipeline_2": EmbeddingPipelineConfig(
#             question_embedding_model=embedding_model,
#             kc_strategy="average", 
#             sa_strategy="stack_antonym",
#             execution_order=("question", "kc", "sa"),
#             sa_kwargs={
#                 "antonym_embeddings": antonym_embeddings,
#                 "text_to_embedding_fn": embed_text
#             }
#         ),
        
#         "pipeline_3": EmbeddingPipelineConfig(
#             question_embedding_model=embedding_model,
#             kc_strategy="sample_single_question",
#             sa_strategy="mirror", 
#             execution_order=("question", "kc", "sa")
#         ),
        
#         "pipeline_4": EmbeddingPipelineConfig(
#             question_embedding_model=embedding_model,
#             kc_strategy="sample_single_question",
#             sa_strategy="stack_antonym",
#             execution_order=("question", "kc", "sa"),
#             sa_kwargs={
#                 "antonym_embeddings": antonym_embeddings,
#                 "text_to_embedding_fn": embed_text
#             }
#         ),
        
#         # Group 2: ("kc", "question", "sa") - 2 pipelines  
#         "pipeline_5": EmbeddingPipelineConfig(
#             question_embedding_model=embedding_model,
#             kc_strategy="stuff_window",
#             sa_strategy="mirror",
#             execution_order=("kc", "question", "sa"),
#             kc_kwargs={
#                 "window_size": auto_window_size,
#                 "question_df": question_df,
#                 "text_to_embedding_fn": embed_text,
#                 "embedding_model": embedding_model
#             }
#         ),
        
#         "pipeline_6": EmbeddingPipelineConfig(
#             question_embedding_model=embedding_model,
#             kc_strategy="stuff_window",
#             sa_strategy="stack_antonym", 
#             execution_order=("kc", "question", "sa"),
#             kc_kwargs={
#                 "window_size": auto_window_size,
#                 "question_df": question_df,
#                 "text_to_embedding_fn": embed_text,
#                 "embedding_model": embedding_model
#             },
#             sa_kwargs={
#                 "antonym_embeddings": antonym_embeddings,
#                 "text_to_embedding_fn": embed_text
#             }
#         ),
        
#         # Group 3: ("sa", "question", "kc") - 2 pipelines
#         "pipeline_7": EmbeddingPipelineConfig(
#             question_embedding_model=embedding_model,
#             kc_strategy="average",
#             sa_strategy="add_words",
#             execution_order=("sa", "question", "kc"),
#             sa_kwargs={
#                 "repetitions": 2,
#                 "text_to_embedding_fn": embed_text
#             }
#         ),
        
#         "pipeline_8": EmbeddingPipelineConfig(
#             question_embedding_model=embedding_model,
#             kc_strategy="sample_single_question", 
#             sa_strategy="add_words",
#             execution_order=("sa", "question", "kc"),
#             sa_kwargs={
#                 "repetitions": 2,
#                 "text_to_embedding_fn": embed_text
#             }
#         ),
        
#         # Group 4: ("sa", "kc", "question") - 1 pipeline
#         "pipeline_9": EmbeddingPipelineConfig(
#             question_embedding_model=embedding_model,
#             kc_strategy="stuff_window",
#             sa_strategy="add_words",
#             execution_order=("sa", "kc", "question"),
#             kc_kwargs={
#                 "window_size": auto_window_size,
#                 "question_df": question_df,
#                 "text_to_embedding_fn": embed_text,
#                 "embedding_model": embedding_model
#             },
#             sa_kwargs={
#                 "repetitions": 2,
#                 "text_to_embedding_fn": embed_text
#             }
#         )
#     }
    
#     return configs


def get_pipeline_configs(question_df, embedding_model, provider):
    """Return all 9 valid pipeline configurations with dynamic setup."""
    
    # Helper function for embedding text using current provider
    def embed_text(text):
        return em._get_embedding(text, model=embedding_model, provider=provider)
    
    # Get auto-detected window size
    from kc_methods import get_window_size_for_model
    auto_window_size = get_window_size_for_model(embedding_model)
    
    print(f"🔧 Configuration:")
    print(f"   Model: {embedding_model}")
    print(f"   Window size: {auto_window_size} chars")
    print(f"   Dimensions: {em.get_embedding_dimension(embedding_model)}")
    
    # Pre-compute antonym embeddings for efficiency
    print("🔄 Pre-computing antonym embeddings...")
    antonym_embeddings = {
        "CORRECT": embed_text("CORRECT"),
        "INCORRECT": embed_text("INCORRECT")
    }
    print(f"✅ Antonym embeddings ready")
    
    configs = {
        # Group 1: ("question", "kc", "sa") - 4 pipelines
        "pipeline_1": EmbeddingPipelineConfig(
            question_embedding_model=embedding_model,
            kc_strategy="average",
            sa_strategy="mirror",
            execution_order=("question", "kc", "sa")
        ),
        
        "pipeline_2": EmbeddingPipelineConfig(
            question_embedding_model=embedding_model,
            kc_strategy="average", 
            sa_strategy="stack_antonym",
            execution_order=("question", "kc", "sa"),
            sa_kwargs={
                "antonym_embeddings": antonym_embeddings
                # Removed text_to_embedding_fn to avoid pickle issues
            }
        ),
        
        "pipeline_3": EmbeddingPipelineConfig(
            question_embedding_model=embedding_model,
            kc_strategy="sample_single_question",
            sa_strategy="mirror", 
            execution_order=("question", "kc", "sa")
        ),
        
        "pipeline_4": EmbeddingPipelineConfig(
            question_embedding_model=embedding_model,
            kc_strategy="sample_single_question",
            sa_strategy="stack_antonym",
            execution_order=("question", "kc", "sa"),
            sa_kwargs={
                "antonym_embeddings": antonym_embeddings
                # Removed text_to_embedding_fn to avoid pickle issues
            }
        ),
        
        # Group 2: ("kc", "question", "sa") - 2 pipelines  
        "pipeline_5": EmbeddingPipelineConfig(
            question_embedding_model=embedding_model,
            kc_strategy="stuff_window",
            sa_strategy="mirror",
            execution_order=("kc", "question", "sa"),
            kc_kwargs={
                "window_size": auto_window_size,
                "question_df": question_df
                # Removed problematic embedding_model kwarg
            }
        ),
        
        "pipeline_6": EmbeddingPipelineConfig(
            question_embedding_model=embedding_model,
            kc_strategy="stuff_window",
            sa_strategy="stack_antonym", 
            execution_order=("kc", "question", "sa"),
            kc_kwargs={
                "window_size": auto_window_size,
                "question_df": question_df
                # Removed problematic embedding_model kwarg
            },
            sa_kwargs={
                "antonym_embeddings": antonym_embeddings
            }
        ),
        
        # Group 3: ("sa", "question", "kc") - 2 pipelines
        "pipeline_7": EmbeddingPipelineConfig(
            question_embedding_model=embedding_model,
            kc_strategy="average",
            sa_strategy="add_words",
            execution_order=("sa", "question", "kc"),
            sa_kwargs={
                "repetitions": 2
                # Removed text_to_embedding_fn
            }
        ),
        
        "pipeline_8": EmbeddingPipelineConfig(
            question_embedding_model=embedding_model,
            kc_strategy="sample_single_question", 
            sa_strategy="add_words",
            execution_order=("sa", "question", "kc"),
            sa_kwargs={
                "repetitions": 2
                # Removed text_to_embedding_fn
            }
        ),
        
        # Group 4: ("sa", "kc", "question") - 1 pipeline
        "pipeline_9": EmbeddingPipelineConfig(
            question_embedding_model=embedding_model,
            kc_strategy="stuff_window",
            sa_strategy="add_words",
            execution_order=("sa", "kc", "question"),
            kc_kwargs={
                "window_size": auto_window_size
                # Removed duplicate question_df and embedding_model
            },
            sa_kwargs={
                "repetitions": 2
                # Removed text_to_embedding_fn
            }
        )
    }
    
    return configs



# =============================================================================
# STEP 3: GENERATE EMBEDDINGS FOR ALL PIPELINES
# =============================================================================

# def generate_embeddings_for_pipeline(pipeline_name, config, question_df, qid_to_kc, output_dir, provider_info):
#     """Generate embeddings for a single pipeline configuration."""
    
#     print(f"\n🚀 Generating embeddings for {pipeline_name}")
#     print(f"   Strategy: {config.kc_strategy} + {config.sa_strategy}")
#     print(f"   Order: {config.execution_order}")
    
#     try:
#         # Create pipeline and run
#         pipeline = EmbeddingPipeline(config)
#         q_vecs, kc_vecs, sa_vecs = pipeline.run(question_df, qid_to_kc)
        
#         # Print results
#         print(f"   ✅ Generated:")
#         print(f"      - {len(q_vecs)} question embeddings") 
#         print(f"      - {len(kc_vecs)} KC embeddings")
#         print(f"      - {len(sa_vecs)} SA embeddings")
        
#         # Get embedding dimensions
#         if q_vecs:
#             q_dim = next(iter(q_vecs.values())).shape[0]
#             print(f"      - Question embedding dim: {q_dim}")
#         if kc_vecs:
#             kc_dim = next(iter(kc_vecs.values())).shape[0] 
#             print(f"      - KC embedding dim: {kc_dim}")
#         if sa_vecs:
#             sa_dim = next(iter(sa_vecs.values())).shape[0]
#             print(f"      - SA embedding dim: {sa_dim}")
#             total_sa_dims = len(sa_vecs) * sa_dim
#             print(f"      - Total SA dimensions: {total_sa_dims}")
        
#         # Save embeddings
#         output_path = output_dir / f"{pipeline_name}_embeddings.pkl"
#         with open(output_path, 'wb') as f:
#             pickle.dump({
#                 'question_embeddings': q_vecs,
#                 'kc_embeddings': kc_vecs, 
#                 'sa_embeddings': sa_vecs,
#                 'config': config,
#                 'provider_info': provider_info
#             }, f)
        
#         print(f"   💾 Saved to: {output_path}")
#         return True
        
#     except Exception as e:
#         print(f"   ❌ Failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return False


def generate_embeddings_for_pipeline(pipeline_name, config, question_df, qid_to_kc, output_dir, provider_info):
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
        
        # Get embedding dimensions - FIXED to handle both tensor and string values
        if q_vecs:
            sample_q_val = next(iter(q_vecs.values()))
            if isinstance(sample_q_val, torch.Tensor):
                q_dim = sample_q_val.shape[0]
                print(f"      - Question embedding dim: {q_dim}")
        
        if kc_vecs:
            sample_kc_val = next(iter(kc_vecs.values()))
            if isinstance(sample_kc_val, torch.Tensor):
                kc_dim = sample_kc_val.shape[0]
                print(f"      - KC embedding dim: {kc_dim}")
        
        if sa_vecs:
            sample_sa_val = next(iter(sa_vecs.values()))
            # FIXED: Handle both tensor and string values
            if isinstance(sample_sa_val, torch.Tensor):
                sa_dim = sample_sa_val.shape[0]
                print(f"      - SA embedding dim: {sa_dim}")
                total_sa_dims = len(sa_vecs) * sa_dim
                print(f"      - Total SA dimensions: {total_sa_dims}")
            else:
                print(f"      - SA embeddings: text format (add_words strategy)")
        
        # Save embeddings - FIXED to avoid pickle issues
        output_path = output_dir / f"{pipeline_name}_embeddings.pkl"
        
        # Create clean config without unpicklable functions
        clean_config = EmbeddingPipelineConfig(
            question_embedding_model=config.question_embedding_model,
            kc_strategy=config.kc_strategy,
            sa_strategy=config.sa_strategy,
            execution_order=config.execution_order,
            kc_kwargs={k: v for k, v in config.kc_kwargs.items() 
                      if not callable(v) and k != 'text_to_embedding_fn'},
            sa_kwargs={k: v for k, v in config.sa_kwargs.items() 
                      if not callable(v) and k != 'text_to_embedding_fn'}
        )
        
        with open(output_path, 'wb') as f:
            pickle.dump({
                'question_embeddings': q_vecs,
                'kc_embeddings': kc_vecs, 
                'sa_embeddings': sa_vecs,
                'config': clean_config,
                'provider_info': provider_info
            }, f)
        
        print(f"   💾 Saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False



def generate_all_embeddings():
    """Main function to generate embeddings for all pipelines."""
    
    print("🎯 Starting embedding generation for all 9 pipelines")
    
    # Step 1: Check requirements and setup provider
    provider, model, embedding_dim = check_requirements()
    provider_info = {
        "provider": provider,
        "model": model,
        "embedding_dim": embedding_dim
    }
    
    # Step 2: Load data
    print("\n📊 Loading your data...")
    question_df, qid_to_kc = load_your_data(data_dir="./my_data")
    print(f"   - Loaded {len(question_df)} questions")
    print(f"   - Loaded {len(set(qid_to_kc.values()))} unique KCs")
    
    # Step 3: Get pipeline configurations
    print("\n⚙️ Setting up pipeline configurations...")
    configs = get_pipeline_configs(question_df, model, provider)
    print(f"   - Configured {len(configs)} pipelines")
    
    # Step 4: Create output directory
    output_dir = Path("generated_embeddings")
    output_dir.mkdir(exist_ok=True)
    print(f"   - Output directory: {output_dir}")
    
    # Step 5: Generate embeddings for each pipeline
    results = {}
    
    for pipeline_name, config in configs.items():
        success = generate_embeddings_for_pipeline(
            pipeline_name, config, question_df, qid_to_kc, output_dir, provider_info
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
        
        # Show provider info
        print(f"\n📋 Provider Information:")
        print(f"   - Provider: {provider.upper()}")
        print(f"   - Model: {model}")
        print(f"   - Dimensions: {embedding_dim}")

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
    
    # Example: Get embedding for specific KC interaction
    if kc_vecs:
        example_kc = list(kc_vecs.keys())[0]
        if (example_kc, True) in sa_vecs and (example_kc, False) in sa_vecs:
            correct_embedding = sa_vecs[(example_kc, True)]
            incorrect_embedding = sa_vecs[(example_kc, False)]
            
            print(f"\n📋 Example embeddings for KC '{example_kc}':")
            print(f"   - Correct response: shape {correct_embedding.shape}")
            print(f"   - Incorrect response: shape {incorrect_embedding.shape}")
            
            # Show the total dimensions for DKT
            total_sa_dims = len(sa_vecs) * correct_embedding.shape[0]
            print(f"   - Total SA dimensions for DKT: {total_sa_dims}")
            print(f"   - Format: 2×{len(kc_vecs)} KCs × {correct_embedding.shape[0]} dims")
    
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