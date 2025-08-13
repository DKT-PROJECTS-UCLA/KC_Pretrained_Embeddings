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
import json  # ← ADD THIS LINE

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

def load_your_complete_mappings(
    mappings_dir: str = "mappings_output",
    keyid2idx_path: str = "/home/mahdi/Projects/pykt-toolkit-pt_emb/data/assist2009/keyid2idx.json"
) -> Tuple[Dict[str, str], Dict[str, int], Dict[str, int]]:
    """
    Load all mappings from your file structure.
    
    Parameters
    ----------
    mappings_dir : str
        Directory containing your mapping files
    keyid2idx_path : str
        Path to keyid2idx.json file with final concept ordering
        
    Returns
    -------
    Tuple[Dict[str, str], Dict[str, int], Dict[str, int]]
        (qid_to_kc_name, kc_name_to_id, kc_id_to_position)
    """
    
    mappings_path = Path(mappings_dir)
    
    print(f"📂 Loading mappings from {mappings_dir}/")
    
    # 1. Load qid_to_kc (question_id -> kc_name)
    qid_to_kc_json_path = mappings_path / "qid_to_kc.json"
    with open(qid_to_kc_json_path, 'r') as f:
        qid_to_kc_name = json.load(f)
    print(f"   ✅ qid_to_kc: {len(qid_to_kc_name)} mappings")
    
    # 2. Load kc_name_to_id
    kc_name_to_id_path = mappings_path / "kc_name_to_id.json"
    with open(kc_name_to_id_path, 'r') as f:
        kc_name_to_id = json.load(f)
    print(f"   ✅ kc_name_to_id: {len(kc_name_to_id)} mappings")
    
    # 3. Load final concept ordering from keyid2idx.json
    with open(keyid2idx_path, 'r') as f:
        keyid2idx = json.load(f)
    
    if 'concepts' not in keyid2idx:
        raise ValueError(f"keyid2idx.json must contain 'concepts' field")
    
    # Convert concepts mapping: kc_id (str) -> position (int)
    concepts_raw = keyid2idx['concepts']
    kc_id_to_position = {kc_id: int(position) for kc_id, position in concepts_raw.items()}
    print(f"   ✅ concepts (final ordering): {len(kc_id_to_position)} positions")
    
    # Show some examples
    print(f"\n📋 Sample mappings:")
    sample_qid = list(qid_to_kc_name.keys())[0]
    sample_kc_name = qid_to_kc_name[sample_qid]
    print(f"   Question: '{sample_qid}' -> KC name: '{sample_kc_name}'")
    
    if sample_kc_name in kc_name_to_id:
        sample_kc_id = kc_name_to_id[sample_kc_name]
        print(f"   KC name: '{sample_kc_name}' -> KC ID: {sample_kc_id}")
        
        if str(sample_kc_id) in concepts_raw:
            sample_position = concepts_raw[str(sample_kc_id)]
            print(f"   KC ID: {sample_kc_id} -> Position: {sample_position}")
            print(f"   Complete chain: '{sample_qid}' -> '{sample_kc_name}' -> {sample_kc_id} -> position {sample_position}")
    
    return qid_to_kc_name, kc_name_to_id, kc_id_to_position




def validate_your_mapping_chain(
    qid_to_kc_name: Dict[str, str],
    kc_name_to_id: Dict[str, int], 
    kc_id_to_position: Dict[str, int]
) -> bool:
    """
    Validate your complete mapping chain.
    
    Returns
    -------
    bool
        True if all mappings are complete and valid
    """
    
    print(f"\n🔍 Validating mapping chain...")
    
    # Get unique KC names from questions
    data_kc_names = set(qid_to_kc_name.values())
    mapping_kc_names = set(kc_name_to_id.keys())
    
    # Get KC IDs
    mapped_kc_ids = set(str(kc_id) for kc_id in kc_name_to_id.values())
    position_kc_ids = set(kc_id_to_position.keys())
    
    print(f"   KC names in questions: {len(data_kc_names)}")
    print(f"   KC names with ID mappings: {len(mapping_kc_names)}")
    print(f"   KC IDs from mappings: {len(mapped_kc_ids)}")
    print(f"   KC IDs with positions: {len(position_kc_ids)}")
    
    # Check for missing mappings
    missing_kc_names = data_kc_names - mapping_kc_names
    missing_kc_ids = mapped_kc_ids - position_kc_ids
    
    is_valid = True
    
    if missing_kc_names:
        print(f"   ❌ KC names without ID mappings: {len(missing_kc_names)}")
        if len(missing_kc_names) <= 5:
            print(f"      {list(missing_kc_names)}")
        else:
            print(f"      {list(missing_kc_names)[:5]} ... and {len(missing_kc_names)-5} more")
        is_valid = False
    
    if missing_kc_ids:
        print(f"   ❌ KC IDs without positions: {len(missing_kc_ids)}")
        print(f"      {list(missing_kc_ids)}")
        is_valid = False
    
    if is_valid:
        print(f"   ✅ All mappings validated successfully!")
        
        # Show mapping statistics
        total_questions = len(qid_to_kc_name)
        total_kc_names = len(data_kc_names)
        total_positions = len(kc_id_to_position)
        
        print(f"\n📊 Mapping Statistics:")
        print(f"   Total questions: {total_questions:,}")
        print(f"   Unique KC names: {total_kc_names}")
        print(f"   Final positions: {total_positions}")
        print(f"   Max position: {max(kc_id_to_position.values())}")
        
    return is_valid


# def create_question_df_from_csv(
#     mappings_dir: str = "mappings_output"
# ) -> pd.DataFrame:
#     """
#     Create question_df from your questions_with_kc.csv file.
    
#     Parameters
#     ----------
#     mappings_dir : str
#         Directory containing questions_with_kc.csv
        
#     Returns
#     -------
#     pd.DataFrame
#         DataFrame with required columns for the pipeline
#     """
    
#     csv_path = Path(mappings_dir) / "questions_with_kc.csv"
    
#     print(f"📊 Loading questions from {csv_path}")
#     df = pd.read_csv(csv_path)
    
#     # Create the required format for the pipeline
#     question_df = pd.DataFrame({
#         'question_id': df['question_id'].astype(str),
#         'question_text': df['problem_body']
#     })
    
#     print(f"   ✅ Loaded {len(question_df)} questions")
#     print(f"   Sample question: {question_df.iloc[0]['question_text'][:100]}...")
    
#     return question_df

def create_question_df_from_csv(
    mappings_dir: str = "mappings_output"
) -> pd.DataFrame:
    """
    Create question_df from your questions_with_kc.csv file.
    FIXED: Ensures question IDs match the format in qid_to_kc.json
    """
    
    csv_path = Path(mappings_dir) / "questions_with_kc.csv"
    
    print(f"📊 Loading questions from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # FIXED: Convert question_id to match qid_to_kc format (add 'q' prefix)
    question_df = pd.DataFrame({
        'question_id': 'q' + df['question_id'].astype(str),  # Add 'q' prefix
        'question_text': df['problem_body']
    })
    
    print(f"   ✅ Loaded {len(question_df)} questions")
    print(f"   Sample question ID: {question_df.iloc[0]['question_id']}")
    print(f"   Sample question: {question_df.iloc[0]['question_text'][:100]}...")
    
    return question_df




# def convert_sa_to_final_ordered_tensor(
#     sa_embeddings: Dict[Tuple[str, bool], torch.Tensor],
#     kc_name_to_id: Dict[str, int],
#     kc_id_to_position: Dict[str, int],
#     verbose: bool = True
# ) -> torch.Tensor:
#     """
#     Convert SA embeddings to final ordered tensor using your exact mappings.
    
#     Parameters
#     ----------
#     sa_embeddings : Dict[Tuple[str, bool], torch.Tensor]
#         SA embeddings: {(kc_name, is_correct): embedding_tensor}
#     kc_name_to_id : Dict[str, int]
#         KC name to ID mapping
#     kc_id_to_position : Dict[str, int]
#         KC ID to position mapping (from keyid2idx.json)
#     verbose : bool
#         Print detailed information
        
#     Returns
#     -------
#     torch.Tensor
#         Final ordered tensor ready for DKT
#     """
    
#     # Determine tensor size
#     max_position = max(kc_id_to_position.values())
#     tensor_size = max_position + 1  # 0-indexed positions
    
#     # Get embedding dimension
#     sample_embedding = next(iter(sa_embeddings.values()))
#     embedding_dim = sample_embedding.shape[0]
    
#     # Create ordered tensor
#     ordered_tensor = torch.zeros(2 * tensor_size, embedding_dim, dtype=torch.float32)
    
#     # Track placements
#     placed_correct = 0
#     placed_incorrect = 0
#     failed_placements = []
    
#     # Process each SA embedding
#     for (kc_name, is_correct), embedding in sa_embeddings.items():
        
#         # KC name -> KC ID
#         if kc_name not in kc_name_to_id:
#             failed_placements.append(f"KC name '{kc_name}' not found in kc_name_to_id")
#             continue
        
#         kc_id = kc_name_to_id[kc_name]
#         kc_id_str = str(kc_id)
        
#         # KC ID -> position
#         if kc_id_str not in kc_id_to_position:
#             failed_placements.append(f"KC ID '{kc_id}' (from '{kc_name}') not found in concepts")
#             continue
        
#         position = kc_id_to_position[kc_id_str]
        
#         # Place in tensor
#         if is_correct:
#             # Correct: positions 0 to tensor_size-1
#             ordered_tensor[position] = embedding
#             placed_correct += 1
#         else:
#             # Incorrect: positions tensor_size to 2*tensor_size-1
#             ordered_tensor[tensor_size + position] = embedding
#             placed_incorrect += 1
    
#     if verbose:
#         print(f"✅ Created final ordered tensor: {ordered_tensor.shape}")
#         print(f"   Tensor layout: [{tensor_size}, {embedding_dim}] × 2")
#         print(f"   Rows 0-{tensor_size-1}: Correct embeddings by keyid2idx position")
#         print(f"   Rows {tensor_size}-{2*tensor_size-1}: Incorrect embeddings by keyid2idx position")
#         print(f"   Successfully placed correct: {placed_correct}")
#         print(f"   Successfully placed incorrect: {placed_incorrect}")
#         print(f"   Total parameters: {ordered_tensor.numel():,}")
        
#         if failed_placements:
#             print(f"⚠️  Failed placements: {len(failed_placements)}")
#             for failure in failed_placements[:3]:
#                 print(f"      {failure}")
#             if len(failed_placements) > 3:
#                 print(f"      ... and {len(failed_placements)-3} more")
        
#         # Show example mappings
#         print(f"\n📋 Example position mappings:")
#         for kc_id, position in list(kc_id_to_position.items())[:5]:
#             print(f"   KC ID {kc_id} -> position {position}")
    
#     return ordered_tensor


def convert_sa_to_final_ordered_tensor(
    sa_embeddings: Dict[Tuple[str, bool], torch.Tensor],
    kc_name_to_id: Dict[str, int],
    kc_id_to_position: Dict[str, int],
    verbose: bool = True
) -> torch.Tensor:
    """
    Convert SA embeddings to final ordered tensor using your exact mappings.
    FIXED: Handles both tensor and string formats from add_words strategy
    """
    
    # FIXED: Filter out non-tensor SA embeddings (from add_words strategy)
    tensor_sa_embeddings = {}
    string_sa_embeddings = {}
    
    for key, value in sa_embeddings.items():
        if isinstance(value, torch.Tensor):
            tensor_sa_embeddings[key] = value
        elif isinstance(value, str):
            string_sa_embeddings[key] = value
        else:
            print(f"⚠️  Unknown SA embedding format for {key}: {type(value)}")
    
    if verbose:
        print(f"🔍 SA embeddings analysis:")
        print(f"   Tensor embeddings: {len(tensor_sa_embeddings)}")
        print(f"   String embeddings: {len(string_sa_embeddings)} (add_words strategy)")
    
    # For add_words strategy, we need to convert the KC embeddings instead
    if len(tensor_sa_embeddings) == 0 and len(string_sa_embeddings) > 0:
        print(f"⚠️  SA embeddings are in text format (add_words strategy)")
        print(f"   This pipeline uses modified question texts instead of tensor embeddings")
        print(f"   Creating dummy tensor for consistency...")
        
        # Create a dummy tensor with expected shape
        max_position = max(kc_id_to_position.values())
        tensor_size = max_position + 1
        embedding_dim = 384  # Default BERT dimension
        
        dummy_tensor = torch.zeros(2 * tensor_size, embedding_dim, dtype=torch.float32)
        
        if verbose:
            print(f"   Created dummy tensor: {dummy_tensor.shape}")
            print(f"   ⚠️  This tensor contains zeros - add_words strategy embeddings are in KC format")
        
        return dummy_tensor
    
    # Process tensor embeddings normally
    if len(tensor_sa_embeddings) == 0:
        raise ValueError("No valid tensor embeddings found in SA embeddings")
    
    # Determine tensor size
    max_position = max(kc_id_to_position.values())
    tensor_size = max_position + 1  # 0-indexed positions
    
    # Get embedding dimension
    sample_embedding = next(iter(tensor_sa_embeddings.values()))
    embedding_dim = sample_embedding.shape[0]
    
    # Create ordered tensor
    ordered_tensor = torch.zeros(2 * tensor_size, embedding_dim, dtype=torch.float32)
    
    # Track placements
    placed_correct = 0
    placed_incorrect = 0
    failed_placements = []
    
    # Process each tensor SA embedding
    for (kc_name, is_correct), embedding in tensor_sa_embeddings.items():
        
        # KC name -> KC ID
        if kc_name not in kc_name_to_id:
            failed_placements.append(f"KC name '{kc_name}' not found in kc_name_to_id")
            continue
        
        kc_id = kc_name_to_id[kc_name]
        kc_id_str = str(kc_id)
        
        # KC ID -> position
        if kc_id_str not in kc_id_to_position:
            failed_placements.append(f"KC ID '{kc_id}' (from '{kc_name}') not found in concepts")
            continue
        
        position = kc_id_to_position[kc_id_str]
        
        # Place in tensor
        if is_correct:
            # Correct: positions 0 to tensor_size-1
            ordered_tensor[position] = embedding
            placed_correct += 1
        else:
            # Incorrect: positions tensor_size to 2*tensor_size-1
            ordered_tensor[tensor_size + position] = embedding
            placed_incorrect += 1
    
    if verbose:
        print(f"✅ Created final ordered tensor: {ordered_tensor.shape}")
        print(f"   Tensor layout: [{tensor_size}, {embedding_dim}] × 2")
        print(f"   Rows 0-{tensor_size-1}: Correct embeddings by keyid2idx position")
        print(f"   Rows {tensor_size}-{2*tensor_size-1}: Incorrect embeddings by keyid2idx position")
        print(f"   Successfully placed correct: {placed_correct}")
        print(f"   Successfully placed incorrect: {placed_incorrect}")
        print(f"   Total parameters: {ordered_tensor.numel():,}")
        
        if failed_placements:
            print(f"⚠️  Failed placements: {len(failed_placements)}")
            for failure in failed_placements[:3]:
                print(f"      {failure}")
            if len(failed_placements) > 3:
                print(f"      ... and {len(failed_placements)-3} more")
    
    return ordered_tensor

def debug_question_ids():
    """Debug function to check question ID formats."""
    
    print("🔍 Debugging question ID formats...")
    
    # Load qid_to_kc
    with open("mappings_output/qid_to_kc.json", 'r') as f:
        qid_to_kc = json.load(f)
    
    # Load CSV
    csv_df = pd.read_csv("mappings_output/questions_with_kc.csv")
    
    # Check formats
    sample_qid_json = list(qid_to_kc.keys())[0]
    sample_qid_csv = str(csv_df['question_id'].iloc[0])
    
    print(f"   Sample QID from JSON: '{sample_qid_json}' (type: {type(sample_qid_json)})")
    print(f"   Sample QID from CSV:  '{sample_qid_csv}' (type: {type(sample_qid_csv)})")
    
    # Check if they match
    csv_qids = set('q' + csv_df['question_id'].astype(str))
    json_qids = set(qid_to_kc.keys())
    
    overlap = csv_qids & json_qids
    only_csv = csv_qids - json_qids
    only_json = json_qids - csv_qids
    
    print(f"   QIDs in CSV (with 'q' prefix): {len(csv_qids)}")
    print(f"   QIDs in JSON: {len(json_qids)}")
    print(f"   Overlapping QIDs: {len(overlap)}")
    print(f"   Only in CSV: {len(only_csv)}")
    print(f"   Only in JSON: {len(only_json)}")
    
    # FIXED: Compare len(overlap) instead of overlap
    if len(overlap) > 0:
        print(f"   ✅ Found {len(overlap)} matching question IDs")
    else:
        print(f"   ❌ No matching question IDs found!")

def generate_final_embeddings():
    """
    Generate embeddings using your complete file structure.
    """
    
    print("🎯 Generating embeddings with your file structure")
    print("="*60)
    
    # Step 1: Load all mappings
    qid_to_kc_name, kc_name_to_id, kc_id_to_position = load_your_complete_mappings()
    
    # Step 2: Validate mappings
    if not validate_your_mapping_chain(qid_to_kc_name, kc_name_to_id, kc_id_to_position):
        print("❌ Mapping validation failed. Please check your files.")
        return
    
    # Step 3: Create question DataFrame
    question_df = create_question_df_from_csv()
    
    # Step 4: Setup embedding system
    provider, model, embedding_dim = check_requirements()
    provider_info = {"provider": provider, "model": model, "embedding_dim": embedding_dim}
    
    # Step 5: Setup pipelines
    print(f"\n⚙️ Setting up pipeline configurations...")
    configs = get_pipeline_configs(question_df, model, provider)
    
    # Step 6: Create output directory
    output_dir = Path("final_embeddings")
    output_dir.mkdir(exist_ok=True)
    
    # Save complete mapping info
    mapping_info = {
        "source_files": {
            "qid_to_kc": "mappings_output/qid_to_kc.json",
            "kc_name_to_id": "mappings_output/kc_name_to_id.json", 
            "concepts": "keyid2idx.json",
            "questions": "mappings_output/questions_with_kc.csv"
        },
        "mapping_statistics": {
            "total_questions": len(qid_to_kc_name),
            "unique_kc_names": len(set(qid_to_kc_name.values())),
            "kc_name_mappings": len(kc_name_to_id),
            "concept_positions": len(kc_id_to_position),
            "max_position": max(kc_id_to_position.values())
        },
        "tensor_layout": {
            "description": "Final tensor layout based on keyid2idx.json",
            "positions_0_to_N": "correct_embeddings_by_keyid2idx_order",
            "positions_N_to_2N": "incorrect_embeddings_by_keyid2idx_order"
        }
    }
    
    with open(output_dir / "mapping_info.json", 'w') as f:
        json.dump(mapping_info, f, indent=2)
    
    # Step 7: Generate embeddings for all pipelines
    results = {}
    final_embeddings = {}
    
    for pipeline_name, config in configs.items():
        print(f"\n🚀 Processing {pipeline_name}")
        print(f"   Strategy: {config.kc_strategy} + {config.sa_strategy}")
        
        try:
            # Generate embeddings
            pipeline = EmbeddingPipeline(config)
            q_vecs, kc_vecs, sa_vecs = pipeline.run(question_df, qid_to_kc_name)
            
            if sa_vecs:
                # Convert to final ordered tensor
                final_tensor = convert_sa_to_final_ordered_tensor(
                    sa_vecs, kc_name_to_id, kc_id_to_position
                )
                
                final_embeddings[pipeline_name] = final_tensor
                
                # Save final DKT-ready tensor
                torch.save(final_tensor, output_dir / f"{pipeline_name}_final_dkt.pt")
                
                # Save original embeddings for reference
                clean_config = EmbeddingPipelineConfig(
                    question_embedding_model=config.question_embedding_model,
                    kc_strategy=config.kc_strategy,
                    sa_strategy=config.sa_strategy,
                    execution_order=config.execution_order,
                    kc_kwargs={k: v for k, v in config.kc_kwargs.items() if not callable(v)},
                    sa_kwargs={k: v for k, v in config.sa_kwargs.items() if not callable(v)}
                )
                
                with open(output_dir / f"{pipeline_name}_original.pkl", 'wb') as f:
                    pickle.dump({
                        'question_embeddings': q_vecs,
                        'kc_embeddings': kc_vecs,
                        'sa_embeddings': sa_vecs,
                        'config': clean_config,
                        'provider_info': provider_info
                    }, f)
                
                print(f"   ✅ Success: {final_tensor.shape}")
                
            results[pipeline_name] = True
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            results[pipeline_name] = False
    
    # Step 8: Save all final embeddings
    if final_embeddings:
        torch.save(final_embeddings, output_dir / "all_final_dkt_embeddings.pt")
        
        # Create comprehensive summary
        sample_tensor = next(iter(final_embeddings.values()))
        
        summary = {
            "generation_timestamp": str(pd.Timestamp.now()),
            "successful_pipelines": len(final_embeddings),
            "total_pipelines": len(results),
            "final_tensor_shape": list(sample_tensor.shape),
            "total_parameters_per_pipeline": sample_tensor.numel(),
            "embedding_provider": provider_info,
            "data_statistics": mapping_info["mapping_statistics"],
            "pipeline_results": results,
            "file_locations": {
                "individual_tensors": "final_embeddings/pipeline_X_final_dkt.pt",
                "all_tensors": "final_embeddings/all_final_dkt_embeddings.pt",
                "original_embeddings": "final_embeddings/pipeline_X_original.pkl"
            }
        }
        
        with open(output_dir / "generation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    # Step 9: Final report
    successful = sum(results.values())
    total = len(results)
    
    print(f"\n📈 FINAL GENERATION REPORT")
    print(f"="*50)
    print(f"Successful pipelines: {successful}/{total}")
    print(f"Questions processed: {len(question_df):,}")
    print(f"KC names: {len(set(qid_to_kc_name.values()))}")
    print(f"Final positions: {len(kc_id_to_position)}")
    
    if final_embeddings:
        sample_shape = next(iter(final_embeddings.values())).shape
        print(f"DKT tensor shape: {sample_shape}")
        print(f"Parameters per pipeline: {sample_shape[0] * sample_shape[1]:,}")
        print(f"Total embedding dimensions: {len(final_embeddings)} × {sample_shape[0] * sample_shape[1]:,}")
    
    print(f"\nPipeline Results:")
    for pipeline_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {pipeline_name}")
    
    if successful > 0:
        print(f"\n🎉 SUCCESS! DKT-ready embeddings generated!")
        print(f"📁 Output directory: final_embeddings/")
        print(f"🎯 Use: final_embeddings/pipeline_X_final_dkt.pt for DKT training")
        print(f"📊 Tensor format: [pos0_correct...posN_correct, pos0_incorrect...posN_incorrect]")
        print(f"🔢 Positions based on keyid2idx.json concepts mapping")


def generate_final_embeddings_with_debug():
    """
    Generate embeddings with debugging for question ID issues.
    """
    
    print("🎯 Generating embeddings with your file structure")
    print("="*60)
    
    # Step 0: Debug question IDs
    debug_question_ids()
    
    # Step 1: Load all mappings
    qid_to_kc_name, kc_name_to_id, kc_id_to_position = load_your_complete_mappings()
    
    # Step 2: Validate mappings
    if not validate_your_mapping_chain(qid_to_kc_name, kc_name_to_id, kc_id_to_position):
        print("❌ Mapping validation failed. Please check your files.")
        return
    
    # Step 3: Create question DataFrame with fixed IDs
    question_df = create_question_df_from_csv()
    
    # Step 3.5: Verify question ID overlap
    df_qids = set(question_df['question_id'])
    mapping_qids = set(qid_to_kc_name.keys())
    overlap = df_qids & mapping_qids
    
    print(f"\n🔍 Question ID verification:")
    print(f"   Question IDs in DataFrame: {len(df_qids)}")
    print(f"   Question IDs in mapping: {len(mapping_qids)}")
    print(f"   Overlapping IDs: {len(overlap)}")
    
    if len(overlap) == 0:
        print("❌ No overlapping question IDs! Cannot proceed.")
        return
    elif len(overlap) < len(df_qids) * 0.8:
        print(f"⚠️  Low overlap ({len(overlap)}/{len(df_qids)} = {len(overlap)/len(df_qids)*100:.1f}%)")
        print("   Proceeding with available questions...")
    else:
        print(f"✅ Good overlap ({len(overlap)}/{len(df_qids)} = {len(overlap)/len(df_qids)*100:.1f}%)")
    
    # Step 4: Setup embedding system
    provider, model, embedding_dim = check_requirements()
    provider_info = {"provider": provider, "model": model, "embedding_dim": embedding_dim}
    
    # Step 5: Setup pipelines
    print(f"\n⚙️ Setting up pipeline configurations...")
    configs = get_pipeline_configs(question_df, model, provider)
    
    # Step 6: Create output directory
    output_dir = Path("final_embeddings")
    output_dir.mkdir(exist_ok=True)
    
    # Save complete mapping info
    mapping_info = {
        "source_files": {
            "qid_to_kc": "mappings_output/qid_to_kc.json",
            "kc_name_to_id": "mappings_output/kc_name_to_id.json", 
            "concepts": "/home/mahdi/Projects/pykt-toolkit-pt_emb/data/assist2009/keyid2idx.json",
            "questions": "mappings_output/questions_with_kc.csv"
        },
        "mapping_statistics": {
            "total_questions_csv": len(question_df),
            "total_questions_mapping": len(qid_to_kc_name),
            "overlapping_questions": len(overlap),
            "unique_kc_names": len(set(qid_to_kc_name.values())),
            "kc_name_mappings": len(kc_name_to_id),
            "concept_positions": len(kc_id_to_position),
            "max_position": max(kc_id_to_position.values())
        },
        "tensor_layout": {
            "description": "Final tensor layout based on keyid2idx.json",
            "positions_0_to_N": "correct_embeddings_by_keyid2idx_order",
            "positions_N_to_2N": "incorrect_embeddings_by_keyid2idx_order"
        }
    }
    
    with open(output_dir / "mapping_info.json", 'w') as f:
        json.dump(mapping_info, f, indent=2)
    
    # Step 7: Generate embeddings for all pipelines
    results = {}
    final_embeddings = {}
    
    for pipeline_name, config in configs.items():
        print(f"\n🚀 Processing {pipeline_name}")
        print(f"   Strategy: {config.kc_strategy} + {config.sa_strategy}")
        
        try:
            # Generate embeddings
            pipeline = EmbeddingPipeline(config)
            q_vecs, kc_vecs, sa_vecs = pipeline.run(question_df, qid_to_kc_name)
            
            if sa_vecs:
                # Convert to final ordered tensor (with fixes)
                final_tensor = convert_sa_to_final_ordered_tensor(
                    sa_vecs, kc_name_to_id, kc_id_to_position
                )
                
                final_embeddings[pipeline_name] = final_tensor
                
                # Save final DKT-ready tensor
                torch.save(final_tensor, output_dir / f"{pipeline_name}_final_dkt.pt")
                
                # Save original embeddings for reference
                clean_config = EmbeddingPipelineConfig(
                    question_embedding_model=config.question_embedding_model,
                    kc_strategy=config.kc_strategy,
                    sa_strategy=config.sa_strategy,
                    execution_order=config.execution_order,
                    kc_kwargs={k: v for k, v in config.kc_kwargs.items() if not callable(v)},
                    sa_kwargs={k: v for k, v in config.sa_kwargs.items() if not callable(v)}
                )
                
                with open(output_dir / f"{pipeline_name}_original.pkl", 'wb') as f:
                    pickle.dump({
                        'question_embeddings': q_vecs,
                        'kc_embeddings': kc_vecs,
                        'sa_embeddings': sa_vecs,
                        'config': clean_config,
                        'provider_info': provider_info
                    }, f)
                
                print(f"   ✅ Success: {final_tensor.shape}")
                
            results[pipeline_name] = True
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            results[pipeline_name] = False
    
    # Step 8: Save all final embeddings and generate summary (same as before)
    if final_embeddings:
        torch.save(final_embeddings, output_dir / "all_final_dkt_embeddings.pt")
        
        sample_tensor = next(iter(final_embeddings.values()))
        
        summary = {
            "generation_timestamp": str(pd.Timestamp.now()),
            "successful_pipelines": len(final_embeddings),
            "total_pipelines": len(results),
            "final_tensor_shape": list(sample_tensor.shape),
            "total_parameters_per_pipeline": sample_tensor.numel(),
            "embedding_provider": provider_info,
            "data_statistics": mapping_info["mapping_statistics"],
            "pipeline_results": results,
        }
        
        with open(output_dir / "generation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    # Step 9: Final report
    successful = sum(results.values())
    total = len(results)
    
    print(f"\n📈 FINAL GENERATION REPORT")
    print(f"="*50)
    print(f"Successful pipelines: {successful}/{total}")
    print(f"Questions processed: {len(question_df):,}")
    print(f"Questions with embeddings: {len(overlap):,}")
    print(f"KC names: {len(set(qid_to_kc_name.values()))}")
    print(f"Final positions: {len(kc_id_to_position)}")
    
    if final_embeddings:
        sample_shape = next(iter(final_embeddings.values())).shape
        print(f"DKT tensor shape: {sample_shape}")
        print(f"Parameters per pipeline: {sample_shape[0] * sample_shape[1]:,}")
    
    print(f"\nPipeline Results:")
    for pipeline_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {pipeline_name}")
    
    if successful > 0:
        print(f"\n🎉 SUCCESS! DKT-ready embeddings generated!")
        print(f"📁 Output directory: final_embeddings/")
        print(f"🎯 Use: final_embeddings/pipeline_X_final_dkt.pt for DKT training")


if __name__ == "__main__":
    # Generate final embeddings using your file structure
    generate_final_embeddings_with_debug()
    
    print(f"\n📋 Files used:")
    print(f"   - mappings_output/qid_to_kc.json")
    print(f"   - mappings_output/kc_name_to_id.json") 
    print(f"   - keyid2idx.json (concepts)")
    print(f"   - mappings_output/questions_with_kc.csv")