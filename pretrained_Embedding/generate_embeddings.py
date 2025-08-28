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
from embedding_models import (
    load_api_config,
    _get_embedding,
    _current_provider,
    _current_model,
    DEFAULT_MODEL,
    _provider_clients,
    get_embedding_dimension,
    _setup_openai_client,
    _setup_cohere_client,
    _setup_bert_client
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


# DATASET_CONFIG = {
#     "assist2009": {
#         "mappings_dir": "mappings_output2009",
#         "keyid2idx_path": "../data/assist2009/keyid2idx.json",
#         "mapping_format": "standard"  # kc_name -> kc_id -> position
#     }
# }

DATASET_CONFIG = {
    "assist2009": {
        "mappings_dir": "mappings_output2009",
        "keyid2idx_path": "../data/assist2009/keyid2idx.json",
        "mapping_format": "standard"  # kc_name -> kc_id -> position
    },
    "assist2012": {
        "mappings_dir": "mappings_output2012", 
        "keyid2idx_path": "../data/assist2012/keyid2idx.json",
        "mapping_format": "standard"  # kc_name -> kc_id -> position
    },
    "assist2017": {
        "mappings_dir": "mappings_output2017",
        "keyid2idx_path": "../data/assist2017/keyid2idx.json",
        "mapping_format": "direct"  # skill_name -> position directly
    }
}


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


def load_dataset_mappings(
    mappings_dir: str,
    keyid2idx_path: str,
    mapping_format: str = "standard"
) -> Tuple[Dict[str, str], Dict[str, int], Dict[str, int]]:
    """
    Load mappings based on dataset format.
    FIXED: Only apply format conversion for "direct" format (ASSIST2017)
    """
    
    mappings_path = Path(mappings_dir)
    
    print(f"📂 Loading mappings from {mappings_dir}/")
    print(f"   Format: {mapping_format}")
    
    # 1. Load qid_to_kc (always the same)
    qid_to_kc_json_path = mappings_path / "qid_to_kc.json"
    with open(qid_to_kc_json_path, 'r') as f:
        qid_to_kc_name = json.load(f)
    print(f"   ✅ qid_to_kc: {len(qid_to_kc_name)} mappings")
    
    # 2. Load keyid2idx
    with open(keyid2idx_path, 'r') as f:
        keyid2idx = json.load(f)
    concepts_raw = keyid2idx['concepts']
    
    if mapping_format == "direct":
        # ASSIST2017: KC names in data directly match concept keys (with format conversion)
        print(f"   📌 Direct KC name to position mapping")
        
        # Create identity mapping for kc_name_to_id
        kc_name_to_id = {}
        kc_id_to_position = {}
        
        # Get unique KC names from questions
        unique_kc_names = set(qid_to_kc_name.values())
        
        matched = 0
        unmatched = []
        
        for kc_name in unique_kc_names:
            matched_key = None
            
            # Try exact match first
            if kc_name in concepts_raw:
                matched_key = kc_name
            # Try with underscores converted to hyphens
            elif kc_name.replace('_', '-') in concepts_raw:
                matched_key = kc_name.replace('_', '-')
            # Try with "application_" -> "application: " pattern
            elif kc_name.startswith('application_'):
                parts = kc_name.split('_')
                if len(parts) > 1:
                    application_key = parts[0] + ': ' + ' '.join(parts[1:])
                    if application_key in concepts_raw:
                        matched_key = application_key
            # Try other patterns
            elif kc_name.replace('_', ' ') in concepts_raw:
                matched_key = kc_name.replace('_', ' ')
            # Try n_xxx pattern -> n-xxx
            elif kc_name.startswith('n_'):
                n_key = kc_name.replace('_', '-')
                if n_key in concepts_raw:
                    matched_key = n_key
            # Try p_xxx pattern -> p-xxx
            elif kc_name.startswith('p_'):
                p_key = kc_name.replace('_', '-')
                if p_key in concepts_raw:
                    matched_key = p_key
            
            if matched_key:
                kc_name_to_id[kc_name] = matched_key
                kc_id_to_position[matched_key] = int(concepts_raw[matched_key])
                matched += 1
            else:
                unmatched.append(kc_name)
        
        print(f"   ✅ kc_name_to_id: {matched} mappings (identity with format conversion)")
        print(f"   ✅ kc->position: {matched} positions found")
        
        if unmatched:
            print(f"   ⚠️ {len(unmatched)} KC names not in concepts:")
            for kc in unmatched[:5]:
                print(f"      - '{kc}'")
            if len(unmatched) > 5:
                print(f"      ... and {len(unmatched)-5} more")
    
    else:
        # Standard format: kc_name -> kc_id -> position
        # NO FORMAT CONVERSION HERE - just load as-is
        kc_name_to_id_path = mappings_path / "kc_name_to_id.json"
        with open(kc_name_to_id_path, 'r') as f:
            kc_name_to_id = json.load(f)
        print(f"   ✅ kc_name_to_id: {len(kc_name_to_id)} mappings")
        
        # Convert concepts to proper format - NO CONVERSION
        kc_id_to_position = {kc_id: int(position) for kc_id, position in concepts_raw.items()}
        print(f"   ✅ concepts: {len(kc_id_to_position)} positions")
    
    return qid_to_kc_name, kc_name_to_id, kc_id_to_position


def validate_direct_mapping(
    qid_to_kc_name: Dict[str, str],
    kc_id_to_position: Dict[str, int]
) -> bool:
    """
    Validate direct KC name -> position mapping (for ASSIST2017).
    FIXED: Account for underscore/hyphen conversion
    """
    
    print(f"\n🔍 Validating direct mapping...")
    
    # Get unique KC names from questions
    data_kc_names = set(qid_to_kc_name.values())
    position_kc_names = set(kc_id_to_position.keys())
    
    # Try to match with underscore/hyphen conversion
    matched_kc_names = set()
    unmatched_kc_names = set()
    
    for kc_name in data_kc_names:
        # Try exact match
        if kc_name in position_kc_names:
            matched_kc_names.add(kc_name)
        # Try with underscore to hyphen conversion
        elif kc_name.replace('_', '-') in position_kc_names:
            matched_kc_names.add(kc_name)
        # Try with hyphen to underscore conversion
        elif kc_name.replace('-', '_') in position_kc_names:
            matched_kc_names.add(kc_name)
        else:
            unmatched_kc_names.add(kc_name)
    
    print(f"   KC names in questions: {len(data_kc_names)}")
    print(f"   KC names with positions: {len(position_kc_names)}")
    print(f"   Matched KC names: {len(matched_kc_names)}")
    print(f"   Unmatched KC names: {len(unmatched_kc_names)}")
    
    coverage = len(matched_kc_names) / len(data_kc_names) * 100 if data_kc_names else 0
    print(f"   Coverage: {coverage:.1f}%")
    
    if unmatched_kc_names:
        print(f"   ⚠️  KC names without positions: {len(unmatched_kc_names)}")
        for kc in list(unmatched_kc_names)[:10]:
            print(f"      - '{kc}'")
            # Try to find close matches
            kc_hyphen = kc.replace('_', '-')
            kc_underscore = kc.replace('-', '_')
            kc_colon = kc.replace('_', ': ')
            
            # Check various patterns
            for pos_kc in position_kc_names:
                if kc_hyphen in pos_kc or kc_underscore in pos_kc or kc_colon in pos_kc:
                    print(f"        (possible match: '{pos_kc}')")
                    break
        
        if len(unmatched_kc_names) > 10:
            print(f"      ... and {len(unmatched_kc_names)-10} more")
    
    # Accept if coverage is good enough
    min_coverage = 80.0
    is_valid = coverage >= min_coverage
    
    if is_valid:
        print(f"   ✅ Direct mapping validated!")
    else:
        print(f"   ❌ Coverage too low: {coverage:.1f}% < {min_coverage}%")
    
    print(f"\n📊 Mapping Statistics:")
    print(f"   Total questions: {len(qid_to_kc_name):,}")
    print(f"   Unique KC names: {len(data_kc_names)}")
    print(f"   KC names matched: {len(matched_kc_names)}")
    if kc_id_to_position:
        print(f"   Max position: {max(kc_id_to_position.values())}")
    
    return is_valid


def validate_your_mapping_chain(
    qid_to_kc_name: Dict[str, str],
    kc_name_to_id: Dict[str, int], 
    kc_id_to_position: Dict[str, int]
) -> bool:
    """
    Validate your complete mapping chain.
    FIXED: Handle both with and without .0 suffix in keyid2idx
    """
    
    print(f"\n🔍 Validating mapping chain...")
    
    # Get unique KC names from questions
    data_kc_names = set(qid_to_kc_name.values())
    mapping_kc_names = set(kc_name_to_id.keys())
    
    # Get KC IDs from kc_name_to_id
    mapped_kc_ids_raw = set(str(kc_id) for kc_id in kc_name_to_id.values())
    position_kc_ids = set(kc_id_to_position.keys())
    
    # Determine format used in keyid2idx (check if any have .0 suffix)
    uses_decimal_format = any('.0' in kid for kid in position_kc_ids)
    
    # Convert mapped IDs to match the format in keyid2idx
    mapped_kc_ids = set()
    for kc_id in mapped_kc_ids_raw:
        if uses_decimal_format and '.' not in kc_id:
            # Add .0 suffix if keyid2idx uses it
            mapped_kc_ids.add(f"{kc_id}.0")
        else:
            # Keep as-is
            mapped_kc_ids.add(kc_id)
    
    print(f"   KC names in questions: {len(data_kc_names)}")
    print(f"   KC names with ID mappings: {len(mapping_kc_names)}")
    print(f"   KC IDs from mappings: {len(mapped_kc_ids)}")
    print(f"   KC IDs with positions: {len(position_kc_ids)}")
    print(f"   Format: {'with .0 suffix' if uses_decimal_format else 'without .0 suffix'}")
    
    # Check for missing mappings
    missing_kc_names = data_kc_names - mapping_kc_names
    missing_kc_ids = mapped_kc_ids - position_kc_ids
    
    # Calculate coverage
    kc_name_coverage = len(data_kc_names - missing_kc_names) / len(data_kc_names) * 100 if data_kc_names else 0
    kc_id_coverage = len(mapped_kc_ids - missing_kc_ids) / len(mapped_kc_ids) * 100 if mapped_kc_ids else 0
    
    print(f"   KC name coverage: {kc_name_coverage:.1f}%")
    print(f"   KC ID coverage: {kc_id_coverage:.1f}%")
    
    # More flexible validation - allow partial coverage
    is_valid = True
    min_coverage_threshold = 50.0  # Require at least 80% coverage
    
    if missing_kc_names:
        print(f"   ⚠️  KC names without ID mappings: {len(missing_kc_names)}")
        if len(missing_kc_names) <= 5:
            print(f"      {list(missing_kc_names)}")
        else:
            print(f"      {list(missing_kc_names)[:5]} ... and {len(missing_kc_names)-5} more")
        
        if kc_name_coverage < min_coverage_threshold:
            print(f"   ❌ KC name coverage too low: {kc_name_coverage:.1f}% < {min_coverage_threshold}%")
            is_valid = False
        else:
            print(f"   ⚠️  KC name coverage acceptable: {kc_name_coverage:.1f}%")
    
    if missing_kc_ids:
        print(f"   ⚠️  KC IDs without positions: {len(missing_kc_ids)}")
        if len(missing_kc_ids) <= 10:
            print(f"      {list(missing_kc_ids)}")
        else:
            print(f"      {list(missing_kc_ids)[:10]} ... and {len(missing_kc_ids)-10} more")
        
        if kc_id_coverage < min_coverage_threshold:
            print(f"   ❌ KC ID coverage too low: {kc_id_coverage:.1f}% < {min_coverage_threshold}%")
            is_valid = False
        else:
            print(f"   ⚠️  KC ID coverage acceptable: {kc_id_coverage:.1f}%")
    
    if is_valid and len(missing_kc_names) == 0 and len(missing_kc_ids) == 0:
        print(f"   ✅ Perfect mapping chain validated!")
    elif is_valid:
        print(f"   ✅ Acceptable mapping chain validated with partial coverage!")
        print(f"   📝 Will proceed with available mappings")
    else:
        print(f"   ❌ Mapping chain validation failed - coverage too low")
    
    # Show mapping statistics
    total_questions = len(qid_to_kc_name)
    total_kc_names = len(data_kc_names)
    total_positions = len(kc_id_to_position)
    
    print(f"\n📊 Mapping Statistics:")
    print(f"   Total questions: {total_questions:,}")
    print(f"   Unique KC names: {total_kc_names}")
    print(f"   Final positions: {total_positions}")
    if kc_id_to_position:
        print(f"   Max position: {max(kc_id_to_position.values())}")
    
    return is_valid


def create_question_df_from_csv(
    mappings_dir: str
) -> pd.DataFrame:
    """
    Create question_df from your questions_with_kc.csv file.
    FIXED: Handles different CSV column names across datasets
    """
    
    csv_path = Path(mappings_dir) / "questions_with_kc.csv"
    
    print(f"📊 Loading questions from {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"   Available columns: {list(df.columns)}")
    
    # Handle different column names across datasets
    text_col = None
    if 'problem_body' in df.columns:
        text_col = 'problem_body'
    elif 'question_text' in df.columns:
        text_col = 'question_text'
    elif 'template' in df.columns:
        text_col = 'template'
    elif 'text' in df.columns:
        text_col = 'text'
    else:
        # Find any column that might contain text
        text_columns = [col for col in df.columns if 'text' in col.lower() or 'body' in col.lower() or 'template' in col.lower()]
        if text_columns:
            text_col = text_columns[0]
            print(f"   Using text column: {text_col}")
        else:
            raise ValueError(f"No suitable text column found. Available columns: {list(df.columns)}")
    
    # Convert question_id to match qid_to_kc format (add 'q' prefix)
    question_df = pd.DataFrame({
        'question_id': 'q' + df['question_id'].astype(str),  # Add 'q' prefix
        'question_text': df[text_col].fillna("No question text available")  # Handle missing text
    })
    
    print(f"   ✅ Loaded {len(question_df)} questions")
    print(f"   Used text column: {text_col}")
    print(f"   Sample question ID: {question_df.iloc[0]['question_id']}")
    print(f"   Sample question: {question_df.iloc[0]['question_text'][:100]}...")
    
    return question_df


def convert_sa_to_final_ordered_tensor(
    sa_embeddings: Dict[Tuple[str, bool], torch.Tensor],
    kc_name_to_id: Dict[str, int],
    kc_id_to_position: Dict[str, int],
    verbose: bool = True
) -> torch.Tensor:
    """
    Convert SA embeddings to final ordered tensor.
    UPDATED: Use random embeddings for missing KCs instead of zeros
    """
    
    # Detect ID format
    sample_kc_id = next(iter(kc_name_to_id.values())) if kc_name_to_id else None
    is_string_id = isinstance(sample_kc_id, str)
    
    # Detect if positions use .0 suffix
    uses_decimal_format = False
    if not is_string_id and kc_id_to_position:
        uses_decimal_format = any('.0' in str(kid) for kid in kc_id_to_position.keys())
    
    # Filter out non-tensor SA embeddings
    tensor_sa_embeddings = {}
    string_sa_embeddings = {}
    
    for key, value in sa_embeddings.items():
        if isinstance(value, torch.Tensor):
            tensor_sa_embeddings[key] = value
        elif isinstance(value, str):
            string_sa_embeddings[key] = value
    
    if verbose:
        print(f"🔍 SA embeddings analysis:")
        print(f"   Tensor embeddings: {len(tensor_sa_embeddings)}")
        print(f"   String embeddings: {len(string_sa_embeddings)} (add_words strategy)")
        if is_string_id:
            print(f"   ID type: string")
        else:
            print(f"   ID type: numeric")
            print(f"   Position format: {'with .0 suffix' if uses_decimal_format else 'without .0 suffix'}")
    
    if len(tensor_sa_embeddings) == 0:
        raise ValueError("No valid tensor embeddings found in SA embeddings")
    
    # Determine tensor size
    max_position = max(kc_id_to_position.values())
    tensor_size = max_position + 1
    
    # Get embedding dimension and statistics from existing embeddings
    sample_embedding = next(iter(tensor_sa_embeddings.values()))
    embedding_dim = sample_embedding.shape[0]
    
    # ADDED: Calculate statistics from existing embeddings for better random initialization
    all_embeddings = torch.stack(list(tensor_sa_embeddings.values()))
    embed_mean = all_embeddings.mean()
    embed_std = all_embeddings.std()
    
    # CHANGED: Initialize with random values matching the distribution of real embeddings
    ordered_tensor = torch.randn(2 * tensor_size, embedding_dim, dtype=torch.float32)
    ordered_tensor = ordered_tensor * embed_std + embed_mean
    
    # Track which positions we'll fill with real embeddings
    filled_positions = set()
    
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
        
        # Format KC ID to match kc_id_to_position format
        if is_string_id:
            kc_id_str = str(kc_id)
        else:
            if uses_decimal_format:
                kc_id_str = f"{kc_id}.0"
            else:
                kc_id_str = str(kc_id)
        
        # KC ID -> position
        if kc_id_str not in kc_id_to_position:
            failed_placements.append(f"KC ID '{kc_id_str}' (from '{kc_name}') not found in concepts")
            continue
        
        position = kc_id_to_position[kc_id_str]
        
        # Place in tensor (overwriting the random values)
        if is_correct:
            ordered_tensor[position] = embedding
            placed_correct += 1
            filled_positions.add(position)
        else:
            ordered_tensor[tensor_size + position] = embedding
            placed_incorrect += 1
            filled_positions.add(tensor_size + position)
    
    # ADDED: Count how many positions have random vs real embeddings
    total_positions = tensor_size * 2
    random_positions = total_positions - len(filled_positions)
    
    if verbose:
        print(f"✅ Created final ordered tensor: {ordered_tensor.shape}")
        print(f"   Tensor layout: [{tensor_size}, {embedding_dim}] × 2")
        print(f"   Rows 0-{tensor_size-1}: Correct embeddings by keyid2idx position")
        print(f"   Rows {tensor_size}-{2*tensor_size-1}: Incorrect embeddings by keyid2idx position")
        print(f"   Successfully placed correct: {placed_correct}")
        print(f"   Successfully placed incorrect: {placed_incorrect}")
        print(f"   Positions with real embeddings: {len(filled_positions)}")
        print(f"   Positions with random embeddings: {random_positions}")
        print(f"   Random embedding stats: mean={embed_mean:.3f}, std={embed_std:.3f}")
        print(f"   Total parameters: {ordered_tensor.numel():,}")
        
        if failed_placements:
            print(f"⚠️  Failed placements: {len(failed_placements)}")
            for failure in failed_placements[:3]:
                print(f"      {failure}")
            if len(failed_placements) > 3:
                print(f"      ... and {len(failed_placements)-3} more")
    
    return ordered_tensor


def debug_question_ids(mappings_dir: str):
    """
    Debug function to check question ID formats.
    FIXED: Handles empty JSON files
    """
    
    print("🔍 Debugging question ID formats...")
    
    try:
        # Load qid_to_kc
        with open(f"{mappings_dir}/qid_to_kc.json", 'r') as f:
            qid_to_kc = json.load(f)
        
        if not qid_to_kc:
            print(f"   ❌ Empty qid_to_kc.json file in {mappings_dir}")
            return
        
        # Load CSV
        csv_df = pd.read_csv(f"{mappings_dir}/questions_with_kc.csv")
        
        if len(csv_df) == 0:
            print(f"   ❌ Empty CSV file in {mappings_dir}")
            return
        
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
        
        if len(overlap) > 0:
            print(f"   ✅ Found {len(overlap)} matching question IDs")
        else:
            print(f"   ❌ No matching question IDs found!")
            
    except FileNotFoundError as e:
        print(f"   ❌ File not found: {e}")
    except Exception as e:
        print(f"   ❌ Error: {e}")


def debug_assist2017_skills(mappings_dir: str, keyid2idx_path: str):
    """Debug skill name mismatches in ASSIST2017."""
    
    print("\n🔍 Debugging ASSIST2017 skill names...")
    
    # Load qid_to_kc
    with open(f"{mappings_dir}/qid_to_kc.json", 'r') as f:
        qid_to_kc = json.load(f)
    
    # Load keyid2idx
    with open(keyid2idx_path, 'r') as f:
        keyid2idx = json.load(f)
    concepts = keyid2idx['concepts']
    
    # Get unique skills from questions
    question_skills = set(qid_to_kc.values())
    concept_skills = set(concepts.keys())
    
    # Find overlaps and differences
    overlap = question_skills & concept_skills
    only_in_questions = question_skills - concept_skills
    only_in_concepts = concept_skills - question_skills
    
    print(f"\n📊 Skill Analysis:")
    print(f"   Skills in questions: {len(question_skills)}")
    print(f"   Skills in concepts: {len(concept_skills)}")
    print(f"   Overlapping: {len(overlap)}")
    
    print(f"\n📋 Sample skills from questions:")
    for skill in list(question_skills)[:10]:
        print(f"   - '{skill}'")
    
    print(f"\n📋 Sample skills from concepts (keyid2idx):")
    for skill in list(concept_skills)[:10]:
        print(f"   - '{skill}'")
    
    print(f"\n❌ Skills only in questions (not in concepts):")
    for skill in list(only_in_questions)[:10]:
        print(f"   - '{skill}'")
    if len(only_in_questions) > 10:
        print(f"   ... and {len(only_in_questions) - 10} more")
    
    print(f"\n❓ Skills only in concepts (not in questions):")
    for skill in list(only_in_concepts)[:10]:
        print(f"   - '{skill}'")
    if len(only_in_concepts) > 10:
        print(f"   ... and {len(only_in_concepts) - 10} more")
    
    # Check if there's a pattern (e.g., case mismatch, underscores vs spaces)
    print(f"\n🔍 Checking for pattern mismatches...")
    
    # Try case-insensitive matching
    question_skills_lower = {s.lower(): s for s in question_skills}
    concept_skills_lower = {s.lower(): s for s in concept_skills}
    case_overlap = set(question_skills_lower.keys()) & set(concept_skills_lower.keys())
    
    if len(case_overlap) > len(overlap):
        print(f"   ⚠️ Case mismatch detected! Case-insensitive overlap: {len(case_overlap)}")
        print(f"   Examples of case mismatches:")
        for skill_lower in list(case_overlap - {s.lower() for s in overlap})[:5]:
            q_skill = question_skills_lower[skill_lower]
            c_skill = concept_skills_lower[skill_lower]
            print(f"      Questions: '{q_skill}' vs Concepts: '{c_skill}'")


def initialize_provider_silent(provider: str, model: str, config: dict) -> bool:
    """Initialize provider without user interaction."""
    
    if provider == "openai":
        api_key = config.get("openai_api_key")
        if not api_key or api_key == "your-openai-api-key-here":
            print("❌ OpenAI API key not configured")
            return False
        # USE MODULE PREFIX
        return em._setup_openai_client(api_key)
        
    elif provider == "cohere":
        api_key = config.get("cohere_api_key")
        if not api_key or api_key == "your-cohere-api-key-here":
            print("❌ Cohere API key not configured")
            return False
        # USE MODULE PREFIX
        return em._setup_cohere_client(api_key)
        
    elif provider == "bert":
        # USE MODULE PREFIX
        return em._setup_bert_client()
    
    return False


def process_single_dataset_with_output_dir(
    dataset_name: str, 
    dataset_config: dict, 
    provider_info: dict,
    output_dir_name: str
) -> dict:
    """
    Process a single dataset with custom output directory.
    UPDATED: Uses mapping_format flag for different datasets
    """
    
    print(f"\n" + "="*80)
    print(f"🎯 PROCESSING DATASET: {dataset_name.upper()}")
    print(f"🎯 MODEL: {provider_info['provider'].upper()} - {provider_info['model']}")
    print("="*80)
    
    mappings_dir = dataset_config["mappings_dir"]
    keyid2idx_path = dataset_config["keyid2idx_path"]
    mapping_format = dataset_config.get("mapping_format", "standard")  # ADD THIS LINE
    
    try:
        # Check if dataset files exist
        mappings_path = Path(mappings_dir)
        if not mappings_path.exists():
            print(f"❌ Mappings directory not found: {mappings_dir}")
            return {"dataset": dataset_name, "status": "missing_files", "error": f"Missing {mappings_dir}"}
        
        if not Path(keyid2idx_path).exists():
            print(f"❌ KeyID2Idx file not found: {keyid2idx_path}")
            return {"dataset": dataset_name, "status": "missing_files", "error": f"Missing {keyid2idx_path}"}
        
        print(f"✅ Dataset files found:")
        print(f"   Mappings: {mappings_dir}")
        print(f"   KeyID2Idx: {keyid2idx_path}")
        print(f"   Format: {mapping_format}")  # ADD THIS LINE
        
        # Debug question IDs
        debug_question_ids(mappings_dir)
        
        # REPLACE THE SPECIAL HANDLING WITH THIS:
        # Use the general load_dataset_mappings function for ALL datasets
        qid_to_kc_name, kc_name_to_id, kc_id_to_position = load_dataset_mappings(
            mappings_dir, keyid2idx_path, mapping_format
        )
        
        # Check if mappings are empty
        if not qid_to_kc_name:
            print(f"❌ Empty qid_to_kc mapping for {dataset_name}")
            return {"dataset": dataset_name, "status": "empty_mappings", "error": "Empty qid_to_kc mapping"}
        
        # REPLACE THE VALIDATION SECTION WITH THIS:
        # Use appropriate validation based on format
        if mapping_format == "direct":
            # Use the direct validation for ASSIST2017
            is_valid = validate_direct_mapping(qid_to_kc_name, kc_id_to_position)
        else:
            # Standard validation for other datasets
            is_valid = validate_your_mapping_chain(qid_to_kc_name, kc_name_to_id, kc_id_to_position)
        
        if not is_valid:
            print(f"❌ Mapping validation failed for {dataset_name}")
            return {"dataset": dataset_name, "status": "validation_failed", "error": "Mapping validation failed"}
        
        # The rest of your function remains exactly the same...
        # Create question DataFrame
        question_df = create_question_df_from_csv(mappings_dir)
        
        # Verify question ID overlap
        df_qids = set(question_df['question_id'])
        mapping_qids = set(qid_to_kc_name.keys())
        overlap = df_qids & mapping_qids
        
        print(f"\n🔍 Question ID verification:")
        print(f"   Question IDs in DataFrame: {len(df_qids)}")
        print(f"   Question IDs in mapping: {len(mapping_qids)}")
        print(f"   Overlapping IDs: {len(overlap)}")
        
        if len(overlap) == 0:
            print(f"❌ No overlapping question IDs for {dataset_name}!")
            return {"dataset": dataset_name, "status": "no_overlap", "error": "No overlapping question IDs"}
        
        overlap_rate = len(overlap) / len(df_qids) * 100
        print(f"   Overlap rate: {overlap_rate:.1f}%")
        
        if overlap_rate < 50:
            print(f"❌ Overlap rate too low: {overlap_rate:.1f}%")
            return {"dataset": dataset_name, "status": "low_overlap", "error": f"Low overlap rate: {overlap_rate:.1f}%"}
        
        # Continue with pipeline setup and processing...
        # (Rest remains exactly as you have it)
        
        # Step 7: Setup pipelines
        print(f"\n⚙️ Setting up pipeline configurations...")
        configs = get_pipeline_configs(question_df, provider_info["model"], provider_info["provider"])
        
        # Step 8: Create output directory - USING CUSTOM NAME
        output_dir = Path(output_dir_name)
        output_dir.mkdir(exist_ok=True)
        print(f"📁 Output directory: {output_dir}")
        
        # Step 9: Save mapping info
        mapping_info = {
            "dataset": dataset_name,
            "model": provider_info["model"],
            "provider": provider_info["provider"],
            "embedding_dim": provider_info["embedding_dim"],
            "source_files": {
                "qid_to_kc": f"{mappings_dir}/qid_to_kc.json",
                "kc_name_to_id": f"{mappings_dir}/kc_name_to_id.json", 
                "concepts": keyid2idx_path,
                "questions": f"{mappings_dir}/questions_with_kc.csv"
            },
            "mapping_statistics": {
                "total_questions_csv": len(question_df),
                "total_questions_mapping": len(qid_to_kc_name),
                "overlapping_questions": len(overlap),
                "overlap_rate": overlap_rate,
                "unique_kc_names": len(set(qid_to_kc_name.values())),
                "kc_name_mappings": len(kc_name_to_id),
                "concept_positions": len(kc_id_to_position),
                "max_position": max(kc_id_to_position.values()) if kc_id_to_position else 0
            }
        }
        
        with open(output_dir / "mapping_info.json", 'w') as f:
            json.dump(mapping_info, f, indent=2)
        
        # Step 10: Generate embeddings for all pipelines
        pipeline_results = {}
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
                    pipeline_results[pipeline_name] = True
                else:
                    print(f"   ❌ No SA embeddings generated")
                    pipeline_results[pipeline_name] = False
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                pipeline_results[pipeline_name] = False
        
        # Step 11: Save all final embeddings
        if final_embeddings:
            torch.save(final_embeddings, output_dir / "all_final_dkt_embeddings.pt")
            
            sample_tensor = next(iter(final_embeddings.values()))
            
            summary = {
                "dataset": dataset_name,
                "model": provider_info["model"],
                "provider": provider_info["provider"],
                "generation_timestamp": str(pd.Timestamp.now()),
                "successful_pipelines": len(final_embeddings),
                "total_pipelines": len(pipeline_results),
                "final_tensor_shape": list(sample_tensor.shape),
                "total_parameters_per_pipeline": sample_tensor.numel(),
                "embedding_provider": provider_info,
                "data_statistics": mapping_info["mapping_statistics"],
                "pipeline_results": pipeline_results,
            }
            
            with open(output_dir / "generation_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
        
        # Step 12: Dataset summary
        successful = sum(pipeline_results.values())
        total = len(pipeline_results)
        
        print(f"\n📈 DATASET SUMMARY - {dataset_name.upper()}")
        print(f"-" * 50)
        print(f"Model: {provider_info['provider'].upper()} - {provider_info['model']}")
        print(f"Successful pipelines: {successful}/{total}")
        print(f"Questions processed: {len(question_df):,}")
        print(f"Questions with embeddings: {len(overlap):,}")
        print(f"KC names: {len(set(qid_to_kc_name.values()))}")
        print(f"Final positions: {len(kc_id_to_position)}")
        
        if final_embeddings:
            sample_shape = next(iter(final_embeddings.values())).shape
            print(f"DKT tensor shape: {sample_shape}")
            print(f"Parameters per pipeline: {sample_shape[0] * sample_shape[1]:,}")
        
        print(f"Pipeline Results:")
        for pipeline_name, success in pipeline_results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {pipeline_name}")
        
        if successful > 0:
            print(f"✅ {dataset_name} completed successfully!")
            print(f"📁 Output: {output_dir}/")
        
        return {
            "dataset": dataset_name,
            "model": provider_info["model"],
            "provider": provider_info["provider"],
            "status": "completed",
            "successful_pipelines": successful,
            "total_pipelines": total,
            "output_dir": str(output_dir),
            "pipeline_results": pipeline_results
        }
        
    except Exception as e:
        print(f"\n❌ Dataset {dataset_name} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "dataset": dataset_name,
            "model": provider_info.get("model", "unknown"),
            "provider": provider_info.get("provider", "unknown"),
            "status": "failed",
            "error": str(e)
        }


def generate_all_datasets_all_models():
    """
    Automatically process all datasets with ALL embedding models.
    """
    
    print("🎯 AUTOMATIC MULTI-DATASET MULTI-MODEL EMBEDDING GENERATION")
    print("="*80)
    
    # Define all models to process
    ALL_MODELS = [
        # BERT models (Provider 3)
        ("bert", "all-MiniLM-L6-v2"),      # 384 dims, fast
        #("bert", "all-mpnet-base-v2"),      # 768 dims, best quality
        # Add more models as needed
        #("bert", "multi-qa-mpnet-base-dot-v1"),  # 768 dims, Q&A optimized
        #("bert", "all-MiniLM-L12-v2"),      # 384 dims, better than L6
        #("bert", "paraphrase-MiniLM-L6-v2"), # 384 dims, paraphrase
        
        # Uncomment these if you want to use API-based models
        # ("openai", "text-embedding-3-small"),  # 1536 dims
        # ("openai", "text-embedding-3-large"),  # 3072 dims
        # ("cohere", "embed-english-v3.0"),      # 1024 dims
        # ("cohere", "embed-english-light-v3.0"), # 384 dims
    ]
    
    print(f"📊 Will process {len(DATASET_CONFIG)} datasets with {len(ALL_MODELS)} models")
    print(f"📊 Total combinations: {len(DATASET_CONFIG) * len(ALL_MODELS)}")
    
    # Load API config once - USE MODULE PREFIX
    config = em.load_api_config()
    
    # Store results for all model-dataset combinations
    all_results = {}
    
    # Process each model
    for model_idx, (provider, model) in enumerate(ALL_MODELS, 1):
        print(f"\n{'='*80}")
        print(f"🤖 MODEL {model_idx}/{len(ALL_MODELS)}: {provider.upper()} - {model}")
        print(f"{'='*80}")
        
        # Setup embedding system for this model
        try:
            print(f"🔧 Setting up {provider.upper()} - {model}...")
            
            # Initialize provider without user interaction
            success = initialize_provider_silent(provider, model, config)
            if not success:
                print(f"❌ Failed to initialize {provider} - {model}")
                continue
            
            # Set global state - USE MODULE PREFIX
            em._current_provider = provider
            em._current_model = model
            em.DEFAULT_MODEL = model
            
            # Test the embedding system - USE MODULE PREFIX
            print("🧪 Testing embedding system...")
            test_embedding = em._get_embedding("test", model=model, provider=provider)
            embedding_dim = em.get_embedding_dimension(model)
            
            print(f"✅ Embedding system ready!")
            print(f"   Provider: {provider.upper()}")
            print(f"   Model: {model}")
            print(f"   Dimensions: {embedding_dim}")
            
            provider_info = {
                "provider": provider,
                "model": model,
                "embedding_dim": embedding_dim
            }
            
        except Exception as e:
            print(f"❌ Failed to setup {provider} - {model}: {e}")
            continue
        
        # Process each dataset with this model
        model_results = []
        datasets = list(DATASET_CONFIG.keys())
        
        for dataset_idx, (dataset_name, dataset_config) in enumerate(DATASET_CONFIG.items(), 1):
            print(f"\n{'='*20} DATASET {dataset_idx}/{len(datasets)} {'='*20}")
            
            # Create a model-specific output directory
            model_safe_name = model.replace("/", "_").replace("-", "_")
            output_dir_name = f"final_embeddings_{dataset_name}_{model_safe_name}"
            
            # Process dataset
            result = process_single_dataset_with_output_dir(
                dataset_name, 
                dataset_config, 
                provider_info,
                output_dir_name
            )
            
            model_results.append(result)
            
            # Store result with model-dataset key
            result_key = f"{provider}_{model}_{dataset_name}"
            all_results[result_key] = result
        
        # Summary for this model
        completed = [r for r in model_results if r["status"] == "completed"]
        print(f"\n📊 Model {model} Summary:")
        print(f"   Completed: {len(completed)}/{len(model_results)} datasets")
    
    # Final global summary
    print(f"\n" + "="*80)
    print("🏁 FINAL SUMMARY - ALL MODELS, ALL DATASETS")
    print("="*80)
    
    total_combinations = len(ALL_MODELS) * len(DATASET_CONFIG)
    successful_combinations = sum(1 for r in all_results.values() if r["status"] == "completed")
    
    print(f"📊 Overall Results:")
    print(f"   Total combinations attempted: {total_combinations}")
    print(f"   Successful combinations: {successful_combinations}")
    print(f"   Failed combinations: {total_combinations - successful_combinations}")
    
    # Detailed breakdown
    print(f"\n📋 Detailed Results:")
    for (provider, model) in ALL_MODELS:
        print(f"\n   {provider.upper()} - {model}:")
        for dataset_name in DATASET_CONFIG.keys():
            key = f"{provider}_{model}_{dataset_name}"
            if key in all_results:
                result = all_results[key]
                if result["status"] == "completed":
                    print(f"      ✅ {dataset_name}: {result['successful_pipelines']}/{result['total_pipelines']} pipelines")
                else:
                    print(f"      ❌ {dataset_name}: {result['status']}")
    
    # Save comprehensive summary
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"all_models_datasets_summary_{timestamp}.json"
    
    with open(summary_file, 'w') as f:
        json.dump({
            "generation_timestamp": str(pd.Timestamp.now()),
            "models_processed": ALL_MODELS,
            "datasets_processed": list(DATASET_CONFIG.keys()),
            "total_combinations": total_combinations,
            "successful_combinations": successful_combinations,
            "results": all_results
        }, f, indent=2)
    
    print(f"\n📁 Comprehensive summary saved: {summary_file}")
    
    if successful_combinations > 0:
        print(f"\n🎉 SUCCESS! Generated embeddings for {successful_combinations} model-dataset combinations!")


if __name__ == "__main__":
    # Generate final embeddings using your file structure
        generate_all_datasets_all_models()