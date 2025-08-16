"""
embedding_models.py
~~~~~~~~~~~~~~~~~~~
Unified embedding system supporting OpenAI, Cohere, and BERT/SentenceTransformers.

Main public callable
--------------------
get_question_embeddings(df, id_col="question_id", text_col="question_text", ...)

Providers
---------
1. OpenAI: text-embedding-3-small/large (requires API key)
2. Cohere: embed-english-v3.0 (requires API key)  
3. BERT: SentenceTransformers models (local, no API key)
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Hashable, Iterable

import pandas as pd
import torch
from tqdm.auto import tqdm

__all__ = [
    'get_question_embeddings',
    'setup_embedding_system', 
    'get_embedding_dimension',
    'get_current_provider_info',
    '_get_embedding',
    '_current_provider',
    '_current_model',
    'DEFAULT_MODEL',
    '_provider_clients',
    'initialize_provider',
    'load_api_config',
    'PROVIDER_MODELS',
    '_setup_openai_client',  
    '_setup_cohere_client',  
    '_setup_bert_client'     
]

_logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------#
# Configuration and Provider Selection
# -----------------------------------------------------------------------------#

# Default models for each provider
PROVIDER_MODELS = {
    "openai": {
        "default": "text-embedding-3-small",
        "models": [
            "text-embedding-3-small",  # 1536 dims
            "text-embedding-3-large",  # 3072 dims  
            "text-embedding-ada-002"   # 1536 dims
        ]
    },
    "cohere": {
        "default": "embed-english-v3.0",
        "models": [
            "embed-english-v3.0",           # 1024 dims
            "embed-english-light-v3.0",     # 384 dims
            "embed-multilingual-v3.0",      # 1024 dims
            "embed-multilingual-light-v3.0" # 384 dims
        ]
    },
    "bert": {
        "default": "all-MiniLM-L6-v2", 
        "models": [
            "all-MiniLM-L6-v2",           # 384 dims, fast
            "all-mpnet-base-v2",          # 768 dims, best quality
            "multi-qa-mpnet-base-dot-v1", # 768 dims, Q&A optimized
            "all-MiniLM-L12-v2",          # 384 dims, better than L6
            "paraphrase-MiniLM-L6-v2"     # 384 dims, paraphrase
        ]
    }
}

# Global provider state
_current_provider = None
_current_model = None
_provider_clients = {}

# -----------------------------------------------------------------------------#
# Configuration Management
# -----------------------------------------------------------------------------#

def load_api_config(config_path: str = "api_config.json") -> dict:
    """Load API keys from JSON config file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        # Create template config file
        template_config = {
            "openai_api_key": "your-openai-api-key-here",
            "cohere_api_key": "your-cohere-api-key-here",
            "note": "Replace the API keys above with your actual keys"
        }
        
        with open(config_file, 'w') as f:
            json.dump(template_config, f, indent=2)
        
        print(f"📄 Created template config file: {config_file}")
        print(f"⚠️  Please edit {config_file} and add your API keys!")
        return template_config
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"✅ Loaded API config from: {config_file}")
        return config
    except Exception as e:
        print(f"❌ Error loading config from {config_file}: {e}")
        return {}

def select_embedding_provider() -> tuple[str, str]:
    """Interactive provider and model selection."""
    
    print("\n" + "="*60)
    print("🚀 EMBEDDING PROVIDER SELECTION")
    print("="*60)
    
    print("\nAvailable embedding providers:")
    print("1. OpenAI (text-embedding-3-small/large)")
    print("   • High quality, 1536-3072 dimensions")
    print("   • Requires API key, costs money")
    print("   • Best semantic understanding")
    
    print("\n2. Cohere (embed-english-v3.0)")  
    print("   • Good quality, 384-1024 dimensions")
    print("   • Requires API key, costs money")
    print("   • Fast and efficient")
    
    print("\n3. BERT/SentenceTransformers (all-MiniLM-L6-v2)")
    print("   • Good quality, 384-768 dimensions") 
    print("   • Free, runs locally")
    print("   • No API key needed")
    
    # Get provider choice
    while True:
        try:
            choice = input("\nSelect provider (1/2/3): ").strip()
            if choice == "1":
                provider = "openai"
                break
            elif choice == "2":
                provider = "cohere"
                break
            elif choice == "3":
                provider = "bert"
                break
            else:
                print("❌ Please enter 1, 2, or 3")
        except KeyboardInterrupt:
            print("\n👋 Cancelled by user")
            exit(0)
    
    # Show available models for selected provider
    provider_info = PROVIDER_MODELS[provider]
    print(f"\n📋 Available {provider.upper()} models:")
    for i, model in enumerate(provider_info["models"], 1):
        marker = " (default)" if model == provider_info["default"] else ""
        print(f"  {i}. {model}{marker}")
    
    # Get model choice  
    print(f"\nPress Enter for default ({provider_info['default']}) or select number:")
    while True:
        try:
            model_choice = input("Model choice: ").strip()
            
            if not model_choice:  # Enter pressed, use default
                model = provider_info["default"]
                break
            
            try:
                model_idx = int(model_choice) - 1
                if 0 <= model_idx < len(provider_info["models"]):
                    model = provider_info["models"][model_idx]
                    break
                else:
                    print(f"❌ Please enter number 1-{len(provider_info['models'])}")
            except ValueError:
                print("❌ Please enter a number or press Enter for default")
                
        except KeyboardInterrupt:
            print("\n👋 Cancelled by user")
            exit(0)
    
    print(f"\n✅ Selected: {provider.upper()} - {model}")
    return provider, model

# -----------------------------------------------------------------------------#
# Provider Implementations
# -----------------------------------------------------------------------------#

def _setup_openai_client(api_key: str):
    """Setup OpenAI client."""
    try:
        import openai
        openai.api_key = api_key
        _provider_clients["openai"] = openai
        print("✅ OpenAI client initialized")
        return True
    except ImportError:
        print("❌ OpenAI library not installed. Install with: pip install openai")
        return False
    except Exception as e:
        print(f"❌ OpenAI setup failed: {e}")
        return False

def _setup_cohere_client(api_key: str):
    """Setup Cohere client."""
    try:
        import cohere
        client = cohere.Client(api_key)
        _provider_clients["cohere"] = client
        print("✅ Cohere client initialized")
        return True
    except ImportError:
        print("❌ Cohere library not installed. Install with: pip install cohere")
        return False
    except Exception as e:
        print(f"❌ Cohere setup failed: {e}")
        return False

def _setup_bert_client():
    """Setup SentenceTransformers client."""
    try:
        from sentence_transformers import SentenceTransformer
        _provider_clients["bert"] = SentenceTransformer
        print("✅ SentenceTransformers available")
        return True
    except ImportError:
        print("❌ SentenceTransformers not installed. Install with: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"❌ SentenceTransformers setup failed: {e}")
        return False

def initialize_provider(provider: str, model: str, config: dict) -> bool:
    """Initialize the selected provider."""
    
    if provider == "openai":
        api_key = config.get("openai_api_key")
        if not api_key or api_key == "your-openai-api-key-here":
            print("❌ Please set your OpenAI API key in api_config.json")
            return False
        return _setup_openai_client(api_key)
        
    elif provider == "cohere":
        api_key = config.get("cohere_api_key") 
        if not api_key or api_key == "your-cohere-api-key-here":
            print("❌ Please set your Cohere API key in api_config.json")
            return False
        return _setup_cohere_client(api_key)
        
    elif provider == "bert":
        return _setup_bert_client()
    
    return False

# -----------------------------------------------------------------------------#
# Embedding Functions
# -----------------------------------------------------------------------------#

@lru_cache(maxsize=8_192)
def _get_embedding(text: str, *, model: str, provider: str) -> torch.Tensor:
    """Get embedding for single text using the specified provider."""
    
    if provider == "openai":
        client = _provider_clients["openai"]
        resp = client.embeddings.create(input=[text], model=model)
        vec = resp.data[0].embedding
        return torch.tensor(vec, dtype=torch.float32)
        
    elif provider == "cohere":
        client = _provider_clients["cohere"]
        resp = client.embed(texts=[text], model=model, input_type="search_document")
        vec = resp.embeddings[0]
        return torch.tensor(vec, dtype=torch.float32)
        
    elif provider == "bert":
        SentenceTransformer = _provider_clients["bert"]
        if model not in _provider_clients:
            _provider_clients[model] = SentenceTransformer(model)
        st_model = _provider_clients[model]
        embedding = st_model.encode(text, convert_to_tensor=True)
        return embedding.cpu()
    
    else:
        raise ValueError(f"Unknown provider: {provider}")

# -----------------------------------------------------------------------------#
# Public API
# -----------------------------------------------------------------------------#

DEFAULT_MODEL = None  # Will be set after provider selection

def setup_embedding_system():
    """Setup the embedding system with provider selection."""
    global _current_provider, _current_model, DEFAULT_MODEL
    
    # Load config
    config = load_api_config()
    
    # Interactive provider selection
    provider, model = select_embedding_provider()
    
    # Initialize provider
    success = initialize_provider(provider, model, config)
    if not success:
        raise RuntimeError(f"Failed to initialize {provider} provider")
    
    # Set globals
    _current_provider = provider
    _current_model = model  
    DEFAULT_MODEL = model
    
    print(f"\n🎯 Embedding system ready!")
    print(f"   Provider: {provider.upper()}")
    print(f"   Model: {model}")
    
    return provider, model

def get_question_embeddings(
    df: pd.DataFrame,
    *,
    id_col: str = "question_id",
    text_col: str = "question_text", 
    model: str | None = None,
    show_progress: bool = True,
) -> Dict[Hashable, torch.Tensor]:
    """
    Generate embeddings using the configured provider.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least the columns given by *id_col* and *text_col*.
    id_col, text_col : str
        Column names for the question identifier and the raw text.
    model : str, optional
        Model name. If None, uses the globally configured model.
    show_progress : bool
        If True, wrap iteration in `tqdm` progress bar.

    Returns
    -------
    Dict[Hashable, torch.Tensor]
        Mapping from question ID to embedding tensor
    """
    # Auto-setup if not already done
    if _current_provider is None:
        setup_embedding_system()
    
    # Use global model if not specified
    if model is None:
        model = _current_model
    
    if id_col not in df or text_col not in df:
        raise KeyError(f"DataFrame must contain '{id_col}' and '{text_col}' columns.")

    # Batch processing for BERT models
    if _current_provider == "bert":
        return _get_bert_embeddings_batch(df, id_col, text_col, model, show_progress)
    
    # Sequential processing for API-based providers
    iterator: Iterable = tqdm(
        df.itertuples(index=False), 
        desc=f"Embedding with {_current_provider.upper()}", 
        disable=not show_progress,
        total=len(df)
    )

    embed_map: Dict[Hashable, list[torch.Tensor]] = {}

    for row in iterator:
        qid = getattr(row, id_col)
        qtext = getattr(row, text_col)
        embed_map.setdefault(qid, []).append(
            _get_embedding(qtext, model=model, provider=_current_provider)
        )

    # Aggregate duplicates
    averaged: Dict[Hashable, torch.Tensor] = {
        qid: torch.stack(vectors).mean(dim=0) for qid, vectors in embed_map.items()
    }

    _logger.info("Generated %d unique question embeddings using %s.", len(averaged), _current_provider.upper())
    return averaged

def _get_bert_embeddings_batch(df, id_col, text_col, model, show_progress):
    """Optimized batch processing for BERT models."""
    SentenceTransformer = _provider_clients["bert"]
    if model not in _provider_clients:
        _provider_clients[model] = SentenceTransformer(model)
    st_model = _provider_clients[model]
    
    # Extract texts and IDs
    texts = df[text_col].tolist()
    ids = df[id_col].tolist()
    
    # Batch encode
    embeddings = st_model.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=show_progress,
        batch_size=32
    )
    
    # Organize by ID
    embed_map: Dict[Hashable, list[torch.Tensor]] = {}
    for qid, embedding in zip(ids, embeddings):
        embed_map.setdefault(qid, []).append(embedding.cpu())

    # Aggregate duplicates
    averaged: Dict[Hashable, torch.Tensor] = {
        qid: torch.stack(vectors).mean(dim=0) for qid, vectors in embed_map.items()
    }

    return averaged

# -----------------------------------------------------------------------------#
# Utility Functions
# -----------------------------------------------------------------------------#

def get_current_provider_info():
    """Get information about current provider setup."""
    if _current_provider is None:
        return None
    
    return {
        "provider": _current_provider,
        "model": _current_model,
        "available_models": PROVIDER_MODELS[_current_provider]["models"]
    }

def get_embedding_dimension(model: str | None = None) -> int:
    """Get embedding dimension for the current/specified model."""
    if model is None:
        model = _current_model
    
    # Dimension mapping
    dimensions = {
        # OpenAI
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072, 
        "text-embedding-ada-002": 1536,
        
        # Cohere  
        "embed-english-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-v3.0": 1024,
        "embed-multilingual-light-v3.0": 384,
        
        # BERT/SentenceTransformers
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "multi-qa-mpnet-base-dot-v1": 768,
        "all-MiniLM-L12-v2": 384,
        "paraphrase-MiniLM-L6-v2": 384
    }
    
    if model in dimensions:
        return dimensions[model]
    
    # For unknown BERT models, load and check
    if _current_provider == "bert":
        try:
            SentenceTransformer = _provider_clients["bert"]
            st_model = SentenceTransformer(model)
            test_embedding = st_model.encode("test", convert_to_tensor=True)
            return test_embedding.shape[0]
        except:
            pass
    
    return 384  # Conservative default

# -----------------------------------------------------------------------------#
# Auto-setup for backwards compatibility
# -----------------------------------------------------------------------------#

if __name__ == "__main__":
    # Demo/test the system
    setup_embedding_system()
    
    # Test embedding
    test_text = "What is 2 + 3?"
    embedding = _get_embedding(test_text, model=_current_model, provider=_current_provider)
    print(f"\n🧪 Test embedding:")
    print(f"   Text: '{test_text}'")
    print(f"   Shape: {embedding.shape}")
    print(f"   First 5 values: {embedding[:5].tolist()}")