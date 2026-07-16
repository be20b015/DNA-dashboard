"""
Genomic Foundation Model (GFM) integration.

Loads a Hugging Face model such as DNABERT-2 (zhihan1996/DNABERT-2-117M)
or HyenaDNA, cached as a global singleton via st.cache_resource so the
(large) weights are only pulled into memory once per server process —
not once per user session or per Streamlit rerun.

NOTE: downloading model weights requires outbound internet access to
huggingface.co. If you're running behind a restricted network, set
HF_HUB_OFFLINE=1 and pre-download the weights, or point
TRANSFORMERS_CACHE at a volume where they're already cached.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

try:
    import streamlit as st
    _cache_resource = st.cache_resource
except ImportError:  # allows this module to be imported/tested outside Streamlit
    def _cache_resource(func):
        return lru_cache(maxsize=1)(func)


DEFAULT_MODEL_NAME = "zhihan1996/DNABERT-2-117M"


@_cache_resource
def load_gfm(model_name: str = DEFAULT_MODEL_NAME):
    """
    Loads tokenizer + model once and caches them as a singleton for the
    life of the server process. Subsequent calls (across all user
    sessions) return the cached object instead of re-loading weights.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()

    return {"tokenizer": tokenizer, "model": model, "device": device}


def embed_sequences(sequences: List[str], model_bundle: dict, batch_size: int = 8) -> np.ndarray:
    """
    Extract fixed-length embeddings for a list of DNA sequences using
    mean token pooling over the model's last hidden state. Returns an
    (n_sequences, hidden_dim) numpy array suitable for downstream
    classification / clustering.
    """
    import torch

    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]
    device = model_bundle["device"]

    all_embeddings = []

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            outputs = model(**inputs)
            hidden_states = outputs[0]  # (batch, seq_len, hidden_dim)

            # Mean pooling, respecting the attention mask so padding
            # tokens don't dilute the embedding.
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = (summed / counts).cpu().numpy()

            all_embeddings.append(pooled)

    return np.vstack(all_embeddings) if all_embeddings else np.empty((0, 0))


def gpu_tokenize_hint() -> str:
    """
    For single-nucleotide-resolution workloads (e.g. per-base variant
    scoring across long sequences), plain CPU tokenization becomes the
    bottleneck. This returns guidance text shown in the UI rather than
    silently switching tokenizers, since it requires an extra dependency
    (dnatok) the user must opt into.
    """
    return (
        "For single-nucleotide-resolution analysis, CPU tokenization is "
        "typically the bottleneck. Consider installing a GPU-accelerated "
        "tokenizer (e.g. `dnatok`) and routing `tokenizer.__call__` "
        "through it to keep the GPU fed during large batch runs."
    )
