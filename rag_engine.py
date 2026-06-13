"""
rag_engine.py
RAG engine using FAISS for vector storage.

Embedding strategy (in priority order):
  1. HuggingFace Inference API  — if HF_TOKEN env var is set (cloud, no download)
  2. TF-IDF + FAISS             — pure local fallback, zero dependencies beyond sklearn

Both produce the same retrieve_context() interface consumed by agents.
"""

from __future__ import annotations
import os
import json
import numpy as np
from typing import Optional

import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

from knowledge_base import DESIGN_KNOWLEDGE


# ── HuggingFace Inference API embeddings (cloud, free tier) ──────────────────

HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"


def _hf_embed(texts: list[str], hf_token: str) -> np.ndarray:
    """Call HuggingFace Inference API for sentence embeddings."""
    headers = {"Authorization": f"Bearer {hf_token}"}
    response = requests.post(
        HF_API_URL,
        headers=headers,
        json={"inputs": texts, "options": {"wait_for_model": True}},
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    return np.array(result, dtype="float32")


# ── TF-IDF local embeddings (pure fallback) ───────────────────────────────────

class TFIDFVectorStore:
    """
    Lightweight vector store using TF-IDF + cosine similarity.
    No model download required. Works fully offline.
    Used when HF Inference API is unavailable.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=512,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.matrix: Optional[np.ndarray] = None
        self.docs: list[dict] = []

    def add_documents(self, docs: list[dict]):
        """docs: list of {content, metadata}"""
        self.docs = docs
        texts = [d["content"] for d in docs]
        self.matrix = self.vectorizer.fit_transform(texts).toarray().astype("float32")
        # L2 normalize for cosine similarity via dot product
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        self.matrix = self.matrix / (norms + 1e-8)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_category: Optional[str] = None,
    ) -> list[dict]:
        """Return top-k most similar docs."""
        if self.matrix is None:
            return []

        q_vec = self.vectorizer.transform([query]).toarray().astype("float32")
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scores = cosine_similarity(q_vec, self.matrix)[0]

        # Apply category filter
        if filter_category:
            for i, doc in enumerate(self.docs):
                if doc.get("metadata", {}).get("category") != filter_category:
                    scores[i] = -1.0

        top_indices = np.argsort(scores)[::-1]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append({
                "content": self.docs[idx]["content"],
                "metadata": self.docs[idx].get("metadata", {}),
                "score": float(scores[idx]),
            })
            if len(results) >= k:
                break
        return results


# ── FAISS vector store with HF embeddings ────────────────────────────────────

class FAISSVectorStore:
    """
    FAISS-backed vector store using HuggingFace Inference API embeddings.
    Used when HF_TOKEN is available.
    """

    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.index: Optional[faiss.Index] = None
        self.docs: list[dict] = []

    def add_documents(self, docs: list[dict]):
        self.docs = docs
        texts = [d["content"] for d in docs]
        # Batch embed (HF API max 100 at a time)
        embeddings = _hf_embed(texts, self.hf_token)
        # L2 normalize
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product = cosine after normalization
        self.index.add(embeddings)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_category: Optional[str] = None,
    ) -> list[dict]:
        if self.index is None:
            return []

        q_emb = _hf_embed([query], self.hf_token)
        faiss.normalize_L2(q_emb)

        fetch_k = k * 3 if filter_category else k
        scores, indices = self.index.search(q_emb, min(fetch_k, len(self.docs)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self.docs[idx]
            meta = doc.get("metadata", {})
            if filter_category and meta.get("category") != filter_category:
                continue
            results.append({
                "content": doc["content"],
                "metadata": meta,
                "score": float(score),
            })
            if len(results) >= k:
                break
        return results


# ── Public interface ──────────────────────────────────────────────────────────

def build_vectorstore(hf_token: str = "") -> TFIDFVectorStore | FAISSVectorStore:
    """
    Build and populate the vector store from the knowledge base.
    Tries HF Inference API first; falls back to TF-IDF.
    """
    docs = [
        {
            "content": entry["content"],
            "metadata": {
                "category": entry["category"],
                "source": entry["source"],
                "severity": entry["severity"],
            },
        }
        for entry in DESIGN_KNOWLEDGE
    ]

    # Try HuggingFace Inference API
    token = hf_token.strip() if hf_token else os.environ.get("HF_TOKEN", "")
    if token:
        try:
            store = FAISSVectorStore(hf_token=token)
            store.add_documents(docs)
            return store
        except Exception as e:
            print(f"[RAG] HF Inference API failed ({e}), falling back to TF-IDF")

    # Local TF-IDF fallback
    store = TFIDFVectorStore()
    store.add_documents(docs)
    return store


def retrieve_context(
    vectorstore,
    query: str,
    category: Optional[str] = None,
    k: int = 4,
) -> str:
    """
    Retrieve top-k relevant knowledge chunks and format for prompt injection.
    """
    try:
        docs = vectorstore.similarity_search(query, k=k, filter_category=category)
    except Exception:
        return ""

    if not docs:
        return ""

    lines = ["[Relevant design standards from RAG knowledge base]"]
    for doc in docs:
        meta = doc.get("metadata", {})
        lines.append(
            f"\n• [{meta.get('source', 'Design Standards')} | "
            f"{meta.get('severity', 'info')}]\n  {doc['content']}"
        )
    return "\n".join(lines)
