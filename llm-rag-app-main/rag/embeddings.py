"""
Gestão de embeddings — FastEmbed local (ONNX, roda na CPU)
----------------------------------------------------------
O Groq só serve modelos de chat, então os embeddings são gerados localmente com
FastEmbed (sem torch, sem servidor). O modelo é baixado do Hugging Face na
primeira execução e fica em cache.

Modelo configurável via EMBEDDING_MODEL no .env
(padrão: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, 384 dims, multilíngue).
"""

from __future__ import annotations

import os

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ── Singleton ─────────────────────────────────────────────────────────────────
_embeddings = None


def get_embeddings():
    """Retorna FastEmbedEmbeddings como singleton (modelo configurado em EMBEDDING_MODEL)."""
    global _embeddings
    if _embeddings is None:
        from langchain_community.embeddings import FastEmbedEmbeddings
        model_name  = os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
        _embeddings = FastEmbedEmbeddings(
            model_name=model_name,
            cache_dir=os.getenv("EMBEDDING_CACHE_DIR") or None,
        )
        print(f"[embeddings] FastEmbed local: {model_name}")
    return _embeddings
