"""Calcula os embeddings do corpus e das perguntas (EN e PT) e grava o indice FAISS.

Roda uma vez, antes do LLM, para nao dividir a GPU com ele.
Uso: python src/embed.py data/smoke
"""
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag import load_jsonl

data = Path(sys.argv[1])
out = data / "index"
out.mkdir(exist_ok=True)
docs = load_jsonl(data / "corpus.jsonl")
cases = load_jsonl(data / "cases.jsonl")

model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", model_kwargs={"torch_dtype": "bfloat16"})
for lang in ("en", "pt"):
    d = model.encode([x[lang] for x in docs], normalize_embeddings=True)
    # A busca usa a pergunta limpa, sem o payload, para que a recuperacao seja igual em todas as condicoes.
    q = model.encode([c[f"question_{lang}"] for c in cases], prompt_name="query", normalize_embeddings=True)
    index = faiss.IndexFlatIP(d.shape[1])  # produto interno de vetores normalizados = cosseno
    index.add(d.astype(np.float32))
    faiss.write_index(index, str(out / f"docs_{lang}.faiss"))
    np.save(out / f"queries_{lang}.npy", q.astype(np.float32))
    print(lang, "docs", d.shape, "queries", q.shape)
