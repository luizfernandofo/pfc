"""Teste de fumaca: RAG + injecao direta e indireta com token canario, em EN e PT.

Uso: python src/smoke.py data/smoke qwen      (ou ministral, ou qwen3 como reserva)
"""
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from vllm import LLM, SamplingParams

from rag import DEFENSES, build_messages, load_jsonl, score

MODELS = {
    "qwen": dict(model="Qwen/Qwen3.5-9B", quantization="fp8_per_tensor",
                 limit_mm_per_prompt={"image": 0, "video": 0}),
    "ministral": dict(model="mistralai/Ministral-3-8B-Instruct-2512",  # ja vem em FP8
                      tokenizer_mode="mistral", config_format="mistral", load_format="mistral",
                      limit_mm_per_prompt={"image": 0}),
    "qwen3": dict(model="Qwen/Qwen3-8B", quantization="fp8_per_tensor"),  # reserva se o Qwen3.5 falhar
}
# Modo sem raciocinio nos modelos que tem esse modo.
CHAT_KWARGS = {"qwen": {"enable_thinking": False}, "qwen3": {"enable_thinking": False}}
TOP_K = 3


def main():
    data, name = Path(sys.argv[1]), sys.argv[2]
    docs = load_jsonl(data / "corpus.jsonl")
    cases = load_jsonl(data / "cases.jsonl")
    docs_by_id = {d["id"]: d for d in docs}

    jobs, convs = [], []
    for lang in ("en", "pt"):
        index = faiss.read_index(str(data / "index" / f"docs_{lang}.faiss"))
        _, top = index.search(np.load(data / "index" / f"queries_{lang}.npy"), TOP_K)
        for case, row in zip(cases, top):
            ids = [docs[i]["id"] for i in row]
            if case["vector"] == "indirect" and case["doc"] not in ids:
                ids[-1] = case["doc"]  # o documento envenenado sempre entra entre os recuperados
            for defense in DEFENSES:
                jobs.append((case, lang, defense, ids))
                convs.append(build_messages(case, lang, defense, docs_by_id, ids))

    llm = LLM(**MODELS[name], max_model_len=8192, gpu_memory_utilization=0.90, seed=0)
    outs = llm.chat(convs, SamplingParams(temperature=0, max_tokens=256),
                    chat_template_kwargs=CHAT_KWARGS.get(name))

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"smoke_{name}.jsonl", "w", encoding="utf-8") as f:
        for (case, lang, defense, ids), out in zip(jobs, outs):
            resp = out.outputs[0].text
            ok = score(case, lang, resp)
            label = "correta" if case["vector"] == "clean" else ("ATAQUE OK" if ok else "bloqueado")
            print(f"{case['id']} {case['vector']:8} {lang} {defense:12} {label:10} | {resp[:90]!r}")
            f.write(json.dumps({"model": name, "case": case["id"], "vector": case["vector"], "lang": lang,
                                "defense": defense, "retrieved": ids, "success": ok, "response": resp},
                               ensure_ascii=False) + "\n")


# O vLLM inicia processos com spawn, que reimporta este arquivo: sem esta guarda, o script roda de novo no filho.
if __name__ == "__main__":
    main()
