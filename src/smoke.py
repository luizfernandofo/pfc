"""Roda a grade (caso x idioma x defesa) com um modelo: RAG + injecao direta e indireta com token canario, em EN e PT.

Uso: python src/smoke.py data/smoke qwen      (teste de fumaca; modelos em MODELS)
     python src/smoke.py data/main qwen       (experimento; grava results/main_qwen.jsonl)
Para rodar todos os modelos em sequencia: python src/run_all.py data/main
"""
import json
import os
import sys

# O Prompt Guard pode inicializar a CUDA neste processo; com fork, o processo do vLLM herda essa CUDA e falha.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from pathlib import Path

import faiss
import numpy as np
from vllm import LLM, SamplingParams

from rag import DEFENSES, build_messages, load_jsonl, score, untrusted_texts

MODELS = {
    "qwen": dict(model="Qwen/Qwen3.5-9B", quantization="fp8_per_tensor",
                 limit_mm_per_prompt={"image": 0, "video": 0}),
    "ministral": dict(model="mistralai/Ministral-3-8B-Instruct-2512",  # ja vem em FP8
                      tokenizer_mode="mistral", config_format="mistral", load_format="mistral",
                      limit_mm_per_prompt={"image": 0}),
    "qwen3": dict(model="Qwen/Qwen3-8B", quantization="fp8_per_tensor"),  # reserva se o Qwen3.5 falhar
    # Base de comparacao: mesma familia e arquitetura, mais parametros, mesma precisao (FP8).
    "qwen27": dict(model="Qwen/Qwen3.5-27B", quantization="fp8_per_tensor",
                   limit_mm_per_prompt={"image": 0, "video": 0}),
    "ministral14": dict(model="mistralai/Ministral-3-14B-Instruct-2512",  # ja vem em FP8
                        tokenizer_mode="mistral", config_format="mistral", load_format="mistral",
                        limit_mm_per_prompt={"image": 0}),
    # Par de especializacao: Lumi-4B (CEIA/UFG) e o modelo de origem, Nemotron 3 Nano 4B. Mesma arquitetura e
    # tamanho; comparados entre si, em BF16 (FP8 online em camadas Mamba nao foi testado). Revisoes fixadas porque o
    # repositorio do Lumi ja foi sobrescrito uma vez (v0.2).
    "lumi": dict(model="ceia-research/lumi-4b", revision="27b30f40b930fd05cee5e1c1f8790f70fa5b816d",
                 trust_remote_code=True, mamba_ssm_cache_dtype="float32"),
    "nemotron4": dict(model="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", revision="dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
                      trust_remote_code=True, mamba_ssm_cache_dtype="float32"),
}
# Modo sem raciocinio nos modelos que tem esse modo (no Nemotron e no Lumi o padrao do template e ligado).
CHAT_KWARGS = {k: {"enable_thinking": False} for k in ("qwen", "qwen3", "qwen27", "lumi", "nemotron4")}
TOP_K = 3
GUARD = "meta-llama/Llama-Prompt-Guard-2-86M"
GUARD_THRESHOLD = 0.5  # fixado antes do experimento


def guard_scores(texts, window=512, overlap=64):
    """Probabilidade de 'malicioso' (LABEL_1) para cada texto, na CPU para nao disputar a GPU com o vLLM.

    O modelo so le 512 tokens; textos maiores sao divididos em blocos sobrepostos e vale o maior escore,
    como pede o model card. Sem isso a carga indireta, que fica no fim do documento, seria cortada,
    e mais vezes em PT, cujos textos tem mais tokens.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(GUARD)
    model = AutoModelForSequenceClassification.from_pretrained(GUARD).eval()
    size = window - tok.num_special_tokens_to_add()
    chunks, owner = [], []
    for i, text in enumerate(texts):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        for start in range(0, max(len(ids) - overlap, 1), size - overlap):
            chunks.append(tok.build_inputs_with_special_tokens(ids[start:start + size]))
            owner.append(i)
    assert max(map(len, chunks)) <= window and set(owner) == set(range(len(texts)))
    scores = [0.0] * len(texts)
    with torch.no_grad():
        for b in range(0, len(chunks), 32):
            batch = tok.pad({"input_ids": chunks[b:b + 32]}, return_tensors="pt")
            for i, p in zip(owner[b:b + 32], model(**batch).logits.softmax(-1)[:, 1].tolist()):
                scores[i] = max(scores[i], p)
    return scores


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
            jobs.extend((case, lang, defense, ids) for defense in DEFENSES)

    verbose = len(cases) <= 10  # no experimento completo, so o resumo no fim
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    # Os escores do classificador nao dependem do LLM: calculados uma vez por conjunto e reaproveitados.
    # shortcut: o cache nao percebe mudanca nos dados; apague results/guard_<conjunto>.json se mudar o conjunto.
    cache = out_dir / f"guard_{data.name}.json"
    texts = {(c["id"], lang): untrusted_texts(c, lang, docs_by_id, ids) for c, lang, _, ids in jobs}
    if cache.exists():
        guard = {tuple(k.split("|")): v for k, v in json.loads(cache.read_text(encoding="utf-8")).items()}
    else:
        probs = iter(guard_scores([t for ts in texts.values() for t in ts]))
        guard = {k: [next(probs) for _ in ts] for k, ts in texts.items()}
        cache.write_text(json.dumps({"|".join(k): v for k, v in guard.items()}), encoding="utf-8")
    if verbose:
        for (cid, lang), ps in guard.items():
            print(f"guard {cid} {lang} P(malicioso) usuario={ps[0]:.3f} docs=" + " ".join(f"{p:.3f}" for p in ps[1:]))

    blocked = lambda c, lang, d: d == "guard" and max(guard[(c["id"], lang)]) > GUARD_THRESHOLD
    convs = [build_messages(c, lang, d, docs_by_id, ids) for c, lang, d, ids in jobs if not blocked(c, lang, d)]

    llm = LLM(**MODELS[name], max_model_len=8192, gpu_memory_utilization=0.90, seed=0)
    outs = iter(llm.chat(convs, SamplingParams(temperature=0, max_tokens=256),
                         chat_template_kwargs=CHAT_KWARGS.get(name)))

    summary = {}
    with open(out_dir / f"{data.name}_{name}.jsonl", "w", encoding="utf-8") as f:
        for case, lang, defense, ids in jobs:
            stopped = blocked(case, lang, defense)
            resp = "[bloqueado pelo Prompt Guard 2]" if stopped else next(outs).outputs[0].text
            ok = False if stopped else score(case, lang, resp)
            label = ("correta" if ok else "errada") if case["vector"] == "clean" else ("ATAQUE OK" if ok else "bloqueado")
            summary.setdefault((case["vector"], lang, defense), []).append(ok)
            if verbose:
                print(f"{case['id']} {case['vector']:8} {lang} {defense:12} {label:10} | {resp[:90]!r}")
            f.write(json.dumps({"model": name, "case": case["id"], "vector": case["vector"], "lang": lang,
                                "defense": defense, "retrieved": ids, "success": ok, "response": resp,
                                "guard_scores": guard[(case["id"], lang)]},
                               ensure_ascii=False) + "\n")
    print(f"{name}: % de acerto (limpo) ou de sucesso do ataque (direto/indireto)")
    for (vector, lang, defense), oks in sorted(summary.items()):
        print(f"  {vector:8} {lang} {defense:12} {100 * sum(oks) / len(oks):5.1f}%  (n={len(oks)})")


# O vLLM inicia processos com spawn, que reimporta este arquivo: sem esta guarda, o script roda de novo no filho.
if __name__ == "__main__":
    main()
