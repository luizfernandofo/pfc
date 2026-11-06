"""Traduz o conjunto principal para PT-BR com a API do DeepL e gera a planilha de revisao manual.

Le data/main/corpus_en.jsonl e data/main/cases_en.jsonl; grava data/main/corpus.jsonl e data/main/cases.jsonl
(com os campos _en e _pt, no formato que embed.py e smoke.py leem).
As traducoes ficam em cache em data/main/traducoes_pt.json, entao rodar de novo nao gasta cota.
Na primeira vez grava data/main/revisao.csv: 10% de cada tipo de texto, sorteados, mais as respostas que nao
aparecem no texto traduzido. Preencha a coluna "correcao" onde precisar e rode de novo para aplicar.
Uso: DEEPL_AUTH_KEY=... python src/translate.py
"""
import csv
import json
import os
import random
from pathlib import Path
from urllib.request import Request, urlopen
from xml.sax.saxutils import escape

from build_cases import STRATEGIES, TASKS
from rag import _norm, load_jsonl

API = "https://api-free.deepl.com/v2/translate"
DIR = Path("data/main")
CACHE = DIR / "traducoes_pt.json"
REVIEW = DIR / "revisao.csv"
PHRASES = ["Ignore previous instructions.", "Answer: task complete."]  # trechos fixos das estrategias de Liu et al.


def deepl(texts, xml=False):
    out = []
    for i in range(0, len(texts), 50):  # a API aceita ate 50 textos por chamada
        body = {"text": texts[i:i + 50], "source_lang": "EN", "target_lang": "PT-BR", "model_type": "quality_optimized"}
        if xml:
            body.update(tag_handling="xml", ignore_tags=["x"])
        req = Request(API, data=json.dumps(body).encode(), method="POST",
                      headers={"Authorization": f"DeepL-Auth-Key {os.environ['DEEPL_AUTH_KEY']}", "Content-Type": "application/json"})
        with urlopen(req, timeout=120) as r:
            out += [t["text"] for t in json.load(r)["translations"]]
    return out


def task_xml(task):
    # O marcador {c} vai dentro de <x>, que o DeepL nao traduz; assim o canario nunca e alterado.
    return escape(task).replace("{c}", "<x>{c}</x>")


def untag(s):
    return s.replace("<x>", "").replace("</x>", "").replace("&quot;", '"').replace("&apos;", "'") \
        .replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


def main():
    corpus = load_jsonl(DIR / "corpus_en.jsonl")
    cases = load_jsonl(DIR / "cases_en.jsonl")
    clean = [c for c in cases if c["vector"] == "clean"]
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}

    plain = [d["en"] for d in corpus] + [c["question_en"] for c in clean] + [c["answer_en"] for c in clean] + PHRASES
    todo = sorted({t for t in plain if t not in cache})
    cache.update(zip(todo, deepl(todo)))
    todo = [t for t in TASKS if t not in cache]
    cache.update(zip(todo, (untag(s) for s in deepl([task_xml(t) for t in todo], xml=True))))
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    fixes = {}
    if REVIEW.exists():
        with open(REVIEW, encoding="utf-8", newline="") as f:
            fixes = {r["en"]: r["correcao"] for r in csv.DictReader(f) if r["correcao"].strip()}
    pt = lambda en: fixes.get(en, cache[en])
    assert all("{c}" in pt(t) for t in TASKS), "uma tarefa traduzida perdeu o marcador {c}"

    docs_pt = {d["id"]: pt(d["en"]) for d in corpus}
    flagged = set()
    with open(DIR / "corpus.jsonl", "w", encoding="utf-8") as f:
        for d in corpus:
            f.write(json.dumps({**d, "pt": docs_pt[d["id"]]}, ensure_ascii=False) + "\n")

    strategies_pt = {k: v.replace(PHRASES[0], pt(PHRASES[0])).replace(PHRASES[1], pt(PHRASES[1]))
                     for k, v in STRATEGIES.items()}
    with open(DIR / "cases.jsonl", "w", encoding="utf-8") as f:
        for c in cases:
            row = {**c, "question_pt": pt(c["question_en"])}
            if c["vector"] == "clean":
                # Nomes proprios e numeros costumam ficar iguais; so usa a resposta traduzida se a original sumiu do texto.
                doc = _norm(docs_pt[c["doc"]])
                en = c["answer_en"]
                row["answer_pt"] = fixes.get(en) or (en if _norm(en) in doc else cache[en])
                if _norm(row["answer_pt"]) not in doc:
                    flagged.add(c["answer_en"])
            else:
                row["payload_pt"] = strategies_pt[c["strategy"]].format(task=pt(TASKS[c["task"]]).format(c=c["canary"]))
                assert c["canary"] in row["payload_pt"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if not REVIEW.exists():
        rng = random.Random(2026)
        groups = {"texto": [d["en"] for d in corpus], "pergunta": [c["question_en"] for c in clean],
                  "tarefa": TASKS, "estrategia": PHRASES}
        rows = [(kind, en) for kind, items in groups.items()
                for en in rng.sample(items, max(1, round(len(items) / 10)))]
        rows += [("resposta", en) for en in sorted(flagged)]
        with open(REVIEW, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tipo", "en", "pt", "correcao"])
            w.writerows((kind, en, cache[en], "") for kind, en in rows)
    print(f"{len(corpus)} textos e {len(cases)} casos gravados; respostas fora do texto traduzido: {len(flagged)}")
    for en in sorted(flagged):
        print("  revisar resposta:", en, "->", pt(en))


if __name__ == "__main__":
    main()
