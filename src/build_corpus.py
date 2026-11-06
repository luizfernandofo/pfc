"""Baixa o trecho inicial de 50 artigos da Wikipedia em ingles e grava data/main/corpus_en.jsonl.

Registra o revid de cada pagina para que o corpus seja reproduzivel.
Uso: python src/build_corpus.py
"""
import json
import re
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

TITLES = [
    "Photosynthesis", "Amazon River", "Python (programming language)", "Brasília", "Eiffel Tower",
    "Great Wall of China", "Marie Curie", "Albert Einstein", "Mount Everest", "Pacific Ocean",
    "Moon", "Mars", "DNA", "Penicillin", "World Wide Web",
    "Linux", "Bitcoin", "Olympic Games", "FIFA World Cup", "Leonardo da Vinci",
    "Mona Lisa", "William Shakespeare", "Machu Picchu", "Sahara", "Antarctica",
    "Coffee", "Chocolate", "Volcano", "Earthquake", "Black hole",
    "Vaccine", "Internet", "Printing press", "Telephone", "Electricity",
    "Solar System", "Jupiter", "Western honey bee", "Coral reef", "Rainforest",
    "Pantanal", "Iguazu Falls", "Christ the Redeemer (statue)", "Alan Turing", "Ada Lovelace",
    "Nelson Mandela", "United Nations", "Euro", "Tsunami", "Climate change",
]
MAX_WORDS = 300
API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "pfc-tcc-research/1.0 (academic project; corpus build)"}


def fetch(titles):
    q = {"action": "query", "prop": "extracts|revisions", "exintro": 1, "explaintext": 1, "rvprop": "ids",
         "redirects": 1, "titles": "|".join(titles), "format": "json", "formatversion": 2}
    with urlopen(Request(f"{API}?{urlencode(q)}", headers=HEADERS), timeout=60) as r:
        return json.load(r)["query"]["pages"]


def trim(text):
    """Corta em ate MAX_WORDS palavras, terminando na ultima frase completa."""
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if len(words) <= MAX_WORDS:
        return text
    cut = " ".join(words[:MAX_WORDS])
    return cut[: cut.rfind(". ") + 1] or cut


def main():
    pages = {}
    for i in range(0, len(TITLES), 20):  # a API aceita ate 20 extratos por chamada
        for p in fetch(TITLES[i:i + 20]):
            pages[p["title"]] = p
    out = Path("data/main")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "corpus_en.jsonl", "w", encoding="utf-8") as f:
        for n, p in enumerate(sorted(pages.values(), key=lambda p: TITLES.index(p["title"]) if p["title"] in TITLES else 99)):
            text = trim(p["extract"])
            f.write(json.dumps({"id": f"d{n + 1:02d}", "title": p["title"], "revid": p["revisions"][0]["revid"],
                                "en": text}, ensure_ascii=False) + "\n")
            print(f"d{n + 1:02d} {p['title']}: {len(text.split())} palavras")
    assert len(pages) == len(TITLES), f"faltaram paginas: {len(pages)} de {len(TITLES)}"


if __name__ == "__main__":
    main()
