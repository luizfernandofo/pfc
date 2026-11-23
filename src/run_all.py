"""Roda o experimento com todos os modelos, um processo por modelo, do menor para o maior.

Uso: python src/run_all.py data/main [--copiar /content/drive/MyDrive/pfc-results] [--modelos qwen,ministral]

- Pula o modelo cujo resultado ja esta completo (retomada depois de queda da sessao do Colab).
- Pula o modelo que nao cabe na GPU, avisando; os pequenos rodam primeiro, para o essencial sair antes.
- Se o Qwen3.5-9B falhar, roda o Qwen3-8B (reserva).
- Depois de cada modelo, copia results/ para --copiar e apaga os pesos do cache para liberar disco.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import auth_check, constants

from rag import DEFENSES, load_jsonl
from smoke import GUARD, MODELS

# Memoria minima (GB) para pesos em FP8 + cache de contexto de 8192 tokens com folga.
MIN_GB = {"qwen": 16, "ministral": 16, "qwen3": 16, "lumi": 16, "nemotron4": 16,
          "ministral14": 22, "qwen27": 38}
# Os dois modelos principais primeiro; depois os pares de comparacao, dos menores para os maiores.
ORDEM = ["qwen", "ministral", "lumi", "nemotron4", "ministral14", "qwen27"]


def gpu():
    # nvidia-smi em vez de torch: este processo nao pode ocupar memoria da GPU que os filhos vao usar.
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.splitlines()[0]
    name, mem, cap = [x.strip() for x in out.split(",")]
    return name, int(mem) / 1024, float(cap)


def completo(path, esperado):
    return path.exists() and sum(1 for _ in open(path, encoding="utf-8")) == esperado


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--copiar", type=Path)
    ap.add_argument("--modelos", default=",".join(ORDEM))
    args = ap.parse_args()
    modelos = args.modelos.split(",")

    name, mem, cap = gpu()
    print(f"GPU: {name}, {mem:.0f} GB, compute capability {cap}")
    if cap < 8.0:
        sys.exit("Esta GPU nao roda FP8 no vLLM (precisa de compute capability >= 8.0; a T4 e 7.5). Use L4, A100 ou G4.")

    # Falha cedo, antes de gastar tempo de GPU: repositorio inexistente ou licenca nao aceita.
    for repo in [MODELS[m]["model"] for m in modelos] + [GUARD, "Qwen/Qwen3-Embedding-4B"]:
        auth_check(repo)

    # Sessao nova do Colab: traz de volta o que ja rodou (resultados e cache do classificador).
    if args.copiar and args.copiar.exists():
        shutil.copytree(args.copiar, "results", dirs_exist_ok=True)

    if not (args.data / "index").exists():
        subprocess.run([sys.executable, "src/embed.py", str(args.data)], check=True)

    esperado = len(load_jsonl(args.data / "cases.jsonl")) * 2 * len(DEFENSES)
    falhas = []
    fila = list(modelos)
    while fila:
        m = fila.pop(0)
        saida = Path("results") / f"{args.data.name}_{m}.jsonl"
        if completo(saida, esperado):
            print(f"== {m}: ja completo, pulando")
            continue
        if mem < MIN_GB[m]:
            print(f"== {m}: precisa de ~{MIN_GB[m]} GB e a GPU tem {mem:.0f} GB, pulando")
            continue
        print(f"== {m}: {MODELS[m]['model']}", flush=True)
        r = subprocess.run([sys.executable, "src/smoke.py", str(args.data), m])
        if r.returncode or not completo(saida, esperado):
            falhas.append(m)
            if m == "qwen" and "qwen3" not in modelos:
                fila.insert(0, "qwen3")
        if args.copiar:
            shutil.copytree("results", args.copiar, dirs_exist_ok=True)
        repo_dir = Path(constants.HF_HUB_CACHE) / ("models--" + MODELS[m]["model"].replace("/", "--"))
        shutil.rmtree(repo_dir, ignore_errors=True)

    print("Falharam: " + ", ".join(falhas) if falhas else "Todos os modelos terminaram.")


if __name__ == "__main__":
    main()
