"""Download de arquivos do Hugging Face com verificação de integridade.

Existe por causa de uma falha real: `all-results-latest.csv` foi baixado
truncado em 7% (177,9 MB de 2.534,8 MB) e nada acusou. O curl retornou
sucesso, o arquivo abriu normalmente no pandas, e o parser descartou apenas
1 linha malformada — indistinguível de um CSV levemente sujo.

A contagem de concentrados feita sobre esse arquivo estava errada por um
fator de ~14x, e isso quase virou a conclusão do projeto.

Regra: nenhum download é considerado válido sem comparar o tamanho local
contra o tamanho declarado pela API de origem.
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = "cannlytics/cannabis_results"
TREE_API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/data?recursive=true"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"


def remote_sizes() -> dict[str, int]:
    """Tamanho declarado pela API do HF, por caminho."""
    with urllib.request.urlopen(TREE_API, timeout=60) as r:
        tree = json.load(r)
    return {e["path"]: e.get("size", 0) for e in tree if e.get("type") == "file"}


def fetch(path: str, dest_dir: Path, expected: int, max_attempts: int = 40) -> bool:
    """Baixa `path`, retomando de onde parou, até bater `expected` bytes.

    Retomar importa: o link do HF cai com frequência em arquivos de GB, e sem
    -C cada queda recomeçaria do zero. O loop é o que transforma "caiu no meio"
    em "continua depois", em vez de um arquivo parcial que passa despercebido.
    """
    dest = dest_dir / Path(path).name
    dest_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_attempts + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have >= expected:
            return True
        print(f"  tentativa {attempt}: {have/1e6:.1f}/{expected/1e6:.1f} MB", flush=True)
        subprocess.run(
            ["curl", "-sL", "--retry", "3", "--retry-delay", "3",
             "--retry-all-errors", "-C", "-", "--connect-timeout", "30",
             "--max-time", "1200", RESOLVE + path, "-o", str(dest)],
            check=False,
        )

    have = dest.stat().st_size if dest.exists() else 0
    return have >= expected


def verify(path: str, dest_dir: Path, expected: int) -> bool:
    """Confere tamanho E se o arquivo termina num registro completo.

    O tamanho pega truncamento; a última linha pega corrupção que mantenha o
    tamanho. Um CSV válido nunca termina no meio de um campo entre aspas.
    """
    dest = dest_dir / Path(path).name
    if not dest.exists():
        print(f"  FALTA: {dest}")
        return False
    got = dest.stat().st_size
    if got != expected:
        pct = 100 * got / expected if expected else 0
        print(f"  TRUNCADO: {got/1e6:.1f} MB de {expected/1e6:.1f} MB ({pct:.1f}%)")
        return False
    if dest.suffix == ".csv":
        with open(dest, "rb") as f:
            f.seek(max(0, got - 4096))
            tail = f.read().decode("utf-8", errors="replace")
        if tail.count('"') % 2 != 0:
            print("  SUSPEITO: aspas desbalanceadas no fim — possível corte")
            return False
    print(f"  OK: {got/1e6:.1f} MB")
    return True


def main(argv: list[str]) -> int:
    raw = Path(__file__).resolve().parents[2] / "data/raw/cannlytics"
    sizes = remote_sizes()

    # Por padrão só o consolidado; --states baixa também os splits por estado.
    wanted = ["data/all/all-results-latest.csv"]
    if "--states" in argv:
        wanted += sorted(
            p for p in sizes
            if p.endswith("-results-latest.csv") and "/all/" not in p
        )

    ok = True
    for path in wanted:
        expected = sizes.get(path)
        if not expected:
            print(f"{path}: não encontrado na API")
            ok = False
            continue
        print(f"{path} ({expected/1e6:.1f} MB)")
        fetch(path, raw, expected)
        ok &= verify(path, raw, expected)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
