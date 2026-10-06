# Cannabis Extraction Method — Data Science research

Investiga se o perfil químico (canabinoides + terpenos) de um concentrado de
Cannabis carrega informação suficiente para inferir o **método de extração** —
separando **solventless** (rosin, live rosin, hash rosin) de **hydrocarbon**
(BHO, live resin, shatter, badder, wax).

**A hipótese não é assumida como verdadeira.** O projeto existe para testá-la,
e "não há dado suficiente" é desfecho aceitável.

## Estado atual

Fase 0 e Fase 1 concluídas. **Gate 1 passou** — o dado existe e em volume:

```
non-solvent based (SOLVENTLESS): 12.798   11.289 com terpenos | 69 produtores | 12 labs
solvent based     (HYDROCARBON): 24.576   21.970 com terpenos | 82 produtores | 11 labs
```

O rótulo vem do campo `product_type` do Cannlytics, **declarado na origem** —
não é inferência sobre nome comercial. Base total: 808.407 linhas, 139.714
concentrados.

**53 produtores aparecem nas duas classes** (após excluir as 30 linhas com
rótulo contraditório), o que permite testar se o modelo aprende processo ou
apenas memoriza marca.

Medições em [`docs/DATASET_AUDIT.md`](docs/DATASET_AUDIT.md).
Próxima etapa em [`ROADMAP.md`](ROADMAP.md): construir o dataset master.

## Dados

| Diretório | Fonte | Tamanho |
|---|---|---|
| `data/raw/cannlytics/` | HF `cannlytics/cannabis_results` | 2.534,8 MB |
| `data/raw/myleslab/` | GitHub MylesLab/cannabis-labelling | 99 MB |
| `data/raw/kushy/` | GitHub kushyapp/cannabis-dataset | 7,3 MB |
| `data/raw/openthc/` | GitHub openthc/data | 1,3 MB |
| `data/raw/zenodo_13823859/` | Zenodo — tem coluna `Metodo` | 216 KB |

`data/` está no `.gitignore`. Reproduza com:

```bash
.venv/bin/python src/ingestion/fetch_cannlytics.py
```

Pendentes: Dryad (bloqueia cliente automatizado; é flor, baixa prioridade) e
os dois Zenodo de LC-MS (~3,7 GB, adiados).

## Estrutura

```
data/{raw,normalized,labeled,processed}/   RAW nunca sobrescrito
docs/DATASET_AUDIT.md                      auditoria com números medidos
docs/evidence/                             logs brutos de cada medição
src/ingestion/fetch_cannlytics.py          download com verificação de integridade
notebooks/  src/  reports/                 análise, código, saída
ROADMAP.md                                 fases e gates
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Regras

1. Todo número publicado tem evidência em `docs/evidence/` ou script que o
   reproduz.
2. Rótulo nunca é inventado — `concentrate` sozinho fica `UNLABELED`.
3. Split sempre por grupo (producer/brand/strain/lab), nunca aleatório.
4. Accuracy sozinha não é métrica — a classe é 1:2.
5. Antes de acreditar em qualquer classificador, verificar se o dataset prevê
   `brand`/`lab`/`strain`. Se prever bem, é confounding até prova em
   contrário.
6. **Download grande é verificado contra o tamanho da origem.** Um arquivo
   truncado em 7% já abriu normalmente, produziu números plausíveis e levou à
   conclusão oposta à verdade neste projeto.
