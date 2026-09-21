# Cannabis Extraction Method — Data Science research

Investiga se o perfil químico (canabinoides + terpenos) de um concentrado de
Cannabis carrega informação suficiente para inferir o **método de extração** —
especificamente para separar **solventless** (rosin, live rosin, hash rosin) de
**hydrocarbon** (BHO, live resin, shatter, badder, wax).

**A hipótese não é assumida como verdadeira.** O projeto existe para testá-la,
e "não há dado público suficiente" é um desfecho aceitável.

## Estado atual

Fase 0 concluída (auditoria de fontes). O projeto está **parado no Gate 1**,
e por um motivo concreto:

> O Cannlytics — a maior fonte pública de COAs — tem 4.140 concentrados, mas
> **nenhum campo de método de extração**. Rotulando pelo nome do produto,
> sobram **24 solventless e 11 hydrocarbon**. Não dá para treinar nem avaliar.

Isso não é limitação da regra de labeling: `product_name` está preenchido em
100% dos concentrados, mas os nomes são strain + peso (`blue dream (1g)`),
não descrição de processo. Medição completa em
[`docs/DATASET_AUDIT.md`](docs/DATASET_AUDIT.md).

A próxima etapa decide o projeto: verificar se **OpenCOA** ou **MassIVE**
fornecem rótulo em volume. Ver [`ROADMAP.md`](ROADMAP.md).

## Dados baixados

| Diretório | Fonte | Tamanho |
|---|---|---|
| `data/raw/cannlytics/` | HF `cannlytics/cannabis_results` | 163 MB |
| `data/raw/myleslab/` | GitHub MylesLab/cannabis-labelling | 99 MB |
| `data/raw/kushy/` | GitHub kushyapp/cannabis-dataset | 7,3 MB |
| `data/raw/openthc/` | GitHub openthc/data | 1,3 MB |
| `data/raw/zenodo_13823859/` | Zenodo (Equador, tem coluna `Metodo`) | 216 KB |

`data/` está no `.gitignore` — o repositório versiona código e evidência, não
os dados brutos. Reproduza com `src/ingestion/`.

Pendente: **Dryad** (API 401 / download 403 para cliente automatizado — requer
download manual) e os dois Zenodo de LC-MS (~3,7 GB, adiados).

## Estrutura

```
data/{raw,normalized,labeled,processed}/   camadas, RAW nunca sobrescrito
docs/DATASET_AUDIT.md                      auditoria com números medidos
docs/evidence/                             logs brutos de cada medição
notebooks/                                 análise exploratória
src/{ingestion,preprocessing,labeling,features,models,evaluation}/
reports/                                   figuras, tabelas, relatório final
ROADMAP.md                                 fases e gates de decisão
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Regras do projeto

1. Todo número publicado tem evidência em `docs/evidence/` ou script que o
   reproduz.
2. Rótulo nunca é inventado — `concentrate` não vira BHO por conveniência.
3. Split sempre por grupo (brand/producer/strain/lab), nunca aleatório: várias
   amostras compartilham marca e strain, e split aleatório vaza.
4. Accuracy sozinha não é métrica — usar balanced accuracy, macro F1 e recall
   por classe.
5. Antes de acreditar em qualquer classificador de método, verificar se o
   dataset prevê `brand`/`lab`/`strain`. Se prever bem, o resultado é
   confounding até prova em contrário.
