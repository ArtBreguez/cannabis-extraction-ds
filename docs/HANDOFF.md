# Handoff — próxima rodada

Estado em 2026-09-21, commit `7b8f32e`. Fases 0 e 1 concluídas, Gate 1 passou.

## Onde parei

O dataset existe e é grande o suficiente. O trabalho de ingestão acabou; o
próximo passo é **construir o dataset master** (Fase 2 do `ROADMAP.md`).

Nada está pela metade. `data/raw/` tem tudo o que foi baixado, verificado por
`src/ingestion/fetch_cannlytics.py`.

## A primeira decisão da próxima rodada

**Tratamento de não-detectado.** É a escolha que mais afeta o resultado e
ainda não foi tomada.

Num COA, um analito pode aparecer como:

| Valor | Significa |
|---|---|
| `ND` | não detectado — abaixo do limite de detecção |
| `<LOQ` | detectado, mas abaixo do limite de quantificação |
| vazio | **não foi testado** — ausência de informação, não de substância |
| `0` | zero medido (raro; normalmente é `ND`) |

Tratar tudo como `0` é o caminho fácil e enviesa o modelo: um lab que não
testa terpeno X vira indistinguível de um produto sem terpeno X. E como o lab
correlaciona com mercado e com método, isso vira leakage disfarçado de sinal.

Proposta a avaliar: **valor + máscara de censura por analito** (duas colunas),
ou imputação por `LOD/2` com flag. Medir antes quantos `ND`/`<LOQ`/vazio
existem por analito e por lab — se um lab concentra os vazios, a máscara vira
obrigatória.

## Segunda decisão

**Validar o rótulo por nome contra o estruturado.** Onde as duas fontes
existirem na mesma linha, elas devem concordar. A taxa de divergência é o
teste empírico da regra por regex.

- Divergência baixa → o caminho por nome amplia cobertura com segurança.
- Divergência alta → sai do dataset principal e vira só sinal exploratório.

Isso precisa ser medido **antes** de usar as 2.412 + 7.652 linhas rotuladas
por nome.

## Terceira: unidades

Labs reportam em percentual ou mg/g, e nem sempre declaram. Normalizar exige
inspecionar por lab. Uma conversão errada silenciosa é do mesmo tipo do
download truncado: passa por tudo e corrompe o resultado.

## O que NÃO precisa mais

- **OpenCOA / MassIVE / ACS** deixaram de ser bloqueantes. Com 12.798 vs
  24.576 rotulados na origem, viraram validação externa opcional.
- **Dryad** é flor, não extrato. Baixa prioridade, e requer download manual.
- **LC-MS (Zenodo, ~3,7 GB)** só se o projeto decidir trabalhar com espectro
  bruto — decisão que não precisa ser tomada agora.

## Armadilha registrada

O download do Cannlytics veio truncado em 7% e **nada acusou**: abriu no
pandas, parseou, produziu números plausíveis, e levou à conclusão oposta à
verdade (24 vs 11 em vez de 12.798 vs 24.576).

Qualquer fonte nova passa por verificação de tamanho contra a origem antes de
virar número em documento.

## Ativos prontos

- `src/ingestion/fetch_cannlytics.py` — download com retomada e verificação
- `docs/evidence/` — logs brutos de toda medição citada
- `docs/DATASET_AUDIT.md` — auditoria das 10 fontes
- `ROADMAP.md` — fases e gates
