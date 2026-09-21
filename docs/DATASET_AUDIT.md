# Auditoria de datasets — evidência coletada em 2026-09-21

Todos os números vêm de execução real contra os arquivos baixados ou contra as
APIs oficiais. Logs brutos em `docs/evidence/`. Nenhum número foi estimado.

> **Nota de correção.** A primeira versão deste documento reportava 6.336
> linhas e 24 vs 11 amostras rotuláveis, e concluía que a pesquisa
> provavelmente não era viável. Aquilo foi medido sobre um download
> **truncado em 7%** (177,9 MB de 2.534,8 MB), cortado no meio de um registro.
> O `curl` retornou sucesso, o pandas abriu o arquivo e o parser descartou
> apenas 1 linha — nada acusou. Detectado ao comparar o tamanho local com o
> declarado pela API do HF. Os números abaixo são do arquivo íntegro,
> verificado por `src/ingestion/fetch_cannlytics.py`.

## Resumo por fonte

| Fonte | Status | Licença | Volume | Rótulo de método? |
|---|---|---|---|---|
| Cannlytics `cannabis_results` | ✅ 2.534,8 MB, íntegro | CC-BY-4.0 | 808.407 linhas | ✅ **sim, estruturado** |
| Zenodo 13823859 (Equador) | ✅ baixado | CC-BY-4.0 | 378 + 648 linhas | ✅ coluna `Metodo` |
| MylesLab `cannabis-labelling` | ✅ 99 MB | ver repo | 296 amostras | ❌ é flor |
| Kushy `cannabis-dataset` | ✅ 7,3 MB | ver repo | — | ❌ metadata de mercado |
| OpenTHC `data` | ✅ 1,3 MB | ver repo | — | ❌ taxonomia |
| Dryad `sxksn0314` | ⚠️ bloqueado (401/403) | CC0-1.0 | 16 MB | ❌ é flor/cultivar |
| Zenodo 19222374 / 17671960 | ⏸️ adiados | CC-BY-4.0 | ~3,7 GB | LC-MS bruto |
| OpenCOA / MassIVE / ACS | ⏸️ não coletados | — | — | a verificar |

## Cannlytics — o dataset central

Medido em `docs/evidence/cannlytics_recount_full.txt`:

```
LINHAS DE DADOS:            808.407   (0 malformadas)
CONCENTRADOS/EXTRATOS:      139.714
  com total_terpenes:        52.952
```

### Caminho A — rótulo estruturado (preferido)

O `product_type` de alguns mercados **já declara o método**, sem depender de
interpretar nome comercial
(`docs/evidence/cannlytics_structured_labels.txt`):

```
non-solvent based  (SOLVENTLESS): 12.798   terpenos: 11.289 | 69 produtores | 12 labs
solvent based      (HYDROCARBON): 24.576   terpenos: 21.970 | 82 produtores | 11 labs
```

Esta é a base de rótulo do projeto. Confiança **HIGH**: é campo declarado na
origem, não inferência.

**55 produtores aparecem nas duas classes.** Isso é excelente: significa que a
mesma casa produz solventless e hydrocarbon, então um modelo não consegue
acertar apenas decorando produtor. É a melhor defesa natural contra o
confounding que o protocolo teme — e permite um teste interno forte
(treinar e testar dentro dos mesmos 55 produtores).

### Caminho B — rótulo por nome do produto (complementar)

```
 15.416  distillate      1.843  shatter       271  wax
  3.013  live_resin      1.501  live_rosin    120  hash_rosin
  1.848  badder            791  rosin          30  ethanol
    677  bho               439  co2
113.765  UNLABELED
```

Agregado: **SOLVENTLESS 2.412** (177 produtores) · **HYDROCARBON 7.652**
(271 produtores). Confiança **HIGH** para `*rosin`, **MEDIUM** para o resto,
conforme a regra do protocolo.

Serve para: ampliar cobertura em mercados sem `product_type` estruturado, e
como **validação cruzada** — onde as duas fontes de rótulo existirem, elas
devem concordar. Divergência é sinal de problema na regra.

## Gate 1 — PASSOU

Critério: ≥150 amostras por classe, de ≥3 produtores distintos por classe.

Real: **12.798 vs 24.576**, com **69 e 82 produtores**. Passa por duas ordens
de grandeza, e o `unseen-brand test` é viável.

## Fontes de rótulo secundárias

- **Zenodo 13823859** tem coluna `Metodo` de origem, mas é extração de
  laboratório (maceração, ultrassom) em flor no Equador — não é rosin vs BHO
  de mercado. Serve para a pergunta *"método altera perfil químico?"* num
  desenho controlado, não para o classificador de produto comercial.
- **MylesLab** é flor (`Label`: hemp/sativa/indica, 296 amostras), não
  extrato. Baseline de variabilidade da matéria-prima, como o protocolo
  previa.

## Pendências

- **Dryad**: API 401, download direto 403. Requer download manual em
  <https://datadryad.org/dataset/doi:10.5061/dryad.sxksn0314>. É flor/cultivar
  — baixa prioridade.
- **LC-MS (Zenodo)**: ~3,7 GB. Só após decidir se o projeto usa MS bruto.
- **OpenCOA / MassIVE / ACS**: já não são bloqueantes. Com o rótulo
  estruturado do Cannlytics, deixam de ser a única esperança e passam a ser
  validação externa.

## Lição operacional

Download grande **exige verificação de integridade contra a origem**. Um
arquivo truncado passa por todas as checagens normais: abre, parseia, produz
números plausíveis — e leva a conclusão oposta à verdade.
`src/ingestion/fetch_cannlytics.py` compara tamanho e verifica se o CSV
termina em registro completo.
