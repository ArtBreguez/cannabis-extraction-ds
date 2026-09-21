# Auditoria de datasets — evidência coletada em 2026-09-21

Todos os números abaixo vêm de execução real contra os arquivos baixados ou
contra as APIs oficiais. Os logs brutos estão em `docs/evidence/`. Nenhum
número foi estimado.

## Resumo por fonte

| Fonte | Status | Licença | Amostras | Tem método de extração? |
|---|---|---|---|---|
| Cannlytics `cannabis_results` | ✅ baixado (163 MB) | CC-BY-4.0 | 6.336 linhas | ❌ não como campo |
| Zenodo 13823859 (Equador) | ✅ baixado | CC-BY-4.0 | 378 + 648 linhas | ✅ **sim, coluna `Metodo`** |
| MylesLab `cannabis-labelling` | ✅ clonado (99 MB) | ver repo | 296 amostras | ❌ é flor, não extrato |
| Kushy `cannabis-dataset` | ✅ clonado (7,3 MB) | ver repo | — | ❌ metadata de mercado |
| OpenTHC `data` | ✅ clonado (1,3 MB) | ver repo | — | ❌ taxonomia de produto |
| Dryad `sxksn0314` | ⚠️ **bloqueado** | CC0-1.0 | 16 MB, 14 arquivos | ❌ é flor/cultivar |
| Zenodo 19222374 (LC-MS) | ⏸️ adiado | CC-BY-4.0 | ~1,9 GB | ✅ `metadata3classes.csv` |
| Zenodo 17671960 (LC-MS) | ⏸️ adiado | CC-BY-4.0 | ~1,8 GB | parcial |
| OpenCOA | ⏸️ não coletado | — | — | a verificar |
| MassIVE | ⏸️ não coletado | — | — | experimento controlado |
| ACS figshare | ⏸️ não coletado | — | — | referência química |

## O achado que muda o projeto

**A hipótese principal não é testável com o Cannlytics no estado atual.**

Medido (`docs/evidence/cannlytics_labeling_attempt.txt`):

```
total de linhas:                              6.336
linhas de CONCENTRADO/EXTRATO:                4.140
concentrados com total_terpenes preenchido:   3.599

rótulos obtidos por regex no product_name:
   4.102  UNLABELED
      19  live_rosin
       7  badder
       4  rosin
       3  distillate
       2  live_resin
       1  wax
       1  hash_rosin
       1  bho

SOLVENTLESS (rosin/live rosin/hash rosin)  = 24
HYDROCARBON (bho/live resin/shatter/badder/wax) = 11
```

**24 vs 11 amostras.** Não dá para treinar nem avaliar classificador com isso,
e muito menos fazer teste de generalização por marca/strain/lab como o
protocolo exige.

### Isso é limite do dataset, não da regra de labeling

Verifiquei antes de concluir (`docs/evidence/cannlytics_product_names.txt`):

- `product_name` vazio em concentrados: **0**
- nomes distintos: **1.943**
- os 40 nomes mais frequentes são todos do tipo `blue dream (1g)`,
  `kimbo cookies (1g)`, `pink lemonade (0.5g)` — **strain + peso**

O `product_type` também não ajuda: 3.133 linhas são
`concentrate, product inhalable` e 964 são `concentrate` puro. A regra do
prompt é explícita: *nunca transformar automaticamente `concentrate` em BHO*.

Ou seja: o campo que descreveria o método simplesmente não existe nesse
dump. Ampliar a regex não resolve — a informação não está no texto.

## Consequência para o roadmap

O Cannlytics continua valioso como **fonte de fingerprint químico** (463
colunas, 3.599 concentrados com terpenos), mas **não pode ser a fonte de
rótulo**. O rótulo precisa vir de:

1. **OpenCOA** — COAs onde o método aparece explicitamente (não coletado ainda)
2. **Zenodo 13823859** — tem coluna `Metodo` de origem, mas é extrato de flor
   em experimento controlado no Equador, não concentrado de mercado
3. **MassIVE** — comparação controlada de métodos (não coletado ainda)

Sem pelo menos uma dessas fontes render volume, a resposta honesta para a
pergunta científica é *"não há dado público suficiente"* — que é um resultado
legítimo e deve ser reportado como tal.

## Pendências

- **Dryad**: API retorna 401 e o download direto retorna 403 para cliente
  automatizado. Precisa de download manual pelo navegador em
  <https://datadryad.org/dataset/doi:10.5061/dryad.sxksn0314>.
  (De qualquer forma é dado de **flor/cultivar**, não de extrato — serve de
  baseline de variabilidade da matéria-prima, como o próprio prompt diz.)
- **LC-MS (Zenodo)**: ~3,7 GB somados. Só faz sentido depois de decidir se o
  projeto vai trabalhar com MS bruto.
- **OpenCOA / MassIVE / ACS**: próximos a investigar, e são os que decidem a
  viabilidade da pesquisa.
