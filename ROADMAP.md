# Roadmap

Fases com **gates**: se o gate não passa, o projeto reporta o resultado
negativo em vez de forçar o próximo passo.

---

## Fase 0 — Auditoria de fontes ✅

`docs/DATASET_AUDIT.md`. 10/10 links vivos, 5 fontes baixadas.

Achado central: o `product_type` do Cannlytics **já declara o método** em
parte dos mercados — `non-solvent based concentrate` (12.798) e
`solvent based concentrate` (24.576). Rótulo de origem, não inferido.

---

## Fase 1 — Viabilidade do rótulo ✅ GATE PASSOU

Critério: ≥150 amostras/classe de ≥3 produtores/classe.
Real: **12.798 vs 24.576**, com **69 e 82 produtores**, **11.289 e 21.970**
com perfil de terpenos.

Passou por duas ordens de grandeza. **OpenCOA e MassIVE deixam de ser
bloqueantes** e viram validação externa.

---

## Fase 2 — Dataset master 🔵 PRÓXIMA

```
RAW → NORMALIZED → LABELED → MODEL_READY
```

`RAW` nunca é sobrescrito. Toda linha carrega `source_dataset`, `source_url`,
`source_id`, `original_product_name`, `original_product_type`,
`labeling_rule`, `label_confidence`, `retrieval_date`.

### Estratégia de labeling em dois caminhos

1. **Estruturado (HIGH)** — `product_type` contendo `non-solvent based` ou
   `solvent based`. Declarado na origem.
2. **Por nome (HIGH/MEDIUM)** — regex em `product_name`: `hash rosin`,
   `live rosin`, `rosin` → solventless; `live resin`, `bho`, `shatter`,
   `badder`, `wax` → hydrocarbon.

**Onde os dois existirem, devem concordar.** Medir a taxa de divergência antes
de confiar no caminho 2 — ela é o teste empírico da regra por nome. Se for
alta, o caminho 2 sai do dataset principal.

`concentrate` sozinho permanece `UNLABELED`. Nunca vira BHO.

### Tarefas

- Harmonizar nomes de analitos entre labs (`beta-caryophyllene`, `BCP`,
  `b_caryophyllene`...).
- **Decidir e documentar o tratamento de não-detectado.** `ND`, `<LOQ`, vazio
  e `0` não são a mesma coisa; tratar tudo como zero enviesa o modelo. Provável
  solução: valor + máscara de censura por analito.
- Normalizar unidades (percentual vs mg/g) — variam por lab.

---

## Fase 3 — EDA e confundidores

**Modelos-alerta antes de qualquer classificador de método.** Se o perfil
químico prevê bem `brand`, `lab`, `state` ou `strain`, qualquer resultado da
Fase 4 é suspeito.

Vantagem já medida: **55 produtores aparecem nas duas classes**. Dá para um
teste interno forte — treinar e testar apenas nesses produtores, onde acertar
por memorização de marca é impossível.

Também: PCA/UMAP para estrutura natural.

---

## Fase 4 — Modelagem

Experimentos: cannabinoides-only, terpenos-only, combinado, PCA/UMAP, SHAP.

**Splits — nunca aleatório:**
- `GroupKFold` por producer, brand, strain, lab
- **unseen-brand** (nenhum produtor do teste aparece no treino) — viável com
  69/82 produtores
- unseen-lab (12 e 11 labs), unseen-state

**Métricas:** balanced accuracy, macro F1, recall por classe, PR-AUC, matriz
de confusão. Accuracy sozinha é proibida — a classe é 1:2.

**Baselines obrigatórias:** majority class e random.

### Gate 4

> O macro F1 no unseen-brand test supera a majority class por margem maior que
> o desvio entre folds?

Se não, a conclusão é que o fingerprint não generaliza para marca nova —
resultado publicável.

---

## Fase 5 — Relatório

`reports/final_report.md` responde:

> Existe informação suficiente nos COAs públicos para diferenciar, de maneira
> generalizável, concentrados solventless de hydrocarbon?

Três desfechos aceitáveis: sim, não, ou não decidível com o dado disponível.

---

## Princípios

1. **Testar a hipótese, não provar.**
2. **Nunca inventar rótulo.** `concentrate` não vira BHO.
3. **Todo número tem evidência** em `docs/evidence/` ou script em `src/`.
4. **Resultado negativo é resultado.**
5. **Leakage é a ameaça principal.** Um modelo que decora produtor parece
   ótimo e não serve para nada.
6. **Verificar integridade de download contra a origem.** Um arquivo truncado
   abre, parseia e produz números plausíveis — e já inverteu a conclusão deste
   projeto uma vez.
