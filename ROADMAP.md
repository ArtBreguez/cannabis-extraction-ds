# Roadmap

Ordem de trabalho e critérios de decisão. Cada fase tem um **gate**: se o gate
não passa, o projeto não avança para a fase seguinte — ele reporta o resultado
negativo, que também é resultado.

---

## Fase 0 — Auditoria de fontes ✅ FEITA

Ver `docs/DATASET_AUDIT.md`. Resultado: 10/10 links vivos, 5 fontes baixadas,
e um achado que reordena tudo — **o Cannlytics não tem campo de método de
extração**, e rotular por nome de produto rende 24 solventless vs 11
hydrocarbon em 4.140 concentrados.

---

## Fase 1 — Viabilidade do rótulo 🔴 GATE CRÍTICO

**A única pergunta que importa agora:** existe volume rotulável em algum lugar?

1. **OpenCOA** — mapear a API/estrutura, medir quantos COAs de concentrado
   trazem método explícito. É a fonte mais promissora para rótulo de mercado.
2. **MassIVE** — baixar metadados do experimento de comparação de métodos.
   Rótulo é confiável por construção (experimento controlado), mas o N tende a
   ser pequeno e não representa produto comercial.
3. **Zenodo 13823859** — já baixado, tem coluna `Metodo`. Entender o desenho
   experimental: são métodos de extração de laboratório (maceração,
   ultrassom), **não** rosin vs BHO de mercado.

### Gate 1

> Existem ≥ 150 amostras por classe (solventless / hydrocarbon) com rótulo de
> confiança HIGH ou MEDIUM, vindas de ≥ 3 marcas/produtores distintos por
> classe?

- **Passa** → Fase 2.
- **Não passa** → escrever `reports/final_report.md` concluindo que o dado
  público disponível não sustenta a pergunta, com os números que provam isso.
  **Isso encerra o projeto com resultado válido.** Não inventar rótulo para
  fabricar dataset.

O limite de 3 marcas por classe não é decorativo: com menos que isso, o
`unseen-brand test` que o protocolo exige é impossível, e qualquer acurácia
alta estaria medindo marca, não método.

---

## Fase 2 — Dataset master

Só começa se o Gate 1 passar.

Camadas, conforme o protocolo:

```
RAW → NORMALIZED → LABELED → MODEL_READY
```

- `RAW` nunca é sobrescrito.
- Toda linha carrega `source_dataset`, `source_url`, `source_id`,
  `original_product_name`, `original_product_type`, `labeling_rule`,
  `label_confidence`, `retrieval_date`.
- Harmonizar nomes de canabinoides/terpenos entre fontes (cada lab usa uma
  convenção — `b_caryophyllene`, `beta-caryophyllene`, `BCP`...).
- Decidir e **documentar** o tratamento de não-detectado: `ND`, `<LOQ`, vazio e
  zero não são a mesma coisa, e tratar todos como 0 enviesa o modelo.

---

## Fase 3 — EDA e confundidores

Antes de qualquer classificador de método, rodar os **modelos-alerta**:

| Alvo | O que significa se acertar demais |
|---|---|
| `brand` | o fingerprint identifica a marca, não o processo |
| `laboratory` | há assinatura analítica por lab |
| `state` | diferença de mercado/regulação, não de método |
| `strain` | está lendo genética |

Se `brand` ou `lab` forem muito previsíveis a partir das features químicas,
qualquer resultado da Fase 4 fica suspeito e precisa de split por grupo mais
agressivo.

Também nesta fase: PCA/UMAP para ver se existe estrutura natural.

---

## Fase 4 — Modelagem

Experimentos do protocolo: cannabinoides-only, terpenos-only, combinado,
PCA/UMAP, SHAP.

**Splits — nunca `train_test_split` aleatório:**

- `GroupKFold` por brand, producer, strain, lab
- teste difícil: **unseen-brand** (nenhuma marca do teste aparece no treino)
- também unseen-strain, unseen-lab, unseen-state

**Métricas:** balanced accuracy, macro F1, per-class recall, PR-AUC,
matriz de confusão. Accuracy sozinha é proibida — com classe desbalanceada
ela mente.

**Baselines obrigatórias:** majority class e random. Um modelo que não bate a
majority class não é modelo.

### Gate 4

> O macro F1 no **unseen-brand test** supera a baseline de majority class por
> margem maior que o desvio entre folds?

Se não superar, a conclusão é que o fingerprint químico não generaliza para
marca nova — resultado publicável e honesto.

---

## Fase 5 — Relatório

`reports/final_report.md` responde diretamente:

> Existe informação suficiente nos COAs públicos para diferenciar, de maneira
> generalizável, concentrados solventless/rosin de hydrocarbon/BHO?

Com três respostas possíveis, todas aceitáveis: **sim**, **não**, ou
**não há dado público suficiente para decidir**. A terceira é hoje a mais
provável, e o projeto deve estar pronto para defendê-la com números.

---

## Princípios

1. **Não provar a hipótese** — testá-la. O objetivo não é achar que dá certo.
2. **Nunca inventar rótulo.** `concentrate` não vira BHO por conveniência.
3. **Todo número tem evidência** em `docs/evidence/` ou é reproduzível por
   script em `src/`.
4. **Resultado negativo é resultado.** "Não dá para responder com esse dado" é
   uma contribuição, desde que sustentada por medição.
5. **Leakage é a ameaça principal**, não overfitting clássico. Um modelo que
   decora marca parece excelente e não serve para nada.
