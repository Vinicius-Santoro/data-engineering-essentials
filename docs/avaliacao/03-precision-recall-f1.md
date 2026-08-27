# Precision, Recall e F1

**Prioridade: Essencial**

Essas métricas usam os quadrantes da matriz de confusão, mas respondem perguntas diferentes.

## Precision

\[
Precision = \frac{TP}{TP+FP}
\]

Pergunta:

> **Entre tudo o que marquei como positivo, quanto realmente era positivo?**

Alta precision = poucos falsos positivos entre os alertas.

### Exemplo operacional

Você só consegue telefonar para 1.000 clientes por dia. Se o modelo selecionar 1.000, talvez você queira alta precision para não desperdiçar a capacidade.

## Recall / Sensitivity

\[
Recall = \frac{TP}{TP+FN}
\]

Pergunta:

> **Entre todos os positivos reais, quantos eu encontrei?**

Alta recall = poucos positivos perdidos.

Em fraude, pode ser muito importante capturar grande parte das fraudes, mesmo gerando mais investigação manual.

## O trade-off via threshold

Suponha scores:

```text
0.95  fraude
0.72  fraude
0.60  legítima
0.41  fraude
0.20  legítima
```

Threshold alto seleciona poucos casos → costuma aumentar precision e reduzir recall. Threshold baixo seleciona mais → costuma aumentar recall e reduzir precision.

```python
from sklearn.metrics import precision_score, recall_score

for t in [0.2, 0.4, 0.5, 0.7, 0.9]:
    pred_t = (prob >= t).astype(int)
    p = precision_score(y_test, pred_t, zero_division=0)
    r = recall_score(y_test, pred_t, zero_division=0)
    print(t, round(p, 3), round(r, 3))
```

## F1

\[
F1 = 2\frac{Precision\cdot Recall}{Precision+Recall}
\]

É a média harmônica entre precision e recall. A média harmônica pune desequilíbrio: uma métrica muito baixa derruba o F1.

## Quando F1 é útil?

- você quer equilibrar precision e recall;
- classes são desbalanceadas;
- true negatives não são o foco principal.

Mas F1 pressupõe implicitamente importância semelhante para precision e recall. Se custos são muito diferentes, uma métrica/custo customizado pode ser melhor.

## F-beta

\[
F_\beta = (1+\beta^2)\frac{PR}{\beta^2P+R}
\]

- `β > 1`: dá mais peso a recall;
- `β < 1`: dá mais peso a precision.

```python
from sklearn.metrics import fbeta_score
print(fbeta_score(y_test, pred, beta=2))
```

## Macro, micro e weighted em multiclasse

### Macro

Calcula a métrica por classe e faz média simples. Classes raras têm o mesmo peso das grandes.

### Weighted

Média ponderada pelo suporte de cada classe.

### Micro

Agrega decisões de todas as classes antes de calcular a métrica.

```python
from sklearn.metrics import f1_score

f1_macro = f1_score(y_test, pred, average="macro")
f1_weighted = f1_score(y_test, pred, average="weighted")
```

Em intents com classes muito desbalanceadas, macro F1 frequentemente revela problemas que weighted F1 pode esconder.

## Precision@K

Se a operação atua apenas nos top `K` scores, métricas de ranking como Precision@K podem ser mais alinhadas do que threshold fixo.

```python
import numpy as np

k = 100
top_idx = np.argsort(prob)[::-1][:k]
precision_at_k = y_test[top_idx].mean()
```

## Anki

- Qual pergunta Precision responde?
- Qual pergunta Recall responde?
- O que tende a acontecer com recall ao baixar threshold?
- Quando macro F1 é preferível a weighted F1?
- Por que F1 pode não representar custos assimétricos?
