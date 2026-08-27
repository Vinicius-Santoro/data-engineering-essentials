# Decision Tree Classifier

**Prioridade: Essencial**

A árvore de classificação cria regras sucessivas para separar classes.

## 1. O que faz?

Exemplo hipotético:

```text
uso_30d <= 2?
├── sim: atraso > 0?
│   ├── sim → alto risco
│   └── não → médio risco
└── não → baixo risco
```

## 2. Intuição

A cada nó, o algoritmo procura uma divisão que deixe os grupos filhos mais “puros” em relação às classes.

Critérios comuns incluem Gini e entropia.

### Gini

\[
Gini = 1 - \sum_k p_k^2
\]

Um nó com uma única classe tem impureza 0.

## 3. Quando usar?

- relações não lineares;
- interações;
- baseline interpretável se árvore rasa;
- base para ensembles.

## 4. Vantagens

- fácil de visualizar;
- sem scaling;
- aceita thresholds e interações naturalmente;
- pouca preparação numérica.

## 5. Limitações

- alta variance;
- propensa a overfitting;
- fronteiras em degraus;
- pequenas mudanças nos dados podem alterar a estrutura.

## 6. Código

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=10,
    class_weight=None,
    random_state=42
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(classification_report(y_test, pred))
```

## Visualização

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(18, 8))
plot_tree(model, max_depth=3, filled=True)
plt.show()
```

## Probabilidades em folhas

`predict_proba` estima proporções de classes nas folhas. Se uma folha contém 80% de classe positiva, a probabilidade pode ficar perto de 0.8.

Folhas muito pequenas geram estimativas extremas e instáveis. `min_samples_leaf` ajuda.

## Poda por custo-complexidade

`ccp_alpha` permite podar a árvore, penalizando complexidade.

```python
DecisionTreeClassifier(ccp_alpha=0.01, random_state=42)
```

Você não precisa decorar o algoritmo de poda; entenda que é outra forma de regularização.

## Anki

- O que significa pureza de um nó?
- Por que uma árvore profunda tem alta variance?
- Qual papel de `min_samples_leaf`?
- Por que scaling não muda muito uma árvore?
