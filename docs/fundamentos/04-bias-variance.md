# Overfitting, underfitting e bias–variance

**Prioridade: Essencial**

Esses conceitos explicam por que aumentar a complexidade de um modelo ajuda até certo ponto e depois pode piorar a capacidade de generalização.

## Underfitting

O modelo é simples demais para capturar o padrão.

```text
erro treino: alto
erro validação: alto
```

Exemplo: tentar representar uma relação fortemente curva com uma reta simples.

## Overfitting

O modelo aprende padrões reais **e ruído específico do treino**.

```text
erro treino: muito baixo
erro validação: alto
```

Uma árvore sem restrições pode criar divisões muito específicas para explicar praticamente cada observação do treino.

## Generalização

Queremos um modelo que aprenda regularidades que continuem válidas fora da amostra.

Uma forma útil de pensar:

```text
complexidade baixa    → underfitting
complexidade adequada → boa generalização
complexidade extrema  → overfitting
```

## Bias e variance — intuição

**Bias alto:** o modelo impõe hipóteses fortes e sistematicamente perde aspectos do padrão.

**Variance alta:** pequenas mudanças no conjunto de treino podem gerar modelos muito diferentes.

Uma árvore profunda costuma ter baixa bias e alta variance. Random Forest reduz variance agregando muitas árvores.

## Experimento com árvore

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

for depth in [1, 2, 3, 5, 10, None]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(depth, round(train_acc, 3), round(test_acc, 3))
```

Não memorize os números. Observe o padrão: o score de treino tende a aumentar com complexidade; o de teste pode parar de melhorar ou piorar.

## Regularização

Regularização é qualquer mecanismo que restrinja a flexibilidade para melhorar generalização.

Exemplos:

- Regressão Logística/Linear: penalidades L1/L2;
- árvore: `max_depth`, `min_samples_leaf`;
- Random Forest: profundidade, tamanho mínimo das folhas, amostragem;
- Gradient Boosting: learning rate, profundidade das árvores, número de estimadores;
- SVM: `C` controla tolerância a violações da margem.

## Mais dados podem ajudar?

Se o problema principal é variance/overfitting, mais dados representativos frequentemente ajudam. Se o modelo sofre forte underfitting, simplesmente adicionar linhas pode não resolver: talvez seja necessário melhorar features ou capacidade do modelo.

## Learning curves

Learning curves comparam desempenho conforme aumenta a quantidade de treino. Elas ajudam a diagnosticar se mais dados provavelmente ajudam.

```python
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(random_state=42),
    X, y,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.2, 1.0, 5)
)

print(train_scores.mean(axis=1))
print(val_scores.mean(axis=1))
```

## Um diagnóstico operacional

| Treino | Validação | Hipótese |
|---|---|---|
| ruim | ruim | underfitting / features fracas |
| excelente | ruim | overfitting / variance alta |
| bom | bom | situação saudável |
| ruim | excelente | investigue bug, leakage ou amostra estranha |

!!! warning "Não use apenas um número"
    Um gap treino–validação é mais informativo quando a avaliação é estável e o split é apropriado. Cross-validation ajuda a distinguir sinal de sorte da amostra.

## Perguntas de recuperação

1. Como distinguir empiricamente underfitting de overfitting?
2. Por que limitar `max_depth` pode melhorar o teste mesmo piorando o treino?
3. Por que Random Forest costuma reduzir variance em relação a uma única árvore?
4. Mais dados sempre resolvem underfitting?
