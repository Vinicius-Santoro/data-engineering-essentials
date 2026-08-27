# Support Vector Machine (SVM)

**Prioridade: Importante**

SVM busca uma fronteira de decisão com **margem grande** entre classes. Com kernels, consegue representar fronteiras não lineares.

## 1. O que faz?

No caso linear, encontra um hiperplano:

\[
w^Tx + b = 0
\]

que separa classes tentando maximizar a margem.

## 2. Intuição

Não basta separar. Entre várias fronteiras possíveis, preferimos uma que deixe espaço entre as classes.

Os pontos mais próximos da fronteira são os **support vectors**. Eles são particularmente importantes para definir a solução.

## 3. Quando usar?

- datasets pequenos/médios;
- muitas features;
- fronteira complexa com kernel;
- texto com representação vetorial esparsa (LinearSVC é um clássico).

## 4. Vantagens

- forte em espaços de alta dimensão;
- fundamento geométrico elegante;
- kernels oferecem flexibilidade.

## 5. Limitações

- scaling essencial;
- kernels podem ficar caros em datasets grandes;
- probabilidades não são nativas da formulação básica;
- `C` e `gamma` podem alterar muito o comportamento.

## 6. Código com RBF

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

svc = Pipeline([
    ("scale", StandardScaler()),
    ("model", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True
    ))
])
svc.fit(X_train, y_train)
```

## `C`

- `C` pequeno → aceita mais violações, margem mais regularizada;
- `C` grande → penaliza violações fortemente, tenta ajustar mais o treino.

## `gamma` no kernel RBF

`gamma` controla quão local é a influência de cada ponto.

```text
gamma baixo → influência ampla → fronteira suave
gamma alto  → influência local → fronteira complexa
```

`C` alto + `gamma` alto pode produzir overfitting.

## LinearSVC

Para muitas features, especialmente texto TF-IDF:

```python
from sklearn.svm import LinearSVC

model = LinearSVC(C=1.0)
model.fit(X_train, y_train)
```

É diferente de `SVC(kernel="linear")` em implementação e escala computacional.

## Anki

- O que são support vectors?
- O que significa maximizar margem?
- Como `C` altera regularização?
- Como `gamma` altera localidade do RBF?
- Por que scaling é tão importante?
