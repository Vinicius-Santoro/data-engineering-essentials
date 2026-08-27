# Pipeline e ColumnTransformer

**Prioridade: Essencial**

`Pipeline` é uma das ferramentas mais importantes do scikit-learn porque transforma preprocessamento e modelo em **uma única unidade treinável**.

## Sem pipeline

Você precisa lembrar de:

```text
imputar treino
imputar teste com estatísticas do treino
escalar treino
escalar teste com estatísticas do treino
codificar treino
codificar teste com categorias aprendidas
ajustar modelo
```

É fácil cometer leakage ou inconsistência.

## Com pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000))
])

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
```

Do lado de fora, o pipeline se comporta como um estimador.

## O que acontece no `fit`?

```text
X_train
  ↓
scaler.fit_transform(X_train)
  ↓
modelo.fit(X_transformado, y_train)
```

## E no `predict`?

```text
X_test
  ↓
scaler.transform(X_test)   ← não refaz fit
  ↓
modelo.predict(...)
```

Essa distinção é o coração da prevenção de leakage.

## Pipeline dentro de cross-validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    pipe,
    X_train,
    y_train,
    cv=5,
    scoring="roc_auc"
)
```

Em cada fold, o scikit-learn clona e ajusta o pipeline apenas com a porção de treino daquele fold.

## Dados mistos

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", numeric, ["idade", "renda"]),
    ("cat", categorical, ["estado", "plano"])
])

model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(max_iter=2000))
])
```

Agora você possui **um objeto** que recebe DataFrame bruto e produz previsões.

## Hiperparâmetros internos

Pipeline usa `__` para acessar etapas:

```python
params = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__class_weight": [None, "balanced"]
}
```

Isso permite usar GridSearch sem desmontar o pipeline.

## Pipeline como contrato de produção

Uma grande vantagem conceitual é reduzir divergência entre treinamento e inferência. O mesmo objeto encapsula:

```text
raw input → transformações → modelo
```

Isso não elimina todos os problemas de produção, mas reduz uma classe enorme de erros.

## Quando não colocar algo no pipeline?

A regra prática: transformações **por linha** ou que aprendem estatísticas a partir dos dados podem ser encapsuladas. Já a construção de labels, split temporal, consultas SQL e lógica de data de corte geralmente pertencem a etapas anteriores do processo e precisam de governança própria.

## Exercício

Qual versão é mais segura durante cross-validation?

=== "A"

    ```python
    X2 = StandardScaler().fit_transform(X)
    cross_val_score(LogisticRegression(), X2, y, cv=5)
    ```

=== "B"

    ```python
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ])
    cross_val_score(pipe, X, y, cv=5)
    ```

??? success "Resposta"
    **B**, porque o scaler é reajustado dentro de cada fold apenas com as observações de treino daquele fold.

## Perguntas para Anki

- O que acontece com cada etapa de um Pipeline durante `fit` e `predict`?
- Por que Pipeline é especialmente importante em cross-validation?
- Para que serve `ColumnTransformer`?
- O que significa `classifier__C` em um GridSearch?
