# Laboratório — classificação ponta a ponta

**Objetivo:** praticar split, pipeline, CV, tuning, métricas e threshold usando o mesmo dataset.

!!! info "Regra do laboratório"
    Primeiro execute exatamente como está. Na segunda rodada, feche esta página e tente reconstruir o fluxo olhando apenas os títulos.

## 1. Dados

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer

raw = load_breast_cancer(as_frame=True)
df = raw.frame.copy()
print(df.shape)
print(df["target"].value_counts(normalize=True))
```

Perguntas antes de seguir:

- `target` é regressão ou classificação?
- as classes são exatamente balanceadas?
- a métrica principal deve ser escolhida apenas olhando frequência?

## 2. Separe teste final

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns="target")
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Não use `X_test` para escolher `C`, features ou threshold durante a etapa seguinte.

## 3. Baseline

```python
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

baseline = DummyClassifier(strategy="prior")
base_scores = cross_validate(
    baseline,
    X_train,
    y_train,
    cv=cv,
    scoring=["accuracy", "roc_auc", "f1"]
)

for metric in ["test_accuracy", "test_roc_auc", "test_f1"]:
    print(metric, base_scores[metric].mean())
```

O baseline com `strategy="prior"` produz probabilidades pela prevalência e classe majoritária nas decisões.

## 4. Logistic Regression com Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

logit = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=5000))
])

scores = cross_validate(
    logit,
    X_train,
    y_train,
    cv=cv,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    return_train_score=True
)

for key, values in scores.items():
    if key.startswith("test_"):
        print(key, values.mean(), values.std())
```

Observe média **e desvio**.

## 5. Tune `C`

```python
from sklearn.model_selection import GridSearchCV

params = {
    "model__C": [0.001, 0.01, 0.1, 1, 10, 100]
}

search = GridSearchCV(
    logit,
    param_grid=params,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    return_train_score=True
)
search.fit(X_train, y_train)

print(search.best_params_)
print(search.best_score_)
```

Pergunta: `C=100` significa regularização mais forte ou mais fraca?

## 6. Compare Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=400,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_scores = cross_validate(
    rf,
    X_train,
    y_train,
    cv=cv,
    scoring=["f1", "roc_auc"],
    n_jobs=-1
)

print("RF AUC", rf_scores["test_roc_auc"].mean())
print("RF F1 ", rf_scores["test_f1"].mean())
```

Não escolha vencedor apenas porque a terceira casa decimal é maior. Observe variabilidade, complexidade e objetivo.

## 7. Teste final

```python
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    ConfusionMatrixDisplay
)

best = search.best_estimator_
best.fit(X_train, y_train)

prob = best.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)

print(classification_report(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, prob))
ConfusionMatrixDisplay.from_predictions(y_test, pred)
```

## 8. Threshold

Suponha que a classe `1` seja a classe que queremos capturar com alto recall.

```python
from sklearn.metrics import precision_score, recall_score, f1_score

for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    pred_t = (prob >= t).astype(int)
    print(
        t,
        "P", round(precision_score(y_test, pred_t), 3),
        "R", round(recall_score(y_test, pred_t), 3),
        "F1", round(f1_score(y_test, pred_t), 3)
    )
```

Neste laboratório usamos teste para **demonstrar** threshold. Em projeto real, escolha threshold em validação e use teste apenas para confirmar o pipeline completo.

## 9. Experimentos obrigatórios

- [ ] remova `StandardScaler` e compare Logistic Regression;
- [ ] teste KNN com e sem scaling;
- [ ] compare `C=0.001` e `C=100`;
- [ ] altere `min_samples_leaf` da Random Forest;
- [ ] encontre threshold com recall ≥ 0.95 e maximize precision entre candidatos;
- [ ] explique por que a escolha acima é uma **restrição operacional**, não apenas tuning matemático.

## Você dominou este lab quando consegue

1. reconstruir o split sem consultar;
2. montar Pipeline de scaling + modelo;
3. executar Stratified K-Fold;
4. explicar cada métrica;
5. diferenciar score e decisão por threshold;
6. dizer onde leakage ocorreria se scaler ficasse fora do pipeline.
