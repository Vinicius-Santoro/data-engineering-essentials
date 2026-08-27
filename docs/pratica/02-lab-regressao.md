# Laboratório — regressão ponta a ponta

**Objetivo:** comparar baseline, modelo linear, árvore e ensemble usando MAE/RMSE e diagnóstico treino–validação.

## 1. Dataset

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

## 2. Baseline

```python
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, cross_validate

cv = KFold(n_splits=5, shuffle=True, random_state=42)
base = DummyRegressor(strategy="median")

scores = cross_validate(
    base,
    X_train,
    y_train,
    cv=cv,
    scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error"]
)

print("MAE", -scores["test_neg_mean_absolute_error"].mean())
print("RMSE", -scores["test_neg_root_mean_squared_error"].mean())
```

Por que scores de erro vêm negativos? O scikit-learn segue a convenção “maior é melhor” em scorers; por isso nega losses durante seleção.

## 3. Regressão Linear

```python
from sklearn.linear_model import LinearRegression

linear = LinearRegression()
lin = cross_validate(
    linear, X_train, y_train,
    cv=cv,
    scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
    return_train_score=True
)

print("MAE", -lin["test_neg_mean_absolute_error"].mean())
print("R2", lin["test_r2"].mean())
```

## 4. Decision Tree e overfitting

```python
from sklearn.tree import DecisionTreeRegressor

for depth in [1, 2, 3, 5, 10, None]:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
    s = cross_validate(
        tree, X_train, y_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
        return_train_score=True
    )
    train_mae = -s["train_score"].mean()
    val_mae = -s["test_score"].mean()
    print(depth, round(train_mae,2), round(val_mae,2))
```

Procure o gap de generalização.

## 5. Random Forest

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=400,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

rf_cv = cross_validate(
    rf, X_train, y_train,
    cv=cv,
    scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
    n_jobs=-1
)

print("RF MAE", -rf_cv["test_neg_mean_absolute_error"].mean())
```

## 6. Gradient Boosting

```python
from sklearn.ensemble import HistGradientBoostingRegressor

hgb = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=15,
    l2_regularization=1.0,
    random_state=42
)

hgb_cv = cross_validate(
    hgb, X_train, y_train,
    cv=cv,
    scoring="neg_mean_absolute_error"
)
print("HGB MAE", -hgb_cv["test_score"].mean())
```

## 7. Avaliação final

Escolha o candidato pela CV, ajuste no treino completo e avalie teste.

```python
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

final = rf  # substitua pelo vencedor da sua execução
final.fit(X_train, y_train)
pred = final.predict(X_test)

print("MAE", mean_absolute_error(y_test, pred))
print("RMSE", root_mean_squared_error(y_test, pred))
print("R²", r2_score(y_test, pred))
```

## 8. Diagnóstico de resíduos

```python
import matplotlib.pyplot as plt

res = y_test - pred
plt.scatter(pred, res)
plt.axhline(0)
plt.xlabel("previsão")
plt.ylabel("resíduo")
plt.show()
```

Também olhe maiores erros:

```python
import pandas as pd

audit = X_test.copy()
audit["real"] = y_test
audit["pred"] = pred
audit["abs_error"] = abs(audit["real"] - audit["pred"])
print(audit.sort_values("abs_error", ascending=False).head(10))
```

## Experimentos

- [ ] compare MAE e RMSE: algum modelo possui erros extremos maiores?
- [ ] teste `min_samples_leaf` da RF em 1, 2, 5, 10, 20;
- [ ] compare Linear vs RF: a não linearidade trouxe ganho real?
- [ ] descubra os 10 maiores erros e procure padrão;
- [ ] crie um DummyRegressor pela média e compare com mediana.

## Meta de domínio

Você deve conseguir explicar por que **um modelo com RMSE pior pode ter MAE parecido**, e por que **treino excelente + validação fraca** não é vitória.
