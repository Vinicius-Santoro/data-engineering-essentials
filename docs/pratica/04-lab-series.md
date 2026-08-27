# Laboratório — série temporal

**Objetivo:** criar uma série com tendência + sazonalidade, construir baselines, fazer holdout temporal e comparar Holt-Winters com ML baseado em lags.

## 1. Série sintética

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
n = 500
idx = pd.date_range("2025-01-01", periods=n, freq="D")

t = np.arange(n)
trend = 0.05 * t
season = 10 * np.sin(2*np.pi*t/7)
noise = rng.normal(0, 3, n)
y = 100 + trend + season + noise

s = pd.Series(y, index=idx, name="y")
s.plot(figsize=(12,4))
plt.show()
```

Antes de modelar, identifique visualmente:

- tendência;
- período sazonal provável;
- escala do ruído.

## 2. Holdout temporal

```python
train = s.iloc[:-60]
test = s.iloc[-60:]

print(train.index.min(), train.index.max())
print(test.index.min(), test.index.max())
```

Nenhum shuffle.

## 3. Baseline naive

Para uma previsão one-step simples:

```python
naive = test.shift(1)
naive.iloc[0] = train.iloc[-1]
```

## 4. Seasonal naive

```python
history = pd.concat([train, test])
seasonal_naive = history.shift(7).loc[test.index]
```

## 5. Avalie

```python
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

for name, pred in {
    "naive": naive,
    "seasonal_naive": seasonal_naive
}.items():
    print(
        name,
        "MAE", mean_absolute_error(test, pred),
        "RMSE", root_mean_squared_error(test, pred)
    )
```

## 6. Holt-Winters

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

hw = ExponentialSmoothing(
    train,
    trend="add",
    seasonal="add",
    seasonal_periods=7
).fit()

hw_pred = hw.forecast(len(test))
print("HW MAE", mean_absolute_error(test, hw_pred))
```

Plote real e previsão.

## 7. Dataset supervisionado com lags

```python
df = s.to_frame()
df["lag_1"] = df["y"].shift(1)
df["lag_7"] = df["y"].shift(7)
df["lag_14"] = df["y"].shift(14)
df["mean_7"] = df["y"].shift(1).rolling(7).mean()
df["std_7"] = df["y"].shift(1).rolling(7).std()
df["dow"] = df.index.dayofweek

df = df.dropna()
```

Observe o `shift(1)` antes da rolling mean. Remova-o e explique exatamente qual informação vaza.

## 8. Split supervisionado temporal

```python
train_df = df.loc[df.index < test.index.min()]
test_df = df.loc[df.index >= test.index.min()]

features = ["lag_1", "lag_7", "lag_14", "mean_7", "std_7", "dow"]

X_train = train_df[features]
y_train = train_df["y"]
X_test = test_df[features]
y_test = test_df["y"]
```

## 9. Gradient Boosting

```python
from sklearn.ensemble import HistGradientBoostingRegressor

model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=15,
    random_state=42
)
model.fit(X_train, y_train)
ml_pred = model.predict(X_test)

print("ML MAE", mean_absolute_error(y_test, ml_pred))
```

Neste exemplo, features de lag no conjunto de teste usam valores **reais passados** disponíveis na tabela. Isso corresponde a uma avaliação one-step/rolling em que cada valor real anterior já foi observado. Para forecast aberto de 60 dias feito de uma vez, você não teria todos esses valores reais e precisaria estratégia recursiva/direta.

## 10. Backtesting com TimeSeriesSplit

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for fold, (tr_idx, va_idx) in enumerate(tscv.split(df), 1):
    tr = df.iloc[tr_idx]
    va = df.iloc[va_idx]

    model.fit(tr[features], tr["y"])
    pred = model.predict(va[features])
    print(fold, mean_absolute_error(va["y"], pred))
```

## Experimentos obrigatórios

- [ ] remova `lag_7` e observe efeito;
- [ ] altere a sazonalidade da série de 7 para 30 dias;
- [ ] compare seasonal naive com Holt-Winters;
- [ ] crie rolling mean incorreta sem shift e identifique leakage;
- [ ] explique a diferença entre avaliação one-step e forecast aberto multi-step.
