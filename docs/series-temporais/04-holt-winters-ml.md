# Holt-Winters e Machine Learning para séries

**Prioridade: Importante**

Há duas abordagens úteis para o 80/20: modelos clássicos que representam nível/tendência/sazonalidade e a transformação da série em uma tabela supervisionada.

# Holt-Winters

Exponential Smoothing atribui mais peso a observações recentes e pode modelar:

- nível;
- tendência;
- sazonalidade.

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(
    train,
    trend="add",
    seasonal="add",
    seasonal_periods=7
).fit()

forecast = model.forecast(len(test))
```

## Sazonalidade aditiva vs multiplicativa

### Aditiva

Amplitude sazonal aproximadamente constante.

```text
+20 todo dezembro
```

### Multiplicativa

Amplitude cresce proporcionalmente ao nível.

```text
+20% todo dezembro
```

Dados precisam ser compatíveis com a formulação multiplicativa, especialmente valores positivos.

# ML com features temporais

Crie uma tabela:

| data | lag_1 | lag_7 | media_7 | dia_semana | y |
|---|---:|---:|---:|---:|---:|

E treine Random Forest, Gradient Boosting ou outro regressor.

```python
df["lag_1"] = df["y"].shift(1)
df["lag_7"] = df["y"].shift(7)
df["mean_7"] = df["y"].shift(1).rolling(7).mean()
df["dow"] = df.index.dayofweek

data = df.dropna()
features = ["lag_1", "lag_7", "mean_7", "dow"]
```

Depois use split temporal.

## A diferença escondida: previsão multi-step

Se você usa `lag_1`, para prever amanhã conhece o valor de hoje. Para prever daqui a 30 dias, os lags intermediários talvez dependam de **suas próprias previsões**.

### Estratégia recursiva

Prevê `t+1`, usa essa previsão para construir `t+2`, e assim por diante. Erros podem acumular.

### Estratégia direta

Treina um modelo específico para cada horizonte.

```text
modelo_h1 → y_{t+1}
modelo_h7 → y_{t+7}
modelo_h30 → y_{t+30}
```

## Compare famílias com o mesmo backtest

A comparação justa é:

```text
Seasonal Naive
ARIMA/SARIMA
Holt-Winters
Gradient Boosting com lags
```

Todos avaliados nos mesmos cortes e métricas.

## Quando começar simples

Séries com sazonalidade muito regular podem ser bem servidas por Holt-Winters. Séries com muitas covariáveis, eventos e relações não lineares podem favorecer ML. Não existe hierarquia universal.

## Anki

- Qual diferença entre sazonalidade aditiva e multiplicativa?
- Como transformar série em problema supervisionado?
- Qual problema da previsão recursiva multi-step?
- Por que todos os modelos devem usar o mesmo backtest?
