# ARIMA, SARIMA e SARIMAX

**Prioridade: Essencial para conceitos · Importante para tuning**

ARIMA modela dependência temporal combinando autoregressão, diferenciação e média móvel dos erros.

## ARIMA(p, d, q)

### AR — `p`

Usa valores passados da própria série.

\[
y_t \sim y_{t-1}, y_{t-2}, ...
\]

### I — `d`

Número de diferenciações usadas para tornar a série mais estacionária.

### MA — `q`

Usa erros passados do modelo.

A parte MA não é simplesmente “média móvel da série”; esse é um erro conceitual comum.

## Código

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train, order=(2, 1, 2))
fit = model.fit()
forecast = fit.forecast(steps=len(test))
```

## Como escolher p, d, q?

Na prática:

1. visualize série;
2. avalie necessidade de diferenciação;
3. ACF/PACF ajudam a gerar hipóteses;
4. compare configurações por AIC/BIC e principalmente validação temporal;
5. inspecione resíduos.

Não transforme padrões de ACF/PACF em regras mecânicas para qualquer série.

## SARIMA

Adiciona estrutura sazonal:

\[
(p,d,q)\times(P,D,Q)_m
\]

`m` é período sazonal: por exemplo 12 em dados mensais com sazonalidade anual, 7 em dados diários com padrão semanal — dependendo do processo.

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(
    train,
    order=(1,1,1),
    seasonal_order=(1,1,1,7),
    enforce_stationarity=False,
    enforce_invertibility=False
)
fit = model.fit(disp=False)
```

## SARIMAX

O `X` indica variáveis exógenas.

```python
model = SARIMAX(
    y_train,
    exog=X_train,
    order=(1,1,1),
    seasonal_order=(1,1,1,7)
)
fit = model.fit(disp=False)
forecast = fit.forecast(steps=len(y_test), exog=X_test)
```

Cuidado: `X_test` precisa representar features realmente conhecidas/disponíveis para o horizonte futuro.

## Diagnóstico de resíduos

Idealmente, resíduos restantes não deveriam exibir forte estrutura previsível.

```python
res = fit.resid
plot_acf(res.dropna(), lags=30)
```

Se autocorrelação forte permanece, o modelo pode ter deixado sinal na mesa.

## Intervalo de previsão

```python
pred = fit.get_forecast(steps=14)
summary = pred.summary_frame()
```

Intervalos comunicam incerteza; quanto maior o horizonte, normalmente mais ampla tende a ficar.

## AIC não é score de produção

AIC/BIC ajudam a comparar modelos probabilísticos ajustados à mesma série sob condições apropriadas, mas a decisão operacional deve incluir **backtesting fora da amostra**.

## Anki

- O que representam p, d e q?
- Por que MA no ARIMA não é uma rolling mean comum?
- O que SARIMA adiciona?
- O que SARIMAX adiciona?
- Por que AIC não substitui validação temporal?
