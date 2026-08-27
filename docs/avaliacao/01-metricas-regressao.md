# Métricas de regressão

**Prioridade: Essencial**

Uma métrica transforma erros de previsão em um número que permite comparar modelos. O ponto central é: **métricas diferentes respondem perguntas diferentes**.

## Resíduo / erro por observação

\[
e_i = y_i - \hat y_i
\]

Se `y=100` e a previsão é `90`, o erro assinado é `+10`. Para avaliação global normalmente usamos magnitude ou quadrado porque erros positivos e negativos poderiam se cancelar.

## MAE — Mean Absolute Error

\[
MAE = \frac{1}{n}\sum_i |y_i-\hat y_i|
\]

Interpretação: erro absoluto médio na **mesma unidade do target**.

Se MAE de previsão de tempo é 12 minutos, uma leitura prática é “o erro absoluto típico médio é cerca de 12 min”.

```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, pred)
```

### Quando gosto de MAE?

- interpretação simples;
- quando custo cresce aproximadamente de forma linear com o erro;
- quando não quero que poucos erros gigantes dominem tanto quanto no RMSE.

## MSE — Mean Squared Error

\[
MSE = \frac{1}{n}\sum_i (y_i-\hat y_i)^2
\]

Penaliza erros grandes fortemente. Sua unidade fica ao quadrado, então costuma ser menos intuitiva para negócio.

## RMSE — Root Mean Squared Error

\[
RMSE = \sqrt{MSE}
\]

Volta para a unidade original, preservando maior sensibilidade a erros grandes.

```python
from sklearn.metrics import root_mean_squared_error
rmse = root_mean_squared_error(y_test, pred)
```

Em versões anteriores do sklearn:

```python
rmse = mean_squared_error(y_test, pred) ** 0.5
```

## Por que RMSE tende a ser ≥ MAE?

O quadrado dá mais peso aos erros extremos. Se dois modelos têm MAE semelhante, mas um tem RMSE muito maior, ele provavelmente produz alguns erros grandes.

### Exemplo

Modelo A erros absolutos:

```text
10, 10, 10, 10
MAE = 10
RMSE = 10
```

Modelo B:

```text
0, 0, 0, 40
MAE = 10
RMSE = 20
```

Mesmo MAE, perfis de risco diferentes.

## R²

\[
R^2 = 1 - \frac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2}
\]

Compara o modelo a uma baseline que prevê a média.

- `R² = 1`: previsão perfeita;
- `R² = 0`: similar à previsão pela média no conjunto avaliado;
- `R² < 0`: pior que essa baseline.

!!! warning "R² não é porcentagem de acertos"
    `R²=0.8` não significa “80% das previsões estão corretas”. Ele descreve redução relativa de soma de quadrados em relação à baseline média.

## MAPE

\[
MAPE = \frac{100}{n}\sum_i \left|\frac{y_i-\hat y_i}{y_i}\right|
\]

É atraente por ser percentual, mas pode explodir quando `y` está perto de zero e ter comportamento assimétrico. Não use automaticamente.

## Escolha guiada pelo custo

Pergunte:

> O custo de errar 100 é aproximadamente dez vezes o de errar 10, ou muito maior que dez vezes?

- custo aproximadamente linear → MAE é natural;
- erros grandes são especialmente perigosos → RMSE pode refletir melhor;
- porcentagem é essencial e zero não é problema → alguma métrica percentual pode ajudar.

## Compare com baseline

```python
from sklearn.dummy import DummyRegressor

base = DummyRegressor(strategy="median")
base.fit(X_train, y_train)
base_pred = base.predict(X_test)
print(mean_absolute_error(y_test, base_pred))
```

Se seu modelo sofisticado mal supera a mediana, investigue features e formulação.

## Distribuição dos erros

Uma média pode esconder subgrupos.

```python
errors = abs(y_test - pred)
print("p50", np.quantile(errors, .50))
print("p90", np.quantile(errors, .90))
print("p99", np.quantile(errors, .99))
```

Também avalie por segmentos relevantes: região, faixa de valor, canal, período.

## Anki

- Qual a diferença prática entre MAE e RMSE?
- Por que MAPE é perigosa perto de zero?
- Como interpretar R² negativo?
- Por que olhar percentis de erro além da média?
