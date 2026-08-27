# Regressão Linear

**Prioridade: Essencial**

A Regressão Linear é um excelente primeiro modelo porque obriga você a pensar em **relação entre features, target, erro e generalização**. Mesmo quando não será o melhor modelo final, ela costuma ser um baseline valioso.

## 1. O que ela faz?

Prevê um valor contínuo como combinação linear das features:

\[
\hat y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_px_p
\]

`β0` é o intercepto. Cada `βj` representa a contribuição associada à feature `xj`, mantendo as demais constantes — interpretação que exige cuidado quando features são correlacionadas.

## 2. Como funciona intuitivamente?

Imagine pontos em um gráfico. A regressão procura a reta/plano/hiperplano cujas previsões fiquem próximas dos valores observados. No Ordinary Least Squares, ela minimiza a soma dos erros quadráticos:

\[
\sum_i (y_i - \hat y_i)^2
\]

Erros grandes pesam bastante porque são elevados ao quadrado.

## 3. Quando usar?

- baseline para regressão;
- relações aproximadamente lineares;
- quando interpretabilidade é importante;
- quando você quer estimar efeitos condicionais com hipóteses estatísticas adequadas;
- problemas em que uma transformação de features torna a relação suficientemente simples.

## 4. Principal vantagem

É simples, rápida, interpretável e extremamente útil para diagnosticar se existe sinal linear nos dados.

## 5. Limitações e cuidados

- não captura não linearidade automaticamente;
- outliers podem influenciar fortemente;
- multicolinearidade torna coeficientes instáveis;
- extrapolação além da faixa observada pode ser perigosa;
- inferência estatística exige hipóteses adicionais sobre resíduos e processo amostral.

!!! note "Predição ≠ inferência causal"
    Um coeficiente positivo não prova que aumentar aquela feature causará aumento no target. Causalidade exige desenho e hipóteses adicionais.

## 6. Como executar?

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.3f}")
```

## Como ler os coeficientes

```python
for i, coef in enumerate(model.coef_):
    print(i, coef)
```

Se uma feature está em unidade original, um coeficiente `β=3.2` sugere que aumentar a feature em 1 unidade está associado a +3.2 na previsão, mantendo outras features fixas.

Mas compare coeficientes com cautela quando as escalas diferem.

## Resíduos

Resíduo:

\[
e_i = y_i - \hat y_i
\]

Você pode inspecionar padrões:

```python
import matplotlib.pyplot as plt

residuos = y_test - pred
plt.scatter(pred, residuos)
plt.axhline(0)
plt.xlabel("Previsão")
plt.ylabel("Resíduo")
plt.show()
```

Se resíduos exibem curva sistemática, talvez a relação não seja bem representada por uma função linear.

## Ridge e Lasso — o 80/20

Regularização adiciona penalidade aos coeficientes.

### Ridge (L2)

Penaliza soma dos quadrados dos coeficientes. Tende a encolher todos, sem zerá-los facilmente.

### Lasso (L1)

Pode levar alguns coeficientes a zero e atuar como seleção embutida de features.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])
pipe.fit(X_train, y_train)
```

Scaling é particularmente importante quando regularização compara magnitudes de coeficientes.

## Experimente

Troque `alpha` por `[0.01, 0.1, 1, 10, 100]` e observe MAE/RMSE. Pergunte: quando a penalização fica forte demais?

## Anki

- O que a Regressão Linear minimiza no OLS?
- Qual a diferença conceitual entre resíduo e erro futuro?
- Por que multicolinearidade afeta interpretação dos coeficientes?
- Ridge e Lasso diferem como?
- Um coeficiente positivo implica causalidade? Por quê?
