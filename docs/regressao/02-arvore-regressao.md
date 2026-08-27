# Decision Tree Regressor

**Prioridade: Essencial**

Árvores de decisão constroem regras hierárquicas e produzem previsões por regiões do espaço de features. São uma ponte importante para Random Forest e Gradient Boosting.

## 1. O que faz?

Divide recursivamente os dados por condições como:

```text
renda <= 5000?
├── sim → idade <= 30?
└── não → uso_app <= 4?
```

Em regressão, cada folha costuma prever a média dos targets das observações que chegaram ali.

## 2. Intuição

A árvore procura cortes que reduzam a heterogeneidade do target dentro dos grupos resultantes. Ela cria uma função **por partes**, capaz de representar não linearidades e interações sem você especificá-las manualmente.

## 3. Quando usar?

- baseline não linear para dados tabulares;
- relações com interações e thresholds;
- quando regras locais são úteis;
- como componente de ensembles.

## 4. Vantagens

- captura não linearidade;
- dispensa scaling na maioria dos casos;
- lida naturalmente com interações;
- fácil de visualizar quando rasa.

## 5. Limitações

- árvores profundas têm alta variance;
- pequenas mudanças nos dados podem mudar bastante a estrutura;
- previsão é em degraus, não suave;
- uma única árvore geralmente perde em performance para ensembles.

## 6. Código

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=10,
    random_state=42
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(mean_absolute_error(y_test, pred))
```

## Os hiperparâmetros que importam primeiro

### `max_depth`

Limita quantos níveis a árvore pode crescer. Profundidade maior = mais flexibilidade.

### `min_samples_leaf`

Exige um número mínimo de amostras em cada folha. Folhas maiores suavizam regras e reduzem a chance de memorizar casos isolados.

### `min_samples_split`

Mínimo de amostras para permitir nova divisão.

## Experimento de complexidade

```python
from sklearn.metrics import mean_squared_error

for depth in [1, 2, 3, 5, 10, None]:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    rmse_train = mean_squared_error(
        y_train, model.predict(X_train)
    ) ** 0.5
    rmse_test = mean_squared_error(
        y_test, model.predict(X_test)
    ) ** 0.5

    print(depth, round(rmse_train, 2), round(rmse_test, 2))
```

Procure o padrão de overfitting: treino melhorando continuamente enquanto teste deixa de acompanhar.

## Feature importance

```python
print(model.feature_importances_)
```

A importância baseada em redução de impureza pode favorecer features com muitas possibilidades de corte e não explica direção do efeito. Use como indício, não como verdade causal.

## Árvore vs regressão linear

| Aspecto | Linear | Árvore |
|---|---|---|
| não linearidade | precisa engenharia | natural |
| interações | precisam ser especificadas | natural |
| extrapolação | possível, com risco | não extrapola suavemente |
| scaling | geralmente desnecessário | desnecessário |
| estabilidade | maior | menor |

## Anki

- Como uma folha produz previsão em regressão?
- Por que `max_depth=None` pode overfitar?
- Por que árvore não precisa de scaling?
- Qual efeito de aumentar `min_samples_leaf`?
- Por que uma árvore é chamada de modelo por partes?
