# SVR e KNN Regressor

**Prioridade: Importante**

Esses dois modelos mostram duas ideias diferentes: **margem/kernel** e **vizinhança por distância**. Você não precisa dominar todos os detalhes matemáticos na primeira volta, mas deve entender quando scaling é obrigatório e como os hiperparâmetros mudam a flexibilidade.

# SVR — Support Vector Regression

## As seis perguntas

**O que faz?** Prediz valores contínuos usando a ideia de margem das Support Vector Machines.

**Intuição:** procura uma função que mantenha muitos pontos dentro de um “tubo” de tolerância `ε`, controlando complexidade e penalizando violações.

**Quando usar?** datasets pequenos/médios, relações não lineares com kernels, quando custo computacional é aceitável.

**Vantagem:** kernels permitem fronteiras/relações complexas sem criar explicitamente todas as features transformadas.

**Limitações:** sensível a escala e a `C`, `gamma`, `epsilon`; pode ficar caro em datasets grandes.

**Como executar?** Use `Pipeline` com scaling.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

svr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1))
])

svr.fit(X_train, y_train)
pred = svr.predict(X_test)
```

## Parâmetros centrais

- `C`: penalização de erros; maior `C` tende a exigir ajuste mais rígido aos dados;
- `gamma`: alcance de influência de cada exemplo no kernel RBF; gamma alto cria comportamento mais local/complexo;
- `epsilon`: largura do tubo em que pequenos erros não são penalizados da mesma forma.

!!! warning "Scaling"
    Sem scaling, features grandes podem dominar distâncias no espaço usado pelo kernel.

# KNN Regressor

## As seis perguntas

**O que faz?** Prediz usando targets de observações vizinhas.

**Intuição:** “casos parecidos tendem a ter respostas parecidas”.

**Quando usar?** datasets pequenos/médios e baixa/moderada dimensionalidade, principalmente como baseline local.

**Vantagem:** modelo intuitivo, praticamente sem hipótese funcional global.

**Limitações:** previsão pode ser lenta; sofre com muitas dimensões; scaling é crítico; densidade irregular dos dados afeta vizinhança.

**Como executar?**:

```python
from sklearn.neighbors import KNeighborsRegressor

knn = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsRegressor(n_neighbors=10, weights="distance"))
])
knn.fit(X_train, y_train)
```

## `k` controla suavidade

- `k` pequeno → muito local, maior variance;
- `k` grande → média muitos vizinhos, maior bias.

## Maldição da dimensionalidade

Em muitas dimensões, distâncias tendem a ficar menos discriminativas. O conceito de “vizinho próximo” pode perder força. Feature selection ou redução de dimensionalidade pode ajudar, mas não é garantia.

## Comparação mental

| SVR | KNN |
|---|---|
| aprende uma função baseada em suporte/kernel | usa exemplos armazenados |
| tuning de `C`, `gamma`, `epsilon` | tuning de `k`, distância, pesos |
| scaling essencial | scaling essencial |
| pode modelar superfícies suaves | comportamento local |

## Anki

- Por que SVR e KNN precisam de scaling?
- O que acontece com KNN quando `k` é muito pequeno?
- O que `C` controla no SVR?
- O que é a maldição da dimensionalidade para KNN?
