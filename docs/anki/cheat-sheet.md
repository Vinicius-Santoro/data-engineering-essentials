# Cheat sheet — Ciência de Dados 80/20

Use esta página para revisão rápida **depois** de estudar os capítulos detalhados.

## Mapa de decisão

```text
Tenho y?
├── não
│   ├── quero grupos → clustering
│   └── quero representação menor → PCA/SVD/t-SNE
└── sim
    ├── y categórico → classificação
    ├── y contínuo → regressão
    └── y ao longo do tempo → considere estrutura temporal

Decisões sequenciais + recompensa + interação?
└── Reinforcement Learning pode fazer sentido
```

## Pipeline universal supervisionado

```text
Pergunta
↓
Unidade + target + data de corte
↓
Split que simula produção
↓
EDA no treino
↓
Preprocessamento em Pipeline
↓
Baseline
↓
Modelos
↓
Cross-validation
↓
Tuning / threshold
↓
Teste final
↓
Deploy + monitoramento
```

## Scikit-learn mental model

```python
estimator.fit(X_train, y_train)
pred = estimator.predict(X_test)
```

Transformador:

```python
transformer.fit(X_train)
X_train2 = transformer.transform(X_train)
X_test2 = transformer.transform(X_test)
```

Pipeline:

```python
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
```

## Leakage

Pergunta de ouro:

> Eu conheceria esta informação exatamente assim no instante da previsão?

Sinais:

- preprocessing antes do split;
- feature criada depois do target;
- rolling sem shift;
- mesmo usuário nos dois lados quando produção exige usuário novo;
- seleção de feature fora do CV;
- PCA/scaler/imputer ajustados usando validação.

## Scaling

| Algoritmo | Scaling |
|---|---|
| Logistic | recomendado |
| KNN | **sim** |
| SVM/SVR | **sim** |
| K-Means | **sim** |
| PCA | **sim** geralmente |
| Linear/Ridge/Lasso | importante com regularização |
| Decision Tree | não precisa geralmente |
| Random Forest | não precisa geralmente |
| Tree Boosting | não precisa geralmente |

## Regressão

### Linear

```python
LinearRegression()
Ridge(alpha=1.0)
Lasso(alpha=0.1)
```

- baseline interpretável;
- linearidade;
- Ridge L2, Lasso L1.

### Decision Tree

```python
DecisionTreeRegressor(max_depth=4, min_samples_leaf=10)
```

- não linear;
- interações naturais;
- alta variance se profunda.

### Random Forest

```python
RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    n_jobs=-1
)
```

- bagging;
- várias árvores decorrelacionadas;
- reduz variance.

### Gradient Boosting

```python
HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=300
)
```

- sequencial;
- corrige erros/gradientes;
- learning rate × número de iterações.

### SVR / KNN

```python
SVR(C=10, gamma="scale")
KNeighborsRegressor(n_neighbors=10)
```

Scaling obrigatório na prática.

## Classificação

### Logistic Regression

```python
LogisticRegression(C=1.0, max_iter=3000)
```

```text
C pequeno → regularização forte
C grande  → regularização fraca
```

### KNN

```python
KNeighborsClassifier(n_neighbors=7)
```

`k` pequeno = alta variance; `k` grande = maior bias.

### Tree / Random Forest

```python
DecisionTreeClassifier(max_depth=4)
RandomForestClassifier(n_estimators=400, min_samples_leaf=2)
```

Sem scaling obrigatório.

### SVM

```python
SVC(kernel="rbf", C=1, gamma="scale")
```

- margem;
- support vectors;
- `C` e `gamma` controlam flexibilidade.

### Naive Bayes

```python
MultinomialNB(alpha=1.0)
```

Clássico para texto/contagens.

## Métricas de classificação

```text
                REAL +     REAL -
PREV +            TP         FP
PREV -            FN         TN
```

\[
Precision=\frac{TP}{TP+FP}
\]

> Dos alertas positivos, quantos eram verdadeiros?

\[
Recall=\frac{TP}{TP+FN}
\]

> Dos positivos reais, quantos encontrei?

\[
F1=2\frac{PR}{P+R}
\]

Accuracy: proporção total correta — cuidado com desbalanceamento.

ROC-AUC: qualidade do ranking ao longo de thresholds.

PR-AUC/AP: foco em precision/recall da classe positiva; muito útil com positivos raros.

## Threshold

```python
pred = (prob >= threshold).astype(int)
```

```text
threshold ↓ → mais positivos → recall tende ↑, precision pode ↓
threshold ↑ → menos positivos → precision tende ↑, recall ↓
```

Escolha por custo/capacidade, não por hábito de 0.5.

## Métricas de regressão

### MAE

Mesmo peso linear para cada erro; unidade original.

### RMSE

Penaliza mais erros grandes; unidade original.

### R²

Compara squared error com baseline da média. Pode ser negativo.

### MAPE

Percentual, mas perigoso perto de zero.

## Cross-validation

```python
scores = cross_val_score(
    pipe, X_train, y_train,
    cv=5,
    scoring="roc_auc"
)
```

- classificação → StratifiedKFold;
- grupos → GroupKFold;
- tempo → TimeSeriesSplit.

Pipeline **dentro** da CV.

## Tuning

```python
GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc")
```

Use teste final só depois.

## Clustering

### K-Means

```python
KMeans(n_clusters=4, n_init="auto")
```

- centroides;
- precisa K;
- scaling;
- clusters compactos.

### DBSCAN

```python
DBSCAN(eps=0.3, min_samples=8)
```

- densidade;
- ruído `-1`;
- formatos irregulares;
- sensível a `eps` e escala.

### Hierárquico

```python
AgglomerativeClustering(n_clusters=4, linkage="ward")
```

estrutura de fusões/dendrograma.

## PCA

```python
Pipeline([
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=0.95))
])
```

- PC1 = direção de maior variância;
- componentes ortogonais;
- cria features novas;
- não usa y;
- dentro do CV se fizer parte do modelo.

## Séries temporais

```text
NUNCA embaralhe o tempo por padrão.
```

Features:

```python
df["lag_1"] = df.y.shift(1)
df["lag_7"] = df.y.shift(7)
df["mean_7"] = df.y.shift(1).rolling(7).mean()
```

Baselines:

```text
naive
seasonal naive
```

ARIMA:

```text
p = autoregressivo
d = diferenciação
q = média móvel dos erros
```

SARIMA adiciona `(P,D,Q,m)` sazonal. SARIMAX adiciona exógenas.

Holt-Winters: nível + tendência + sazonalidade por exponential smoothing.

## Reinforcement Learning

```text
estado → ação → recompensa → novo estado
```

- política `π(a|s)`;
- retorno = recompensa futura acumulada/descontada;
- `γ` = valor do futuro;
- Q-Learning = value-based;
- REINFORCE = policy-based;
- actor-critic = política + valor;
- PPO = atualizações de política controladas.

## As 10 perguntas antes de um modelo

1. O que uma linha representa?
2. Qual é `y`?
3. Quando a previsão acontece?
4. Quais features existem naquele instante?
5. Como meu split imita produção?
6. Qual baseline preciso superar?
7. Qual erro custa mais?
8. Onde pode haver leakage?
9. Como vou validar estabilidade?
10. Como vou avaliar/monitorar depois do deploy?
