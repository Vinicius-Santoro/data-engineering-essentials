# Cross-validation

**Prioridade: Essencial**

Cross-validation responde uma pergunta simples: **o desempenho observado é consistente ou depende demais de uma divisão específica dos dados?**

## O problema de um único split

Você divide:

```text
80% treino | 20% validação
```

Por sorte, o conjunto de validação pode ser mais fácil ou mais difícil. Outro `random_state` produz outro score.

## K-Fold

Em 5-fold:

```text
Rodada 1  [VAL][TRN][TRN][TRN][TRN]
Rodada 2  [TRN][VAL][TRN][TRN][TRN]
Rodada 3  [TRN][TRN][VAL][TRN][TRN]
Rodada 4  [TRN][TRN][TRN][VAL][TRN]
Rodada 5  [TRN][TRN][TRN][TRN][VAL]
```

Cada observação participa da validação uma vez e do treino `k-1` vezes.

No final:

```text
0.86, 0.88, 0.84, 0.87, 0.89
média = 0.868
desvio ≈ estabilidade
```

## Código

```python
from sklearn.model_selection import KFold, cross_val_score

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="neg_mean_absolute_error"
)

print("scores:", -scores)
print("média:", -scores.mean())
print("desvio:", scores.std())
```

## Classificação: StratifiedKFold

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Mantém aproximadamente a proporção das classes em cada fold.

## Por que Pipeline é obrigatório quando há preprocessing aprendido?

Errado:

```python
X_scaled = StandardScaler().fit_transform(X_train)
cross_val_score(model, X_scaled, y_train, cv=5)
```

A validação de cada fold influenciou o scaler.

Correto:

```python
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression())
])

cross_val_score(pipe, X_train, y_train, cv=5)
```

## Cross-validation não substitui teste final

Use CV para desenvolvimento/tuning **dentro do treino**. Depois, uma avaliação final em dados congelados fornece uma estimativa menos contaminada pelas decisões de desenvolvimento.

## Tipos especiais

### GroupKFold

Evita que a mesma entidade apareça nos dois lados.

```python
from sklearn.model_selection import GroupKFold, cross_val_score

cv = GroupKFold(n_splits=5)
scores = cross_val_score(
    model, X, y,
    groups=cliente_id,
    cv=cv
)
```

### TimeSeriesSplit

Mantém ordem temporal.

```text
Fold 1: [train] [val]
Fold 2: [---train---] [val]
Fold 3: [------train------] [val]
```

```python
from sklearn.model_selection import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=5)
```

## Quantos folds?

5 ou 10 são escolhas comuns, não leis.

Mais folds:

- treino de cada fold maior;
- custo computacional maior;
- propriedades de bias/variance da estimativa mudam.

Para o 80/20, `5-fold` é um default prático para muitos problemas tabulares não temporais.

## O desvio importa

Compare:

```text
Modelo A: 0.90 ± 0.01
Modelo B: 0.91 ± 0.08
```

B tem média ligeiramente maior, mas grande instabilidade. Isso pode indicar sensibilidade à amostra ou heterogeneidade relevante.

Não trate o desvio entre folds como um intervalo de confiança formal automaticamente; use-o como diagnóstico descritivo.

## Repeated CV

Para datasets menores, repetir K-Fold com diferentes embaralhamentos pode fornecer visão mais robusta:

```python
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
```

## Anki

- Qual problema K-Fold reduz em relação a um split único?
- Por que CV não substitui teste final?
- Quando usar GroupKFold?
- Quando usar TimeSeriesSplit?
- Por que preprocessing deve ficar dentro do Pipeline durante CV?
