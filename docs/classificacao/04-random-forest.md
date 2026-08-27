# Random Forest Classifier

**Prioridade: Essencial**

A Random Forest combina várias árvores para produzir uma classificação mais robusta.

## 1. O que faz?

Treina árvores em amostras bootstrap e subconjuntos aleatórios de features. A classe final vem de agregação/votação e probabilidades podem ser médias das árvores.

## 2. Intuição

Perguntar a várias árvores diferentes reduz dependência das peculiaridades de uma só.

```text
árvore A → fraude
árvore B → não fraude
árvore C → fraude
árvore D → fraude
------------------
resultado → fraude
```

## 3. Quando usar?

- dados tabulares;
- baseline forte;
- relações não lineares;
- interações;
- quando preprocessing mínimo é desejável.

## 4. Vantagens

- robusta;
- pouco sensível a scaling;
- bom desempenho com tuning moderado;
- probabilidades e importâncias disponíveis.

## 5. Limitações

- menos interpretável;
- pode gerar probabilidades mal calibradas em alguns cenários;
- custo maior que modelo linear;
- importâncias por impureza podem ser enviesadas.

## 6. Código

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

rf = RandomForestClassifier(
    n_estimators=400,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight=None,
    random_state=42,
    n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="roc_auc")
print(scores.mean(), scores.std())
```

## `class_weight="balanced"`

Ajusta pesos aproximadamente de forma inversa à frequência das classes. Pode ajudar o treinamento a valorizar a classe minoritária, mas não garante melhor resultado operacional.

Teste com a métrica correta e ajuste threshold.

## Probabilidade e threshold

```python
rf.fit(X_train, y_train)
prob = rf.predict_proba(X_test)[:, 1]
pred_30 = (prob >= 0.30).astype(int)
```

A floresta produz score; a decisão de 30%, 50% ou 80% pertence ao sistema de decisão.

## Feature importance: prefira comparação cuidadosa

Permutation importance mede quanto a métrica piora quando embaralhamos uma feature.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    rf, X_test, y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42
)
```

Ainda é uma medida associativa e pode se comportar de forma complexa com features correlacionadas.

## Anki

- Por que Random Forest costuma generalizar melhor que árvore isolada?
- Para que serve `max_features`?
- `class_weight` substitui threshold tuning?
- Qual o problema de interpretar feature importance como causalidade?
