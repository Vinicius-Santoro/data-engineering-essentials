# Random Forest Regressor

**Prioridade: Essencial**

Random Forest combina muitas árvores para obter um modelo mais estável. É um dos baselines mais fortes para dados tabulares.

## 1. O que faz?

Treina várias árvores em amostras e subconjuntos de features diferentes e agrega as previsões. Em regressão, normalmente usa a média.

## 2. Intuição

Uma árvore profunda é poderosa, mas instável. Se treinarmos várias árvores ligeiramente diferentes, seus erros individuais não serão idênticos. A média pode cancelar parte desse ruído.

```text
árvore 1 → 100
árvore 2 → 120
árvore 3 → 105
árvore 4 → 115
             ↓
média       110
```

## Bagging

Cada árvore recebe uma amostra bootstrap: sorteio com reposição do conjunto de treino. Além disso, em cada divisão a Random Forest considera apenas um subconjunto de features. Isso **descorrelaciona** as árvores.

Se todas fossem idênticas, fazer média não traria grande ganho.

## 3. Quando usar?

- dados tabulares;
- baseline forte não linear;
- relações com interações;
- quando você quer pouco preprocessing;
- quando interpretabilidade global aproximada é suficiente.

## 4. Vantagens

- robusta e relativamente fácil de usar;
- pouca necessidade de scaling;
- reduz variance da árvore isolada;
- captura relações complexas.

## 5. Limitações

- maior uso de memória/CPU;
- modelo menos transparente que uma árvore pequena;
- não extrapola bem para targets além das regiões vistas;
- feature importance padrão pode enganar;
- em datasets muito grandes, boosting otimizado pode ser mais eficiente/preciso.

## 6. Código

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    max_features=1.0,
    random_state=42,
    n_jobs=-1
)

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

print("MAE CV:", -scores.mean())
```

## Hiperparâmetros 80/20

### `n_estimators`

Número de árvores. Mais árvores normalmente reduzem variância da estimativa até estabilizar, ao custo de processamento.

### `max_depth`

Controla profundidade individual.

### `min_samples_leaf`

Regulariza folhas pequenas.

### `max_features`

Controla quantas features cada split considera. Menos features aumentam diversidade entre árvores, mas podem enfraquecer cada árvore.

## OOB score

Como cada árvore não vê todas as observações no bootstrap, amostras “out-of-bag” podem fornecer uma estimativa adicional de desempenho.

```python
model = RandomForestRegressor(
    n_estimators=300,
    oob_score=True,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print(model.oob_score_)
```

Use OOB como ferramenta útil, não como substituto automático de toda estratégia de validação.

## Random Forest vs árvore única

O ponto principal é o **trade-off bias–variance**. Uma floresta mantém árvores flexíveis, mas reduz variance pela agregação.

## Experimente

1. fixe `n_estimators=300`;
2. teste `min_samples_leaf = 1, 2, 5, 10, 20`;
3. compare MAE de treino e CV;
4. observe o ponto em que regularização começa a causar underfitting.

## Anki

- Por que bootstrap e `max_features` ajudam Random Forest?
- O que aconteceria se todas as árvores fossem altamente correlacionadas?
- `n_estimators` maior sempre causa mais overfitting?
- Qual hiperparâmetro você usaria para evitar folhas extremamente específicas?
