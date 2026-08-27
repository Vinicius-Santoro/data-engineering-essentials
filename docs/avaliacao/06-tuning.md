# Grid Search, Randomized Search e tuning

**Prioridade: Essencial**

Tuning escolhe hiperparâmetros usando uma estratégia de validação. A regra mais importante é: **hiperparâmetro é escolhido na validação, não no teste final**.

## Parâmetro vs hiperparâmetro

- **parâmetro aprendido**: coeficiente da regressão, splits da árvore;
- **hiperparâmetro**: `C`, `max_depth`, `n_estimators`, `k`.

O algoritmo aprende parâmetros durante `fit`. Você define/seleciona hiperparâmetros externamente.

## GridSearchCV

Experimenta combinações explícitas.

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

params = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__class_weight": [None, "balanced"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = GridSearchCV(
    pipe,
    param_grid=params,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    return_train_score=True
)

search.fit(X_train, y_train)
print(search.best_params_)
print(search.best_score_)
```

## O que acontece internamente?

Para cada combinação:

```text
combinação A → CV fold 1...5 → média
combinação B → CV fold 1...5 → média
combinação C → CV fold 1...5 → média
...
melhor média → selecionada
```

Depois, por padrão, `refit=True` treina o melhor pipeline usando todo `X_train`.

## Teste final

```python
best = search.best_estimator_
prob_test = best.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_test, prob_test))
```

Só agora usamos o teste congelado.

## RandomizedSearchCV

Quando o espaço é grande, amostra combinações aleatórias.

```python
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV

params = {
    "model__C": loguniform(1e-4, 1e2),
    "model__gamma": loguniform(1e-5, 1)
}
```

É especialmente útil para hiperparâmetros contínuos em várias ordens de magnitude.

## Tuning demais também overfita a validação

Se você executa centenas de experimentos e sempre escolhe pelo mesmo conjunto/mesma CV, existe “overfitting do processo de seleção”. O teste final ajuda a revelar isso.

Em avaliações rigorosas, **nested cross-validation** separa uma CV interna para tuning e externa para estimar desempenho.

## Nested CV — ideia

```text
Outer Fold
├── treino outer
│   └── Inner CV → escolhe hiperparâmetros
└── validação outer → avalia escolha
```

É mais caro e não precisa ser seu primeiro reflexo em todos os projetos, mas é importante saber que existe.

## Não tune tudo

O 80/20:

- Logistic: `C`, talvez `class_weight`;
- árvore: `max_depth`, `min_samples_leaf`;
- Random Forest: `min_samples_leaf`, `max_features`, `max_depth`, `n_estimators` suficiente;
- boosting: `learning_rate`, complexidade das árvores, número de iterações;
- KNN: `n_neighbors`, pesos;
- SVM: `C`, `gamma`, kernel.

Comece com poucos hiperparâmetros de alto impacto.

## Leia `cv_results_`

```python
import pandas as pd

results = pd.DataFrame(search.cv_results_)
cols = ["params", "mean_train_score", "mean_test_score", "std_test_score"]
print(results[cols].sort_values("mean_test_score", ascending=False).head(10))
```

Compare treino vs validação e estabilidade, não apenas `best_params_`.

## Anki

- Qual a diferença entre parâmetro e hiperparâmetro?
- Por que não tune usando teste final?
- Quando RandomizedSearch é melhor que GridSearch?
- O que `refit=True` faz?
- O que nested CV tenta estimar?
