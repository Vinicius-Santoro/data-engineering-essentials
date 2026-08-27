# K-Nearest Neighbors (KNN)

**Prioridade: Importante**

KNN classifica uma nova observação olhando os rótulos de seus vizinhos mais próximos.

## 1. O que faz?

Para um novo ponto:

1. calcula distância para exemplos de treino;
2. encontra os `k` mais próximos;
3. realiza votação entre suas classes.

## 2. Intuição

Se perfis próximos no espaço de features tendem a compartilhar classe, a vizinhança contém sinal.

```text
● ● ●       × ×
 ● ? ●      × ×
```

Se `?` está cercado por `●`, KNN tende a classificá-lo como `●`.

## 3. Quando usar?

- datasets pequenos/médios;
- baixa/moderada dimensionalidade;
- fronteiras locais;
- baseline intuitivo.

## 4. Vantagens

- simples;
- não impõe fronteira linear;
- treinamento praticamente trivial;
- naturalmente multiclasse.

## 5. Limitações

- scaling essencial;
- predição cara com muitos exemplos;
- sensível a features irrelevantes;
- sofre com alta dimensionalidade;
- classes densas podem dominar.

## 6. Código

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

knn = Pipeline([
    ("scale", StandardScaler()),
    ("model", KNeighborsClassifier(
        n_neighbors=7,
        weights="distance"
    ))
])

knn.fit(X_train, y_train)
pred = knn.predict(X_test)
```

## `k`: bias vs variance

```text
k muito pequeno → fronteira irregular, variance alta
k moderado       → compromisso
k muito grande   → fronteira suave demais, bias alto
```

Escolha por cross-validation.

```python
from sklearn.model_selection import GridSearchCV

params = {"model__n_neighbors": [3, 5, 7, 11, 21, 31]}
search = GridSearchCV(knn, params, cv=5, scoring="f1")
search.fit(X_train, y_train)
print(search.best_params_)
```

## Métricas de distância

Euclidiana é comum, mas não é a única. Em dados específicos, Manhattan ou outras métricas podem fazer sentido. A escolha precisa corresponder ao significado do espaço de features.

## Features categóricas

One-hot + distância Euclidiana é possível, mas nem sempre representa uma noção semântica de similaridade ideal. KNN exige pensar bastante no significado de “perto”.

## Anki

- Por que KNN é chamado de lazy learner?
- Como `k` afeta bias e variance?
- Por que scaling é obrigatório em geral?
- O que acontece em alta dimensionalidade?
