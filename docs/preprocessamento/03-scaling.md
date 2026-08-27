# Scaling e transformações

**Prioridade: Essencial**

Scaling muda a escala numérica das features. Ele não é necessário para todos os algoritmos, mas é crítico para modelos que dependem de **distância, magnitude de coeficientes ou otimização sensível à escala**.

## O problema

Considere:

```text
idade:   18 a 80
renda:   1.500 a 100.000
score:   0 a 1
```

Em uma distância Euclidiana, diferenças de renda podem dominar completamente diferenças de idade ou score.

## StandardScaler

Transforma aproximadamente para média 0 e desvio padrão 1:

\[
z = \frac{x - \mu}{\sigma}
\]

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

O `fit` aprende `μ` e `σ` no treino.

## MinMaxScaler

Mapeia para um intervalo, tipicamente `[0,1]`.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
```

É sensível a outliers porque mínimo e máximo podem ser extremos.

## RobustScaler

Usa mediana e intervalo interquartil, sendo mais robusto a outliers.

```python
from sklearn.preprocessing import RobustScaler
```

## Quem costuma precisar?

| Modelo | Scaling? | Motivo |
|---|---|---|
| KNN | **sim** | distância |
| SVM / SVR | **sim** | margem/kernel e otimização |
| Logistic Regression | recomendado | estabilidade e regularização |
| Linear Regression | não obrigatório para previsão | ajuda interpretação/otimização em variantes |
| PCA | geralmente sim | variância depende da escala |
| K-Means | geralmente sim | distância |
| Decision Tree | geralmente não | divisões por limiar |
| Random Forest | geralmente não | árvores |
| Gradient Boosting com árvores | geralmente não | árvores |

## Por que árvores não precisam normalmente?

Uma árvore pergunta algo como:

```text
renda <= 5320?
```

Se renda for transformada linearmente, a ordem das observações continua a mesma; a árvore pode encontrar um limiar equivalente.

## Scaling e regularização

Na Logistic Regression com penalização L2, coeficientes são penalizados por magnitude. Se uma feature está em escala de milhares e outra em décimos, a comparação da penalização fica menos intuitiva. Scaling torna o problema mais equilibrado.

## Experimento KNN

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

sem = KNeighborsClassifier(n_neighbors=5)
sem.fit(X_train, y_train)
print("sem scaling:", accuracy_score(y_test, sem.predict(X_test)))

com = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])
com.fit(X_train, y_train)
print("com scaling:", accuracy_score(y_test, com.predict(X_test)))
```

Não generalize a diferença numérica para todo dataset; use o experimento para observar **por que escala é parte do pipeline de KNN**.

## Outras transformações

Distribuições muito assimétricas podem se beneficiar de `log1p`, PowerTransformer ou quantile transformations, dependendo do modelo e objetivo.

```python
import numpy as np
x_log = np.log1p(x)
```

Transformações não devem ser aplicadas automaticamente: visualize, entenda e valide.

## Perguntas para Anki

- Por que KNN é sensível à escala?
- Por que Decision Tree geralmente não é?
- Qual a diferença intuitiva entre StandardScaler e MinMaxScaler?
- Onde está o leakage se o scaler é ajustado antes do split?
- Por que PCA costuma ser precedido por scaling?
