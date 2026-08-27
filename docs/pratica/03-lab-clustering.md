# Laboratório — clustering e PCA

**Objetivo:** sentir na prática como scaling, escolha de `K` e geometria alteram um problema não supervisionado.

## 1. Gere dados

```python
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, _ = make_blobs(
    n_samples=1200,
    centers=4,
    cluster_std=[0.8, 1.0, 1.4, 0.7],
    random_state=42
)

plt.scatter(X[:,0], X[:,1], s=10)
plt.show()
```

Como os dados são sintéticos, sabemos que quatro centros foram usados — mas finja que você não sabe durante os primeiros testes.

## 2. Scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Para criar um experimento mais instrutivo, distorça uma dimensão:

```python
X_bad_scale = X.copy()
X_bad_scale[:, 0] *= 100
```

Compare clustering com e sem scaling em `X_bad_scale`.

## 3. K-Means em diferentes K

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    print(k, round(km.inertia_,2), round(sil,3))
```

Perguntas:

- qual `K` maximiza silhouette?
- inertia sempre diminui quando K cresce? Por quê?
- o maior K é automaticamente melhor?

## 4. Visualize

```python
km = KMeans(n_clusters=4, n_init="auto", random_state=42)
labels = km.fit_predict(X_scaled)

plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels, s=10)
plt.scatter(
    km.cluster_centers_[:,0],
    km.cluster_centers_[:,1],
    marker="X",
    s=200
)
plt.show()
```

## 5. DBSCAN

```python
from sklearn.cluster import DBSCAN

for eps in [0.15, 0.25, 0.35, 0.50]:
    labels = DBSCAN(eps=eps, min_samples=8).fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = (labels == -1).mean()
    print(eps, n_clusters, round(noise,3))
```

Observe como pequenas mudanças em `eps` alteram a solução.

## 6. Dados em luas

K-Means funciona melhor em geometria compacta. Crie um caso em que DBSCAN brilha:

```python
from sklearn.datasets import make_moons

X_moons, _ = make_moons(n_samples=800, noise=0.06, random_state=42)
X_moons = StandardScaler().fit_transform(X_moons)

km_labels = KMeans(n_clusters=2, n_init="auto", random_state=42).fit_predict(X_moons)
db_labels = DBSCAN(eps=0.18, min_samples=8).fit_predict(X_moons)
```

Plote os dois resultados. A diferença visual ajuda a fixar a hipótese geométrica do K-Means.

## 7. PCA em Wine

```python
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA

X_wine, y_wine = load_wine(return_X_y=True)
X_wine_scaled = StandardScaler().fit_transform(X_wine)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_wine_scaled)

print(pca.explained_variance_ratio_)
print("total", pca.explained_variance_ratio_.sum())

plt.scatter(X_pca[:,0], X_pca[:,1], c=y_wine)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
```

Aqui usamos `y` apenas para colorir e interpretar visualmente; PCA não recebeu `y` no `fit_transform`.

## 8. PCA + classificação

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
    ("model", LogisticRegression(max_iter=3000))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(cross_val_score(pipe, X_wine, y_wine, cv=cv, scoring="accuracy"))
```

PCA fica dentro do pipeline porque deve ser ajustado apenas no treino de cada fold.

## Experimentos obrigatórios

- [ ] multiplique uma feature por 1.000 e veja K-Means sem scaling;
- [ ] compare K-Means e DBSCAN em `make_moons`;
- [ ] plote silhouette por K;
- [ ] execute PCA sem scaling no Wine e compare variância/componentes;
- [ ] compare Logistic Regression com e sem PCA usando a mesma CV.
