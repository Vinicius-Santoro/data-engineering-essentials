# K-Means e K-Medoids

**Prioridade: Essencial para K-Means · Reconhecer K-Medoids**

## K-Means — as seis perguntas

### 1. O que faz?

Divide observações em `K` grupos, minimizando aproximadamente a soma das distâncias quadráticas aos centroides.

### 2. Intuição

O algoritmo alterna:

```text
1. escolha centros
2. atribua cada ponto ao centro mais próximo
3. recalcule cada centro como média do cluster
4. repita até estabilizar
```

### 3. Quando usar?

Grupos compactos, features numéricas e distância Euclidiana com significado razoável.

### 4. Vantagens

Rápido, simples e escalável.

### 5. Limitações

- precisa escolher `K`;
- sensível a escala;
- sensível a outliers;
- favorece clusters aproximadamente esféricos e de variâncias/tamanhos relativamente comparáveis;
- inicialização pode levar a soluções locais.

### 6. Código

```python
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, _ = make_blobs(
    n_samples=800,
    centers=4,
    cluster_std=1.2,
    random_state=42
)

X_scaled = StandardScaler().fit_transform(X)

model = KMeans(n_clusters=4, n_init="auto", random_state=42)
labels = model.fit_predict(X_scaled)

print("inertia:", model.inertia_)
print("silhouette:", silhouette_score(X_scaled, labels))
```

## Escolhendo K

### Elbow

Plote inertia para vários `K` e procure uma mudança de inclinação.

```python
inertias = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
```

O “cotovelo” pode ser subjetivo.

### Silhouette

Compare score para diferentes `K`, mas confirme se a solução é interpretável e estável.

## Centroides

Após scaling, centroides estão no espaço escalado. Para interpretar em unidade original:

```python
centros_originais = scaler.inverse_transform(model.cluster_centers_)
```

Isso é excelente para construir perfis de segmentos.

# K-Medoids

K-Medoids escolhe uma observação real como representante (medoid) em vez da média geométrica.

Vantagens conceituais:

- pode usar outras métricas de distância;
- tende a ser mais robusto a outliers que K-Means;
- representante é uma observação real.

`scikit-learn` principal não inclui KMedoids; uma implementação conhecida existe em `scikit-learn-extra`, mas o conceito é mais importante que instalar outra biblioteca agora.

## Anki

- Quais são as duas etapas repetidas do K-Means?
- Por que K-Means é sensível a outliers?
- O que é inertia?
- Como K-Medoids difere de K-Means?
