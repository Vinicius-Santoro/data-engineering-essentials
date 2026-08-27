# DBSCAN e clustering hierárquico

**Prioridade: Importante**

# DBSCAN

DBSCAN agrupa regiões densas e marca pontos isolados como ruído.

## As seis perguntas

**O que faz?** encontra componentes conectados por densidade.

**Intuição:** um ponto “core” possui vizinhos suficientes dentro de um raio. Pontos core próximos expandem o cluster; pontos que não se conectam a regiões densas podem virar ruído.

**Quando usar?** clusters de formato irregular, presença de outliers, quando você não quer especificar `K` diretamente.

**Vantagem:** detecta ruído e formas não esféricas.

**Limitações:** `eps` é sensível à escala; densidades muito diferentes são difíceis; alta dimensionalidade degrada distâncias.

**Código:**

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)
labels = DBSCAN(eps=0.35, min_samples=8).fit_predict(X_scaled)
```

Por convenção, label `-1` representa ruído.

## `eps`

Raio máximo para considerar vizinhança. Muito pequeno → muitos pontos viram ruído. Muito grande → clusters diferentes podem se fundir.

## `min_samples`

Quantidade mínima de amostras na vizinhança para caracterizar uma região densa. Valor maior exige densidade mais forte.

# Clustering hierárquico

Cria uma hierarquia de agrupamentos.

## Aglomerativo

Começa com cada ponto como cluster e vai fundindo pares.

```text
A B C D
↓
AB C D
↓
AB CD
↓
ABCD
```

O dendrograma representa essas fusões.

## Linkage

Define distância entre clusters:

- `single`: menor distância entre pontos;
- `complete`: maior distância;
- `average`: média;
- `ward`: busca minimizar aumento de variância e usa distância Euclidiana.

```python
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=4, linkage="ward")
labels = model.fit_predict(X_scaled)
```

## K-Means × DBSCAN × Hierárquico

| Situação | Comece pensando em |
|---|---|
| clusters compactos, dataset grande | K-Means |
| formas irregulares + ruído | DBSCAN |
| estrutura em níveis / dendrograma | hierárquico |

## Anki

- O que `-1` significa no DBSCAN?
- Como `eps` e `min_samples` mudam clusters?
- Qual vantagem DBSCAN tem sobre K-Means?
- O que linkage define no clustering hierárquico?
