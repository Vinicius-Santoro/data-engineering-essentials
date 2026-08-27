# SVD, Kernel PCA e t-SNE

**Prioridade: Reconhecer / Importante em contexto**

PCA é o núcleo do 80/20. Estas técnicas ampliam o mapa.

# SVD

Singular Value Decomposition fatoriza uma matriz:

\[
X = U\Sigma V^T
\]

Ela está intimamente relacionada ao PCA. Em matrizes esparsas, `TruncatedSVD` é muito útil porque não exige centralização densa.

## Texto / TF-IDF

Uma matriz documento-termo pode ter dezenas de milhares de colunas. `TruncatedSVD` reduz dimensões preservando estrutura linear importante.

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(min_df=2)),
    ("svd", TruncatedSVD(n_components=100, random_state=42))
])
X_reduced = pipe.fit_transform(textos)
```

Essa ideia é associada a Latent Semantic Analysis quando aplicada a texto.

# Kernel PCA

PCA encontra projeções lineares. Kernel PCA usa kernel para capturar estrutura não linear.

```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(
    n_components=2,
    kernel="rbf",
    gamma=0.1
)
X_kpca = kpca.fit_transform(X_scaled)
```

Cuidados:

- tuning de kernel/gamma;
- custo computacional;
- componentes ainda menos interpretáveis;
- raramente é meu primeiro passo em dados tabulares modernos.

# t-SNE

t-SNE é uma técnica não linear focada em visualização, buscando preservar vizinhanças locais.

```python
from sklearn.manifold import TSNE

X_2d = TSNE(
    n_components=2,
    perplexity=30,
    init="pca",
    learning_rate="auto",
    random_state=42
).fit_transform(X_scaled)
```

## O cuidado mais importante com t-SNE

Distâncias globais e tamanhos aparentes entre clusters no mapa 2D podem ser enganosos. O método é excelente para explorar vizinhanças, mas **não trate a figura como uma representação métrica fiel de todo o espaço original**.

`perplexity`, inicialização e random seed podem alterar o mapa.

## UMAP?

UMAP é muito popular para visualização e redução não linear, mas não faz parte do núcleo que combinamos. Vale estudar depois de dominar PCA/t-SNE.

## Comparação

| Técnica | Linear? | Uso 80/20 |
|---|---:|---|
| PCA | sim | compressão + modelo + visualização |
| TruncatedSVD | sim | matrizes esparsas/texto |
| Kernel PCA | não | estrutura não linear específica |
| t-SNE | não | visualização exploratória |

## Anki

- Por que TruncatedSVD é útil em TF-IDF?
- Qual diferença principal entre PCA e Kernel PCA?
- Por que não devemos interpretar distância global de t-SNE literalmente?
