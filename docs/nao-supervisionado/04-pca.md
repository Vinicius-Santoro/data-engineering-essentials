# PCA — Principal Component Analysis

**Prioridade: Essencial**

PCA reduz dimensionalidade criando novas variáveis — **componentes principais** — que capturam direções de maior variância.

## 1. O que faz?

Transforma `p` features correlacionadas em componentes ortogonais ordenados por variância explicada.

```text
x1, x2, x3, x4
      ↓ PCA
PC1, PC2
```

Você perde alguma informação se mantiver poucos componentes, mas ganha uma representação menor.

## 2. Intuição geométrica

Imagine uma nuvem alongada de pontos em 2D. A maior variação ocorre ao longo da direção comprida. PCA gira os eixos para alinhar o primeiro componente com essa direção.

`PC2` é perpendicular ao `PC1` e captura a maior variância restante.

## 3. Quando usar?

- compressão;
- visualização em 2D/3D;
- reduzir multicolinearidade;
- preprocessing para alguns modelos;
- explorar estrutura latente linear.

## 4. Vantagens

- determinístico e rápido;
- componentes ortogonais;
- quantifica variância explicada;
- útil com features correlacionadas.

## 5. Limitações

- componentes são combinações e perdem interpretação direta;
- maximizar variância não significa maximizar utilidade para `y`;
- linear;
- sensível à escala.

## 6. Código

```python
from sklearn.datasets import load_wine
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X, y = load_wine(return_X_y=True)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=2))
])

X_2d = pipe.fit_transform(X)
print(X_2d.shape)
```

## Variância explicada

```python
pca = pipe.named_steps["pca"]
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())
```

Se os dois componentes explicam 65%, então 35% da variância total padronizada ficou nas dimensões descartadas.

## Escolha por proporção

```python
PCA(n_components=0.95)
```

Mantém o número mínimo de componentes necessário para explicar aproximadamente 95% da variância.

## PCA dentro de CV

Se PCA faz parte de um modelo supervisionado, ajuste-o dentro do pipeline/cross-validation. PCA não usa `y`, mas aprende estrutura de `X`; usar dados de validação no fit ainda é leakage de preprocessing.

```python
model = Pipeline([
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
    ("clf", LogisticRegression(max_iter=2000))
])
```

## Loadings

`components_` mostra pesos de features nos componentes. Isso ajuda a interpretar quais variáveis contribuem para cada eixo, embora a interpretação possa continuar abstrata.

## PCA não é feature selection

Feature selection escolhe colunas originais. PCA cria **novas colunas** como combinações lineares.

## Anki

- O que PC1 maximiza?
- Por que PCA geralmente precisa de scaling?
- PCA é supervisionado?
- Por que ainda deve ficar dentro do CV?
- Qual diferença entre PCA e feature selection?
