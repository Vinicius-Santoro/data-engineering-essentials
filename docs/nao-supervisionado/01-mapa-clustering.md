# O mapa de clustering

**Prioridade: Essencial**

Clustering procura estrutura nos dados **sem um target fornecido**. A pergunta deixa de ser “qual classe devo prever?” e passa a ser “existem grupos úteis segundo uma definição de similaridade?”.

## Clustering não descobre uma verdade única

Os mesmos clientes podem ser agrupados por:

- comportamento transacional;
- renda e patrimônio;
- canais digitais;
- produtos contratados;
- risco;
- combinação desses aspectos.

Cada escolha de features e distância muda a noção de “parecido”. Portanto, clusters são **construções analíticas**, não categorias naturais garantidas.

## Três famílias do 80/20

| Algoritmo | Ideia | Bom quando |
|---|---|---|
| K-Means | centros e distância | grupos aproximadamente compactos/esféricos |
| DBSCAN | densidade | clusters irregulares + ruído |
| Hierárquico | árvore de fusões | explorar estrutura em níveis |

K-Medoids lembra K-Means, mas usa observações reais como centros e pode ser mais robusto a outliers dependendo da distância.

## Scaling muda o significado

Se clustering usa distância e as features são:

```text
idade: 18–80
renda: 1.000–200.000
frequencia: 0–30
```

renda pode dominar. Um pipeline típico:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("cluster", KMeans(n_clusters=4, random_state=42, n_init="auto"))
])
labels = pipe.fit_predict(X)
```

## Como avaliar sem `y`?

Não existe uma única métrica universal. Algumas opções internas:

- silhouette score;
- Davies–Bouldin;
- inertia (K-Means);
- estabilidade dos grupos;
- separação visual após redução dimensional.

Mais importante: **utilidade e interpretabilidade no domínio**.

Um cluster matematicamente separado pode ser inútil para a ação de negócio.

## Silhouette

Para cada ponto, compara proximidade ao próprio cluster com proximidade ao cluster vizinho.

Valores próximos de 1 sugerem boa separação, perto de 0 indicam fronteira e negativos sugerem possível alocação ruim.

```python
from sklearn.metrics import silhouette_score
score = silhouette_score(X_scaled, labels)
```

## Cuidado com escolher features pelo resultado

Se você tenta dezenas de transformações até conseguir clusters “bonitos”, pode criar uma narrativa pós-hoc. Defina objetivo, população e variáveis com justificativa.

## Perguntas para Anki

- Por que clustering não tem uma “resposta correta” única?
- Qual papel do scaling?
- Como K-Means e DBSCAN diferem conceitualmente?
- Por que silhouette não basta para validar segmentação de negócio?
