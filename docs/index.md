# Ciência de Dados 80/20

<div class="hero" markdown>

## Uma base para **entender, praticar e lembrar**

Este material não tenta transformar cada tópico de Ciência de Dados em uma disciplina acadêmica completa. O objetivo é diferente: dominar o pequeno conjunto de ideias que explica grande parte do trabalho prático com Machine Learning — e praticá-las o suficiente para que deixem de parecer uma coleção de comandos.

</div>

!!! tip "Como usar"
    Não leia como um livro que precisa ser terminado antes de abrir o Python. O ciclo recomendado é **ler → prever o que vai acontecer → executar → explicar com suas palavras → revisar no Anki**.

## O mapa

<div class="grid cards" markdown>

-   :material-map-outline:{ .lg .middle } **Fundamentos**

    ---

    Aprenda a pensar em `X`, `y`, treino, validação, teste, generalização, overfitting e pipeline.

-   :material-filter-cog-outline:{ .lg .middle } **Preparação dos dados**

    ---

    Data leakage, missing values, encoding, scaling, `Pipeline` e `ColumnTransformer`.

-   :material-chart-bell-curve-cumulative:{ .lg .middle } **Modelos supervisionados**

    ---

    Regressão Linear, Logística, árvores, Random Forest, Gradient Boosting, KNN, SVM e Naive Bayes.

-   :material-ruler-square-compass:{ .lg .middle } **Avaliação**

    ---

    Métricas, threshold, cross-validation, tuning, feature selection e interpretação.

-   :material-dots-hexagon:{ .lg .middle } **Não supervisionado**

    ---

    K-Means, DBSCAN, clustering hierárquico, PCA e outras reduções.

-   :material-chart-timeline-variant:{ .lg .middle } **Séries temporais**

    ---

    Validação temporal, ARIMA, SARIMA, SARIMAX, Holt-Winters e ML com lags.

-   :material-gamepad-variant-outline:{ .lg .middle } **Reinforcement Learning**

    ---

    Estado, ação, recompensa, retorno, Q-Learning, policy gradients e PPO no nível 80/20.

-   :material-flask-outline:{ .lg .middle } **Laboratórios**

    ---

    Projetos pequenos ponta a ponta para praticar o processo, não apenas a sintaxe.

</div>

## A regra das seis perguntas

Para cada algoritmo, tente responder sem consultar:

1. **O que ele faz?**
2. **Como funciona intuitivamente?**
3. **Quando eu usaria?**
4. **Qual sua principal vantagem?**
5. **Quais limitações e cuidados?**
6. **Como eu o executo e avalio?**

Se você consegue responder as seis e implementar um exemplo simples, já possui uma representação mental útil do algoritmo.

## Prioridades

| Sinal | Significado |
|---|---|
| **Essencial** | precisa entender e praticar várias vezes |
| **Importante** | precisa reconhecer, explicar e implementar com consulta |
| **Reconhecer** | entenda a ideia e saiba onde se encaixa |

!!! success "Meta realista"
    O objetivo não é decorar APIs. É reconhecer **o problema**, escolher um **pipeline defensável**, treinar um **baseline**, avaliar com a **métrica certa** e diagnosticar o que está errado.

## Ambiente sugerido para os códigos

```bash
python -m venv .venv
source .venv/Scripts/activate  # Git Bash no Windows
pip install numpy pandas matplotlib scikit-learn statsmodels shap
```

Os exemplos privilegiam datasets embutidos no `scikit-learn` ou dados sintéticos para que você consiga executá-los sem baixar arquivos externos.

## Comece aqui

A sequência recomendada é:

**Fundamentos → Preparação dos dados → Avaliação → Classificação/Regressão → Não supervisionado → Séries temporais → RL → Laboratórios.**

Você pode estudar os algoritmos antes da avaliação, mas entender **split, leakage e métricas** cedo evita decorar práticas erradas.
