# Gradient Boosting

**Prioridade: Essencial**

Boosting também combina árvores, mas a lógica é diferente de Random Forest. Em vez de treinar árvores independentes e fazer média, ele constrói modelos **sequencialmente**, cada etapa tentando corrigir os erros acumulados.

## 1. O que faz?

Produz uma soma de modelos fracos:

\[
F_M(x) = F_0(x) + \eta h_1(x) + \eta h_2(x) + \cdots + \eta h_M(x)
\]

`η` é o learning rate.

## 2. Intuição

Imagine:

```text
modelo inicial → erra alguns padrões
      ↓
nova árvore aprende parte do erro residual
      ↓
nova árvore corrige o que ainda falta
      ↓
...
```

Cada árvore costuma ser rasa. O poder surge da sequência.

## 3. Quando usar?

- dados tabulares com relações complexas;
- quando performance é prioritária;
- problemas de regressão ou classificação;
- quando você aceita tuning um pouco mais cuidadoso.

## 4. Vantagens

- frequentemente excelente em dados tabulares;
- captura não linearidades e interações;
- flexível em funções de perda.

## 5. Limitações

- mais sensível a hiperparâmetros;
- treinamento sequencial reduz paralelismo conceitual;
- pode overfitar se árvores forem profundas ou houver etapas demais;
- outliers podem afetar certas losses.

## 6. Código

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score

model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=15,
    l2_regularization=1.0,
    random_state=42
)

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_root_mean_squared_error"
)
print("RMSE CV:", -scores.mean())
```

`HistGradientBoosting` é uma implementação eficiente baseada em histogramas dentro do scikit-learn.

## Learning rate × número de árvores

Esses hiperparâmetros interagem:

- learning rate alto: cada etapa corrige muito; convergência rápida, maior risco de passar do ponto;
- learning rate baixo: passos menores; geralmente requer mais árvores/iterações.

Uma heurística comum é reduzir `learning_rate` e compensar com mais estimadores, validando o resultado.

## Random Forest vs Gradient Boosting

| Random Forest | Gradient Boosting |
|---|---|
| árvores relativamente independentes | árvores sequenciais |
| reduz variance por média | corrige resíduos/gradientes |
| tuning geralmente mais simples | tuning mais sensível |
| forte baseline | frequentemente performance superior |

## Residual como ideia

Em squared error, você pode imaginar cada nova árvore aprendendo aproximadamente os resíduos:

\[
r_i = y_i - \hat y_i
\]

O algoritmo geral usa gradientes da loss, por isso “Gradient” Boosting.

## Experimento

Compare:

```text
learning_rate=0.2, max_iter=100
learning_rate=0.05, max_iter=400
learning_rate=0.01, max_iter=1000
```

Não existe vencedor universal. A ideia é sentir o trade-off entre tamanho do passo e número de etapas.

## Anki

- Qual a diferença estrutural entre bagging e boosting?
- Por que learning rate e número de estimadores devem ser pensados juntos?
- O que significa dizer que boosting corrige erros sequencialmente?
- Por que árvores rasas são comuns em boosting?
