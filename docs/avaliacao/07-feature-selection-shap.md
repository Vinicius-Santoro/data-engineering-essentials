# Feature selection, importância e SHAP

**Prioridade: Importante**

Depois de treinar um modelo, surgem duas perguntas diferentes:

1. **quais features ajudam a previsão?**
2. **como uma feature influenciou uma previsão específica?**

Feature selection e interpretabilidade respondem partes diferentes dessas perguntas.

## Por que selecionar features?

- reduzir ruído;
- acelerar treinamento/inferência;
- reduzir risco de overfitting em alguns cenários;
- simplificar coleta e manutenção;
- melhorar interpretabilidade.

Mas remover features úteis também pode piorar performance.

## Métodos filter

Avaliam features antes/fora do modelo principal, por exemplo correlação, testes univariados ou mutual information.

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif

selector = SelectKBest(mutual_info_classif, k=10)
```

Coloque dentro do pipeline para evitar leakage.

## Métodos embedded

O próprio treinamento faz alguma seleção/regularização.

- Lasso / L1;
- árvores;
- alguns modelos com penalização.

```python
LogisticRegression(penalty="l1", solver="liblinear", C=0.1)
```

## Recursive Feature Elimination

Remove features iterativamente com base no estimador.

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

rfe = RFE(LogisticRegression(max_iter=2000), n_features_to_select=10)
```

Custo pode ser alto.

## Permutation importance

Pergunta: **quanto a métrica piora se destruirmos a informação desta feature?**

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=20,
    random_state=42
)
```

Se duas features carregam informação redundante, embaralhar apenas uma pode parecer pouco importante porque a outra continua disponível.

## SHAP — ideia 80/20

SHAP usa ideias de valores de Shapley da teoria dos jogos para atribuir a cada feature uma contribuição relativa à previsão, com propriedades matemáticas específicas.

Conceitualmente:

```text
valor base
+ contribuição idade
+ contribuição renda
+ contribuição atraso
= saída explicada
```

### TreeExplainer

```python
import shap

explainer = shap.TreeExplainer(rf)
shap_values = explainer(X_sample)
shap.plots.beeswarm(shap_values)
```

A API exata pode variar conforme versão/modelo; consulte a documentação instalada ao usar.

## O que SHAP não é

- prova de causalidade;
- garantia de que a feature é “boa”;
- solução automática para leakage;
- interpretação independente de correlação entre features.

Uma feature vazada pode ter SHAP altíssimo — isso só mostra que o modelo a usa.

## Global vs local

### Global

Quais features geralmente têm maiores contribuições? Como a direção varia no dataset?

### Local

Por que esta observação específica recebeu score 0.82?

Ambos são úteis, mas respondem perguntas diferentes.

## Seleção dentro de CV

Errado:

```python
selector.fit(X, y)
X2 = selector.transform(X)
cross_val_score(model, X2, y, cv=5)
```

O seletor viu targets da validação.

Correto: seletor no `Pipeline`.

## Anki

- Qual diferença entre feature selection e interpretabilidade?
- Como permutation importance funciona intuitivamente?
- Por que features correlacionadas complicam importância?
- SHAP prova causalidade?
- Por que seleção supervisionada precisa ficar dentro da CV?
