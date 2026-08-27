# O mapa de Machine Learning

**Prioridade: Essencial**

Machine Learning é um conjunto de métodos que aprende padrões a partir de dados para produzir previsões, decisões, representações ou políticas de ação. A melhor forma de não se perder é começar pela **pergunta que você quer responder**, e não pelo nome do algoritmo.

## Quatro famílias úteis

| Família | Existe `y` conhecido? | Exemplo |
|---|---:|---|
| Classificação | sim | fraude ou não fraude |
| Regressão | sim | valor esperado de uma venda |
| Não supervisionado | não | segmentar clientes |
| Reforço | recompensa, não rótulo por linha | aprender uma política de ações |

Séries temporais atravessam essa divisão: uma previsão temporal pode usar modelos estatísticos próprios ou algoritmos supervisionados, mas a **ordem do tempo** muda a forma correta de validar.

## O que são `X` e `y`?

Considere uma tabela:

| idade | renda | atraso_30d | inadimplente |
|---:|---:|---:|---:|
| 24 | 3200 | 0 | 0 |
| 51 | 9800 | 2 | 1 |
| 32 | 6100 | 0 | 0 |

Se queremos prever `inadimplente`:

```text
X = idade, renda, atraso_30d
y = inadimplente
```

- **feature**: variável utilizada como entrada;
- **target/label**: variável que queremos prever;
- **amostra/observação**: uma linha;
- **modelo**: função parametrizada que mapeia `X → previsão`.

## Aprender significa ajustar parâmetros

Uma regressão linear pode aprender coeficientes. Uma árvore aprende regras de divisão. Uma rede neural aprende pesos. KNN é um caso interessante: ele “treina” principalmente armazenando exemplos e faz grande parte do trabalho durante a predição.

O padrão do `scikit-learn` torna os modelos parecidos por fora:

```python
modelo.fit(X_train, y_train)     # aprende
pred = modelo.predict(X_test)    # usa o que aprendeu
```

Essa uniformidade é intencional: você deve enxergar os algoritmos como componentes intercambiáveis dentro de um processo maior.

## Treinar não é o objetivo

Um modelo pode memorizar perfeitamente o treino e falhar em dados novos. Portanto, a pergunta real é:

> **O padrão aprendido generaliza para observações que não participaram do ajuste?**

É por isso que treino/teste, cross-validation e prevenção de leakage são tão importantes.

## Um baseline antes de complexidade

Imagine classificação binária com 70% de classe `0`. Um classificador que sempre responde `0` alcança 70% de accuracy sem aprender nada útil.

Antes de celebrar um score, compare com algo simples:

```python
from sklearn.dummy import DummyClassifier

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
print(baseline.score(X_test, y_test))
```

O baseline responde: **meu modelo sofisticado realmente acrescentou valor?**

## Aprendizado supervisionado em três frases

1. Você possui exemplos com entrada e resposta conhecida.
2. O algoritmo ajusta parâmetros para reduzir um erro no treino.
3. Você avalia em dados não utilizados naquele ajuste para estimar generalização.

## Exemplo mínimo

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

modelo = LogisticRegression(max_iter=5000)
modelo.fit(X_train, y_train)

pred = modelo.predict(X_test)
print(accuracy_score(y_test, pred))
```

!!! warning "Ainda não copie isso como receita universal"
    Logistic Regression normalmente se beneficia de scaling, e accuracy pode ser inadequada em classes desbalanceadas. Esses detalhes aparecem nos próximos capítulos. O exemplo serve apenas para mostrar o fluxo.

## Perguntas de recuperação

1. Por que `fit` e `predict` são operações conceitualmente diferentes?
2. Qual a diferença entre feature e target?
3. Por que um score alto no treino não garante um bom modelo?
4. Qual o objetivo de um baseline?
5. Se não existe `y`, classificação ainda é uma formulação adequada?

??? success "Respostas curtas"
    1. `fit` aprende parâmetros; `predict` aplica o modelo ajustado.  
    2. Feature é entrada; target é a resposta a prever.  
    3. O modelo pode ter memorizado particularidades do treino.  
    4. Estabelecer um nível mínimo de desempenho a superar.  
    5. Não no sentido supervisionado tradicional; considere métodos não supervisionados ou reformule o problema.
