# Naive Bayes

**Prioridade: Importante**

Naive Bayes usa o Teorema de Bayes e assume independência condicional entre features dado a classe. A hipótese é “ingênua”, mas o modelo pode funcionar surpreendentemente bem, especialmente em texto.

## 1. O que faz?

Queremos:

\[
P(y|x_1,\dots,x_p)
\]

Pelo Teorema de Bayes:

\[
P(y|x) \propto P(y)P(x|y)
\]

Com a hipótese naive:

\[
P(x|y) = \prod_j P(x_j|y)
\]

## 2. Intuição

Para classificar uma mensagem como spam, o modelo combina evidências como presença/frequência de palavras, assumindo simplificadamente que cada evidência contribui independentemente condicionada à classe.

## 3. Quando usar?

- classificação de texto;
- datasets pequenos;
- baseline extremamente rápido;
- features de contagem ou contínuas dependendo da variante.

## 4. Vantagens

- muito rápido;
- funciona bem em alta dimensionalidade esparsa;
- precisa de relativamente poucos dados;
- implementação simples.

## 5. Limitações

- independência condicional raramente é literalmente verdadeira;
- probabilidades podem ser extremas/mal calibradas;
- escolha da distribuição adequada importa.

## 6. Variantes

### MultinomialNB

Comum para contagens/TF-IDF não negativo.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
    ("nb", MultinomialNB(alpha=1.0))
])
model.fit(textos_train, y_train)
```

### GaussianNB

Assume distribuição Gaussiana por feature condicionada à classe.

```python
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
```

### BernoulliNB

Útil com features binárias.

## `alpha`

Em MultinomialNB, smoothing evita probabilidade zero para termos não observados em uma classe.

## Texto: por que funciona?

Mesmo que palavras sejam dependentes, a tarefa de **ranking/classificação** pode ser boa quando padrões de frequência distinguem classes. Um modelo pode prever bem mesmo com hipóteses generativas simplificadas.

## Anki

- Qual é a hipótese “naive”?
- Por que MultinomialNB combina bem com bag-of-words/TF-IDF?
- Para que serve smoothing `alpha`?
- Qual a diferença básica entre MultinomialNB e GaussianNB?
