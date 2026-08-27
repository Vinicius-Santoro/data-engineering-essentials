# Matriz de confusão

**Prioridade: Essencial**

A matriz de confusão é a base para entender métricas de classificação. Antes de decorar fórmulas, aprenda a nomear os quatro resultados possíveis.

## O quadro

Para uma classe positiva `1`:

| | Real positivo | Real negativo |
|---|---:|---:|
| **Prevê positivo** | True Positive (TP) | False Positive (FP) |
| **Prevê negativo** | False Negative (FN) | True Negative (TN) |

## Exemplo: fraude

- **TP**: marcou fraude e era fraude;
- **FP**: bloqueou compra legítima;
- **FN**: deixou fraude passar;
- **TN**: liberou compra legítima.

A palavra “positivo” não significa bom; significa apenas a classe que definimos como positiva.

## Código

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, pred)
print(cm)

ConfusionMatrixDisplay.from_predictions(y_test, pred)
```

No sklearn binário, o arranjo padrão é:

```text
[[TN, FP],
 [FN, TP]]
```

## Accuracy nasce daqui

\[
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
\]

Ela responde: **qual proporção total foi classificada corretamente?**

## Por que accuracy pode enganar?

Dataset:

```text
9.900 transações legítimas
100 fraudes
```

Um modelo que sempre diz “legítima”:

```text
accuracy = 99%
recall de fraude = 0%
```

Excelente accuracy, modelo inútil para detectar fraude.

## Custo dos quadrantes

A matriz deixa o problema de negócio explícito.

### Saúde / triagem

FN pode significar deixar uma condição passar.

### Spam

FP pode significar mandar email legítimo para spam.

### Marketing

FP pode consumir orçamento de contato; FN pode perder oportunidade.

Não existe universalmente “o pior erro”.

## Multiclasse

Com classes A, B e C:

```text
        Pred A Pred B Pred C
Real A
Real B
Real C
```

A diagonal representa acertos. Fora da diagonal você vê **quais classes são confundidas entre si**, informação que uma única accuracy não mostra.

## Normalização

```python
ConfusionMatrixDisplay.from_predictions(
    y_test, pred,
    normalize="true"
)
```

Normalizar por classe real ajuda a ler proporções quando classes têm tamanhos diferentes.

## Antes de calcular Precision/Recall

Escolha explicitamente:

1. qual é a classe positiva;
2. qual erro custa mais;
3. qual capacidade operacional existe;
4. qual threshold foi usado.

## Anki

- Defina TP, FP, FN e TN em um cenário de fraude.
- Por que accuracy pode ser excelente e o modelo inútil?
- O que a diagonal da matriz multiclasse representa?
- Por que a definição da classe positiva importa?
