# Treino, validação e teste

**Prioridade: Essencial**

Separar dados não é burocracia. É uma tentativa de simular a situação que realmente importa: **o modelo encontrando observações que não viu durante o aprendizado**.

## O problema de avaliar no próprio treino

```python
modelo.fit(X, y)
pred = modelo.predict(X)
```

Esse score mede o quanto o modelo se ajustou aos próprios exemplos. Um modelo extremamente flexível pode simplesmente memorizar padrões acidentais.

## Train e test

A versão mais simples:

```text
100% dos dados
├── 80% treino → ajustar o modelo
└── 20% teste  → estimar desempenho em dados novos
```

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

`random_state` permite repetir a mesma divisão. Ele não melhora o modelo; melhora **reprodutibilidade**.

## Por que `stratify=y`?

Em classificação, especialmente com classes desbalanceadas, queremos preservar aproximadamente a proporção das classes.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Se 10% dos exemplos são positivos, stratification ajuda a manter perto de 10% no treino e no teste.

## Train, validation e test

Quando você usa o mesmo teste repetidamente para escolher modelo e hiperparâmetros, ele deixa de ser um teste realmente “final”. Você começa, indiretamente, a otimizar decisões para ele.

Uma divisão conceitual:

```text
TREINO      → aprender parâmetros
VALIDAÇÃO   → escolher modelo/hiperparâmetros/threshold
TESTE       → uma avaliação final, depois das decisões
```

Na prática, **cross-validation** frequentemente cumpre o papel de validação dentro do conjunto de treino.

## Exemplo correto de processo

```text
Dados disponíveis
↓
separe teste final
↓
no restante: cross-validation + tuning
↓
escolha pipeline
↓
refit no treino completo
↓
avalie uma vez no teste final
```

## Aleatório nem sempre é correto

### Dados temporais

Se você prevê o futuro, não deve treinar em 2026 e testar em 2024.

```text
2023 | 2024 | 2025 | 2026
TREINO──────────────│ TESTE
```

### Usuários repetidos

Se o mesmo cliente aparece em várias linhas, um split aleatório pode colocar interações quase idênticas do mesmo cliente em treino e teste. O score pode ficar otimista.

Considere `GroupKFold`, `GroupShuffleSplit` ou uma separação por entidade.

### Dados geográficos / lojas / hospitais

O mesmo princípio vale: se a produção exige generalizar para uma nova unidade, a validação deve simular esse cenário.

## O split deve imitar produção

Não existe “80/20” magicamente correto. A pergunta é:

> **Que tipo de novidade o modelo encontrará em produção?**

- novas linhas aleatórias do mesmo processo → split aleatório pode servir;
- futuro → split temporal;
- novos usuários → split por usuário;
- novas lojas → split por loja.

## Mini-experimento

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)

for seed in [1, 7, 42, 99]:
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    print(seed, y_test.mean())
```

Observe que a composição exata muda, mesmo com stratification. Isso ajuda a entender por que um único split pode produzir uma estimativa instável.

## Perguntas para Anki

- Qual a função do conjunto de teste?
- Por que usar o teste repetidamente para tuning é problemático?
- Quando `stratify=y` ajuda?
- Por que um split aleatório pode ser inválido em séries temporais?
- O que significa dizer que o split deve simular produção?
