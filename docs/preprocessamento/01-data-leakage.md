# Data leakage

**Prioridade: Essencial**

Data leakage acontece quando o processo de treinamento recebe informação que **não estaria legitimamente disponível no momento em que uma previsão real fosse feita**. O resultado típico é um desempenho de validação artificialmente alto que desaparece em produção.

## A pergunta que detecta leakage

> **Eu conseguiria calcular esta informação exatamente assim no instante da previsão?**

Se a resposta for não, investigue.

## Tipo 1 — preprocessing antes do split

### Errado

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # aprende média/desvio do dataset inteiro

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

O scaler viu as observações que depois serão teste.

### Correto

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

A distinção:

- `fit`: aprende média e desvio;
- `transform`: aplica os valores já aprendidos;
- `fit_transform`: aprende + aplica.

## Tipo 2 — target leakage

Queremos prever se um cliente cancelará.

Features:

```text
idade
renda
uso_30d
motivo_cancelamento  ← problema
```

`motivo_cancelamento` só existe depois de o evento ocorrer. O modelo praticamente recebe parte da resposta.

Outros exemplos:

- prever fraude usando `status_contestacao_final`;
- prever óbito usando um procedimento realizado depois do desfecho;
- prever atraso usando multa que só é gerada após o atraso.

## Tipo 3 — leakage temporal

Previsão feita em janeiro usando informação de fevereiro.

```text
PREVISÃO
   ↓
Jan | Fev
      ↑
feature calculada com futuro
```

Pode acontecer de forma sutil em agregações:

```sql
AVG(valor) OVER (PARTITION BY cliente)
```

Se essa média utiliza transações posteriores à data de corte, há leakage.

A feature correta precisa de janela temporal:

```text
valor_medio_30d_antes_da_data_de_corte
```

## Tipo 4 — duplicatas e entidades compartilhadas

Você possui várias mensagens da mesma conversa e faz split por linha. Mensagens quase idênticas podem cair em treino e teste.

O modelo aparentemente generaliza para “novos exemplos”, mas na prática está vendo variações da mesma entidade.

Solução possível: split por `id_sessao`, cliente ou grupo relevante.

## Tipo 5 — leakage dentro de cross-validation

Este erro é muito comum:

```python
X_scaled = StandardScaler().fit_transform(X_train)

scores = cross_val_score(model, X_scaled, y_train, cv=5)
```

O scaler foi ajustado usando todos os exemplos de `X_train`, incluindo observações que serão validação em cada fold.

### Solução: Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000))
])

scores = cross_val_score(pipe, X_train, y_train, cv=5)
```

Agora, em cada fold, o scaler é ajustado apenas na parte de treino daquele fold.

## Por que o leakage é tão perigoso?

Porque ele não costuma gerar um erro de execução. Pelo contrário: **o código roda e o score fica ótimo**.

Sinais de alerta:

- performance improvável;
- feature extremamente preditiva sem explicação causal/operacional;
- queda enorme em produção;
- score despenca ao fazer split por tempo ou entidade;
- uma feature só existe depois do target.

## Exercício mental

Você quer prever, na abertura de um chamado, se ele será reaberto nos próximos 7 dias. Quais features são suspeitas?

1. canal de abertura;
2. categoria prevista na abertura;
3. quantidade de reaberturas futuras;
4. duração final do chamado;
5. histórico de reaberturas anteriores do cliente.

??? success "Resposta"
    3 e 4 são leakages óbvios. 5 pode ser legítima se calculada apenas com eventos anteriores à abertura atual. 1 e 2 são geralmente disponíveis no instante correto.

## Perguntas para Anki

- Defina leakage sem usar a palavra “vazamento”.
- Por que `fit_transform` no dataset inteiro antes do split é incorreto?
- Como `Pipeline` evita leakage em cross-validation?
- Qual a diferença entre target leakage e temporal leakage?
- Por que split por entidade pode ser necessário mesmo sem uma variável temporal?
