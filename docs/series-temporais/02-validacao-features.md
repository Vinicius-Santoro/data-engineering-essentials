# Validação temporal e feature engineering

**Prioridade: Essencial**

Este é o capítulo mais importante de séries temporais. Um modelo excelente com validação errada é pior que um baseline honesto.

## Split errado

```python
train_test_split(X, y, shuffle=True)
```

Se as linhas são tempo, embaralhar permite treinar em observações posteriores e avaliar em anteriores.

## Holdout temporal

```text
2024          2025          2026
|------ TREINO ------|------ TESTE ------|
```

```python
train = df[df.index < "2026-01-01"]
test  = df[df.index >= "2026-01-01"]
```

## Walk-forward / backtesting

Avalie vários cortes temporais:

```text
Fold 1: [train] [val]
Fold 2: [----train----] [val]
Fold 3: [--------train--------] [val]
```

```python
from sklearn.model_selection import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=5)
```

Em aplicações reais, configure janela/horizonte para imitar a cadência de retreino e previsão.

## Lags como features

Transformamos série em problema supervisionado.

```python
df["lag_1"] = df["y"].shift(1)
df["lag_7"] = df["y"].shift(7)
df["lag_28"] = df["y"].shift(28)
```

## Rolling features

O detalhe crucial é usar **apenas passado**.

Errado:

```python
df["media_7"] = df["y"].rolling(7).mean()
```

Na linha `t`, isso inclui `y_t`. Se `y_t` é justamente o target a prever, há leakage.

Seguro para previsão de `y_t`:

```python
df["media_7"] = df["y"].shift(1).rolling(7).mean()
```

Primeiro deslocamos; depois calculamos a janela.

## Features de calendário

```python
df["dia_semana"] = df.index.dayofweek
df["mes"] = df.index.month
df["fim_semana"] = (df.index.dayofweek >= 5).astype(int)
```

Datas podem ser conhecidas no futuro e portanto são legítimas, desde que o calendário real esteja disponível.

## Features exógenas

Temperatura, preço, campanha e feriados podem ajudar. Mas pergunte:

> Eu conheço o **valor futuro real** dessa feature no instante da previsão?

Se não, talvez precise usar previsão dessa feature ou valor planejado, não o valor realizado futuro.

## Janela expansiva vs deslizante

### Expansiva

Usa todo histórico disponível.

### Deslizante

Usa apenas os últimos `N` períodos, útil quando dados antigos deixam de representar o regime atual.

## Métricas por horizonte

Um modelo pode ser ótimo em `t+1` e fraco em `t+30`. Avalie por horizonte:

```text
MAE h=1
MAE h=7
MAE h=14
MAE h=30
```

## Anki

- Por que shuffle é perigoso em séries?
- Como criar rolling mean sem leakage?
- Por que feature exógena futura pode vazar?
- O que walk-forward simula?
- Qual diferença entre janela expansiva e deslizante?
