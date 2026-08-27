# Séries temporais — fundamentos

**Prioridade: Essencial**

Séries temporais mudam uma regra fundamental: **o futuro não pode ajudar a prever o passado**. A ordem dos dados faz parte do problema.

## Componentes úteis

### Tendência

Movimento de longo prazo.

```text
100, 110, 120, 130, 140...
```

### Sazonalidade

Padrão que se repete em intervalo conhecido: dia da semana, mês, hora do dia.

### Ciclo

Flutuações de duração não necessariamente fixa, como ciclos econômicos.

### Ruído

Variações não explicadas pelo padrão modelado.

## Autocorrelação

Uma série pode depender de seus próprios valores anteriores.

```text
y_t relacionado a y_{t-1}, y_{t-7}, y_{t-12}...
```

ACF mede correlação da série com diferentes lags.

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(serie.dropna(), lags=40)
plot_pacf(serie.dropna(), lags=40)
```

## Estacionariedade — intuição

Modelos ARIMA clássicos funcionam melhor quando propriedades como média/estrutura de dependência são aproximadamente estáveis ao longo do tempo após transformações/diferenciação adequadas.

Uma série com tendência forte não é estacionária em média.

## Diferenciação

```text
y_t - y_{t-1}
```

Remove parte de tendências ao modelar mudanças em vez de nível.

```python
serie_diff = serie.diff().dropna()
```

## Baseline temporal

Nunca compare ARIMA apenas contra “nada”. Use baselines:

### Naive

```text
previsão amanhã = valor de hoje
```

### Seasonal naive

```text
previsão segunda = valor da segunda anterior
```

Em séries sazonais, essa baseline pode ser surpreendentemente forte.

## Frequência

Uma série precisa de índice temporal coerente. Dados faltantes e timestamps irregulares mudam o significado de lags.

```python
s = df.set_index("data")["vendas"].asfreq("D")
```

`asfreq` pode introduzir missing; decida conscientemente como tratá-los.

## Forecast horizon

Prever `t+1` é diferente de prever os próximos 30 passos. Estratégias multi-step podem ser recursivas, diretas ou multi-output.

Defina horizonte antes de validar.

## Anki

- Qual diferença entre tendência e sazonalidade?
- O que é lag?
- Por que baseline seasonal naive é importante?
- O que diferenciação tenta remover?
- Por que a frequência temporal precisa estar bem definida?
