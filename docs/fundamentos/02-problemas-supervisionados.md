# Classificação, regressão e desenho das features

**Prioridade: Essencial**

A escolha entre classificação e regressão vem da **natureza da variável alvo**, não do algoritmo que você gosta de usar.

## Classificação

O objetivo é prever uma categoria.

```text
fraude?              → sim / não
cliente vai cancelar?→ sim / não
produto              → A / B / C
sentimento            → positivo / neutro / negativo
```

O modelo pode produzir uma classe final ou uma probabilidade/score.

```python
classe = modelo.predict(X_novo)
prob = modelo.predict_proba(X_novo)
```

A probabilidade é especialmente valiosa porque permite escolher um **threshold** conforme o custo dos erros.

## Regressão

O alvo é numérico e contínuo ou aproximadamente contínuo.

```text
preço de imóvel      → 485000.00
tempo de atendimento → 312.4 segundos
demanda amanhã       → 1832 unidades
```

As métricas precisam medir a distância entre valor observado e previsto, por exemplo MAE ou RMSE.

## E números que são classes?

Uma coluna `0, 1, 2` pode representar categorias. O tipo numérico do banco não define o problema.

```text
0 = baixo
1 = médio
2 = alto
```

Pode ser classificação ordinal. Sempre investigue o **significado semântico** do target.

## Features não são “todas as colunas disponíveis”

Uma feature deve ser:

- disponível no momento da predição;
- semanticamente plausível;
- de qualidade suficiente;
- produzida sem olhar informação proibida do futuro/target.

Imagine prever cancelamento no início do mês. A coluna `data_cancelamento` obviamente não pode entrar. Menos óbvio: uma coluna `dias_desde_ultimo_pagamento` calculada usando dados registrados depois da data de corte também não pode.

## Data de corte

Uma prática poderosa é perguntar:

> **Em que instante eu estou simulando que a previsão acontece?**

Se a previsão acontece em `2026-08-01 00:00`, todas as features precisam ser construídas com informações que existiam até aquele instante.

Esse raciocínio previne uma enorme classe de leakages.

## Feature engineering

Feature engineering transforma informações brutas em representações que tornam o padrão mais acessível ao modelo.

Exemplo:

```text
transações brutas
↓
qtde_transacoes_30d
valor_medio_30d
dias_desde_ultima_compra
proporcao_transacoes_noturnas
```

A qualidade da feature pode ser mais importante do que trocar Random Forest por outro ensemble.

## Um dataset, duas perguntas

Com os mesmos dados de clientes podemos construir:

### Problema A — classificação

> O cliente vai cancelar em até 30 dias?

`y = 0/1`

### Problema B — regressão

> Quantos reais o cliente gastará nos próximos 30 dias?

`y = valor`

A modelagem muda porque a pergunta mudou.

## Exercício mental

Classifique os problemas:

1. prever nota de 0 a 10;
2. prever se uma transação é fraude;
3. prever qual de 12 intents representa uma mensagem;
4. estimar tempo até resolução em minutos;
5. descobrir grupos de clientes sem rótulos prévios.

??? success "Resposta"
    1. geralmente regressão; 2. classificação binária; 3. classificação multiclasse; 4. regressão; 5. clustering/não supervisionado.

## Checklist antes de modelar

- [ ] Qual decisão será tomada com a previsão?
- [ ] Qual é exatamente o target?
- [ ] Qual é a unidade de observação?
- [ ] Quando a previsão ocorre?
- [ ] Todas as features existem naquele momento?
- [ ] Qual erro é mais caro?
- [ ] Existe um baseline simples?

!!! tip "Pergunta de ouro"
    Se você não consegue explicar **quem é uma linha, quando a previsão acontece e o que significa `y`**, ainda não é hora de escolher o algoritmo.
