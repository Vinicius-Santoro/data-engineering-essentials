# Causalidade vs predição

**Prioridade: Importante**

Um modelo pode prever muito bem sem responder o que aconteceria **se intervíssemos** em uma variável. Essa distinção evita interpretações perigosas.

## Perguntas diferentes

### Predição

> Qual a probabilidade de este cliente cancelar?

### Causalidade

> Se eu oferecer um benefício a este cliente, isso reduzirá a probabilidade de cancelamento?

A primeira pode ser respondida por padrões associativos. A segunda exige estimar um efeito de intervenção.

## Correlação não implica causalidade

Vendas de sorvete e afogamentos podem aumentar no verão. Sorvete não precisa causar afogamentos; temperatura é uma variável que influencia ambos.

```text
temperatura ─→ sorvete
      └──────→ afogamentos
```

## Confundimento

Uma variável `Z` é confounder quando influencia exposição/tratamento e outcome, criando associação que pode ser interpretada incorretamente.

## Exemplo de churn

Seu modelo mostra que clientes que receberam ligações têm maior churn. Conclusão errada:

> “Ligar causa churn.”

Talvez a empresa ligue justamente para clientes que já demonstraram sinais fortes de risco. A associação pode refletir **seleção do tratamento**.

## Experimento randomizado

Randomização quebra, em expectativa, associação entre tratamento e fatores pré-existentes.

```text
população elegível
     ↓ sorteio
controle     tratamento
     ↓           ↓
 comparar outcome
```

A/B tests são uma forma de experimento randomizado.

## Potential outcomes — ideia

Para cada unidade imaginamos:

- `Y(1)`: resultado se receber tratamento;
- `Y(0)`: resultado se não receber.

Efeito individual:

\[
Y(1)-Y(0)
\]

Problema fundamental: observamos apenas um dos dois para cada unidade.

## Dados observacionais

Quando randomização não é possível, métodos causais tentam ajustar diferenças sob hipóteses: propensity scores, matching, inverse probability weighting, regressão ajustada, doubly robust, instrumental variables, diff-in-diff etc.

O 80/20 aqui é: **não aplicar esses métodos como receita sem entender as hipóteses de identificação**.

## Feature importance não é efeito causal

Se `renda` tem alta importância, isso significa que o modelo usa informação associada à renda para prever. Não significa que “aumentar renda em R$1.000 causará” a mudança sugerida pelo modelo.

## Predição continua valiosa

Não é uma crítica ao ML preditivo. Muitas aplicações precisam apenas de ranking/previsão. O erro é responder uma pergunta causal com uma ferramenta validada apenas para previsão.

## Anki

- Dê um exemplo de pergunta preditiva e causal com os mesmos dados.
- O que é confundimento?
- Por que feature importance não é efeito causal?
- Qual o papel da randomização?
- O que são `Y(1)` e `Y(0)`?
