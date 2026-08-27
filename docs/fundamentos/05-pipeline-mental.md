# Pipeline mental de um projeto

**Prioridade: Essencial**

Antes do `fit`, existe uma sequência de decisões. Aprender Ciência de Dados fica muito mais fácil quando você enxerga essa sequência como um sistema.

## O fluxo

```text
1. Pergunta de negócio
        ↓
2. Unidade de observação + target + data de corte
        ↓
3. Split que simula produção
        ↓
4. EDA no conjunto de treino
        ↓
5. Preprocessamento
        ↓
6. Baseline
        ↓
7. Modelos candidatos
        ↓
8. Cross-validation + métrica
        ↓
9. Tuning / threshold / feature engineering
        ↓
10. Teste final
        ↓
11. Deploy + monitoramento
```

## 1. Defina a decisão

“Quero usar Machine Learning” não é uma pergunta. Algo como isto é:

> Para cada cliente ativo às 08h, estimar a probabilidade de cancelamento nos próximos 30 dias para priorizar uma ação de retenção.

Agora temos:

- unidade: cliente ativo;
- instante: 08h/data de corte;
- horizonte: 30 dias;
- target: cancelou ou não;
- uso: priorização.

## 2. Congele o teste cedo

Se possível, separe a avaliação final antes de explorar milhares de decisões. Você reduz a chance de adaptar inconscientemente o processo ao teste.

## 3. EDA também pode vazar

Uma nuance: olhar distribuições do teste geralmente não “treina” diretamente o modelo, mas decisões humanas também podem incorporar informação. Em projetos rigorosos, desenvolva principalmente com treino/validação e trate teste como avaliação final.

## 4. Preprocessamento pertence ao processo de treinamento

Se o preprocessamento aprende algo — média, mediana, categorias, componentes principais — ele precisa ser ajustado apenas nos dados permitidos de cada fold.

Por isso `Pipeline` é tão importante.

## 5. Baseline primeiro

Exemplos:

- classificação: classe majoritária, regressão logística simples;
- regressão: média/mediana, regressão linear;
- séries: último valor, média sazonal.

## 6. Métrica é parte do problema

Um modelo não é “bom” sem contexto. A métrica deve refletir o erro que importa.

- perda financeira por falso negativo → recall pode ser importante;
- capacidade limitada para atuar nos positivos → precision pode pesar mais;
- erro absoluto em reais → MAE é interpretável;
- grandes erros são especialmente caros → RMSE penaliza mais.

## 7. Compare candidatos de forma justa

Use o **mesmo split**, o mesmo preprocessamento conceitual e a mesma métrica para comparar modelos.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="roc_auc"
)

print(scores.mean(), scores.std())
```

## 8. Teste final apenas no fim

Depois de escolher pipeline e hiperparâmetros, ajuste no treino disponível e avalie no teste congelado.

## 9. Produção muda tudo

Depois do deploy:

- distribuição das features pode mudar;
- prevalência da classe pode mudar;
- regra do negócio pode mudar;
- atraso das fontes pode mudar;
- custo de decisão pode mudar.

Isso é **model monitoring** e **data drift**.

## Checklist mental

Antes de qualquer experimento, responda:

- [ ] Qual é a unidade de uma linha?
- [ ] Qual é o target?
- [ ] Quando a previsão acontece?
- [ ] Quais dados existiam nesse instante?
- [ ] Como o teste simula produção?
- [ ] Qual baseline preciso superar?
- [ ] Qual métrica corresponde ao objetivo?
- [ ] Alguma transformação aprende estatísticas?
- [ ] O pipeline evita leakage dentro do CV?
- [ ] Como vou interpretar erros e incerteza?

!!! success "Se isso estiver claro"
    Escolher o primeiro algoritmo se torna uma decisão muito menor e mais fácil.
