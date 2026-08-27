# Checklist de um projeto de Machine Learning

Use esta página antes de considerar um experimento “pronto”.

## 1. Problema

- [ ] Consigo explicar a decisão de negócio em uma frase?
- [ ] Defini a unidade de observação?
- [ ] Defini o instante/data de corte da previsão?
- [ ] Defini horizonte do target?
- [ ] Sei quem consome a previsão?
- [ ] Sei qual ação será tomada?

## 2. Target

- [ ] O target é construído de forma reproduzível?
- [ ] O target representa exatamente a pergunta?
- [ ] Existe ambiguidade ou atraso na rotulagem?
- [ ] Classes/valores impossíveis foram investigados?

## 3. Features

Para cada feature importante:

- [ ] existe no instante da previsão?
- [ ] é calculada apenas com passado permitido?
- [ ] tem definição estável entre treino e produção?
- [ ] missing significa o mesmo nos dois ambientes?
- [ ] categorias novas são esperadas?

## 4. Leakage

- [ ] fiz split antes de preprocessing aprendido?
- [ ] imputers/scalers/encoders estão em Pipeline?
- [ ] seleção de features está dentro da CV?
- [ ] PCA está dentro da CV?
- [ ] mesma entidade pode aparecer em treino/teste?
- [ ] há informação futura em agregações?
- [ ] alguma feature é consequência do target?

## 5. Split

- [ ] o split simula produção?
- [ ] temporal → ordem preservada?
- [ ] entidades repetidas → group split?
- [ ] classificação desbalanceada → estratificação quando apropriada?
- [ ] teste final está congelado?

## 6. Baseline

- [ ] defini baseline trivial?
- [ ] comparei com baseline simples de modelo?
- [ ] consigo explicar quanto valor incremental o modelo adiciona?

## 7. Métrica

- [ ] a métrica corresponde ao custo/benefício real?
- [ ] olhei mais de uma métrica relevante?
- [ ] classificação → matriz de confusão?
- [ ] desbalanceamento → Precision/Recall/PR-AUC?
- [ ] regressão → distribuição dos erros e não só média?
- [ ] séries → métrica por horizonte?

## 8. Validação

- [ ] usei CV apropriada?
- [ ] olhei média e variabilidade entre folds?
- [ ] treino e validação indicam overfitting?
- [ ] hiperparâmetros foram escolhidos sem olhar teste final?
- [ ] muitos experimentos podem ter overfitado a validação?

## 9. Threshold / decisão

- [ ] threshold foi escolhido por requisito/custo?
- [ ] sei quantos casos serão acionados por período?
- [ ] avaliei Precision/Recall no ponto operacional?
- [ ] probabilidades precisam ser calibradas?

## 10. Error analysis

- [ ] examinei maiores erros?
- [ ] comparei segmentos relevantes?
- [ ] procurei classes sistematicamente confundidas?
- [ ] erros revelam problema de label, feature ou processo?

## 11. Interpretabilidade

- [ ] distingui associação de causalidade?
- [ ] importância foi calculada no conjunto apropriado?
- [ ] features correlacionadas foram consideradas?
- [ ] explicação local/global está alinhada à pergunta?

## 12. Produção

- [ ] preprocessing é reproduzível?
- [ ] versão do modelo e dados é rastreável?
- [ ] latência/custo são aceitáveis?
- [ ] features existem com o mesmo SLA?
- [ ] logging permite auditar previsões?
- [ ] existe fallback se feature falhar?

## 13. Monitoramento

- [ ] monitoro distribuição das features?
- [ ] monitoro missing/categorias novas?
- [ ] monitoro score/predições?
- [ ] quando labels chegam, monitoro performance?
- [ ] existe gatilho de investigação/retreino?

## Um resumo em 10 perguntas

Se você tiver apenas dois minutos, responda:

1. O que uma linha representa?
2. O que `y` representa?
3. Quando a previsão acontece?
4. O que não pode ser conhecido nesse momento?
5. Como o teste imita produção?
6. Qual baseline precisa ser superado?
7. Qual erro custa mais?
8. Onde preprocessing aprende estatísticas?
9. Como sei se estou overfitando?
10. Como o sistema será monitorado?

Se alguma resposta estiver vaga, o projeto ainda tem uma dívida de modelagem.
