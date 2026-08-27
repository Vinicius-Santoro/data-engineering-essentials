# Banco de perguntas para Anki

Estas perguntas foram escritas para **recuperação ativa**. Não transforme todas em cartões de uma vez. Crie cartões conforme estuda e apague os que ficarem redundantes.

!!! tip "Regra"
    Antes de abrir a resposta, fale ou escreva uma resposta completa. Reconhecer a resposta depois de vê-la não é o mesmo que recuperá-la da memória.

## Fundamentos

??? question "1. Qual diferença entre parâmetro e hiperparâmetro?"
    Parâmetro é aprendido durante `fit` (coeficientes, splits). Hiperparâmetro é configurado/selecionado externamente (`C`, `max_depth`, `k`).

??? question "2. O que são X e y?"
    `X` contém features/entradas; `y` contém o target/rótulo que o modelo supervisionado tenta prever.

??? question "3. O que significa generalização?"
    Manter bom desempenho em observações representativas que não participaram do ajuste do modelo/processamento.

??? question "4. Por que score de treino não basta?"
    Um modelo flexível pode memorizar ruído do treino; precisamos estimar desempenho fora da amostra.

??? question "5. O que é baseline?"
    Uma referência simples que o modelo precisa superar, como classe majoritária, média/mediana ou um modelo linear simples.

??? question "6. Como distinguir classificação de regressão?"
    Pela natureza da pergunta/target: categorias → classificação; valor numérico contínuo → regressão.

??? question "7. O que é unidade de observação?"
    O que cada linha representa: cliente em uma data, transação, mensagem, pedido etc.

??? question "8. Por que definir data de corte?"
    Para determinar exatamente quais informações existiam no instante da previsão e impedir uso do futuro.

??? question "9. O que é underfitting?"
    Modelo simples/incapaz de capturar o padrão: erro alto tanto em treino quanto em validação.

??? question "10. O que é overfitting?"
    Modelo se ajusta demais ao treino, incluindo ruído: treino muito bom e generalização pior.

??? question "11. Bias alto costuma se relacionar a quê?"
    Hipóteses rígidas/simplicidade excessiva e underfitting.

??? question "12. Variance alta costuma se relacionar a quê?"
    Grande sensibilidade à amostra de treino e maior risco de overfitting.

## Split e leakage

??? question "13. Para que serve o conjunto de teste?"
    Fornecer avaliação final em dados que não orientaram escolhas de desenvolvimento/tuning.

??? question "14. Por que usar o teste repetidamente para escolher modelos é ruim?"
    As decisões começam a se adaptar ao teste, tornando a estimativa final otimista.

??? question "15. Quando stratify=y é útil?"
    Em classificação, para preservar aproximadamente proporções das classes nos splits.

??? question "16. O que random_state resolve?"
    Reprodutibilidade do sorteio/divisão/inicialização; não melhora magicamente generalização.

??? question "17. Defina data leakage em uma frase operacional."
    O processo usa informação que não estaria legitimamente disponível no momento da previsão real.

??? question "18. Por que fit_transform no dataset inteiro antes do split é leakage?"
    A transformação aprende estatísticas também das observações que depois formarão o teste/validação.

??? question "19. O que é target leakage?"
    Feature contém direta ou indiretamente informação produzida pelo próprio target/desfecho futuro.

??? question "20. O que é leakage temporal?"
    Usar informação posterior ao instante em que a previsão deveria ter sido feita.

??? question "21. Por que split por linha pode vazar identidade?"
    Registros do mesmo usuário/sessão podem cair dos dois lados, tornando teste semelhante demais ao treino.

??? question "22. Como Pipeline evita leakage em CV?"
    Cada fold ajusta suas transformações (`fit`) apenas na parte de treino daquele fold antes de transformar validação.

## Preprocessing

??? question "23. Qual diferença entre fit, transform e fit_transform?"
    `fit` aprende parâmetros; `transform` aplica parâmetros já aprendidos; `fit_transform` faz os dois em sequência.

??? question "24. Por que imputação deve ficar dentro do Pipeline?"
    Média/mediana/moda são aprendidas nos dados e não devem incorporar validação/teste.

??? question "25. Quando missing indicator pode ajudar?"
    Quando a própria ausência carrega informação preditiva relevante.

??? question "26. Por que one-hot é adequado para categorias nominais?"
    Evita impor uma ordem/distância artificial como SP=0, RJ=1, MG=2.

??? question "27. O que handle_unknown='ignore' resolve?"
    Categorias novas no transform não quebram o encoder; são tratadas como vetor sem categoria conhecida naquele bloco.

??? question "28. Por que KNN precisa de scaling?"
    Distância é influenciada pela magnitude; features em escala grande dominariam vizinhança.

??? question "29. Por que árvore geralmente não precisa de scaling?"
    Ela usa ordenação e thresholds; transformações monotônicas de escala preservam possíveis separações equivalentes.

??? question "30. Por que PCA geralmente precisa de scaling?"
    PCA maximiza variância; features com escala maior poderiam dominar componentes só por unidade de medida.

??? question "31. Para que serve ColumnTransformer?"
    Aplicar pipelines diferentes a grupos de colunas, como numéricas e categóricas, em uma única transformação.

## Regressão

??? question "32. O que OLS da Regressão Linear minimiza?"
    Soma dos erros quadráticos entre valores observados e previstos.

??? question "33. Coeficiente positivo prova causalidade?"
    Não. É uma associação condicional no modelo sob suas hipóteses; causalidade requer identificação/desenho adicionais.

??? question "34. Qual diferença entre Ridge e Lasso?"
    Ridge usa penalidade L2 e encolhe coeficientes; Lasso usa L1 e pode zerar alguns.

??? question "35. Como uma árvore de regressão prevê em uma folha?"
    Tipicamente pela média dos targets das amostras de treino naquela folha.

??? question "36. O que max_depth controla?"
    Complexidade/profundidade da árvore; profundidade maior permite regras mais específicas.

??? question "37. O que min_samples_leaf faz?"
    Exige folhas com tamanho mínimo, regularizando divisões muito específicas.

??? question "38. Por que Random Forest reduz variance?"
    Agrega muitas árvores diversificadas; erros não perfeitamente correlacionados se compensam parcialmente.

??? question "39. Qual papel do bootstrap na Random Forest?"
    Produzir amostras diferentes para cada árvore e aumentar diversidade.

??? question "40. Qual diferença entre bagging e boosting?"
    Bagging treina modelos de forma mais independente e agrega; boosting constrói modelos sequencialmente para corrigir erros/gradientes.

??? question "41. Como learning_rate interage com número de estimadores no boosting?"
    Passos menores normalmente exigem mais etapas; passos maiores corrigem mais por etapa e podem overfitar/instabilizar mais cedo.

??? question "42. Por que SVR precisa de scaling?"
    Margem/kernel e distâncias são sensíveis à magnitude das features.

??? question "43. Como k afeta KNN Regressor?"
    k pequeno é local/alta variance; k grande suaviza mais e aumenta bias.

## Classificação

??? question "44. Por que Regressão Logística é classificação?"
    Ela modela score/log-odds e transforma em probabilidade de classe, que depois pode ser convertida em decisão.

??? question "45. O que a sigmoide faz?"
    Converte qualquer score real em valor entre 0 e 1.

??? question "46. O que C pequeno significa na Logistic/SVM do sklearn?"
    Regularização mais forte; `C` é inverso da força da penalização.

??? question "47. Qual diferença entre predict e predict_proba?"
    `predict` retorna classe segundo regra/threshold interno; `predict_proba` retorna probabilidades estimadas por classe.

??? question "48. O que são support vectors?"
    Pontos mais relevantes próximos/violando a margem que determinam a fronteira da SVM.

??? question "49. O que gamma alto faz no kernel RBF?"
    Torna influência mais local, permitindo fronteira mais complexa e potencialmente maior overfitting.

??? question "50. Qual é a hipótese naive do Naive Bayes?"
    Features são condicionalmente independentes dado a classe.

??? question "51. Por que MultinomialNB funciona bem com texto?"
    Modela evidências baseadas em contagens/frequências por classe e lida bem com alta dimensionalidade esparsa.

## Matriz e métricas

??? question "52. O que é falso positivo em fraude?"
    Marcar como fraude uma transação que era legítima.

??? question "53. O que é falso negativo em fraude?"
    Classificar como legítima uma transação que era fraude.

??? question "54. Qual pergunta Precision responde?"
    Entre os casos previstos como positivos, quantos realmente eram positivos?

??? question "55. Qual pergunta Recall responde?"
    Entre todos os positivos reais, quantos foram encontrados?

??? question "56. O que tende a ocorrer ao baixar o threshold?"
    Mais exemplos são marcados positivos: recall geralmente sobe e precision frequentemente cai.

??? question "57. O que F1 combina?"
    Média harmônica de precision e recall.

??? question "58. Quando macro F1 é útil?"
    Quando queremos dar peso semelhante a cada classe, inclusive raras.

??? question "59. Por que accuracy falha em classes muito desbalanceadas?"
    A classe majoritária pode dominar acertos mesmo com desempenho zero na minoria importante.

??? question "60. Qual diferença entre ROC-AUC e threshold?"
    AUC resume ranking em vários thresholds; threshold define a decisão operacional específica.

??? question "61. Quando PR-AUC costuma ser especialmente informativa?"
    Quando a classe positiva é rara e precision/recall sobre positivos são o foco.

??? question "62. Ranking bom implica calibração boa?"
    Não. Um modelo pode ordenar positivos acima de negativos e ainda atribuir probabilidades numericamente incorretas.

??? question "63. Como interpretar MAE?"
    Erro absoluto médio na unidade original do target.

??? question "64. Por que RMSE penaliza erros grandes mais que MAE?"
    Porque ele parte de erros ao quadrado antes de tirar a raiz.

??? question "65. R² negativo significa o quê?"
    No conjunto avaliado, o modelo foi pior que a baseline que prevê a média segundo a definição usual de R².

??? question "66. Por que MAPE é problemática perto de y=0?"
    Divide pelo valor real; denominadores pequenos explodem o erro percentual.

## Cross-validation e tuning

??? question "67. Qual problema K-Fold reduz?"
    Dependência da avaliação em um único split aleatório.

??? question "68. Por que olhar média e desvio dos folds?"
    Média resume desempenho; variabilidade mostra estabilidade/sensibilidade à amostra.

??? question "69. Quando usar GroupKFold?"
    Quando exemplos da mesma entidade/grupo não podem aparecer em treino e validação.

??? question "70. Quando usar TimeSeriesSplit?"
    Quando ordem temporal precisa ser preservada para simular previsão futura.

??? question "71. GridSearchCV faz o quê?"
    Avalia combinações explícitas de hiperparâmetros usando cross-validation e seleciona conforme a métrica.

??? question "72. Quando RandomizedSearch é útil?"
    Espaços grandes/contínuos, quando enumerar todas as combinações é caro.

??? question "73. O que refit=True faz?"
    Após seleção, reajusta o melhor estimador usando todo o conjunto fornecido ao search.

??? question "74. O que é nested CV?"
    CV externa estima desempenho enquanto uma CV interna faz tuning em cada treino externo.

??? question "75. Como tuning pode overfitar validação?"
    Muitas decisões são escolhidas pelo mesmo sinal de validação e podem capturar sorte/ruído específico da avaliação.

## Clustering e PCA

??? question "76. Por que clustering não possui uma verdade única?"
    O agrupamento depende de features, escala, distância e objetivo; diferentes noções de similaridade produzem partições diferentes.

??? question "77. O que K-Means minimiza intuitivamente?"
    Distância quadrática das observações aos centroides de seus clusters (inertia dentro dos clusters).

??? question "78. Por que inertia sempre tende a cair ao aumentar K?"
    Mais centros permitem representar pontos com menor distância, chegando a zero se cada ponto pudesse ser seu próprio cluster.

??? question "79. Por que K-Means não é ideal para duas luas?"
    Sua geometria baseada em centroides favorece grupos compactos/esféricos e não acompanha bem formas curvas não convexas.

??? question "80. O que DBSCAN chama de ruído no sklearn?"
    Observações com label `-1` que não foram conectadas a regiões densas conforme parâmetros.

??? question "81. O que eps controla no DBSCAN?"
    Raio da vizinhança usada para definir densidade/conectividade.

??? question "82. O que PCA cria?"
    Novas features ortogonais como combinações lineares das originais, ordenadas por variância explicada.

??? question "83. PCA é feature selection?"
    Não; seleção mantém features originais, PCA cria componentes novos.

??? question "84. PCA usa y?"
    PCA padrão é não supervisionado e usa apenas X.

??? question "85. Por que PCA ainda deve ficar dentro de CV?"
    Porque aprende estrutura de X; validação não deve influenciar os componentes treinados.

??? question "86. Por que t-SNE não deve ser interpretado globalmente como mapa métrico fiel?"
    Ele prioriza preservação de vizinhanças locais e distâncias globais/tamanhos de clusters podem ser distorcidos.

## Séries temporais

??? question "87. Qual erro central de usar train_test_split com shuffle em série temporal?"
    Permite que o futuro entre no treino para avaliar o passado.

??? question "88. O que é seasonal naive?"
    Baseline que prevê usando o valor do mesmo período sazonal anterior, como semana passada.

??? question "89. O que é lag_7?"
    Valor da série observado sete períodos antes.

??? question "90. Por que rolling mean deve frequentemente usar shift(1)?"
    Para impedir que o próprio target do instante atual entre na feature usada para prevê-lo.

??? question "91. O que p, d, q representam no ARIMA?"
    Ordem autoregressiva, número de diferenciações e ordem de média móvel dos erros.

??? question "92. O que SARIMA adiciona?"
    Termos autoregressivos, diferenciação e média móvel sazonais com período m.

??? question "93. O que SARIMAX adiciona?"
    Variáveis exógenas além da estrutura SARIMA.

??? question "94. Por que valor exógeno futuro pode causar leakage?"
    O valor realizado pode não ser conhecido no instante da previsão; usar o futuro real fornece informação impossível.

??? question "95. AIC substitui backtesting?"
    Não. AIC é critério de ajuste/model selection sob uma formulação; desempenho futuro deve ser testado fora da amostra temporal.

??? question "96. Qual diferença entre sazonalidade aditiva e multiplicativa?"
    Aditiva tem amplitude aproximadamente constante; multiplicativa cresce/decresce proporcionalmente ao nível.

??? question "97. O que é previsão recursiva multi-step?"
    Prever um passo e usar previsões anteriores como entrada para passos seguintes, acumulando possível erro.

## Causalidade e interpretação

??? question "98. Qual diferença entre pergunta preditiva e causal?"
    Preditiva pergunta o que acontecerá/qual score; causal pergunta o que mudaria sob uma intervenção.

??? question "99. O que é confounder?"
    Variável que influencia tratamento/exposição e outcome, podendo criar associação enganosa.

??? question "100. Por que SHAP alto não prova causalidade?"
    SHAP explica uso associativo da feature pelo modelo; não identifica efeito de intervenção.

## Reforço

??? question "101. O que é estado?"
    Representação da situação do ambiente usada pelo agente para decidir.

??? question "102. O que é política?"
    Mapeamento/distribuição de ações condicionada ao estado.

??? question "103. Qual diferença entre recompensa e retorno?"
    Recompensa é sinal imediato; retorno agrega recompensas futuras, normalmente descontadas.

??? question "104. O que gamma controla?"
    Peso relativo dado a recompensas futuras no retorno.

??? question "105. O que Q(s,a) representa?"
    Retorno esperado associado a escolher ação a em estado s e seguir a política depois.

??? question "106. Qual diferença entre value-based e policy-based?"
    Value-based aprende valores e deriva ações; policy-based otimiza diretamente uma política parametrizada.

??? question "107. O que actor e critic fazem?"
    Actor escolhe/parametriza ações; critic estima valor e fornece sinal de melhoria.

??? question "108. O que PPO tenta evitar intuitivamente?"
    Atualizações de política grandes demais que desestabilizem o treinamento.

## Cartões de integração

??? question "109. Você recebeu dados tabulares mistos e quer Logistic Regression. Qual pipeline mínimo mental?"
    Split apropriado → numéricas: imputer+scaler; categóricas: imputer+one-hot; ColumnTransformer → Logistic Regression → CV com métrica adequada → threshold/avaliação final.

??? question "110. Seu CV está ótimo e produção péssima. Cite cinco hipóteses."
    Leakage; split não representativo; drift; diferença de preprocessing/serving; target/label shift; bug de features; entidades duplicadas; latência/fonte indisponível; overfit de validação.

??? question "111. KNN piorou muito após adicionar 100 features irrelevantes. Por quê?"
    Distâncias em alta dimensão ficam menos discriminativas e features irrelevantes perturbam vizinhança.

??? question "112. Árvore tem 100% treino e CV bem menor. Primeiras ações?"
    Regularizar (`max_depth`, `min_samples_leaf`, poda), revisar leakage/split, comparar ensemble e analisar learning curves/features.

??? question "113. Random Forest e Logistic têm AUC semelhante. Como escolher?"
    Considere estabilidade, calibração, latência, interpretabilidade, custo, facilidade operacional e desempenho no ponto de threshold real — não só AUC.

??? question "114. Por que 'melhor algoritmo' sem definir métrica e split é uma pergunta incompleta?"
    Porque desempenho depende do objetivo, custo dos erros e distribuição simulada pela validação.

??? question "115. Qual é o pipeline mental mais importante deste curso?"
    Problema → unidade/target/data de corte → split → preprocessing seguro → baseline → modelo → CV/métrica → tuning/threshold → teste final → monitoramento.
