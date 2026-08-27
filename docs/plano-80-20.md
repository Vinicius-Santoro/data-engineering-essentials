# Como estudar este material

A documentação foi desenhada para ser relida. Na primeira passagem, sua meta é construir um **mapa mental**; nas seguintes, você aumenta precisão e velocidade.

## Ciclo de uma sessão

<div class="study-path" markdown>

**1. Recuperação (5–10 min)** → tente explicar o capítulo anterior sem olhar.  
**2. Leitura (20–35 min)** → foque em entender o problema que o conceito resolve.  
**3. Prática (20–40 min)** → execute e altere o código.  
**4. Fechamento (5–10 min)** → escreva 3–5 perguntas para Anki.

</div>

### Antes de executar um código

Pare e faça uma previsão:

- Qual variável será `X` e qual será `y`?
- O modelo precisa de scaling?
- Que métrica faz sentido?
- Onde pode haver leakage?
- O que você espera que aconteça se aumentar a complexidade?

Essa previsão transforma execução passiva em **prática deliberada**.

## Três níveis de domínio

### Nível 1 — reconhecer

Você consegue dizer, por exemplo:

> “DBSCAN é um algoritmo de clustering baseado em densidade e consegue marcar ruído.”

É suficiente para tópicos secundários na primeira volta.

### Nível 2 — explicar

Você consegue explicar **por que** DBSCAN encontra grupos de formatos irregulares, o papel de `eps` e `min_samples` e em que cenário K-Means seria inadequado.

### Nível 3 — operar

Você recebe dados, prepara o pipeline, experimenta hiperparâmetros, interpreta o resultado e identifica limitações.

!!! note "A regra para o 80/20"
    Busque nível 3 para fundamentos, pipeline, leakage, métricas, regressão/logística, árvores, Random Forest, cross-validation, K-Means, PCA e validação temporal. Para os demais, nível 1–2 já é uma excelente primeira meta.

## Como criar bons cartões de Anki

Evite cartões que só testam reconhecimento textual.

=== "Fraco"

    **Frente:** O que é cross-validation?  
    **Verso:** Técnica que divide os dados em folds.

=== "Melhor"

    **Frente:** Por que um único train/test split pode fornecer uma avaliação instável? Como K-Fold reduz esse problema?  
    **Verso:** O resultado pode depender da amostra que caiu no teste. K-Fold repete a avaliação usando diferentes subconjuntos como validação e resume a distribuição dos scores.

=== "Aplicado"

    **Frente:** Você aplicou `StandardScaler().fit_transform(X)` antes do `train_test_split`. Qual o problema?  
    **Verso:** O scaler aprendeu média/desvio usando também observações que depois formarão o teste; isso cria leakage. Ajuste o scaler apenas no treino, idealmente dentro de um `Pipeline`.

## Frequência de prática

Para algoritmos essenciais, prefira **repetições curtas** a uma única sessão longa.

| Dia | Ação |
|---|---|
| D0 | ler + executar exemplo |
| D1 | reimplementar com pouca consulta |
| D3 | explicar + alterar hiperparâmetro |
| D7 | comparar com outro algoritmo |
| D14 | resolver um mini-problema sem roteiro |

## O que não vale a pena decorar

- ordem exata dos argumentos de uma função;
- nomes de todos os hiperparâmetros;
- fórmulas longas sem entender o significado;
- dezenas de modelos raros antes de dominar o pipeline.

Vale memorizar:

- **ideia central**;
- hipóteses importantes;
- o que muda o comportamento do modelo;
- erros comuns;
- quais métricas respondem à pergunta do negócio;
- padrão `fit → predict → evaluate`.

## Um princípio para toda a documentação

> **O modelo é apenas uma etapa.** A qualidade do split, das features, da validação e da definição da métrica frequentemente importa mais do que trocar um algoritmo por outro.
