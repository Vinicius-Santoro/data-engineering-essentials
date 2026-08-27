# Aprendizado por reforço — fundamentos

**Prioridade: Reconhecer com boa intuição**

Reinforcement Learning (RL) é diferente do supervisionado: não existe necessariamente um dataset fixo de `(X, y)`. Um **agente** interage com um ambiente, escolhe ações e recebe recompensas.

## Os elementos

### Estado `s`

Representação da situação atual.

### Ação `a`

Escolha disponível ao agente.

### Recompensa `r`

Sinal numérico recebido após uma ação/transição.

### Política `π(a|s)`

Regra que diz quais ações o agente tende a escolher em cada estado.

### Retorno

Soma de recompensas futuras, frequentemente descontadas:

\[
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3}+\cdots
\]

`γ` é discount factor.

## O objetivo

Não maximizar apenas a recompensa imediata, mas o **retorno esperado**.

Um agente pode aceitar uma pequena perda agora para obter ganho maior depois.

## Exemplo mental

Em um jogo:

```text
estado → posição atual
atividade → mover esquerda/direita
recompensa → +1 objetivo, -1 colisão
política → como escolher movimentos
```

## MDP

Markov Decision Process formaliza muitos problemas de RL por:

```text
(S, A, P, R, γ)
```

- estados;
- ações;
- dinâmica de transição;
- recompensa;
- desconto.

A propriedade de Markov diz, simplificando, que o estado contém informação suficiente para prever a dinâmica futura dado a ação, sem precisar de todo histórico bruto.

## Exploração vs exploitation

O agente precisa equilibrar:

- **exploration**: testar ações para aprender;
- **exploitation**: usar a melhor ação conhecida.

Sempre explorar desperdiça recompensa; nunca explorar pode prender o agente em uma estratégia medíocre.

## `epsilon-greedy`

```text
com probabilidade ε → ação aleatória
caso contrário      → melhor ação conhecida
```

Uma estratégia comum reduz `ε` ao longo do treinamento.

## Valor de estado e ação

### V(s)

Retorno esperado a partir do estado seguindo uma política.

### Q(s,a)

Retorno esperado ao tomar ação `a` no estado `s` e depois seguir a política.

Essa distinção leva diretamente ao Q-Learning.

## Quando RL faz sentido?

- decisões sequenciais;
- ação muda o estado futuro;
- recompensa pode ser atrasada;
- existe possibilidade segura/econômica de interação ou simulação.

Se você possui apenas uma tabela estática com rótulo por linha, supervisionado provavelmente é a formulação natural.

## Dificuldades reais

- recompensa mal desenhada gera comportamento indesejado;
- exploração pode ser cara/perigosa;
- distribuição de dados depende da política;
- avaliação offline é difícil;
- treinamento pode ser instável.

## Anki

- Qual diferença entre recompensa imediata e retorno?
- O que é política?
- O que `γ` controla?
- Exploração e exploitation diferem como?
- Quando um problema tabular não precisa de RL?
