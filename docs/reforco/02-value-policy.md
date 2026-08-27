# Value-based, policy-based e PPO

**Prioridade: Reconhecer**

O objetivo aqui é saber onde as famílias se encaixam, não reproduzir toda a matemática de Deep RL.

# Q-Learning — value-based

Q-Learning aprende uma função `Q(s,a)` para estimar qualidade das ações.

Atualização clássica:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma\max_{a'}Q(s',a') - Q(s,a)]
\]

Interpretação:

```text
valor atual
+ taxa de aprendizado ×
  (alvo baseado em recompensa + melhor futuro - valor atual)
```

## Exemplo tabular conceitual

```python
Q[state, action] += alpha * (
    reward
    + gamma * Q[next_state].max()
    - Q[state, action]
)
```

Em espaços pequenos e discretos, uma tabela pode funcionar.

# DQN

Quando estados são grandes/contínuos, uma rede neural aproxima `Q(s,a)`.

DQN adiciona técnicas importantes para estabilidade, como experience replay e target network.

Você precisa reconhecer:

```text
Q-Learning tabular
      ↓ aproximação
Deep Q-Network
```

# Policy Gradient

Em vez de aprender valores e derivar a ação, modela diretamente uma política parametrizada `πθ(a|s)` e ajusta parâmetros para aumentar retorno esperado.

REINFORCE é o exemplo clássico.

## Por que policy-based?

- ações contínuas;
- políticas estocásticas naturais;
- otimização direta da distribuição de ações.

Mas estimativas de gradiente podem ter alta variance.

# Actor-Critic

Combina:

- **actor**: política que escolhe ações;
- **critic**: estima valor e fornece sinal para melhorar o actor.

```text
estado
  ↓
actor → ação
  ↓
ambiente → recompensa, novo estado
  ↓
critic avalia
  ↓
actor melhora
```

# PPO

Proximal Policy Optimization é um algoritmo actor-critic/policy optimization que limita atualizações para evitar mudar a política agressivamente demais.

O 80/20:

> PPO tenta realizar melhorias de política mantendo cada atualização em uma região relativamente segura/estável.

Não precisa decorar a objective clipped agora.

## Comparação

| Família | Aprende principalmente | Exemplos |
|---|---|---|
| value-based | valor de ações | Q-Learning, DQN |
| policy-based | política | REINFORCE |
| actor-critic | política + valor | A2C, PPO |

## Exemplo com Gymnasium

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
state, info = env.reset(seed=42)

done = False
while not done:
    action = env.action_space.sample()  # política aleatória
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

Isso ainda não aprende; serve para entender a interface `estado → ação → recompensa → próximo estado`.

## Anki

- O que Q-Learning aprende?
- Qual diferença principal entre value-based e policy-based?
- O que actor e critic fazem?
- Qual problema intuitivo PPO tenta evitar?
- Por que DQN é necessário quando uma Q-table fica inviável?
