# 01 - Q-learning

**Q-learning** is a foundational model-free reinforcement learning (RL) algorithm. It enables agents to learn optimal policies by updating action-value functions through experience, without requiring a model of the environment.

## Q-learning Update Rule

The core of Q-learning is the update rule for the action-value function $`Q(s, a)`$:

```math
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]
```
Where:
- $`s_t`$: current state
- $`a_t`$: action taken
- $`r_{t+1}`$: reward received
- $`s_{t+1}`$: next state
- $`\alpha`$: learning rate
- $`\gamma`$: discount factor

## Q-learning Algorithm

1. Initialize $`Q(s, a)`$ arbitrarily
2. For each episode:
    - Initialize $`s`$
    - For each step:
        - Choose $`a`$ (e.g., $\epsilon$-greedy)
        - Take action $`a`$, observe $`r, s'`$
        - Update $`Q(s, a)`$ using the update rule
        - $`s \leftarrow s'`$

## Example: Q-learning in Python

```python
import numpy as np

def q_learning(env, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    for episode in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
            if np.random.rand() < epsilon:
                a = env.action_space.sample()
            else:
                a = np.argmax(Q[s])
            s_next, r, done, _ = env.step(a)
            Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])
            s = s_next
    return Q
```

## Summary
- Q-learning learns optimal action-value functions from experience
- It is off-policy and does not require a model of the environment 