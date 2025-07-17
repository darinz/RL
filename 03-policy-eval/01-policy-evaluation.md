# 01 - Policy Evaluation in Tabular RL

**Policy evaluation** is the process of computing the value function $`V^{\pi}(s)`$ for a given policy $`\pi`$ in a Markov Decision Process (MDP).

## Value Function

The value function for a policy $`\pi`$ is:

```math
V^{\pi}(s) = \mathbb{E}_\pi [G_t | s_t = s]
```
Where $`G_t`$ is the return from time $`t`$.

## Bellman Expectation Equation

The value function satisfies the Bellman expectation equation:

```math
V^{\pi}(s) = \sum_a \pi(a|s) \sum_{s', r} P(s', r | s, a) [r + \gamma V^{\pi}(s')]
```

## Iterative Policy Evaluation Algorithm

1. Initialize $`V(s)`$ arbitrarily
2. Repeat until convergence:
    - For each state $`s`$:
        - $`V(s) \leftarrow \sum_a \pi(a|s) \sum_{s', r} P(s', r | s, a) [r + \gamma V(s')]`

## Example: Iterative Policy Evaluation in Python

```python
import numpy as np

def policy_evaluation(P, R, policy, gamma=0.9, theta=1e-6):
    n_states, n_actions = R.shape
    V = np.zeros(n_states)
    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            V[s] = sum(
                policy[s, a] * sum(P[s, a, s_prime] * (R[s, a] + gamma * V[s_prime])
                                    for s_prime in range(n_states))
                for a in range(n_actions)
            )
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V
```

## Summary
- Policy evaluation computes the value function for a given policy
- It is a key step in many RL algorithms 