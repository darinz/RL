# 01 - Value Iteration

**Value Iteration** is a dynamic programming algorithm for finding the optimal policy and value function in a Markov Decision Process (MDP).

## The Bellman Optimality Equation

The core of value iteration is the Bellman optimality equation for the state-value function $`V^*(s)`$:

```math
V^*(s) = \max_a \sum_{s', r} P(s', r | s, a) [r + \gamma V^*(s')]
```
Where:
- $`s`$: current state
- $`a`$: action
- $`s'`$: next state
- $`r`$: reward
- $`P(s', r | s, a)`$: transition probability
- $`\gamma`$: discount factor

## Value Iteration Algorithm

1. Initialize $`V(s)`$ arbitrarily (e.g., zeros)
2. Repeat until convergence:
    - For each state $`s`$:
        - $`V(s) \leftarrow \max_a \sum_{s', r} P(s', r | s, a) [r + \gamma V(s')]`
3. Derive the optimal policy:
    - $`\pi^*(s) = \arg\max_a \sum_{s', r} P(s', r | s, a) [r + \gamma V^*(s')]`

## Example: Value Iteration in Python

```python
import numpy as np

def value_iteration(P, R, gamma=0.9, theta=1e-6):
    n_states, n_actions = R.shape
    V = np.zeros(n_states)
    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            V[s] = max(
                sum(P[s, a, s_prime] * (R[s, a] + gamma * V[s_prime])
                    for s_prime in range(n_states))
                for a in range(n_actions)
            )
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V
```

## Summary
- Value iteration finds the optimal value function and policy for an MDP
- It uses the Bellman optimality equation and iterates until convergence 