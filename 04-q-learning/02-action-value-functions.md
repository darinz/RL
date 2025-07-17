# 02 - Action-Value Functions in RL

The **action-value function** $`Q^{\pi}(s, a)`$ gives the expected return for taking action $`a`$ in state $`s`$ and following policy $`\pi`$ thereafter.

## Definition

```math
Q^{\pi}(s, a) = \mathbb{E}_\pi [G_t | s_t = s, a_t = a]
```
Where $`G_t`$ is the return from time $`t`$.

## Bellman Expectation Equation for Q

```math
Q^{\pi}(s, a) = \sum_{s', r} P(s', r | s, a) [r + \gamma \sum_{a'} \pi(a'|s') Q^{\pi}(s', a')]
```

## Optimal Action-Value Function

The optimal action-value function $`Q^*(s, a)`$ satisfies:

```math
Q^*(s, a) = \sum_{s', r} P(s', r | s, a) [r + \gamma \max_{a'} Q^*(s', a')]
```

## Example: Q-table Initialization in Python

```python
import numpy as np
n_states = 5
n_actions = 2
Q = np.zeros((n_states, n_actions))
```

## Summary
- Action-value functions are central to model-free RL
- Q-learning directly estimates the optimal action-value function 