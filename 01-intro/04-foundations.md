# 04 - Foundations of Reinforcement Learning

Reinforcement learning is based on the interaction between agents and environments, using rewards to learn optimal behavior.

## Policy

A **policy** $`\pi(a|s)`$ defines the agent's behavior:

- $`\pi(a|s)`$: probability of taking action $`a`$ in state $`s`$

## Value Functions

- **State-value function** $`V^{\pi}(s)`$:

```math
V^{\pi}(s) = \mathbb{E}_\pi [G_t | s_t = s]
```

- **Action-value function** $`Q^{\pi}(s,a)`$:

```math
Q^{\pi}(s,a) = \mathbb{E}_\pi [G_t | s_t = s, a_t = a]
```

## Bellman Equations

- **State-value Bellman equation:**

```math
V^{\pi}(s) = \sum_a \pi(a|s) \sum_{s',r} P(s',r|s,a) [r + \gamma V^{\pi}(s')]
```

## Example: Policy in Python

```python
import numpy as np

def random_policy(state, action_space):
    return np.random.choice(action_space)
```

## Summary
- Policies and value functions are central to RL
- Bellman equations relate values across states and actions 