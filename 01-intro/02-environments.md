# 02 - Environments in Reinforcement Learning

The **environment** is everything outside the agent. It defines the rules, dynamics, and rewards.

## What is an Environment?

- Provides the state $`s`$ to the agent
- Receives actions $`a`$ from the agent
- Returns next state $`s'`$ and reward $`r`$

## Markov Decision Process (MDP)

Most RL environments are modeled as MDPs:

```math
(S, A, P, R, \gamma)
```
Where:
- $`S`$: set of states
- $`A`$: set of actions
- $`P(s'|s,a)`$: transition probability
- $`R(s,a)`$: reward function
- $`\gamma`$: discount factor

## Example: Simple Environment in Python

```python
class SimpleEnv:
    def __init__(self):
        self.state = 0
    def step(self, action):
        reward = 1 if action == 1 else 0
        self.state += action
        return self.state, reward
    def reset(self):
        self.state = 0
        return self.state
```

## Summary
- The environment defines the world and rules for the agent
- MDPs are a common formalism for RL environments 