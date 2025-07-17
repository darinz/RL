# 01 - Agents in Reinforcement Learning

An **agent** is an entity that interacts with an environment to achieve a goal. In reinforcement learning (RL), the agent observes the state of the environment, takes actions, and receives rewards.

## What is an Agent?

An agent is defined by its ability to:
- Observe the environment (state $`s`$)
- Take actions $`a`$
- Receive rewards $`r`$
- Update its policy $`\pi`$ based on experience

## Agent-Environment Interaction

The agent interacts with the environment in discrete time steps $`t`$:

```math
s_{t+1}, r_{t+1} \sim P(\cdot | s_t, a_t)
```

Where:
- $`s_t`$ is the state at time $`t`$
- $`a_t`$ is the action taken at time $`t`$
- $`r_{t+1}`$ is the reward received after taking action $`a_t`$
- $`P`$ is the environment's transition probability

## Example: Simple Agent in Python

```python
class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
    def act(self, state):
        return self.action_space.sample()
```

This agent selects actions randomly from the action space.

## Summary
- Agents are decision-makers in RL
- They interact with environments, receive rewards, and learn policies 