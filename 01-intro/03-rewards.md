# 03 - Rewards in Reinforcement Learning

**Rewards** are signals that tell the agent how well it is doing at each step.

## What is a Reward?

- Scalar feedback $`r_t`$ received after taking action $`a_t`$ in state $`s_t`$
- Guides the agent's learning

## Return and Discounting

The **return** $`G_t`$ is the total accumulated reward:

```math
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^\infty \gamma^k r_{t+k+1}
```
Where $`\gamma \in [0,1]`$ is the discount factor.

## Example: Reward Calculation in Python

```python
def compute_return(rewards, gamma):
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
    return G
```

## Summary
- Rewards drive agent behavior
- The return is the sum of discounted rewards 