# 03 - Batch Reinforcement Learning

**Batch RL** (or offline RL) trains agents using a fixed dataset of transitions, without further interaction with the environment.

## Problem Setting

Given a dataset $`D = \{(s, a, r, s')\}`$ collected by some policy, learn a policy that performs well in the environment.

## Challenges
- Distributional shift: The dataset may not cover all relevant states/actions
- Extrapolation error: Estimating values for unseen state-action pairs

## Batch Q-learning Update

```math
Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]
```
Where updates are only performed on transitions in $`D`$.

## Example: Batch Q-learning in Python

```python
for (s, a, r, s_next) in dataset:
    Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])
```

## Summary
- Batch RL uses offline data to train agents
- It is important for applications where online interaction is costly or unsafe 