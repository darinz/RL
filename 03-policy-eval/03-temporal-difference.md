# 03 - Temporal-Difference (TD) Policy Evaluation

**Temporal-Difference (TD) methods** combine ideas from Monte Carlo and dynamic programming. They update value estimates based on observed transitions, without waiting for the end of an episode.

## TD(0) Update Rule

The TD(0) update for state $`s_t`$ is:

```math
V(s_t) \leftarrow V(s_t) + \alpha [r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]
```
Where:
- $`\alpha`$: learning rate
- $`r_{t+1}`$: reward received after $`s_t`$
- $`\gamma`$: discount factor

## Example: TD(0) Policy Evaluation in Python

```python
def td0_policy_evaluation(episodes, n_states, gamma=0.9, alpha=0.1):
    V = np.zeros(n_states)
    for episode in episodes:
        for t in range(len(episode) - 1):
            s, a, r = episode[t]
            s_next, _, _ = episode[t + 1]
            V[s] += alpha * (r + gamma * V[s_next] - V[s])
    return V
```

## Summary
- TD methods update value estimates incrementally
- They can learn online and from incomplete episodes 