# 02 - Monte Carlo Policy Evaluation

**Monte Carlo (MC) methods** estimate value functions by averaging returns from sampled episodes.

## Monte Carlo Estimation

For a state $`s`$, the value is estimated as:

```math
V(s) \approx \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_t^{(i)}
```
Where:
- $`N(s)`$: number of times state $`s`$ is visited
- $`G_t^{(i)}`$: return following the $`i`$-th visit to $`s`$

## First-Visit and Every-Visit MC
- **First-visit MC**: Average returns only for the first time each state is visited in an episode
- **Every-visit MC**: Average returns for every occurrence of the state

## Example: Monte Carlo Policy Evaluation in Python

```python
def mc_policy_evaluation(episodes, n_states, gamma=0.9):
    V = np.zeros(n_states)
    returns = [[] for _ in range(n_states)]
    for episode in episodes:
        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = r + gamma * G
            if s not in visited:  # First-visit MC
                returns[s].append(G)
                V[s] = np.mean(returns[s])
                visited.add(s)
    return V
```

## Summary
- MC methods use sampled episodes to estimate value functions
- They do not require knowledge of the environment's dynamics 