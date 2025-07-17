# 03 - Dynamic Programming for MDPs

**Dynamic Programming (DP)** provides a set of algorithms for solving MDPs when the model (transition probabilities and rewards) is known and the state/action spaces are small (tabular).

## Principle of Optimality

DP relies on the principle of optimality:

> An optimal policy has the property that, whatever the initial state and action are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision.

## Types of DP Algorithms

- **Value Iteration**: Iteratively updates value estimates using the Bellman optimality equation.
- **Policy Iteration**: Alternates between policy evaluation and policy improvement.

## Bellman Equations

- **State-value function:**

```math
V^{\pi}(s) = \sum_a \pi(a|s) \sum_{s', r} P(s', r | s, a) [r + \gamma V^{\pi}(s')]
```

- **Action-value function:**

```math
Q^{\pi}(s,a) = \sum_{s', r} P(s', r | s, a) [r + \gamma Q^{\pi}(s', a')]
```

## Example: Tabular DP in Python

```python
# See value_iteration and policy_iteration for concrete DP algorithms
# Here is a generic Bellman update for value function

def bellman_update(P, R, V, gamma, s, a):
    return sum(P[s, a, s_prime] * (R[s, a] + gamma * V[s_prime])
               for s_prime in range(len(V)))
```

## Summary
- DP methods require a known model and tabular state/action spaces
- They are foundational for understanding RL algorithms 