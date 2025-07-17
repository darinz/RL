# 02 - Policy Iteration

**Policy Iteration** is another dynamic programming method for solving MDPs. It alternates between policy evaluation and policy improvement.

## Policy Evaluation

Given a policy $`\pi`$, compute its value function $`V^{\pi}(s)`$:

```math
V^{\pi}(s) = \sum_a \pi(a|s) \sum_{s', r} P(s', r | s, a) [r + \gamma V^{\pi}(s')]
```

## Policy Improvement

Update the policy by acting greedily with respect to $`V^{\pi}`$:

```math
\pi'(s) = \arg\max_a \sum_{s', r} P(s', r | s, a) [r + \gamma V^{\pi}(s')]
```

## Policy Iteration Algorithm

1. Initialize policy $`\pi`$ arbitrarily
2. Repeat until policy is stable:
    - **Policy Evaluation:** Compute $`V^{\pi}`$
    - **Policy Improvement:** Update $`\pi`$ using $`V^{\pi}`$

## Example: Policy Iteration in Python

```python
import numpy as np

def policy_iteration(P, R, gamma=0.9, theta=1e-6):
    n_states, n_actions = R.shape
    policy = np.zeros(n_states, dtype=int)
    V = np.zeros(n_states)
    while True:
        # Policy Evaluation
        while True:
            delta = 0
            for s in range(n_states):
                v = V[s]
                a = policy[s]
                V[s] = sum(P[s, a, s_prime] * (R[s, a] + gamma * V[s_prime])
                           for s_prime in range(n_states))
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        # Policy Improvement
        policy_stable = True
        for s in range(n_states):
            old_action = policy[s]
            policy[s] = np.argmax([
                sum(P[s, a, s_prime] * (R[s, a] + gamma * V[s_prime])
                    for s_prime in range(n_states))
                for a in range(n_actions)
            ])
            if old_action != policy[s]:
                policy_stable = False
        if policy_stable:
            break
    return policy, V
```

## Summary
- Policy iteration alternates between evaluating a policy and improving it
- It converges to the optimal policy and value function 