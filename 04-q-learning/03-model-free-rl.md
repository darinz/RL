# 03 - Model-Free Reinforcement Learning

**Model-free RL** methods learn optimal policies without requiring a model of the environment's dynamics (transition probabilities or reward function).

## Model-Free vs. Model-Based RL

- **Model-based RL**: Learns or uses a model $`P(s', r | s, a)`$ to plan and make decisions
- **Model-free RL**: Learns directly from experience, without a model

## Types of Model-Free Methods

- **Value-based**: Learn value functions (e.g., Q-learning, SARSA)
- **Policy-based**: Learn policies directly (e.g., policy gradient methods)

## Example: Model-Free Q-learning in Python

```python
# See 01-q-learning.md for a full Q-learning example
# Here is a simple model-free update
Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])
```

## Summary
- Model-free RL is widely used due to its simplicity and generality
- It does not require knowledge of the environment's dynamics 