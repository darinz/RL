# 01 - Sample Efficiency in RL

**Sample efficiency** measures how effectively an RL algorithm learns from limited data. High sample efficiency means achieving good performance with fewer environment interactions.

## Measuring Sample Efficiency

Sample efficiency is often evaluated by the area under the learning curve (reward vs. environment steps) or the number of samples needed to reach a performance threshold.

## Improving Sample Efficiency
- Use prior knowledge or demonstrations
- Reuse past experiences (see experience replay)
- Use model-based approaches

## Example: Tracking Sample Efficiency in Python

```python
rewards = []
for episode in range(num_episodes):
    total_reward = run_episode()
    rewards.append(total_reward)
# Plot rewards vs. episodes to visualize sample efficiency
```

## Summary
- Sample efficiency is crucial for real-world RL applications
- Methods that improve sample efficiency are highly valuable 