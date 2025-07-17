# 02 - Experience Replay

**Experience replay** is a technique to improve sample efficiency by storing and reusing past transitions during training.

## Replay Buffer

A **replay buffer** stores tuples $`(s, a, r, s')`$ collected during agent-environment interaction. During learning, mini-batches are sampled from the buffer to update the agent.

## Benefits
- Breaks correlation between consecutive samples
- Increases data efficiency
- Enables off-policy learning

## Example: Experience Replay Buffer in Python

```python
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
    def push(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

## Summary
- Experience replay is widely used in deep RL (e.g., DQN)
- It allows efficient reuse of past experiences 