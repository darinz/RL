# 01 - Imitation Learning

**Imitation learning** enables agents to learn behaviors by mimicking expert demonstrations, rather than learning solely from reward signals.

## Behavioral Cloning

A common approach is **behavioral cloning**, where the agent learns a policy $`\pi(a|s)`$ by supervised learning from state-action pairs $(s, a)$ provided by an expert.

## Loss Function

The typical loss for behavioral cloning is:

```math
L(\theta) = -\sum_{(s, a) \in D} \log \pi_\theta(a|s)
```
Where $`D`$ is the dataset of demonstrations.

## Example: Behavioral Cloning in Python

```python
import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(state_dim, action_dim)
    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

# Training loop
optimizer = torch.optim.Adam(policy.parameters())
for states, actions in dataloader:
    logits = policy(states)
    loss = nn.CrossEntropyLoss()(logits, actions)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Summary
- Imitation learning uses expert demonstrations to train agents
- Behavioral cloning treats imitation as supervised learning 