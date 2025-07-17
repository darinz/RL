# 02 - RL for LLM Training

Reinforcement learning is used to train LLMs for objectives that are hard to specify with supervised learning alone.

## RLHF: Reinforcement Learning from Human Feedback

**RLHF** is a popular approach for aligning LLMs with human preferences.

### Steps in RLHF
1. Collect human feedback on model outputs
2. Train a reward model $`R(y|x)`$ to predict human preferences
3. Use RL (e.g., Proximal Policy Optimization, PPO) to optimize the LLM with respect to the reward model

## PPO Objective (Simplified)

```math
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t) \right]
```
Where $`r_t(\theta)`$ is the probability ratio and $`\hat{A}_t`$ is the advantage estimate.

## Example: PPO Update for LLM (Pseudo-code)

```python
for batch in data_loader:
    logprobs_old = model.get_logprobs(batch)
    rewards = reward_model(batch)
    loss = ppo_loss(model, batch, logprobs_old, rewards)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Summary
- RLHF is widely used to align LLMs with human values
- PPO is a common RL algorithm for LLM fine-tuning 