# 03 - REINFORCE Algorithm

**REINFORCE** is a classic Monte Carlo policy gradient algorithm. It updates policy parameters using sampled returns from complete episodes.

## REINFORCE Update Rule

The update for policy parameters $`\theta`$ is:

```math
\theta \leftarrow \theta + \alpha \, G_t \, \nabla_\theta \log \pi_\theta(a_t|s_t)
```
Where:
- $`G_t`$: return following time $`t`$
- $`\alpha`$: learning rate

## REINFORCE Algorithm Steps

1. Initialize policy parameters $`\theta`$
2. For each episode:
    - Generate an episode $`(s_0, a_0, r_1, ..., s_T)`$ using $`\pi_\theta`$
    - For each step $`t`$ in the episode:
        - Compute $`G_t`$
        - Update $`\theta`$ using the update rule

## Example: REINFORCE in Python (Pseudo-code)

```python
for episode in range(num_episodes):
    states, actions, rewards = [], [], []
    state = env.reset()
    done = False
    while not done:
        action, log_prob = policy(state)
        next_state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
    # Compute returns
    returns = compute_returns(rewards, gamma)
    # Update policy
    for log_prob, G in zip(log_probs, returns):
        loss = -log_prob * G
        loss.backward()
    optimizer.step()
```

## Summary
- REINFORCE uses sampled returns to update policy parameters
- It is simple but can have high variance 