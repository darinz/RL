# 02 - Learning from Human Input

Agents can learn from various forms of human input, such as demonstrations, preferences, or corrections.

## Types of Human Input
- **Demonstrations**: Sequences of state-action pairs
- **Preferences**: Human feedback on which trajectories are better
- **Corrections**: Direct interventions in agent behavior

## Inverse Reinforcement Learning (IRL)

**IRL** aims to infer the reward function $`R(s, a)`$ that explains observed expert behavior.

## IRL Objective

```math
\max_R \sum_{(s, a) \in D} R(s, a) - \log Z(R)
```
Where $`Z(R)`$ is a normalization term (partition function).

## Example: Preference-based Learning (Pseudo-code)

```python
# Given pairs of trajectories and human preferences
for traj1, traj2, pref in preferences:
    # Update policy to favor preferred trajectories
    loss = -log_prob(preferred_traj)
    loss.backward()
    optimizer.step()
```

## Summary
- Human input can guide agent learning beyond rewards
- IRL seeks to recover reward functions from demonstrations 