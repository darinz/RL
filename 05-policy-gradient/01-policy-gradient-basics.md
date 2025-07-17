# 01 - Policy Gradient Basics

**Policy gradient methods** directly optimize the parameters of a policy in reinforcement learning (RL), rather than learning value functions.

## Policy Parameterization

A policy $`\pi_\theta(a|s)`$ is parameterized by $`\theta`$ (e.g., neural network weights). The goal is to find $`\theta`$ that maximizes expected return:

```math
J(\theta) = \mathbb{E}_{\pi_\theta} [G_t]
```

## Gradient Ascent

We use gradient ascent to update the policy parameters:

```math
\theta \leftarrow \theta + \alpha \, \nabla_\theta J(\theta)
```
Where $`\alpha`$ is the learning rate.

## Example: Policy Parameter Update in Python

```python
# Assume theta is a numpy array of parameters and grad is the gradient
alpha = 0.01
theta += alpha * grad
```

## Summary
- Policy gradients optimize policies directly
- They use gradient ascent to maximize expected return 