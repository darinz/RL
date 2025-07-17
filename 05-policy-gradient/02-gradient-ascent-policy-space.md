# 02 - Gradient Ascent in Policy Space

**Gradient ascent** is used to optimize the policy parameters $`\theta`$ to maximize the expected return $`J(\theta)`$.

## Policy Gradient Theorem

The policy gradient theorem provides a way to compute the gradient:

```math
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} [\nabla_\theta \log \pi_\theta(a|s) \, Q^{\pi_\theta}(s, a)]
```

- $`\nabla_\theta \log \pi_\theta(a|s)`$: gradient of the log-probability of the action
- $`Q^{\pi_\theta}(s, a)`$: action-value function under policy $`\pi_\theta`$

## Example: Computing Policy Gradient in Python

```python
# Assume log_prob is the log-probability, Q is the action-value, and grad is the gradient
policy_grad = log_prob * Q
# In practice, use automatic differentiation (e.g., PyTorch, TensorFlow)
```

## Summary
- Gradient ascent updates policy parameters in the direction of higher expected return
- The policy gradient theorem is the foundation for policy gradient algorithms 