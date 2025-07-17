# 01 - Monte Carlo Tree Search (MCTS) Basics

**Monte Carlo Tree Search (MCTS)** is a heuristic search algorithm for decision processes, widely used in game-playing AI.

## Four Steps of MCTS
1. **Selection**: Traverse the tree from the root to a leaf using a tree policy (e.g., UCB1)
2. **Expansion**: Add a new child node to the tree
3. **Simulation**: Run a rollout (random playout) from the new node to a terminal state
4. **Backpropagation**: Update node statistics along the path

## UCB1 Tree Policy

The Upper Confidence Bound (UCB1) formula balances exploration and exploitation:

```math
UCB1 = \bar{X}_j + c \sqrt{\frac{2 \ln n}{n_j}}
```
Where:
- $`\bar{X}_j`$: average reward of child $`j`$
- $`n`$: number of times parent node was visited
- $`n_j`$: number of times child $`j`$ was visited
- $`c`$: exploration constant

## Example: MCTS Node in Python

```python
class MCTSNode:
    def __init__(self, state):
        self.state = state
        self.children = []
        self.visits = 0
        self.value = 0.0
```

## Summary
- MCTS is a powerful search method for sequential decision problems
- It uses random simulations to estimate action values 