# 02 - MCTS for Go

MCTS has been instrumental in advancing AI for complex games like Go, where the search space is vast.

## Why MCTS for Go?
- Go has a huge branching factor and deep game tree
- Traditional search methods (e.g., minimax) are infeasible
- MCTS can efficiently explore promising moves

## Integrating MCTS with Neural Networks
- AlphaGo combined MCTS with deep neural networks for policy and value estimation
- The neural network guides simulations and evaluates positions

## Example: MCTS Step for Go in Python (Pseudo-code)

```python
def mcts_step(node):
    # Selection
    while node.children:
        node = select_child_ucb1(node)
    # Expansion
    if not is_terminal(node.state):
        node.children = expand(node.state)
        node = random.choice(node.children)
    # Simulation
    reward = rollout(node.state)
    # Backpropagation
    backpropagate(node, reward)
```

## Summary
- MCTS enables strong play in Go by focusing search on promising moves
- Deep learning further enhances MCTS performance 