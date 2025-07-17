# 03 - MCTS Breakthroughs in Game-Playing AI

MCTS has enabled major breakthroughs in AI for games like Go, Chess, and others.

## AlphaGo and Beyond
- AlphaGo (2016) combined MCTS with deep neural networks, defeating top human players
- AlphaZero generalized this approach to other games

## Key Innovations
- Policy/value networks guide MCTS
- Self-play for training
- Generalization to multiple games

## Example: AlphaGo-like MCTS Loop (Pseudo-code)

```python
for game in range(num_games):
    state = env.reset()
    while not done:
        action = mcts_with_nn(state)
        state, reward, done, _ = env.step(action)
```

## Summary
- MCTS, combined with deep learning, has revolutionized game-playing AI
- These methods are now applied to a wide range of decision-making problems 