
# 2048 Game – Monte Carlo Tree Search Variants

This project implements several Monte Carlo Tree Search (MCTS) variants for the popular 2048 game. The code compares classic Monte Carlo, UCT, Nested Monte Carlo Search (NMCS), and Nested Rollout Policy Adaptation (NRPA) under a fixed-horizon configuration.

## Overview

2048 is a popular single-player puzzle game played on a 4×4 grid. The objective is to merge tiles with the same value by sliding them in one of four directions (up, down, left, or right). After each move, a new tile (of value 2 or, with 10% probability, 4) appears in a random empty cell. The game ends when no valid moves remain.

Due to the randomness in tile generation, 2048 poses a challenging, stochastic decision-making problem. Traditional greedy or search methods may not fully capture the long-term value of a move. MCTS methods, by simulating future outcomes, can plan under uncertainty. In this project, we explore and compare several MCTS variants, including:

- **Classic Monte Carlo and UCT:** Employ random rollouts (with a fixed rollout depth) during the simulation phase.
- **Nested Monte Carlo Search (NMCS):** Uses multi-level recursive simulations. At the lowest level (level 0), a random rollout is performed, and higher levels recursively select the best move from simulated playouts. Only the first move of the simulated sequence is committed in the actual game.
- **Nested Rollout Policy Adaptation (NRPA):** Applies a recursively adapted policy to bias rollouts. Although theoretically promising, its performance is sensitive to the stochastic and delayed reward structure of 2048.

## Features

- **Fixed-Horizon Rollouts:** All algorithms use a fixed-depth (or fixed-horizon) approach to limit the number of moves simulated, making real-time play computationally tractable.
- **Decoupled Simulation and Committed Moves:** Only the first move from the simulation is committed, while the simulation itself may run for several moves.
- **Multiple MCTS Variants:** Comparison of classic Monte Carlo, UCT, NMCS, and NRPA on the 2048 game.
- **Verbose Mode (Optional):** When enabled, the script clears the console and displays the game board after every committed move.

## Methodology

The 2048 game engine is implemented using a simple 4×4 NumPy array. While this naïve representation is straightforward, it introduces overhead in move generation. More efficient implementations (using bitboards, for example) could be explored to speed up simulation time.

For NMCS and NRPA the simulation depth (lookahead) is controlled independently from the number of moves that are actually committed in the game. In other words, while the algorithm may simulate a fixed number of moves ahead (for example, 50 moves), only the very first move of the simulation is used to update the game state. This allows the algorithm to re-run the search with a new board configuration after every move, rather than committing to an entire sequence.


## Running the Code

To run the classic MCTS and UCT algorithms, use:

```bash
python classicMC_and_UCT.py
```

For NMCS and NRPA, use:

```bash
python NMCS_and_NRPA.py
```

The NRPA and NMCS modules include a verbose flag. When enabled (by setting verbose=True) it helps in visually tracking the progress of the search.


## Authors
Alejandro Jorba \
Efstathios Chatziloizos
