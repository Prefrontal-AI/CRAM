# CRAM: Compact Reasoning Associative Memory

**Code coming soon.**

This repository will hold the code, trained models and results for the paper "Classical Energies Learn to Reason 1,000× Faster than Looped Models" by Benjamin Hoover, Hugh Blayney and James C. R. Whittington (Prefrontal).

Read the blog post: https://prefrontal.ai/news/cram/

## What is CRAM?

CRAM is a small trainable energy function, together with the recurrent dynamics that descend it at inference time.
It is a learned version of the energy that Hopfield and Tank proposed in the 1980s.
As in a Hopfield network, its weights memorize associations between pairs of valid cells.
Descent on the energy completes a puzzle from its clues.

Looped models, which apply the same network repeatedly to refine an answer, are a prominent approach to reasoning puzzles such as Sudoku-Extreme and Maze-Unique.
On these puzzles, CRAM gives:

- **Fast training.**
  CRAM solves 99.996% of Sudoku-Extreme test puzzles after 42 ms of training on one GPU, and 100% of Maze-Unique test puzzles after 132 ms.
  This is at least 1,000× faster than looped models on these tasks.
- **Cheap inference.**
  CRAM matches the baseline looped models at their highest accuracy on Sudoku-Extreme with at least 282× fewer inference FLOPs (floating-point operations).
- **Training without differentiating through a solver.**
- **Built-in early stopping**, with compute that adapts to the difficulty of the problem.
- **Interpretable weights** that learn the constraints of the puzzle.

## Contact

Benjamin Hoover (ben@prefrontal.ai) and James C. R. Whittington (james@prefrontal.ai).
