<p align="center">
  <a href="https://prefrontal.ai">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/prefrontal-logo-dark.svg">
      <img src="assets/prefrontal-logo-light.svg" alt="Prefrontal" width="300">
    </picture>
  </a>
</p>

<h1 align="center">CRAM: Compact Reasoning Associative Memory</h1>

<p align="center">
  <a href="https://arxiv.org/pdf/2611.17309"><img src="https://img.shields.io/badge/arXiv-2611.17309-b31b1b?logo=arxiv&logoColor=white" alt="Paper on arXiv"></a>
  <a href="https://prefrontal.ai/news/cram/"><img src="https://img.shields.io/badge/Blog-Reasoning%20at%20Hyperspeed-256abf?logo=rss&logoColor=white" alt="Blog post"></a>
  <img src="https://img.shields.io/badge/Code-coming%20soon-8a8f98?logo=github&logoColor=white" alt="Code coming soon">
</p>

<p align="center">
  📄 <a href="https://arxiv.org/pdf/2611.17309"><b>Paper</b></a> &nbsp;·&nbsp;
  📝 <a href="https://prefrontal.ai/news/cram/"><b>Blog post</b></a> &nbsp;·&nbsp;
  🔖 <a href="#citation"><b>Cite</b></a>
</p>

**Code coming soon.**

This repository will hold the code, trained models and results for the paper "Classical Energies Learn to Reason 1,000× Faster than Looped Models" by Benjamin Hoover, Hugh Blayney and James C. R. Whittington (Prefrontal).

![CRAM overview: training pulls down the energy of solutions and pushes up the energy of negative samples; reasoning descends the energy from random starts until a stopping threshold](assets/overview.png)

**CRAM learns an energy from solved puzzles in milliseconds, and the energy tells the search when to stop.**
*Training* (left) pulls down the energy of each solution and pushes up the energy of negative samples, without differentiating through a reasoning trajectory.
*Reasoning* (right) clamps the clues of a new puzzle, starts several chains from random fills of the other cells, and minimizes the energy one cell at a time.
The search stops when the energy of any chain falls below a stopping threshold fitted from the training solutions.

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

## Citation

```bibtex
@article{hoover2026cram,
  title         = {Classical Energies Learn to Reason 1,000$\times$ Faster than Looped Models},
  author        = {Hoover, Benjamin and Blayney, Hugh and Whittington, James C. R.},
  year          = {2026},
  journal       = {arXiv preprint arXiv:2611.17309},
  eprint        = {2611.17309},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2611.17309}
}
```
