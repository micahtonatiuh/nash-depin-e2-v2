"""
fairness_metrics.py — Fairness metrics for E2.
Nash-DePIN.

Implements the two central metrics of the paper:
  - Gini coefficient: inequality of the reward distribution (0=fair, 1=unfair)
  - Jain's Fairness Index: normalized fairness (1=perfectly fair, 1/n=maximally unfair)
"""

import numpy as np
from typing import List, Sequence


def gini_coefficient(rewards: Sequence[float]) -> float:
    """
    Gini coefficient of the reward distribution across agents.

    G = sum_i sum_j |x_i - x_j| / (2 n^2 mean(x))

    - 0   = perfect equality (all agents receive the same)
    - ->1 = one agent takes everything (monopolization)

    A shift is applied when rewards are negative, since the classical Gini
    assumes non-negative values. Shifting by the minimum preserves the
    relative differences.
    """
    x = np.asarray(rewards, dtype=float)
    n = len(x)
    if n == 0:
        return 0.0

    if x.min() < 0:
        x = x - x.min()

    total = x.sum()
    if total <= 1e-12:
        return 0.0  # all zero -> trivially equal

    diff_sum = np.abs(x[:, None] - x[None, :]).sum()
    gini = diff_sum / (2.0 * n * n * x.mean())
    return float(np.clip(gini, 0.0, 1.0))


def jain_fairness_index(rewards: Sequence[float]) -> float:
    """
    Jain's fairness index (1984).

    J = (sum x_i)^2 / (n * sum x_i^2)

    - 1    = perfect fairness (all equal)
    - 1/n  = maximum unfairness (one agent takes everything)

    For n=2: J=1.0 if equal, J=0.5 if one has everything.
    As with Gini, negative values are shifted.
    """
    x = np.asarray(rewards, dtype=float)
    n = len(x)
    if n == 0:
        return 1.0

    if x.min() < 0:
        x = x - x.min()

    sum_sq = np.sum(x ** 2)
    if sum_sq <= 1e-12:
        return 1.0  # all zero -> trivially equal

    jain = (x.sum() ** 2) / (n * sum_sq)
    return float(np.clip(jain, 1.0 / n, 1.0))


def task_balance_ratio(task_counts: Sequence[int]) -> float:
    """
    Task balance ratio: tasks of the least-loaded agent / most-loaded agent.

    - 1.0 = perfect balance (both complete the same number of tasks)
    - ->0 = one robot monopolizes the tasks

    Complementary metric that shows monopolization directly.
    """
    counts = np.asarray(task_counts, dtype=float)
    if counts.max() <= 0:
        return 1.0
    return float(counts.min() / counts.max())


def compute_all_fairness(rewards: Sequence[float],
                         task_counts: Sequence[int] = None) -> dict:
    """Computes all fairness metrics at once."""
    result = {
        "gini": gini_coefficient(rewards),
        "jain": jain_fairness_index(rewards),
        "total_reward": float(np.sum(rewards)),
        "reward_per_agent": [float(r) for r in rewards],
    }
    if task_counts is not None:
        result["task_balance"] = task_balance_ratio(task_counts)
        result["task_counts"] = [int(t) for t in task_counts]
    return result
