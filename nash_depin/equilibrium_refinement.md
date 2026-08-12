# Egalitarian Equilibrium-Selection Refinement

Method documentation for the E2 experiment.

## Motivation

In the E2 setting a single contested task (a scarce resource) is available.
Nash Q-Learning **without** an equilibrium-selection criterion produced bimodal
fairness outcomes: across 10 seeds, some converged to fair rotation equilibria
(Gini < 0.05) and others to monopoly equilibria (Gini > 0.33).

The cause is theoretical, not an implementation defect: a general-sum game over
a scarce resource admits **multiple Nash equilibria**. For competition over one
task, all of the following are valid equilibria:

- "Agent 0 always takes the task" (monopoly, unfair)
- "Agent 1 always takes the task" (monopoly, unfair)
- "Both take it with probability p" (rotation, fair)

Without a selection criterion, learning converges to whichever equilibrium the
initial conditions and exploration trajectory lead to.

## The refinement

We apply an **egalitarian equilibrium-selection criterion**: among the Nash
equilibria available in each stage game, we select the one that minimizes the
payoff disparity

```
sigma* = argmin_{sigma in NashEquilibria}  |V_0(sigma) - V_1(sigma)|
```

This criterion:

1. **Does not alter the equilibrium set** — it only chooses, among equally
   valid equilibria, the most equitable one.
2. Is grounded in equilibrium-selection theory: it is related to the Nash
   (1950) bargaining solution and to egalitarian/Rawlsian refinements in
   cooperative games.
3. Is reported explicitly as part of the method.

## Why QMIX/VDN cannot apply this refinement

The refinement requires evaluating the INDIVIDUAL payoffs V_0 and V_1 of each
candidate equilibrium. QMIX and VDN, by construction:

- train on a SHARED team reward (r_team = sum_i r_i), and
- represent only the joint value Q_tot (additive in VDN, monotonic mixing in
  QMIX).

They do not maintain separable individual values that would allow |V_0 - V_1|
to be measured, and therefore cannot select the equitable equilibrium even
where one exists. They are structurally confined to the team optimum, which
under a scarce resource is monopoly: one agent specializing as the task-getter
maximizes the total by minimizing wasted movement and collisions.

## Resulting claim

> Nash Q-Learning with egalitarian equilibrium selection reaches an equitable
> task distribution in the majority of runs, whereas QMIX and VDN are
> structurally confined to monopolization (Gini ~ 0.36, largely independent of
> initialization), because their value decomposition does not represent the
> individual payoffs required to reason about equity.

This is a difference of *kind* (structural capability), not merely of *degree*.

## Implementation

`nash_depin/nash_solver.py`, function `joint_nash(A, B, fast, equity)`:

- `equity=True` enables the refinement.
- On the exact path (nashpy) it enumerates all equilibria and selects the most
  equitable one.
- On the fast path (best-response) it uses multiple initializations (uniform
  plus asymmetric) to discover equilibria on both sides, then selects the most
  equitable candidate.

Enabled in `nash_agent_e2.py` via `USE_EQUITY = True`.

## Limitations reported

**Residual bimodality.** The refinement reduces but does not eliminate
convergence to monopoly. In the reported 10-seed run, 3 seeds (5, 6, 8) still
converge to monopoly equilibria with Gini ~ 0.34-0.35. These were verified to
be *stable* equilibria rather than non-converged policies (see
`experiments/validate_e2_convergence.py` and
`e2_convergence_validation/convergence_summary.json`: 10/10 seeds stable).
Because the resulting Gini distribution is bimodal, the median is used as the
primary statistic and non-parametric tests are used for the comparisons; see
`experiments/analyze_e2_stats_v2.py`.

**Incomplete enumeration on the fast path.** Best-response iteration with
multiple initializations is a heuristic that discovers the principal candidate
equilibria but does not guarantee complete enumeration. The exact path (nashpy,
`fast=False`) can be used on a subset of states for verification, at a
substantial computational cost.
