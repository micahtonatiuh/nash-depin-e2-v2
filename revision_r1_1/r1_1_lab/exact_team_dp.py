"""
exact_team_dp.py — R1.1: exact analysis of the E2 environment (Paper 1 revision).

Does NOT modify any original code: it imports MixedMotiveGridWorld from the
public repo (nash-depin-e2-v2, tag v2.1) and uses its step() as the transition
and reward oracle, with the exact E2 config (5x5, num_tasks=1, +10/-5/+2,
coop_distance=1, gamma=0.95, starts R0=(0,0), R1=(4,4)).

Analyses
  A. Team MDP (r_tot = r0 + r1): exact value iteration, optimal joint actions.
  B. Two stationary policies -- monopoly vs. per-step handover (alternation):
     team value, per-agent values and Gini, from the E2 start.
  C. Handover stage matrix Q*_tot(s_h, ., .): monotonicity / additivity test.
  D. Team optimum under eps-execution noise (eps = 0.05, the E2 floor):
     which allocation does the robust team optimum pick?
  E. General-sum game (individual rewards): are monopoly and alternation
     Markov-perfect equilibria (one-shot deviation check)?

Usage: python exact_team_dp.py --repo /path/to/nash-depin-e2-v2
"""
import argparse, itertools, json, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--out", default="exact_team_dp_results.json")
args = ap.parse_args()
sys.path.insert(0, args.repo)
from nash_depin.environment_mixed import MixedMotiveGridWorld  # noqa: E402

G, GAMMA, EPS, H = 5, 0.95, 0.05, 200
env = MixedMotiveGridWorld(grid_size=G, n_agents=2, n_tasks=1, task_reward=10.0,
                           collision_penalty=-5.0, cooperation_reward=2.0,
                           coop_distance=1, max_steps=10**9, seed=0)
env.reset()
assert env._task_positions == [(0, 0)], env._task_positions
cells = [(r, c) for r in range(G) for c in range(G)]
S = [(p, q) for p in cells for q in cells if p != q]          # reachable states
sidx = {s: k for k, s in enumerate(S)}
A = list(range(5)); JA = list(itertools.product(A, A))
NAMES = {0: "up", 1: "down", 2: "left", 3: "right", 4: "stay"}

# ---- transition/reward tables from the repo's own step() -------------------
nS, nJ = len(S), len(JA)
NXT = np.zeros((nS, nJ), dtype=int); R = np.zeros((nS, nJ, 2))
for s in S:
    for j, ja in enumerate(JA):
        env.agent_positions = [s[0], s[1]]; env.reset_tasks = None
        env.tasks[0].completed = False; env.tasks[0].position = (0, 0)
        env.step_count = 0
        _, _, rew, _, _ = env.step(ja)
        NXT[sidx[s], j] = sidx[tuple(env.agent_positions)]
        R[sidx[s], j] = rew
RT = R.sum(-1)
START = sidx[((0, 0), (G - 1, G - 1))]
print(f"states={nS}, joint actions={nJ}, max one-step r_tot={RT.max():.1f}")

# ---- A. team value iteration ---------------------------------------------
V = np.zeros(nS)
for _ in range(5000):
    Q = RT + GAMMA * V[NXT]; Vn = Q.max(1)
    if np.abs(Vn - V).max() < 1e-12: break
    V = Vn
QT = RT + GAMMA * V[NXT]
Vstar = V[START]; upper = 14.0 / (1 - GAMMA)
print(f"\n[A] V*_team(start) = {Vstar:.4f}  (bound 14/(1-g) = {upper:.4f})")

def opt_joint(s, tol=1e-9):
    k = sidx[s]; m = QT[k].max()
    return [(NAMES[JA[j][0]], NAMES[JA[j][1]]) for j in range(nJ) if QT[k, j] >= m - tol]

s_mono = ((0, 0), (1, 0))
print(f"    optimal joint actions in {s_mono}: {opt_joint(s_mono)}")

# ---- B. policy evaluation for stationary deterministic profiles ----------
def evaluate(pol, eps=0.0):
    """pol: state-index -> joint-action index. eps: each agent's action is
    replaced by a uniform random one w.p. eps (as in eps-greedy execution)."""
    P = np.zeros((nS, nS)); r = np.zeros((nS, 2))
    for k in range(nS):
        a0, a1 = JA[pol[k]]
        d0 = np.full(5, eps / 5); d0[a0] += 1 - eps
        d1 = np.full(5, eps / 5); d1[a1] += 1 - eps
        for j, (b0, b1) in enumerate(JA):
            p = d0[b0] * d1[b1]
            if p: P[k, NXT[k, j]] += p; r[k] += p * R[k, j]
    Vi = np.linalg.solve(np.eye(nS) - GAMMA * P, r)
    # undiscounted 200-step episode returns from START
    dist = np.zeros(nS); dist[START] = 1; tot = np.zeros(2)
    for _ in range(H): tot += dist @ r; dist = dist @ P
    return Vi, tot

def gini2(x):
    x = np.asarray(x, float); return abs(x[0] - x[1]) / (2 * x.sum()) if x.sum() else 0.0

ja = {v: k for k, v in enumerate(JA)}
UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4

def toward(p, q):
    """one greedy step from p toward q (rows first)."""
    if p[0] > q[0]: return UP
    if p[0] < q[0]: return DOWN
    if p[1] > q[1]: return LEFT
    if p[1] < q[1]: return RIGHT
    return STAY

def team_greedy_tiebreak(prefer):
    """team-optimal policy (argmax of Q*_tot) with a tie-break preference."""
    pol = np.zeros(nS, dtype=int)
    for k in range(nS):
        m = QT[k].max(); opts = [j for j in range(nJ) if QT[k, j] >= m - 1e-9]
        pol[k] = prefer(S[k], opts)
    return pol

mono = team_greedy_tiebreak(lambda s, o: ja[(STAY, STAY)] if ja[(STAY, STAY)] in o else o[0])

def alt_pref(s, o):
    # in a handover state, prefer the joint move that passes the task cell
    p, q = s
    if p == (0, 0) and abs(q[0]) + abs(q[1]) == 1:
        c = ja[(RIGHT if q == (1, 0) else DOWN, toward(q, (0, 0)))]
        if c in o: return c
    if q == (0, 0) and abs(p[0]) + abs(p[1]) == 1:
        c = ja[(toward(p, (0, 0)), RIGHT if p == (1, 0) else DOWN)]
        if c in o: return c
    return ja[(STAY, STAY)] if ja[(STAY, STAY)] in o else o[0]

alt = team_greedy_tiebreak(alt_pref)
res = {"V_team_start": Vstar, "upper_bound": upper}
for name, pol in [("monopoly", mono), ("alternation", alt)]:
    Vi, tot = evaluate(pol)
    Ve, tote = evaluate(pol, EPS)
    res[name] = dict(team_V=float(Vi[START].sum()), V=Vi[START].tolist(),
                     ep200=tot.tolist(), gini200=gini2(tot),
                     team_V_eps=float(Ve[START].sum()), V_eps=Ve[START].tolist(),
                     ep200_eps=tote.tolist(), gini200_eps=gini2(tote))
    print(f"\n[B] {name:11s} team V={Vi[START].sum():8.3f}  per-agent V={np.round(Vi[START],2)}"
          f"  200-step returns={np.round(tot,1)}  Gini={gini2(tot):.3f}")
    print(f"    under eps={EPS} execution noise: team V={Ve[START].sum():8.3f}"
          f"  200-step={np.round(tote,1)}  Gini={gini2(tote):.3f}")

def joint_class(s, a0, a1):
    """Effective outcome of a joint action from a state with one robot on the
    task cell: 'hold' (same robot stays on the task), 'handover' (the other
    robot ends on the task) or 'none'. Robust to tied action labels such as
    'up'/'left'/'stay' against a wall, which all mean 'stay'."""
    k0 = sidx[s]; nxt = S[NXT[k0, ja[(a0, a1)]]]
    holder = 0 if s[0] == (0, 0) else 1
    if nxt[holder] == (0, 0): return "hold"
    if nxt[1 - holder] == (0, 0): return "handover"
    return "none"

# ---- C. handover stage matrix ---------------------------------------------
k = sidx[s_mono]; M = QT[k].reshape(5, 5)
sub = M[np.ix_([STAY, RIGHT], [STAY, UP])]
print(f"\n[C] Q*_tot at s_h={s_mono}, rows a0 in (stay,right), cols a1 in (stay,up):\n{np.round(sub,3)}")
d_stay, d_up = sub[1, 0] - sub[0, 0], sub[1, 1] - sub[0, 1]
print(f"    effect of a0: stay->right is {d_stay:+.3f} if a1=stay, {d_up:+.3f} if a1=up"
      f"  -> sign flip = {np.sign(d_stay) != np.sign(d_up)} (non-monotonic interaction)")
# best additive (VDN) fit of the full 5x5 matrix, least squares
X = np.zeros((25, 10))
for j, (a0, a1) in enumerate(JA): X[j, a0] = 1; X[j, 5 + a1] = 1
w, *_ = np.linalg.lstsq(X, M.reshape(-1), rcond=None)
fit = (X @ w).reshape(5, 5); amax = np.unravel_index(fit.argmax(), fit.shape)
print(f"    best additive fit: argmax = ({NAMES[amax[0]]},{NAMES[amax[1]]}),"
      f" max abs residual = {np.abs(fit - M).max():.3f}")
res["handover_submatrix"] = sub.tolist(); res["sign_flip"] = bool(np.sign(d_stay) != np.sign(d_up))
res["additive_fit_argmax"] = [NAMES[amax[0]], NAMES[amax[1]]]
res["additive_fit_class"] = joint_class(s_mono, int(amax[0]), int(amax[1]))
print(f"    -> effective outcome of that argmax: {res['additive_fit_class']}")

# ---- D. team optimum under eps-execution noise ------------------------------
# executed joint action distribution given intended (a0,a1)
T = np.zeros((nJ, nJ))
for j, (a0, a1) in enumerate(JA):
    d0 = np.full(5, EPS / 5); d0[a0] += 1 - EPS
    d1 = np.full(5, EPS / 5); d1[a1] += 1 - EPS
    T[j] = np.outer(d0, d1).reshape(-1)
Ve = np.zeros(nS)
for _ in range(5000):
    Qe = (RT + GAMMA * Ve[NXT]) @ T.T; Vn = Qe.max(1)
    if np.abs(Vn - Ve).max() < 1e-12: break
    Ve = Vn
robust = Qe.argmax(1)
Vr, totr = evaluate(robust, EPS)
print(f"\n[D] eps-robust team optimum: intended action at {s_mono} = "
      f"({NAMES[JA[robust[k]][0]]},{NAMES[JA[robust[k]][1]]});"
      f" 200-step returns={np.round(totr,1)}  Gini={gini2(totr):.3f}")
gap = Qe[k].max() - Qe[k, ja[(RIGHT, UP)]]
print(f"    Q_eps(stay,stay) - Q_eps(handover) at s_h = {gap:.3f}")
res["eps_robust"] = dict(action=[NAMES[x] for x in JA[robust[k]]],
                         action_class=joint_class(s_mono, *JA[robust[k]]), ep200=totr.tolist(),
                         gini200=gini2(totr), handover_deficit=float(gap))

# ---- E. general-sum: Nash / Markov-perfect check by exact best response ----
def best_response_gap(pol, i):
    """Fix the other agent's stationary policy; solve agent i's MDP exactly.
    Returns (gain at START = Nash condition, max gain over all states = MPE)."""
    Vi, _ = evaluate(pol)
    other = [JA[pol[kk]][1 - i] for kk in range(nS)]
    js = np.array([[ja[(b, other[kk])] if i == 0 else ja[(other[kk], b)] for b in A]
                   for kk in range(nS)])
    Rb = R[np.arange(nS)[:, None], js, i]; Nb = NXT[np.arange(nS)[:, None], js]
    W = np.zeros(nS)
    for _ in range(5000):
        Wn = (Rb + GAMMA * W[Nb]).max(1)
        if np.abs(Wn - W).max() < 1e-12: break
        W = Wn
    return float(W[START] - Vi[START, i]), float((W - Vi[:, i]).max())
for name, pol in [("monopoly", mono), ("alternation", alt)]:
    g = [best_response_gap(pol, i) for i in range(2)]
    print(f"\n[E] {name}: best-response gain at start (R0,R1) = "
          f"({g[0][0]:.4f}, {g[1][0]:.4f}); max over all states = ({g[0][1]:.3f}, {g[1][1]:.3f})")
    res[name]["br_gain_start"] = [g[0][0], g[1][0]]; res[name]["br_gain_any_state"] = [g[0][1], g[1][1]]

# ---- F. Monte Carlo: per-episode Gini under eps noise (1000 x 200 steps) ----
def mc(pol, eps, n=1000, seed=0):
    rng = np.random.default_rng(seed); ginis = []; tasks = []
    for _ in range(n):
        k = START; tot = np.zeros(2); tk = np.zeros(2)
        for _ in range(H):
            a = list(JA[pol[k]])
            for i in range(2):
                if rng.random() < eps: a[i] = rng.integers(5)
            j = ja[tuple(a)]; r = R[k, j]; tot += r
            nk = NXT[k, j]
            for i in range(2):
                if S[nk][i] == (0, 0) and R[k, j, i] >= 5: tk[i] += 1
            k = nk
        ginis.append(gini2(tot)); tasks.append(tk.max() / max(tk.sum(), 1))
    return np.array(ginis), np.array(tasks)
for name, pol in [("monopoly", mono), ("alternation", alt), ("eps_robust", robust)]:
    gi, ts = mc(pol, EPS)
    print(f"[F] {name:11s} eps={EPS}: per-episode Gini median {np.median(gi):.3f} "
          f"[{np.percentile(gi,25):.3f}, {np.percentile(gi,75):.3f}]; top task share median {np.median(ts):.2f}")
    res.setdefault(name, {})["mc_episode_gini"] = [float(np.median(gi)), float(np.percentile(gi,25)), float(np.percentile(gi,75))]
    res[name]["mc_top_task_share"] = float(np.median(ts))

json.dump(res, open(args.out, "w"), indent=2)
print(f"\nsaved {args.out}")
