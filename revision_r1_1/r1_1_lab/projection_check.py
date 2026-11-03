"""Weighted additive (VDN) projection of Q*_tot at the handover state s_h,
under partner-action distributions from epsilon-greedy around a greedy joint action."""
import runpy, sys, io, contextlib, numpy as np
sys.argv = ["x", "--repo", sys.argv[1], "--out", "/dev/null"]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path("exact_team_dp.py")
QT, sidx, JA, NAMES, jc = g["QT"], g["sidx"], g["JA"], g["NAMES"], g["joint_class"]
M = QT[sidx[((0, 0), (1, 0))]].reshape(5, 5)
X = np.zeros((25, 10))
for j, (a0, a1) in enumerate(JA): X[j, a0] = 1; X[j, 5 + a1] = 1
def fit(greedy, eps):
    d0 = np.full(5, eps / 5); d0[greedy[0]] += 1 - eps
    d1 = np.full(5, eps / 5); d1[greedy[1]] += 1 - eps
    w = np.sqrt(np.outer(d0, d1).reshape(-1))
    coef, *_ = np.linalg.lstsq(X * w[:, None], M.reshape(-1) * w, rcond=None)
    q0, q1 = coef[:5], coef[5:]
    a0, a1 = int(q0.argmax()), int(q1.argmax())
    # label of the argmax (ties between equivalent wall moves are broken by
    # floating-point noise) plus its effective outcome [hold/handover]
    return f"({NAMES[a0]},{NAMES[a1]}) [{jc(((0, 0), (1, 0)), a0, a1)}]"
for eps in [1.0, 0.5, 0.2, 0.05]:
    print(f"eps={eps:4}: data around handover(right,up) -> greedy {fit((3,0),eps)};"
          f"  around hold(stay,stay) -> greedy {fit((4,4),eps)}")
