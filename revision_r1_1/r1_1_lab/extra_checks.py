"""Extra checks for R1.1 iteration 2: physical start, undiscounted 200-step equality,
closure of the 14-reward set under hold and handover."""
import runpy, sys, io, contextlib, numpy as np
sys.argv = ["x", "--repo", sys.argv[1], "--out", "/dev/null"]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path("exact_team_dp.py")
S, sidx, NXT, R, RT, V, mono, alt, JA, ja = [g[k] for k in "S sidx NXT R RT V mono alt JA ja".split()]
# 1) physical start
ps = sidx[((4, 0), (0, 4))]
def roll(pol, k, H=200):
    tot = np.zeros(2)
    for _ in range(H): j = pol[k]; tot += R[k, j]; k = NXT[k, j]
    return tot
print("V* physical start:", round(V[ps], 4))
for n, p in [("mono", mono), ("alt", alt)]:
    for lab, k in [("E2", sidx[((0,0),(4,4))]), ("phys", ps)]:
        t = roll(p, k); print(f"{n:4s} {lab:4s} 200-step returns {t}, team {t.sum()}")
# 2) closure: 14-reward configurations
C = [((0,0),(0,1)), ((0,0),(1,0)), ((0,1),(0,0)), ((1,0),(0,0))]
UP,DOWN,LEFT,RIGHT,STAY = 0,1,2,3,4
moves = {((0,0),(0,1)): (DOWN, LEFT), ((0,0),(1,0)): (RIGHT, UP),
         ((0,1),(0,0)): (LEFT, DOWN), ((1,0),(0,0)): (UP, RIGHT)}
for s in C:
    k = sidx[s]
    for lab, a in [("hold", (STAY, STAY)), ("handover", moves[s])]:
        j = ja[a]; print(s, lab, "r_tot", RT[k, j], "->", S[NXT[k, j]])
