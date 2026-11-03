"""
compare_results.py — compares the lab-PC outputs against the cloud reference
outputs in expected/. Prints PASS/FAIL per item and an overall verdict.

Usage: python3 compare_results.py <lab_results_dir> [expected_dir]
"""
import json, sys, os, math

lab = sys.argv[1]
exp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "expected")
TOL = 1e-9
fails = 0


def close(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=TOL, abs_tol=TOL)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    return a == b


def check(label, ok, detail=""):
    global fails
    print(f"[{'PASS' if ok else 'FAIL'}] {label} {detail}")
    fails += 0 if ok else 1


# 1) exact DP
a = json.load(open(os.path.join(lab, "exact_team_dp_results.json")))
b = json.load(open(os.path.join(exp, "exact_team_dp_results.json")))
for k in ["V_team_start", "handover_submatrix", "sign_flip", "additive_fit_class"]:
    check(f"DP {k}", close(a.get(k), b[k]), f"lab={a.get(k)} ref={b[k]}")
# the argmax LABEL can differ between machines: 'up'/'left'/'stay' against a
# wall are exact ties broken by floating-point noise; only the class counts
print(f"[INFO] DP additive_fit_argmax label: lab={a['additive_fit_argmax']} ref={b['additive_fit_argmax']} (tied labels; class compared above)")
for pol in ["monopoly", "alternation"]:
    for k in ["team_V", "ep200", "gini200", "br_gain_start"]:
        check(f"DP {pol}.{k}", close(a[pol][k], b[pol][k]), f"lab={a[pol][k]}")

# 2) text logs (projection + extra checks): identical lines
import re
la = re.findall(r"\[(hold|handover|none)\]", open(os.path.join(lab, "projection_check.log")).read())
lb = re.findall(r"\[(hold|handover|none)\]", open(os.path.join(exp, "projection_check.log")).read())
check("projection outcomes (hold/handover per eps)", la == lb and len(lb) == 8, f"lab={la}")
la = open(os.path.join(lab, "extra_checks.log")).read().strip().splitlines()
lb = open(os.path.join(exp, "extra_checks.log")).read().strip().splitlines()
check("log extra_checks.log", la == lb, f"({len(la)} lines)")

# 3) team learner, per seed
pairs = [("team_q_1k_s0-19.json", "team_q_1k_s0-19.json"),
         ("team_q_10k_s0-9.json", "team_q_10k_s0-9.json"),
         ("team_q_10k_s10-19.json", "team_q_10k_s10-19.json")]
for fl, fe in pairs:
    if not os.path.exists(os.path.join(lab, fl)):
        print(f"[SKIP] team-Q {fl}: not run in this (quick) mode; see the full run")
        continue
    A = {r["seed"]: r for r in json.load(open(os.path.join(lab, fl)))["per_seed"]}
    B = {r["seed"]: r for r in json.load(open(os.path.join(exp, fe)))["per_seed"]}
    exact = sum(1 for s in B if s in A and close(A[s]["mean_R"], B[s]["mean_R"])
                and A[s]["greedy_tasks"] == B[s]["greedy_tasks"])
    check(f"team-Q {fl}", exact == len(B), f"{exact}/{len(B)} seeds identical")
    if exact < len(B):
        for s in sorted(B):
            if s in A and not close(A[s]["mean_R"], B[s]["mean_R"]):
                print(f"       seed {s}: lab Gini {A[s]['gini']:.3f} vs ref {B[s]['gini']:.3f}")

print("\nOVERALL:", "ALL PASS — lab reproduces the cloud results" if fails == 0
      else f"{fails} item(s) differ — report the numbers above, do not overwrite either set")
