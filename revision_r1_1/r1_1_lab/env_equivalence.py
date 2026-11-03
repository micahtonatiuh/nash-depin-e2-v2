"""
env_equivalence.py — behavioural equivalence of the E2 environment between
the original lab folder and the public v2.1 repo.

Uses only the public interface (constructor, reset, step), so it does not
depend on internal or translated names. Both environments are driven with
the same seeded random joint actions; at every step the next state, the
per-agent rewards, the done flag and the info values must coincide.

Usage: python3 env_equivalence.py <public_repo> <original_lab_repo> [episodes]
"""
import importlib.util, inspect, sys, os
import numpy as np

pub, orig = sys.argv[1], sys.argv[2]
EPISODES = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
ARGS = [5, 2, 1, 10.0, -5.0, 2.0, 1, 200]          # E2 config, positional order


def load(repo, tag):
    path = os.path.join(repo, "nash_depin", "environment_mixed.py")
    spec = importlib.util.spec_from_file_location(f"env_{tag}", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    cls = [c for _, c in inspect.getmembers(mod, inspect.isclass)
           if c.__module__ == mod.__name__ and hasattr(c, "reset") and hasattr(c, "step")]
    assert len(cls) == 1, f"{tag}: expected one environment class, found {cls}"
    params = list(inspect.signature(cls[0].__init__).parameters)[1:]
    print(f"{tag}: class {cls[0].__name__}{tuple(params)}")
    return cls[0]


A, B = load(pub, "public"), load(orig, "original")
ea = A(*ARGS, seed=0)
try:
    eb = B(*ARGS, seed=0)
except TypeError:          # original may name the seed argument differently
    eb = B(*ARGS, 0)
rng = np.random.default_rng(12345)
steps = mism = 0; pairs = set()
for ep in range(EPISODES):
    sa, oa = ea.reset(); sb, ob = eb.reset()
    if sa != sb or not np.array_equal(oa, ob):
        mism += 1; print(f"reset mismatch ep {ep}: {sa} vs {sb}"); break
    done = False
    while not done:
        act = tuple(int(x) for x in rng.integers(0, 5, size=2))
        pairs.add((sa, act))
        ra = ea.step(act); rb = eb.step(act)
        same = (ra[0] == rb[0] and np.array_equal(ra[1], rb[1]) and list(ra[2]) == list(rb[2])
                and ra[3] == rb[3] and list(ra[4].values()) == list(rb[4].values()))
        steps += 1
        if not same:
            mism += 1
            if mism <= 5:
                print(f"MISMATCH ep {ep} step {steps}: action {act}\n  public  {ra[0]} {ra[2]} {ra[4]}\n  original {rb[0]} {rb[2]} {rb[4]}")
        sa, done = ra[0], ra[3]
print(f"\nsteps compared: {steps:,}  | distinct (state, joint action) pairs visited: {len(pairs):,} of 15,000")
print(f"[{'PASS' if mism == 0 else 'FAIL'}] environment behaviour {'identical' if mism == 0 else f'differs in {mism} steps'}")
