"""
ast_check.py — checks that the environment and fairness code of the original
lab folder (used for E2) is logically identical to the public v2.1 repo.

Two levels:
  L1  comments and docstrings ignored (the public repo was translated)
  L2  additionally, every string literal is masked (log messages, labels,
      dict keys). If L1 differs but L2 is identical, the only differences
      are text, not logic.
For any file that differs, a unified diff of the normalized code is written
to <out_dir>/ast_diff_<file>.txt so the exact change can be inspected.

Usage: python3 ast_check.py <public_repo> <original_lab_repo> [out_dir]
"""
import ast, sys, os, difflib

FILES = ["nash_depin/environment_mixed.py", "nash_depin/fairness_metrics.py"]


class _MaskStrings(ast.NodeTransformer):
    def visit_Constant(self, node):
        return ast.copy_location(ast.Constant("<str>"), node) if isinstance(node.value, str) else node

    def visit_JoinedStr(self, node):
        return ast.copy_location(ast.Constant("<str>"), node)


def tree(path, mask=False):
    t = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(t):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    if mask:
        t = ast.fix_missing_locations(_MaskStrings().visit(t))
    return t


pub, orig = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else "."
for f in FILES:
    p, o = os.path.join(pub, f), os.path.join(orig, f)
    if not os.path.exists(o):
        print(f"[SKIP] {f}: not found in {orig}")
        continue
    l1 = ast.dump(tree(p)) == ast.dump(tree(o))
    l2 = ast.dump(tree(p, True)) == ast.dump(tree(o, True))
    if l1:
        print(f"[PASS] {f}: logic identical (L1)")
        continue
    verdict = "only string literals differ (L2 identical) -> text, not logic" if l2 \
        else "LOGIC DIFFERS (L2) -> inspect the diff file"
    print(f"[{'TEXT' if l2 else 'DIFF'}] {f}: {verdict}")
    a = ast.unparse(tree(o)).splitlines(); b = ast.unparse(tree(p)).splitlines()
    name = os.path.join(out, "ast_diff_" + os.path.basename(f) + ".txt")
    with open(name, "w") as fh:
        fh.write("\n".join(difflib.unified_diff(a, b, "original_lab", "public_v2.1", lineterm="")))
    print(f"       diff written to {name}")
