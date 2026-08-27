"""Check field_f1 / nTED against hand-computable cases and Donut's definitions."""
from Metrics.field_f1 import (corpus_field_f1, nted_accuracy, normalize,
                              flatten_fields, tree_edit_distance, build_tree,
                              _ins_rem_cost, _update_cost)

ok = fail = 0


def check(name, got, want, tol=1e-9):
    global ok, fail
    good = abs(got - want) <= tol
    print(f"  {'PASS' if good else 'FAIL'}  {name:52s} got={got:.4f} want={want:.4f}")
    ok, fail = ok + good, fail + (not good)


print("--- field-F1: exact-match semantics ---")
a = {"bank": "X", "total": "10.00"}
check("identical -> 1.0", corpus_field_f1([a], [a]), 1.0)
check("disjoint  -> 0.0", corpus_field_f1([{"z": "1"}], [a]), 0.0)
# 2 of 3 gold fields predicted, no spurious: tp=2 fn=1 fp=0 -> 2/(2+0.5)=0.8
g3 = {"a": "1", "b": "2", "c": "3"}
check("2 of 3 correct -> 0.8", corpus_field_f1([{"a": "1", "b": "2"}], [g3]), 0.8)
# 2 right + 1 wrong vs 3 gold: tp=2 fp=1 fn=1 -> 2/(2+1)=0.667
check("2 right + 1 spurious -> 0.667",
      corpus_field_f1([{"a": "1", "b": "2", "d": "9"}], [g3]), 2 / 3, tol=1e-3)
# value must match exactly, not approximately
check("value off by one char -> 0.0",
      corpus_field_f1([{"a": "1.00"}], [{"a": "1.0"}]), 0.0)
# key must match too
check("right value wrong key -> 0.0",
      corpus_field_f1([{"b": "1"}], [{"a": "1"}]), 0.0)

print("--- list handling: multiset, order-agnostic (indices stripped) ---")
L1 = {"items": [{"n": "a"}, {"n": "b"}]}
L2 = {"items": [{"n": "b"}, {"n": "a"}]}
check("reordered list rows -> 1.0", corpus_field_f1([L1], [L2]), 1.0)
# duplicates are counted, not collapsed: pred has 1 of 2 identical rows
D2 = {"items": [{"n": "a"}, {"n": "a"}]}
D1 = {"items": [{"n": "a"}]}
check("1 of 2 duplicate rows -> 0.667", corpus_field_f1([D1], [D2]), 2 / 3, tol=1e-3)

print("--- flatten strips list indices (Donut behaviour) ---")
flat = flatten_fields(normalize(L1))
print("      ", flat)
check("both rows share the parent path",
      float(all(k == "items.n" for k, _ in flat)), 1.0)

print("--- nTED ---")
check("identical -> 1.0", nted_accuracy(a, a), 1.0)
check("empty pred -> 0.0", nted_accuracy({}, a), 0.0)
check("empty gold -> 1.0 (denominator 0)", nted_accuracy({}, {}), 1.0)
part = nted_accuracy({"bank": "X"}, a)
print(f"  INFO  partial pred nTED = {part:.4f}  (0 < x < 1 expected)")
ok += (0 < part < 1); fail += not (0 < part < 1)

print("--- vendored Zhang-Shasha sanity ---")
t = build_tree(normalize(a))
check("TED(tree, itself) = 0",
      tree_edit_distance(t, t, _ins_rem_cost, _ins_rem_cost, _update_cost), 0.0)

print(f"\n  {ok} passed, {fail} failed")
