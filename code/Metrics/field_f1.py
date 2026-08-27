"""
field_f1.py — field-level F1 (+ nTED) for structured extraction, self-contained.

Two metrics, matching Donut's JSONParseEvaluator so numbers are comparable to
published work and the prior UCL thesis:

  field-F1 : order-agnostic field-level F1. Flatten the tree to (field-path, value)
             pairs with LIST INDICES STRIPPED (menu lines matched as a multiset),
             normalise, and take exact-match F1. Baseline ~0.00 (wrong schema).
  nTED     : normalised tree-edit-distance accuracy,
             max(0, 1 - TED(pred, gt) / TED(empty, gt)),
             with Donut's leaf-aware costs (character edit distance on values).
             Baseline ~0.10.

value-F1 (in cord_extract) stays a reading diagnostic; these two are the metrics
reported across baseline / fine-tuned / reversion / WiSE-FT.

The tree-edit-distance is a vendored Zhang-Shasha (no zss dependency).
"""
from __future__ import annotations

from Metrics.text_metrics import levenshtein


# ---------------------------------------------------------------------------
# Zhang-Shasha tree edit distance (canonical algorithm, plain-Python)
# ---------------------------------------------------------------------------

class Node:
    def __init__(self, label):
        self.label = label
        self.children = []

    def addkid(self, node):
        self.children.append(node)
        return self


def _children(n):
    return n.children


class _Annotated:
    """Post-order enumeration + leftmost descendants + keyroots (per Zhang-Shasha)."""

    def __init__(self, root):
        self.nodes, self.ids, self.lmds = [], [], []
        stack, pstack, j = [(root, [])], [], 0
        while stack:
            n, anc = stack.pop()
            nid = j
            for c in _children(n):
                stack.append((c, [nid] + anc))
            pstack.append(((n, nid), anc))
            j += 1
        lmds, keyroots, i = {}, {}, 0
        while pstack:
            (n, nid), anc = pstack.pop()
            self.nodes.append(n)
            self.ids.append(nid)
            if not _children(n):
                lmd = i
                for a in anc:
                    if a not in lmds:
                        lmds[a] = i
                    else:
                        break
            else:
                lmd = lmds[nid]
            self.lmds.append(lmd)
            keyroots[lmd] = i
            i += 1
        self.keyroots = sorted(keyroots.values())


def tree_edit_distance(root1, root2, insert_cost, remove_cost, update_cost):
    A, B = _Annotated(root1), _Annotated(root2)
    An, Bn, Al, Bl = A.nodes, B.nodes, A.lmds, B.lmds
    treedists = [[0.0] * len(Bn) for _ in range(len(An))]

    def treedist(i, j):
        m, n = i - Al[i] + 2, j - Bl[j] + 2
        ioff, joff = Al[i] - 1, Bl[j] - 1
        fd = [[0.0] * n for _ in range(m)]
        for x in range(1, m):
            fd[x][0] = fd[x - 1][0] + remove_cost(An[x + ioff])
        for y in range(1, n):
            fd[0][y] = fd[0][y - 1] + insert_cost(Bn[y + joff])
        for x in range(1, m):
            for y in range(1, n):
                if Al[i] == Al[x + ioff] and Bl[j] == Bl[y + joff]:
                    fd[x][y] = min(fd[x - 1][y] + remove_cost(An[x + ioff]),
                                   fd[x][y - 1] + insert_cost(Bn[y + joff]),
                                   fd[x - 1][y - 1] + update_cost(An[x + ioff], Bn[y + joff]))
                    treedists[x + ioff][y + joff] = fd[x][y]
                else:
                    p, q = Al[x + ioff] - 1 - ioff, Bl[y + joff] - 1 - joff
                    fd[x][y] = min(fd[x - 1][y] + remove_cost(An[x + ioff]),
                                   fd[x][y - 1] + insert_cost(Bn[y + joff]),
                                   fd[p][q] + treedists[x + ioff][y + joff])

    for i in A.keyroots:
        for j in B.keyroots:
            treedist(i, j)
    return treedists[-1][-1]


# ---------------------------------------------------------------------------
# Donut tree construction, normalisation, and costs
# ---------------------------------------------------------------------------

LEAF = "<leaf>"


def normalize(data):
    """Donut's normalize_dict: wrap scalar/dict values in lists, drop empties, sort
    dict keys by (len, key). Reproduced exactly so nTED matches published numbers
    (values are compared later by character edit distance; lists stay ordered)."""
    if not data:
        return {}
    if isinstance(data, dict):
        out = {}
        for k in sorted(data.keys(), key=lambda x: (len(str(x)), str(x))):
            v = normalize(data[k])
            if v:
                out[str(k)] = v if isinstance(v, list) else [v]
        return out
    if isinstance(data, list):
        if all(isinstance(x, dict) for x in data):
            return [x for x in (normalize(i) for i in data) if x]
        return [str(x).strip() for x in data
                if isinstance(x, (str, int, float)) and str(x).strip()]
    return [str(data).strip()]


def build_tree(data, name="<root>"):
    """Donut's construct_tree_from_dict on normalize()d data (values are lists)."""
    node = Node(name)
    if isinstance(data, dict):
        for k, v in data.items():
            kid = Node(k)
            node.addkid(kid)
            if isinstance(v, list) and all(isinstance(it, dict) for it in v):
                for it in v:
                    kid.addkid(build_tree(it, "<subtree>"))
            else:
                for val in v:
                    kid.addkid(Node(LEAF + str(val)))
    elif isinstance(data, list):
        for it in data:
            node.addkid(build_tree(it, "<subtree>") if isinstance(it, dict)
                        else Node(LEAF + str(it)))
    return node


def _ins_rem_cost(node):
    if LEAF in node.label:
        return len(node.label.replace(LEAF, ""))
    return 1


def _update_cost(a, b):
    la, lb = a.label, b.label
    al, bl = LEAF in la, LEAF in lb
    if al and bl:
        return levenshtein(la.replace(LEAF, ""), lb.replace(LEAF, ""))
    if al != bl:
        leaf = la if al else lb
        return 1 + len(leaf.replace(LEAF, ""))
    return 0 if la == lb else 1


def nted_accuracy(pred, gt):
    """max(0, 1 - TED(pred, gt) / TED(empty, gt)), Donut's normalisation."""
    p = build_tree(normalize(pred) if pred else {})
    g = build_tree(normalize(gt) if gt else {})
    empty = build_tree({})
    denom = tree_edit_distance(empty, g, _ins_rem_cost, _ins_rem_cost, _update_cost)
    if denom == 0:
        return 1.0
    dist = tree_edit_distance(p, g, _ins_rem_cost, _ins_rem_cost, _update_cost)
    return max(0.0, 1.0 - dist / denom)


# ---------------------------------------------------------------------------
# Field-level F1 (order-agnostic, indices stripped, exact value match)
# ---------------------------------------------------------------------------

def flatten_fields(data, key="", out=None):
    """(field-path, value) pairs; list items share their parent path (indices
    stripped) so menu lines are matched as a multiset, not by position."""
    out = [] if out is None else out
    if isinstance(data, dict):
        for k, v in data.items():
            path = f"{key}.{str(k)}".lstrip(".")
            flatten_fields(v, path, out)
    elif isinstance(data, list):
        for v in data:
            flatten_fields(v, key, out)
    else:
        out.append((key, str(data).strip()))
    return out


def corpus_field_f1(preds, golds):
    """Donut cal_f1 over a corpus: normalise, flatten to (path, value), exact match,
    F1 = tp/(tp+(fp+fn)/2)."""
    tp = fpfn = 0
    for pred, gold in zip(preds, golds):
        p = flatten_fields(normalize(pred) if pred else {})
        g = flatten_fields(normalize(gold) if gold else {})
        for field in p:
            if field in g:
                tp += 1
                g.remove(field)
            else:
                fpfn += 1
        fpfn += len(g)
    return tp / (tp + fpfn / 2) if (tp + fpfn) else 0.0


def corpus_nted(preds, golds):
    if not preds:
        return 0.0
    return sum(nted_accuracy(p, g) for p, g in zip(preds, golds)) / len(preds)
