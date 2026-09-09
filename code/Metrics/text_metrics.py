"""
metrics.py — evaluation metrics for document extraction / QA.

Reusable across every experiment (baseline, LoRA/DoRA, reversion). Two families:
  - ANLS + exact match : for string answers (DocVQA-style QA, single-field values)
  - field-level micro-F1: for structured extraction (predicted vs gold field dicts)

ANLS (Average Normalized Levenshtein Similarity) is the standard DocVQA metric:
it rewards near-correct text (OCR-ish slips) rather than demanding exact strings.
Pure-Python, no external dependencies.
"""
from __future__ import annotations

import re
import unicodedata


def _norm(s) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    return re.sub(r"\s+", " ", s)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred, golds, threshold: float = 0.5) -> float:
    """ANLS for one prediction vs one or more acceptable gold answers."""
    if isinstance(golds, str):
        golds = [golds]
    p = _norm(pred)
    best = 0.0
    for g in golds:
        g = _norm(g)
        denom = max(len(p), len(g)) or 1
        best = max(best, 1.0 - levenshtein(p, g) / denom)
    return best if best >= threshold else 0.0


def exact_match(pred, golds) -> float:
    if isinstance(golds, str):
        golds = [golds]
    return float(_norm(pred) in {_norm(g) for g in golds})


def corpus_anls(preds, golds, threshold: float = 0.5) -> float:
    scores = [anls(p, g, threshold) for p, g in zip(preds, golds)]
    return sum(scores) / len(scores) if scores else 0.0


def corpus_exact_match(preds, golds) -> float:
    scores = [exact_match(p, g) for p, g in zip(preds, golds)]
    return sum(scores) / len(scores) if scores else 0.0


def field_f1(pred: dict, gold: dict, value_threshold: float = 0.5) -> dict:
    """Micro precision/recall/F1 over (field -> value) pairs.

    A predicted field counts as a true positive if the key matches a gold key
    AND the value's ANLS vs the gold value clears `value_threshold`.
    """
    def flat(d):
        return {_norm(k): (v if isinstance(v, (list, dict)) else v)
                for k, v in (d or {}).items()}

    P, G = flat(pred), flat(gold)
    tp = sum(1 for k, gv in G.items()
             if k in P and anls(str(P[k]), str(gv), value_threshold) > 0)
    prec = tp / len(P) if P else 0.0
    rec = tp / len(G) if G else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "n_pred": len(P), "n_gold": len(G)}


def corpus_field_f1(preds, golds, value_threshold: float = 0.5) -> dict:
    """Aggregate micro-F1 over a corpus of (pred_dict, gold_dict) pairs."""
    tp = n_pred = n_gold = 0
    for p, g in zip(preds, golds):
        r = field_f1(p, g, value_threshold)
        tp += r["tp"]; n_pred += r["n_pred"]; n_gold += r["n_gold"]
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gold if n_gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "n_pred": n_pred, "n_gold": n_gold}


def value_f1(pred: dict, gold: dict, threshold: float = 0.7) -> dict:
    """Key-agnostic F1: match VALUES only (greedy, ANLS-based), ignoring key names.
    Measures reading ability without penalising a different schema."""
    pv, gv = [_norm(v) for v in pred.values()], [_norm(v) for v in gold.values()]
    used = [False] * len(pv)
    tp = 0
    for g in gv:
        best_j, best_s = -1, 0.0
        for j, p in enumerate(pv):
            if used[j]:
                continue
            s = 1.0 - levenshtein(p, g) / (max(len(p), len(g)) or 1)
            if s > best_s:
                best_s, best_j = s, j
        if best_j >= 0 and best_s >= threshold:
            used[best_j] = True
            tp += 1
    return {"tp": tp, "n_pred": len(pv), "n_gold": len(gv)}


def corpus_value_f1(preds, golds, threshold: float = 0.7) -> dict:
    tp = n_pred = n_gold = 0
    for p, g in zip(preds, golds):
        r = value_f1(p, g, threshold)
        tp += r["tp"]; n_pred += r["n_pred"]; n_gold += r["n_gold"]
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gold if n_gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "n_pred": n_pred, "n_gold": n_gold}


if __name__ == "__main__":
    # tiny self-test
    assert exact_match("Credit Agricole", "credit  agricole") == 1.0
    assert anls("Credit Agricol", "Credit Agricole") > 0.9
    assert anls("HSBC", "Credit Agricole") == 0.0
    r = field_f1({"bank": "HSBC", "total": "100"}, {"bank": "HSBC", "total": "100", "date": "x"})
    assert r["tp"] == 2 and abs(r["recall"] - 2/3) < 1e-9
    # value-F1 ignores key names: same values, different keys -> full match
    v = value_f1({"a": "60.000", "b": "TICKET CP"}, {"menu.price": "60.000", "menu.nm": "TICKET CP"})
    assert v["tp"] == 2, v
    print("metrics.py self-test passed")
    print("metrics.py self-test passed:", r)
