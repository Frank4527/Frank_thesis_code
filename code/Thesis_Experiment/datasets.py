"""Single source of truth for every dataset used by the thesis experiments.

Each experiment script says WHICH datasets it wants; nothing else hardcodes a path
or a prompt. Adding an experiment = adding an entry here, not copying a script.

The prompt travels WITH the dataset: previously each driver hardcoded its own
instruction (bank="receipt", quarterly="document"), which is how the two eval
families drifted apart. One prompt per dataset, used identically at train and
eval time.
"""
import json
import os
import random

TD = "/home/frank/data/Thesis_datasets"

# The 2026-07 project standard: one generic instruction for every extraction dataset.
DOC_PROMPT = "Extract all fields from this document as JSON."

DATASETS = {
    #  name          folder under TD        prompt      short label used in result filenames
    "bank_full":  ("Bank_Full",         DOC_PROMPT, "bank"),
    "quarterly":  ("Quarterly_rep_En",  DOC_PROMPT, "quarterly"),
    "bank_de":    ("Bank_De",           DOC_PROMPT, "de"),
    "bank_en":    ("Bank_En",           DOC_PROMPT, "en"),
    "bank_po":    ("Bank_Po",           DOC_PROMPT, "po"),
    "cord":       ("Cord",              DOC_PROMPT, "cord"),
}

SPLITS = ("training", "validation", "test")

# Short names the existing experiment scripts already pass on the command line.
# Keeping them means one dataset-driven script serves every experiment with no
# change to any caller.
ALIASES = {
    "de": "bank_de", "po": "bank_po", "en": "bank_en",
    "bank": "bank_full", "Bank_Full": "bank_full", "Quarterly_rep_En": "quarterly",
}


# Datasets that were randomly re-split three times for the multi-seed runs.
# Everything else (per-language bank sets, cord) has a single fixed split.
SEEDED = {"bank_full", "quarterly"}
VALID_SEEDS = (1, 2, 3)


def data_seed():
    """Which random re-split to use, from $DATA_SEED (default 1).

    This is the DATA split seed and is deliberately separate from the training
    seed ($SEED in config.sh): seed 1 is the original split that produced every
    result before 2026-08, so the default keeps old runs reproducible.
    """
    raw = os.environ.get("DATA_SEED")
    if raw is None:
        raise RuntimeError(
            "DATA_SEED is not set. It has no default on purpose: falling back to "
            "seed 1 inside a seed-2/3 run would evaluate on documents the model "
            "was trained on. Set it explicitly (DATA_SEED=1 reproduces every "
            "pre-2026-08 result).")
    try:
        s = int(raw)
    except ValueError:
        raise ValueError(f"DATA_SEED must be an integer, got {raw!r}")
    if s not in VALID_SEEDS:
        raise ValueError(f"DATA_SEED must be one of {VALID_SEEDS}, got {s}")
    return s


def resolve(name):
    return ALIASES.get(name, name)


def folder(name):
    """Directory under TD for this dataset, honouring DATA_SEED where it applies."""
    key = resolve(name)
    base = _entry(key)[0]
    if key not in SEEDED:
        return base
    seeded = f"{base}_seed{data_seed()}"
    if not os.path.isdir(os.path.join(TD, seeded)):
        raise FileNotFoundError(
            f"DATA_SEED={data_seed()} requested but {os.path.join(TD, seeded)} is missing")
    return seeded


def _entry(name):
    name = resolve(name)
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    return DATASETS[name]


def _audit(path):
    """Record every manifest actually opened, so seed isolation can be proven."""
    dest = os.environ.get("MANIFEST_AUDIT")
    if not dest:
        return
    try:
        with open(dest, "a") as f:
            f.write(f"{os.environ.get('DATA_SEED', '?')}\t{path}\n")
    except OSError:
        pass          # auditing must never break a run


def manifest(name, split):
    """Absolute path to one split's manifest.jsonl."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    p = os.path.join(TD, folder(name), split, "manifest.jsonl")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    _audit(p)
    return p


def prompt(name):
    """The extraction instruction this dataset is trained and scored with."""
    return _entry(name)[1]


def label(name):
    """Short name used in result filenames / table rows."""
    return _entry(name)[2]


def rows(name, split):
    with open(manifest(name, split)) as f:
        return [json.loads(l) for l in f if l.strip()]


def n(name, split):
    return sum(1 for l in open(manifest(name, split)) if l.strip())


def make_mixture(new_name, old_name, ratio, out_path, split="training", seed=0):
    """Write a training manifest = ALL of `new_name` + `ratio` of `old_name`.

    This is the one primitive behind both new baselines:
      rehearsal(r) : ratio = 0.01 / 0.05   (replay a slice of the old task)
      joint        : ratio = 1.0           (train on both tasks at once)

    The replayed slice is sampled with a fixed seed, so a rerun reproduces it
    exactly. Returns (n_new, n_old_sampled, out_path).
    """
    new_rows = rows(new_name, split)
    old_rows = rows(old_name, split)
    k = int(round(len(old_rows) * ratio))
    k = min(k, len(old_rows))
    sampled = random.Random(seed).sample(old_rows, k) if k else []

    mixed = new_rows + sampled
    random.Random(seed).shuffle(mixed)          # interleave, don't append in a block

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for r in mixed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(new_rows), k, out_path


if __name__ == "__main__":
    print(f"DATA_SEED={data_seed()}")
    for nm in DATASETS:
        try:
            tag = f"  [{folder(nm)}]" if nm in SEEDED else ""
            print(f"{nm:12s} " + "  ".join(f"{s}={n(nm, s)}" for s in SPLITS) + tag)
        except FileNotFoundError as e:
            print(f"{nm:12s} MISSING {e}")
