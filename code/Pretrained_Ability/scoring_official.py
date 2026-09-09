"""Official scoring for OCRBench and MMBench, ported from the source implementations.

OCRBench -- ported from Yuliang-Liu/MultimodalOCR, OCRBench/example.py:
    default : answer.lower().strip().replace("\\n"," ")  in  predict.lower().strip().replace("\\n"," ")
    HME100k : answer.strip().replace("\\n"," ").replace(" ","")  in  predict likewise, NO lowercasing
    score   : 1 point per sample, total out of 1000, with five component subtotals.

MMBench -- ported from open-compass/VLMEvalKit, vlmeval/utils/matching_util.py:
    can_infer = can_infer_option, falling back to can_infer_text.
    NOTE ON CIRCULAR EVAL: the official leaderboard runs each question four times with
    rotated options and credits it only if all four are right. We run VANILLA (single
    pass) because this study reports the CHANGE across reversion dials, not a
    leaderboard-comparable absolute. Absolutes here will therefore read a little higher
    than published circular numbers; the comparison across dials is unaffected.
    The official pipeline also falls back to an LLM extractor when can_infer fails; we
    do not, and count those as parse failures instead -- which is the point of the
    confound analysis, so they are reported rather than hidden.
"""
import copy as cp
import re
import string

# ----------------------------------------------------------------- OCRBench
OCRBENCH_COMPONENTS = {
    "Text Recognition": (["Regular Text Recognition", "Irregular Text Recognition",
                          "Artistic Text Recognition", "Handwriting Recognition",
                          "Digit String Recognition", "Non-Semantic Text Recognition"], 300),
    "Scene Text-centric VQA": (["Scene Text-centric VQA"], 200),
    "Doc-oriented VQA": (["Doc-oriented VQA"], 200),
    "Key Information Extraction": (["Key Information Extraction"], 200),
    "Handwritten Mathematical Expression Recognition":
        (["Handwritten Mathematical Expression Recognition"], 100),
}


def ocrbench_correct(predict, answers, dataset_name):
    """Exactly the official rule, including the HME100k special case."""
    if not isinstance(answers, (list, tuple)):
        answers = [answers]
    if dataset_name == "HME100k":
        p = str(predict).strip().replace("\n", " ").replace(" ", "")
        for a in answers:
            if str(a).strip().replace("\n", " ").replace(" ", "") in p:
                return True
        return False
    p = str(predict).lower().strip().replace("\n", " ")
    for a in answers:
        if str(a).lower().strip().replace("\n", " ") in p:
            return True
    return False


# ----------------------------------------------------------------- MMBench
_VERBOSE_ANSWER_RE = re.compile(r"(?i)(?:correct\s+)?answer\s+is\s+\**([ABCD])\**")


def can_infer_option(answer, choices):
    """Port of VLMEvalKit can_infer_option. `choices` is a dict letter -> text."""
    if "Failed to obtain answer via API" in answer:
        return False
    reject = ["Sorry, I can't help with images of people yet.", "I can't process this file.",
              "I'm sorry, but without the image provided", "Cannot determine the answer"]
    for err in reject:
        if err in answer:
            return "Z"

    def count_choice(splits, ch, prefix="", suffix=""):
        return sum(1 for c in ch if prefix + c + suffix in splits)

    answer_mod = cp.copy(answer)
    for c in ".()[],:;!*#{}":
        answer_mod = answer_mod.replace(c, " ")
    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if ch in splits and splits.index(ch) > (len(splits) - 5):
                return ch
    elif count == 0 and count_choice(splits, {"Z", ""}) == 1:
        return "Z"

    m = _VERBOSE_ANSWER_RE.search(answer or "")
    if m and m.group(1).upper() in choices:
        return m.group(1).upper()
    return False


def can_infer_text(answer, choices):
    """Port of VLMEvalKit can_infer_text, including the length guard."""
    answer = answer.lower()
    if len(answer) > 2 * sum(len(str(v)) for v in choices.values()):
        return False
    ch = {k: str(v).lower() for k, v in choices.items()}
    cands = [k for k in ch if ch[k] in answer]
    return cands[0] if len(cands) == 1 else False


def can_infer(answer, choices):
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)
