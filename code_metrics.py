from collections.abc import Sequence
from typing import TypedDict


class Counts(TypedDict):
    TP: int
    FP: int
    FN: int
    TN: int
    precision: float
    recall: float
    f1: float


def counts(truth: Sequence[bool], scores: Sequence[float], threshold: float) -> Counts:
    """Compute image-level counts at the inclusive threshold."""
    tp = fp = fn = tn = 0
    for actual, score in zip(truth, scores, strict=True):
        predicted = score >= threshold
        tp += actual and predicted
        fp += not actual and predicted
        fn += actual and not predicted
        tn += not actual and not predicted
    return Counts(
        TP=tp, FP=fp, FN=fn, TN=tn,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    )


def select_threshold(truth: Sequence[bool], scores: Sequence[float], grid: Sequence[float]) -> float:
    """Maximize validation F1 and prefer the lower threshold on ties."""
    return max(grid, key=lambda t: (counts(truth, scores, t)["f1"], -t))
