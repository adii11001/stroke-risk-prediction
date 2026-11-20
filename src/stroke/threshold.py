"""Decision-threshold selection."""
import numpy as np
from sklearn.metrics import precision_recall_curve


def threshold_for_recall(y_true, proba, target_recall: float = 0.80) -> float:
    """Return the highest probability threshold whose recall is >= target_recall.

    The highest qualifying threshold keeps as many false positives out as
    possible while still catching the required share of stroke cases.
    """
    _, recall, thresholds = precision_recall_curve(y_true, proba)
    # recall[i] corresponds to thresholds[i]; the final recall entry has no threshold
    ok = np.where(recall[:-1] >= target_recall)[0]
    if len(ok) == 0:
        return float(thresholds.min())
    return float(thresholds[ok.max()])
