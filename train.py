"""Train, tune and evaluate stroke-risk classifiers.

Usage:  python train.py [--target-recall 0.80]

Protocol
--------
1. Stratified 80/20 train/test split (the test set is used only for final scoring).
2. For each model: GridSearchCV with 5-fold StratifiedKFold, refit on ROC-AUC.
3. Decision threshold: chosen on out-of-fold training predictions as the highest
   threshold reaching the target recall, then applied unchanged to the test set.
4. Metrics (recall, precision, F1, ROC-AUC, PR-AUC, accuracy) at 0.5 and at the
   tuned threshold are written to results/, together with plots.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,  # noqa: E402
                             RocCurveDisplay, accuracy_score, average_precision_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,  # noqa: E402
                                     cross_val_predict, train_test_split)

sys.path.insert(0, str(Path(__file__).parent / "src"))
from stroke.data import load_data, split_xy  # noqa: E402
from stroke.models import SEED, candidates  # noqa: E402
from stroke.threshold import threshold_for_recall  # noqa: E402

RESULTS = Path(__file__).parent / "results"
warnings.filterwarnings("ignore", category=UserWarning)


def scores(y, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "recall": recall_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred),
        "accuracy": accuracy_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
    }


def main(target_recall: float):
    RESULTS.mkdir(exist_ok=True)
    X, y = split_xy(load_data())
    print(f"Records: {len(y)} | stroke cases: {y.sum()} ({y.mean():.1%}) | "
          f"missing BMI: {X['bmi'].isna().sum()}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows, fitted = [], {}
    for name, (pipe, grid) in candidates(pos_weight).items():
        search = GridSearchCV(pipe, grid, cv=cv, n_jobs=-1,
                              scoring={"roc_auc": "roc_auc", "recall": "recall", "f1": "f1"},
                              refit="roc_auc")
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        i = search.best_index_
        oof = cross_val_predict(best, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        thr = threshold_for_recall(y_tr, oof, target_recall)
        proba = best.predict_proba(X_te)[:, 1]
        fitted[name] = (best, proba, thr)

        default, tuned = scores(y_te, proba, 0.5), scores(y_te, proba, thr)
        row = {"model": name,
               "cv_roc_auc": search.cv_results_["mean_test_roc_auc"][i],
               "cv_recall@0.5": search.cv_results_["mean_test_recall"][i],
               "cv_f1@0.5": search.cv_results_["mean_test_f1"][i],
               **{f"test_{k}@0.5": v for k, v in default.items() if k != "threshold"},
               **{f"test_{k}@tuned": v for k, v in tuned.items()},
               "best_params": json.dumps({k.replace("clf__", ""): v
                                          for k, v in search.best_params_.items()})}
        rows.append(row)
        print(f"{name:30s} CV AUC {row['cv_roc_auc']:.3f} | test AUC {tuned['roc_auc']:.3f} | "
              f"@{thr:.3f}: recall {tuned['recall']:.2f} precision {tuned['precision']:.2f} "
              f"F1 {tuned['f1']:.2f} (F1@0.5 {default['f1']:.2f})", flush=True)

    table = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False)
    table.round(4).to_csv(RESULTS / "model_comparison.csv", index=False)

    # Best model is chosen on cross-validated ROC-AUC (training data only).
    best_name = table.iloc[0]["model"]
    best_model, best_proba, best_thr = fitted[best_name]
    summary = {"best_model": best_name, "selected_on": "cv_roc_auc",
               "target_recall": target_recall,
               "test": {k: round(float(v), 4) for k, v in scores(y_te, best_proba, best_thr).items()},
               "test_at_0.5": {k: round(float(v), 4) for k, v in scores(y_te, best_proba, 0.5).items()},
               "params": json.loads(table.iloc[0]["best_params"])}
    (RESULTS / "best_model.json").write_text(json.dumps(summary, indent=2))
    print("\nBest model:", json.dumps(summary, indent=2))

    plot(fitted, y_te, best_name, best_model, best_proba, best_thr)


def plot(fitted, y_te, best_name, best_model, best_proba, best_thr):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for name, (_, proba, _) in fitted.items():
        RocCurveDisplay.from_predictions(y_te, proba, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_te, proba, name=name, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_title("ROC curves (test set)")
    axes[1].axhline(y_te.mean(), color="k", ls="--", lw=0.8)
    axes[1].set_title("Precision-recall curves (test set)")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "roc_pr_curves.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, t, label in [(axes[0], 0.5, "threshold 0.50"),
                         (axes[1], best_thr, f"tuned threshold {best_thr:.3f}")]:
        ConfusionMatrixDisplay.from_predictions(y_te, (best_proba >= t).astype(int),
                                                display_labels=["No stroke", "Stroke"],
                                                cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"{best_name}: {label}")
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrices.png", dpi=130)
    plt.close(fig)

    clf = best_model.named_steps["clf"]
    names = best_model.named_steps["pre"].get_feature_names_out()
    imp = getattr(clf, "feature_importances_", None)
    if imp is None and hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0])
    if imp is not None:
        s = pd.Series(imp, index=names).sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        s.plot.barh(ax=ax, color="#3b6ea5")
        ax.set_title(f"{best_name}: top feature importances")
        fig.tight_layout()
        fig.savefig(RESULTS / "feature_importance.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target-recall", type=float, default=0.80)
    main(p.parse_args().target_recall)
