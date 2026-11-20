"""Candidate models and their hyperparameter grids.

Class imbalance (~4.9% positives) is handled with class weights by default.
Two SMOTE variants (via imbalanced-learn) are included for comparison.
"""
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .data import build_preprocessor

SEED = 42


def candidates(pos_weight: float):
    """Return {name: (pipeline, param_grid)}. pos_weight = n_negative / n_positive."""

    def pipe(model):
        return Pipeline([("pre", build_preprocessor()), ("clf", model)])

    def smote_pipe(model):
        return ImbPipeline([("pre", build_preprocessor()),
                            ("smote", SMOTE(random_state=SEED)),
                            ("clf", model)])

    xgb_base = dict(eval_metric="logloss", tree_method="hist", random_state=SEED, n_jobs=1)

    return {
        "Logistic Regression": (
            pipe(LogisticRegression(class_weight="balanced", max_iter=2000)),
            {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3]},
        ),
        "Decision Tree": (
            pipe(DecisionTreeClassifier(class_weight="balanced", random_state=SEED)),
            {"clf__max_depth": [3, 4, 5, 6, 8],
             "clf__min_samples_leaf": [10, 25, 50, 100]},
        ),
        "Random Forest": (
            pipe(RandomForestClassifier(class_weight="balanced_subsample",
                                        n_estimators=300, random_state=SEED, n_jobs=1)),
            {"clf__max_depth": [4, 6, 8, None],
             "clf__min_samples_leaf": [5, 10, 25],
             "clf__max_features": ["sqrt", 0.5]},
        ),
        "XGBoost": (
            pipe(XGBClassifier(scale_pos_weight=pos_weight, **xgb_base)),
            {"clf__n_estimators": [100, 200, 400],
             "clf__max_depth": [2, 3, 4],
             "clf__learning_rate": [0.01, 0.03, 0.1],
             "clf__subsample": [0.8],
             "clf__colsample_bytree": [0.8],
             "clf__min_child_weight": [1, 5]},
        ),
        "Logistic Regression + SMOTE": (
            smote_pipe(LogisticRegression(max_iter=2000)),
            {"clf__C": [0.01, 0.1, 1]},
        ),
        "XGBoost + SMOTE": (
            smote_pipe(XGBClassifier(**xgb_base)),
            {"clf__n_estimators": [100, 200],
             "clf__max_depth": [2, 3],
             "clf__learning_rate": [0.03, 0.1]},
        ),
    }
