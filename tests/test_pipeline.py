import numpy as np
import pandas as pd
import pytest

from stroke.data import load_data, split_xy, build_preprocessor, NUMERIC, CATEGORICAL, BINARY
from stroke.threshold import threshold_for_recall


@pytest.fixture(scope="module")
def df():
    return load_data()


def test_load_data_shape_and_target(df):
    assert len(df) == 5110
    assert int(df["stroke"].sum()) == 249
    assert "id" not in df.columns
    assert int(df["bmi"].isna().sum()) == 201


def test_split_xy_separates_target(df):
    X, y = split_xy(df)
    assert "stroke" not in X.columns
    assert set(NUMERIC + CATEGORICAL + BINARY) == set(X.columns)
    assert y.mean() == pytest.approx(249 / 5110)


def test_preprocessor_imputes_encodes_scales(df):
    X, _ = split_xy(df)
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    assert not np.isnan(Xt).any()
    # numeric columns come first and are standardised
    num = Xt[:, : len(NUMERIC)]
    assert np.allclose(num.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(num.std(axis=0), 1, atol=1e-6)
    # one-hot expands categoricals -> more columns than raw features
    assert Xt.shape[1] > X.shape[1]


def test_preprocessor_handles_unseen_category(df):
    X, _ = split_xy(df)
    pre = build_preprocessor().fit(X)
    row = X.iloc[[0]].copy()
    row["work_type"] = "Astronaut"
    assert not np.isnan(pre.transform(row)).any()


def test_threshold_for_recall_meets_target():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.5, 0.7, 0.9])
    t = threshold_for_recall(y, p, target_recall=0.75)
    pred = p >= t
    recall = (pred & (y == 1)).sum() / (y == 1).sum()
    assert recall >= 0.75
    # highest such threshold: 0.5 gives recall 0.75 with one FP (0.6)
    assert t == pytest.approx(0.5)


def test_threshold_for_recall_full_recall():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.9, 0.2, 0.1, 0.8])
    t = threshold_for_recall(y, p, target_recall=1.0)
    assert ((p >= t) & (y == 1)).sum() == 2
