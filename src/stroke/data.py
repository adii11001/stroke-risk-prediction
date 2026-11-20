"""Data loading and preprocessing for the stroke dataset."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "healthcare-dataset-stroke-data.csv"
TARGET = "stroke"
NUMERIC = ["age", "avg_glucose_level", "bmi"]
BINARY = ["hypertension", "heart_disease"]
CATEGORICAL = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Read the CSV, parse 'N/A' BMI values as missing and drop the id column."""
    df = pd.read_csv(path, na_values=["N/A"])
    return df.drop(columns=["id"])


def split_xy(df: pd.DataFrame):
    return df[NUMERIC + BINARY + CATEGORICAL], df[TARGET].astype(int)


def build_preprocessor() -> ColumnTransformer:
    """Median-impute + scale numerics, pass binaries through, one-hot categoricals."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC),
            ("bin", "passthrough", BINARY),
            ("cat", categorical, CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )
