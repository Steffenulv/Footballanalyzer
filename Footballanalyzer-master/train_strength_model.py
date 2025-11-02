import argparse
from sklearn.base import BaseEstimator, TransformerMixin
from ml.features import FeatureBuilder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os
import joblib


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Builds features from raw season totals."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns_ = None

    def fit(self, X, y=None):
        feats = self._prepare_features(X)
        self.feature_columns_ = feats.columns
        self.scaler.fit(feats)
        return self

    def transform(self, X):
        feats = self._prepare_features(X)
        feats = feats[self.feature_columns_]
        Z = self.scaler.transform(feats)
        return pd.DataFrame(Z, columns=self.feature_columns_, index=X.index)

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        mp = np.maximum(df["mp"].astype(float), 1.0)
        df["win_ratio"] = df["w"].astype(float) / mp
        df["goals_per_game"] = df["gf"].astype(float) / mp
        df["goals_against_per_game"] = df["ga"].astype(float) / mp
        cols = ["mp", "win_ratio", "goals_per_game", "goals_against_per_game", "gd"]
        return df[cols]


def prepare_target(df: pd.DataFrame) -> np.ndarray:
    """Binary label: 1 if pts > league mean that season, else 0."""
    labels = []
    for season, block in df.groupby("sesonger"):
        mean_pts = block["pts"].mean()
        labels.extend((block["pts"] > mean_pts).astype(int).tolist())
    return np.array(labels)


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("features", FeatureBuilder()),
        ("clf", LogisticRegression(max_iter=2000))
    ])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to serie_a_full_data.csv")
    p.add_argument("--out", default="models/model.joblib", help="Where to save model")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = pd.read_csv(args.data)
    df.columns = df.columns.str.lower()

    required = {"mp","w","d","l","gf","ga","gd","pts","sesonger"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise SystemExit(f"Missing columns: {missing}")

    X = df[list(required)]
    y = prepare_target(df)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = build_pipeline()
    pipe.fit(Xtr, ytr)

    joblib.dump(pipe, args.out)
    print(f"✅ Model saved to {args.out}")

    pred = pipe.predict(Xte)
    acc = accuracy_score(yte, pred)
    print(f"Validation accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
