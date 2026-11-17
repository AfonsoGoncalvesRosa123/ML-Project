# %% [markdown]
# # Predicting flame extinction with sound waves: modelling (milestone 2)
#
# Continues milestone 1 (data preparation). Same steps as before (winsorisation,
# two new features, one-hot encoding, scaling), now inside ONE pipeline so that
# everything learned from the data (outlier limits, scaler) comes only from the
# training set. That is what prevents data leakage.
#
# Usage:  python ml_milestone2_modelling.py path/to/dataset.csv [output_dir]
# In VS Code / Jupyter the "# %%" lines work as notebook cells.

# %% Imports
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
NUMERIC = ["distance", "desibel", "airflow", "frequency"]  # "desibel" is the dataset's spelling
CATEGORICAL = ["fuel"]


# %% Load data
def load_data(csv_path):
    # Accepts .csv or Excel (.xlsx needs the openpyxl package; the first sheet is read)
    if str(csv_path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(csv_path)
    else:
        df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    # size is ordinal (1 = small ... 7 = large). If it came as text, map it to integers first.
    df["size"] = pd.to_numeric(df["size"], errors="raise")
    df["status"] = df["status"].astype(int)
    return df


# %% Custom transformer: winsorise (IQR) + create the two new features
class WinsoriseAndEngineer(BaseEstimator, TransformerMixin):
    """Learns IQR limits on the training data, clips outliers (keeps every row)
    and adds energy_index and intensity_per_size."""

    def __init__(self, columns=tuple(NUMERIC), k=1.5):
        self.columns = columns
        self.k = k

    def fit(self, X, y=None):
        self.limits_ = {}
        for col in self.columns:
            q1, q3 = X[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.limits_[col] = (q1 - self.k * iqr, q3 + self.k * iqr)
        return self

    def transform(self, X):
        X = X.copy()
        for col, (low, high) in self.limits_.items():
            X[col] = X[col].clip(low, high)
        X["energy_index"] = X["desibel"] * X["airflow"]
        X["intensity_per_size"] = X["desibel"] / X["size"]
        return X


def build_pipeline(model):
    numeric_cols = NUMERIC + ["size", "energy_index", "intensity_per_size"]
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline(
        [
            ("features", WinsoriseAndEngineer()),
            ("preprocess", preprocess),
            ("model", model),
        ]
    )


# %% Models to compare (a baseline plus three real models)
MODELS = {
    "Baseline (most frequent)": DummyClassifier(strategy="most_frequent"),
    "Logistic regression": LogisticRegression(max_iter=1000),
    "Random forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    "Hist. gradient boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
}

PARAM_SPACES = {
    "Random forest": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_leaf": [1, 2, 5],
        "model__max_features": ["sqrt", 0.5, 1.0],
    },
    "Hist. gradient boosting": {
        "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "model__max_iter": [100, 200, 400],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__l2_regularization": [0.0, 0.1, 1.0],
    },
}


def main(csv_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    X = df.drop(columns="status")
    y = df["status"]

    # Same split as milestone 1: 80/20, stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | "
          f"extinguished in train: {y_train.mean():.1%}")

    # %% 1) Compare models with 5-fold stratified cross-validation (train set only)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in MODELS.items():
        res = cross_validate(
            build_pipeline(model), X_train, y_train, cv=cv,
            scoring=["accuracy", "f1", "roc_auc"], n_jobs=1,
        )
        rows.append({
            "model": name,
            "accuracy": res["test_accuracy"].mean(),
            "f1": res["test_f1"].mean(),
            "roc_auc": res["test_roc_auc"].mean(),
            "f1_std": res["test_f1"].std(),
        })
        print(f"{name:28s} acc={rows[-1]['accuracy']:.3f}  f1={rows[-1]['f1']:.3f}  "
              f"auc={rows[-1]['roc_auc']:.3f}")
    cv_table = pd.DataFrame(rows).sort_values("f1", ascending=False)
    cv_table.to_csv(out / "cv_results.csv", index=False)

    # %% 2) Tune the best tree-based model
    best_name = cv_table[cv_table["model"].isin(PARAM_SPACES)].iloc[0]["model"]
    print(f"\nTuning: {best_name}")
    search = RandomizedSearchCV(
        build_pipeline(MODELS[best_name]), PARAM_SPACES[best_name],
        n_iter=15, scoring="f1", cv=cv, random_state=RANDOM_STATE, n_jobs=1,
    )
    search.fit(X_train, y_train)
    print("Best parameters:", search.best_params_)
    print(f"CV F1 after tuning: {search.best_score_:.3f}")
    final = search.best_estimator_

    # %% 3) Final evaluation on the test set (used only once)
    pred = final.predict(X_test)
    proba = final.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, pred, target_names=["not extinguished", "extinguished"],
                                   output_dict=True)
    print("\n" + classification_report(y_test, pred, target_names=["not extinguished", "extinguished"]))
    auc = roc_auc_score(y_test, proba)
    print(f"ROC AUC (test): {auc:.3f}")
    (out / "test_metrics.json").write_text(json.dumps(
        {"model": best_name, "best_params": search.best_params_, "roc_auc": auc, "report": report},
        indent=2, default=str))

    # %% 4) Figures for the README
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["not extinguished", "extinguished"], cmap="Blues", ax=ax)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax, name=best_name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC curve (test set)")
    fig.tight_layout(); fig.savefig(out / "roc_curve.png", dpi=150); plt.close(fig)

    # Permutation importance on the ORIGINAL columns: which experimental settings matter most?
    imp = permutation_importance(final, X_test, y_test, scoring="f1", n_repeats=10,
                                 random_state=RANDOM_STATE, n_jobs=1)
    order = np.argsort(imp.importances_mean)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(X_test.columns[order], imp.importances_mean[order], xerr=imp.importances_std[order])
    ax.set_xlabel("Drop in F1 when the column is shuffled")
    ax.set_title("Which variables matter most?")
    fig.tight_layout(); fig.savefig(out / "permutation_importance.png", dpi=150); plt.close(fig)

    print(f"\nFiles saved in: {out.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python ml_milestone2_modelling.py path/to/dataset.csv [output_dir]")
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
