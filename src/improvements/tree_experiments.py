import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parents[2]
TARGET = "Revenue"
GRIDS = {
    "max_depth": {"max_depth": [3, 5, 7, 10, 15, 20, None]},
    "min_samples_leaf": {"min_samples_leaf": [1, 2, 5, 10, 20, 50, 100]},
    # Select the splitting criterion by CV instead of using the held-out test set.
    "criterion": {"criterion": ["gini", "entropy"]},
}


def load_data():
    train = pd.read_csv(BASE_DIR / "dataset/online_shoppers_train.csv")
    test = pd.read_csv(BASE_DIR / "dataset/online_shoppers_test.csv")
    if TARGET not in train or TARGET not in test:
        raise ValueError("Train and test must contain Revenue.")
    features = train.drop(columns=TARGET).columns
    if not features.equals(test.drop(columns=TARGET).columns):
        raise ValueError("Train/test feature columns or order do not match.")
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError("Missing values found.")
    for frame in (train, test):
        if set(frame[TARGET].unique()) != {0, 1}:
            raise ValueError("Revenue must contain binary labels 0 and 1.")
    return train[features], train[TARGET], test[features], test[TARGET]

def evaluate(model, X_train, y_train, X_test, y_test):
    predicted = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y_test, predicted)
    return {
        "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
        "test_accuracy": accuracy,
        "error_rate": 1 - accuracy,
        "precision": precision_score(y_test, predicted, zero_division=0),
        "recall": recall_score(y_test, predicted, zero_division=0),
        "f1": f1_score(y_test, predicted, zero_division=0),
        "depth": model.get_depth(), "leaves": model.get_n_leaves(),
        "nodes": model.tree_.node_count,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }, predicted

def main():
    X_train, y_train, X_test, y_test = load_data()
    for folder in ("dataset", "model", "img", "doc"):
        (BASE_DIR / folder).mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    searches = {}
    # Identical folds for every experiment; test labels never enter selection.
    for name, grid in GRIDS.items():
        print(f"Cross-validation: {name}", flush=True)
        search = GridSearchCV(
            DecisionTreeClassifier(random_state=42), grid,
            scoring={"accuracy": "accuracy", "f1": "f1", "recall": "recall"},
            refit="accuracy", cv=cv, n_jobs=1, return_train_score=True,
            error_score="raise",
        ).fit(X_train, y_train)
        searches[name] = search
        pd.DataFrame(search.cv_results_).to_csv(
            BASE_DIR / f"dataset/improvements_cv_{name}.csv", index=False,
        )

    # Freeze all choices before final evaluation. Baseline is retrained with
    # the original configuration, without overwriting the existing artifact.
    models = {"baseline": DecisionTreeClassifier(random_state=42).fit(X_train, y_train)}
    models.update({name: search.best_estimator_ for name, search in searches.items()})
    baseline_cv_accuracy = cross_val_score(
        DecisionTreeClassifier(random_state=42), X_train, y_train,
        scoring="accuracy", cv=cv, n_jobs=1,
    ).mean()
    rows = []
    predictions = pd.DataFrame({"actual": y_test})
    for name, model in models.items():
        metrics, predicted = evaluate(model, X_train, y_train, X_test, y_test)
        params = searches[name].best_params_ if name in searches else {"criterion": "gini"}
        cv_accuracy = (searches[name].best_score_ if name in searches else
                       baseline_cv_accuracy)
        rows.append({"model": name, "parameters": json.dumps(params),
                     "cv_accuracy": cv_accuracy, **metrics})
        predictions[name] = predicted
        joblib.dump(model, BASE_DIR / f"model/improvements_{name}.joblib")
    results = pd.DataFrame(rows)
    results.to_csv(BASE_DIR / "dataset/improvements_results.csv", index=False)
    predictions.to_csv(BASE_DIR / "dataset/improvements_predictions.csv", index=False)
    metadata = {
        "sklearn_version": sklearn.__version__, "random_state": 42,
        "train_samples": len(y_train), "test_samples": len(y_test),
        "features": X_train.shape[1], "cv_folds": 5,
        "selection_metric": "accuracy", "positive_class": 1,
        "grids": GRIDS,
        "best_parameters": {name: search.best_params_ for name, search in searches.items()},
        "selected_by_cv": max(searches, key=lambda name: searches[name].best_score_),
    }
    (BASE_DIR / "doc/improvements_run.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    results.set_index("model")[["train_accuracy", "test_accuracy", "f1"]].plot.bar(ax=axes[0])
    axes[0].set(ylim=(0, 1.05), title="Decision tree improvement: performance", ylabel="Score", xlabel="")
    results.set_index("model")[["depth", "leaves"]].plot.bar(ax=axes[1], logy=True)
    axes[1].set(title="Tree complexity (log scale)", ylabel="Count", xlabel="")
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(BASE_DIR / "img/improvements_comparison.png", dpi=160)
    plt.close(fig)
    print(results.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
