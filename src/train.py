from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data import combine_text_columns, load_dataset
from src.evaluate import save_evidently_classification_report


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_pipeline(c: float, min_df: int, ngram_max: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents=None,
                    min_df=min_df,
                    ngram_range=(1, ngram_max),
                    sublinear_tf=True,
                    max_features=150_000,
                ),
            ),
            ("classifier", LinearSVC(C=c, class_weight="balanced")),
        ]
    )


def train(config_path: str | Path) -> dict:
    config = load_config(config_path)
    data_cfg, train_cfg = config["data"], config["training"]
    frame = load_dataset(data_cfg["path"], data_cfg["separator"], data_cfg["encoding"])
    target = data_cfg["target_column"]
    frame = frame.dropna(subset=[target]).copy()
    counts = frame[target].value_counts()
    excluded_classes = {
        str(label): int(count)
        for label, count in counts[counts < train_cfg["min_class_count"]].items()
    }
    eligible = counts[counts >= train_cfg["min_class_count"]].index
    frame = frame[frame[target].isin(eligible)].copy()
    if frame[target].nunique() < 2:
        raise ValueError("At least two target classes with enough samples are required.")

    x = combine_text_columns(frame, data_cfg["text_columns"])
    y = frame[target].astype(str)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=train_cfg["test_size"],
        random_state=train_cfg["random_state"],
        stratify=y,
    )
    smallest_class = int(y_train.value_counts().min())
    folds = min(train_cfg["cv_folds"], smallest_class)
    if folds < 2:
        raise ValueError("Training split needs at least two examples per class for cross-validation.")
    cv = StratifiedKFold(folds, shuffle=True, random_state=train_cfg["random_state"])

    def objective(trial: optuna.Trial) -> float:
        pipeline = make_pipeline(
            c=trial.suggest_float("c", 0.05, 10.0, log=True),
            min_df=trial.suggest_int("min_df", 1, 5),
            ngram_max=trial.suggest_int("ngram_max", 1, 2),
        )
        scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        return float(np.mean(scores))

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run() as run:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=train_cfg["n_trials"])
        model = make_pipeline(**study.best_params)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        macro_f1 = f1_score(y_test, prediction, average="macro")
        weighted_f1 = f1_score(y_test, prediction, average="weighted")

        output_dir = Path("reports/generated")
        model_dir = Path("models")
        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        report = classification_report(y_test, prediction, output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(output_dir / "classification_report.csv")
        labels = sorted(y.unique())
        pd.DataFrame(
            confusion_matrix(y_test, prediction, labels=labels), index=labels, columns=labels
        ).to_csv(output_dir / "confusion_matrix.csv")
        save_evidently_classification_report(
            y_test, prediction, output_dir / "evidently_classification.html"
        )
        joblib.dump(model, model_dir / "best_model.joblib")

        mlflow.log_params(study.best_params)
        mlflow.log_metrics(
            {
                "cv_macro_f1": study.best_value,
                "test_macro_f1": macro_f1,
                "test_weighted_f1": weighted_f1,
            }
        )
        mlflow.log_artifacts(str(output_dir), artifact_path="evaluation")
        mlflow.sklearn.log_model(model, name="model")

        summary = {
            "run_id": run.info.run_id,
            "rows": len(frame),
            "classes": int(y.nunique()),
            "excluded_classes": excluded_classes,
            "best_params": study.best_params,
            "cv_macro_f1": study.best_value,
            "test_macro_f1": macro_f1,
            "test_weighted_f1": weighted_f1,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    print(json.dumps(train(args.config), indent=2, ensure_ascii=False))
