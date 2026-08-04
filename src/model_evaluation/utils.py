"""Shared utilities for baseline, new-model and tuning notebooks.

The functions in this module are intentionally lightweight wrappers around
scikit-learn, pandas and joblib functionality. They keep model artefact names,
cross-validation summaries and text-export steps consistent across notebooks.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)

DEFAULT_MODEL_OUTPUT_DIR: Path | None = None


def set_default_model_output_dir(model_output_dir: str | Path) -> None:
    """Set the default output directory for notebook-style model calls."""

    global DEFAULT_MODEL_OUTPUT_DIR
    DEFAULT_MODEL_OUTPUT_DIR = Path(model_output_dir)


def _resolve_model_output_dir(
    model_output_dir: str | Path | None,
) -> Path:
    """Resolve an explicit or previously configured model output directory."""

    if model_output_dir is not None:
        return Path(model_output_dir)

    if DEFAULT_MODEL_OUTPUT_DIR is not None:
        return DEFAULT_MODEL_OUTPUT_DIR

    return Path.cwd() / "models"


def safe_model_filename(name: str) -> str:
    """Convert a model name into a filesystem-friendly filename stem."""

    filename = str(name).strip().lower()
    filename = re.sub(r"[^a-z0-9äöüß]+", "_", filename)
    filename = filename.strip("_")

    return filename or "model"


def make_json_safe(value: Any) -> Any:
    """Convert nested Python objects into a JSON-serialisable structure."""

    if isinstance(value, Mapping):
        return {
            str(key): make_json_safe(inner_value)
            for key, inner_value in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(inner_value) for inner_value in value]

    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())

    if isinstance(value, Path):
        return str(value)

    if inspect.isfunction(value) or inspect.ismethod(value):
        return {
            "callable": f"{value.__module__}.{value.__qualname__}"
        }

    if inspect.isclass(value):
        return {
            "class": f"{value.__module__}.{value.__qualname__}"
        }

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def create_model_signature(
    model: Any,
    extra_params: Mapping[str, Any] | None = None,
) -> str:
    """Create a stable short hash from model and run configuration.

    ``extra_params`` should contain evaluation settings that affect result
    comparability, for example CV fold count, shuffle setting, random state or
    the selected scoring metrics. This prevents a 3-fold and a 4-fold result
    from being treated as the same cached run.
    """

    if hasattr(model, "get_params"):
        raw_parameters = model.get_params(deep=True)
        parameters = {}

        for key, value in raw_parameters.items():
            # Container entries such as Pipeline.steps or
            # ColumnTransformer.transformers contain estimator repr strings.
            # Those repr strings may include callable memory addresses and can
            # change between sessions even when the effective configuration is
            # identical. The nested ``name__parameter`` entries below contain
            # the actual hyperparameters and are kept.
            container_parameter_names = {
                "steps",
                "transformers",
                "transformer_list",
                "named_steps",
                "named_transformers",
            }

            if key.split("__")[-1] in container_parameter_names:
                continue

            # Skip estimator objects themselves. Their own parameters appear as
            # nested keys, e.g. ``classifier__C`` or ``preprocessor__...``.
            if hasattr(value, "get_params"):
                continue

            parameters[key] = value
    else:
        parameters = {"repr": repr(model)}

    if extra_params:
        parameters["_run_config"] = make_json_safe(extra_params)

    payload = json.dumps(
        make_json_safe(parameters),
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


def expected_model_path(
    model_name: str,
    model_signature: str,
    subdirectory: str,
    extension: str = ".joblib",
    model_output_dir: str | Path | None = None,
) -> Path:
    """Return the expected artefact path for a signed model."""

    return (
        _resolve_model_output_dir(model_output_dir)
        / subdirectory
        / f"{safe_model_filename(model_name)}_{model_signature}{extension}"
    )


def fit_and_save_model(
    model: Any,
    model_name: str,
    X: Any,
    y: Any,
    subdirectory: str,
    model_signature: str | None = None,
    model_output_dir: str | Path | None = None,
) -> Path:
    """Fit a cloned model and save it as a signed joblib artefact."""

    signature = model_signature or create_model_signature(model)
    resolved_model_output_dir = _resolve_model_output_dir(model_output_dir)
    output_directory = resolved_model_output_dir / subdirectory
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = expected_model_path(
        model_name=model_name,
        model_signature=signature,
        subdirectory=subdirectory,
        model_output_dir=resolved_model_output_dir,
    )

    fitted_model = clone(model)
    fitted_model.fit(X, y)
    dump(fitted_model, output_path)

    return output_path


def signed_result_exists(
    results_dataframe: pd.DataFrame,
    model_column: str,
    model_name: str,
    model_signature: str,
) -> bool:
    """Check whether a result table already contains a signed model result."""

    required_columns = {model_column, "model_signature"}

    if results_dataframe.empty or not required_columns.issubset(
        results_dataframe.columns
    ):
        return False

    return (
        results_dataframe[model_column].eq(model_name)
        & results_dataframe["model_signature"].eq(model_signature)
    ).any()


def signed_model_artifact_exists(
    model_name: str,
    model_signature: str,
    subdirectory: str,
    extension: str = ".joblib",
    model_output_dir: str | Path | None = None,
) -> tuple[bool, Path]:
    """Check whether the signed model artefact exists on disk."""

    model_path = expected_model_path(
        model_output_dir=model_output_dir,
        model_name=model_name,
        model_signature=model_signature,
        subdirectory=subdirectory,
        extension=extension,
    )

    return model_path.exists(), model_path


def _metric_names(scoring: Mapping[str, Any] | Sequence[str]) -> list[str]:
    """Return metric names from a scoring dictionary or sequence."""

    if hasattr(scoring, "keys"):
        return list(scoring.keys())

    return list(scoring)


def _safe_metric_value(metric_name: str, y_true: Any, y_pred: Any) -> float:
    """Calculate one of the project metrics from predictions."""

    if metric_name == "accuracy":
        return float(accuracy_score(y_true, y_pred))

    if metric_name == "macro_f1":
        return float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        )

    if metric_name == "weighted_f1":
        return float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        )

    if metric_name == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))

    raise ValueError(f"Unknown metric: {metric_name}")


def evaluate_existing_model_on_training_data(
    model_path: str | Path,
    model_name: str,
    model_signature: str,
    X: Any,
    y: Any,
    scoring: Mapping[str, Any] | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate an existing fitted model on training data only.

    This is useful when a model artefact exists but the corresponding CV result
    row is missing. Validation values are intentionally left empty because they
    cannot be reconstructed from a single fitted model.
    """

    fitted_model = load(model_path)
    predictions = fitted_model.predict(X)

    if scoring is None:
        scoring = [
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "balanced_accuracy",
        ]

    summary: dict[str, Any] = {
        "Modell": model_name,
        "model_signature": model_signature,
        "model_path": str(model_path),
        "CV_Folds": 0,
        "fit_time_mean": pd.NA,
        "fit_time_std": pd.NA,
        "score_time_mean": pd.NA,
        "score_time_std": pd.NA,
        "result_source": "model_file_train_evaluation",
    }

    for metric_name in _metric_names(scoring):
        train_value = round(
            _safe_metric_value(metric_name, y, predictions),
            4,
        )

        summary[f"train_{metric_name}_mean"] = train_value
        summary[f"train_{metric_name}_std"] = pd.NA
        summary[f"val_{metric_name}_mean"] = pd.NA
        summary[f"val_{metric_name}_std"] = pd.NA
        summary[f"{metric_name}_mean"] = pd.NA
        summary[f"{metric_name}_std"] = pd.NA
        summary[f"generalization_gap_{metric_name}"] = pd.NA

    return summary


def create_cv_summary_row(
    model_name: str,
    cv_results: Mapping[str, Any],
    model_signature: str | None = None,
    scoring: Mapping[str, Any] | Sequence[str] | None = None,
    n_splits: int | None = None,
) -> dict[str, Any]:
    """Summarise cross-validation results in the project naming convention."""

    if scoring is None:
        scoring = [
            key.removeprefix("test_")
            for key in cv_results.keys()
            if key.startswith("test_")
        ]

    if n_splits is None:
        n_splits = len(cv_results["fit_time"])

    summary: dict[str, Any] = {
        "Modell": model_name,
        "CV_Folds": n_splits,
        "fit_time_total_seconds": round(
            float(np.sum(cv_results["fit_time"])),
            2,
        ),
        "fit_time_mean": round(float(np.mean(cv_results["fit_time"])), 2),
        "fit_time_std": round(float(np.std(cv_results["fit_time"])), 2),
        "fit_time_mean_seconds": round(
            float(np.mean(cv_results["fit_time"])),
            2,
        ),
        "fit_time_std_seconds": round(
            float(np.std(cv_results["fit_time"])),
            2,
        ),
        "score_time_mean": round(
            float(np.mean(cv_results["score_time"])),
            2,
        ),
        "score_time_std": round(
            float(np.std(cv_results["score_time"])),
            2,
        ),
        "score_time_mean_seconds": round(
            float(np.mean(cv_results["score_time"])),
            2,
        ),
    }

    if model_signature is not None:
        summary["model_signature"] = model_signature

    for metric_name in _metric_names(scoring):
        val_key = (
            f"test_{metric_name}"
            if f"test_{metric_name}" in cv_results
            else f"val_{metric_name}"
        )
        train_key = f"train_{metric_name}"

        val_scores = np.asarray(cv_results[val_key], dtype=float)

        summary[f"val_{metric_name}_mean"] = round(
            float(np.mean(val_scores)),
            4,
        )
        summary[f"val_{metric_name}_std"] = round(
            float(np.std(val_scores)),
            4,
        )

        # Backward-compatible aliases for older result-reading cells.
        summary[f"{metric_name}_mean"] = summary[f"val_{metric_name}_mean"]
        summary[f"{metric_name}_std"] = summary[f"val_{metric_name}_std"]

        if train_key in cv_results:
            train_scores = np.asarray(cv_results[train_key], dtype=float)
            train_mean = round(float(np.mean(train_scores)), 4)

            summary[f"train_{metric_name}_mean"] = train_mean
            summary[f"train_{metric_name}_std"] = round(
                float(np.std(train_scores)),
                4,
            )
            summary[f"generalization_gap_{metric_name}"] = round(
                train_mean - summary[f"val_{metric_name}_mean"],
                4,
            )

    return summary


def create_cv_fold_results_df(
    model_name: str,
    cv_results: Mapping[str, Any],
    model_signature: str | None = None,
    scoring: Mapping[str, Any] | Sequence[str] | None = None,
) -> pd.DataFrame:
    """Create one row per CV fold using the baseline-result convention."""

    if scoring is None:
        scoring = [
            key.removeprefix("test_")
            for key in cv_results.keys()
            if key.startswith("test_")
        ]

    fold_metric_results: dict[str, Any] = {}

    for metric_name in _metric_names(scoring):
        val_key = (
            f"test_{metric_name}"
            if f"test_{metric_name}" in cv_results
            else f"val_{metric_name}"
        )
        train_key = f"train_{metric_name}"

        fold_metric_results[f"val_{metric_name}"] = cv_results[val_key]

        if train_key in cv_results:
            fold_metric_results[f"train_{metric_name}"] = cv_results[
                train_key
            ]
            fold_metric_results[f"generalization_gap_{metric_name}"] = (
                cv_results[train_key] - cv_results[val_key]
            )

    fold_results = pd.DataFrame(fold_metric_results)

    fold_results.insert(
        0,
        "score_time_seconds",
        cv_results["score_time"],
    )
    fold_results.insert(
        0,
        "fit_time_seconds",
        cv_results["fit_time"],
    )
    fold_results.insert(
        0,
        "fold",
        range(1, len(fold_results) + 1),
    )
    fold_results.insert(0, "model", model_name)
    fold_results.insert(1, "model_signature", model_signature)

    return fold_results


def concat_non_empty_dataframes(
    dataframes: Iterable[pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate only non-empty DataFrames to avoid pandas warnings."""

    non_empty_dataframes = [
        dataframe
        for dataframe in dataframes
        if isinstance(dataframe, pd.DataFrame) and not dataframe.empty
    ]

    if not non_empty_dataframes:
        return pd.DataFrame()

    return pd.concat(
        non_empty_dataframes,
        ignore_index=True,
    )


def ensure_val_metric_aliases(
    dataframe: pd.DataFrame,
    metric_names: Iterable[str],
) -> pd.DataFrame:
    """Ensure validation metrics are available under explicit ``val_*`` names.

    Older result files sometimes used ``macro_f1_mean`` or
    ``test_macro_f1_mean`` for cross-validation scores. For interpretation,
    the project now treats ``val_*`` as the canonical validation prefix. The
    older names are kept as compatibility aliases but are not preferred for
    display or sorting.
    """

    result = dataframe.copy()

    for metric_name in metric_names:
        for suffix in ["mean", "std"]:
            val_column = f"val_{metric_name}_{suffix}"
            alias_columns = [
                f"{metric_name}_{suffix}",
                f"test_{metric_name}_{suffix}",
            ]

            if val_column not in result.columns:
                result[val_column] = pd.NA

            for alias_column in alias_columns:
                if alias_column in result.columns:
                    result[val_column] = result[val_column].fillna(
                        result[alias_column]
                    )

            # Keep the old alias for backward compatibility, but derive it
            # from the explicit validation column.
            alias_column = f"{metric_name}_{suffix}"
            if alias_column not in result.columns:
                result[alias_column] = result[val_column]
            else:
                result[alias_column] = result[alias_column].fillna(
                    result[val_column]
                )

    return result


def select_validation_result_columns(
    dataframe: pd.DataFrame,
    extra_columns: Sequence[str] | None = None,
    include_train_metrics: bool = True,
    include_generalization_gap: bool = True,
    include_time_metrics: bool = True,
) -> list[str]:
    """Select display columns while hiding legacy non-``val_*`` aliases."""

    selected_columns: list[str] = []

    for column in extra_columns or []:
        if column in dataframe.columns and column not in selected_columns:
            selected_columns.append(column)

    for column in dataframe.columns:
        include_column = column.startswith("val_")

        if include_train_metrics:
            include_column = include_column or column.startswith("train_")

        if include_generalization_gap:
            include_column = (
                include_column
                or column.startswith("generalization_gap_")
            )

        if include_time_metrics:
            include_column = (
                include_column
                or column.endswith("_seconds")
            )

        if include_column and column not in selected_columns:
            selected_columns.append(column)

    return selected_columns


def get_model_evaluation_display_columns(
    dataframe: pd.DataFrame,
    model_column_candidates: Sequence[str] | None = None,
    include_columns: Sequence[str] | None = None,
    metric_prefixes: Sequence[str] | None = None,
    include_fit_time_total: bool = True,
) -> list[str]:
    """Return compact display columns for model evaluation tables.

    The helper is intentionally generic so that baseline, ablation,
    hyperparameter tuning and final test evaluation tables can use the same
    display logic. By default it shows model identifiers, selected metadata,
    train/validation/test metrics and the total fit time where available.
    """

    if model_column_candidates is None:
        model_column_candidates = [
            "Modell",
            "model",
            "model_name",
            "Ablation",
        ]

    if include_columns is None:
        include_columns = [
            "CV_Folds",
            "Experimenttyp",
            "model_type",
            "trial_number",
            "validation_rows",
            "test_rows",
        ]

    if metric_prefixes is None:
        metric_prefixes = [
            "train_",
            "val_",
            "test_",
        ]

    selected_columns: list[str] = []

    for column in model_column_candidates:
        if column in dataframe.columns and column not in selected_columns:
            selected_columns.append(column)

    for column in include_columns:
        if column in dataframe.columns and column not in selected_columns:
            selected_columns.append(column)

    metric_columns = [
        column
        for column in dataframe.columns
        if any(column.startswith(prefix) for prefix in metric_prefixes)
    ]

    for column in metric_columns:
        if column not in selected_columns:
            selected_columns.append(column)

    if include_fit_time_total:
        for column in [
            "fit_time_total_seconds",
            "training_time_seconds",
        ]:
            if column in dataframe.columns and column not in selected_columns:
                selected_columns.append(column)

    return selected_columns


def display_model_evaluation_table(
    dataframe: pd.DataFrame,
    model_column_candidates: Sequence[str] | None = None,
    include_columns: Sequence[str] | None = None,
    metric_prefixes: Sequence[str] | None = None,
    sort_by: str | None = None,
    ascending: bool = False,
    include_fit_time_total: bool = True,
) -> pd.DataFrame:
    """Display a model evaluation table without truncated columns.

    The function returns the displayed DataFrame as well, so notebooks can
    reuse it for further inspection if needed.
    """

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    result = dataframe.copy()

    if sort_by is not None and sort_by in result.columns:
        result = result.sort_values(
            sort_by,
            ascending=ascending,
            na_position="last",
        )

    display_columns = get_model_evaluation_display_columns(
        result,
        model_column_candidates=model_column_candidates,
        include_columns=include_columns,
        metric_prefixes=metric_prefixes,
        include_fit_time_total=include_fit_time_total,
    )

    displayed_result = result[display_columns]

    try:
        from IPython.display import display

        display(displayed_result)
    except ImportError:
        print(displayed_result)

    return displayed_result


def combine_columns_as_text(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    include_column_names: bool = True,
) -> pd.Series:
    """Combine several feature columns into one text representation."""

    text_parts = []

    for column in columns:
        values = dataframe[column].fillna("").astype(str)

        if include_column_names:
            column_label = column.replace("_standardised", "")
            values = column_label + ": " + values

        text_parts.append(values)

    if not text_parts:
        return pd.Series(
            "",
            index=dataframe.index,
            dtype="string",
        )

    return (
        pd.concat(text_parts, axis=1)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def create_classification_report_df(
    y_true: Any,
    y_pred: Any,
) -> pd.DataFrame:
    """Create a DataFrame with precision, recall and F1 per class."""

    return pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()


def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    title: str = "Konfusionsmatrix",
    figsize: tuple[int, int] = (14, 14),
) -> None:
    """Plot a confusion matrix for final model inspection."""

    fig, ax = plt.subplots(figsize=figsize)

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        xticks_rotation=90,
        ax=ax,
    )

    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def make_fasttext_label(value: Any) -> str:
    """Convert a class label into fastText's required label format."""

    label = unicodedata.normalize("NFKD", str(value))
    label = label.encode("ascii", "ignore").decode("ascii")
    label = re.sub(r"[^A-Za-z0-9]+", "_", label)
    label = label.strip("_")

    return f"__label__{label}"


def write_fasttext_file(
    path: str | Path,
    texts: Sequence[str] | pd.Series,
    labels: Sequence[Any] | pd.Series,
    label_function=make_fasttext_label,
) -> None:
    """Write labelled text data in fastText supervised-training format."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for text, label in zip(texts, labels):
            clean_text = re.sub(r"\s+", " ", str(text)).strip()
            file.write(f"{label_function(label)} {clean_text}\n")
