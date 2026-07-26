from pathlib import Path

import pandas as pd
from evidently.legacy.metric_preset import ClassificationPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report


def save_evidently_classification_report(
    y_true: pd.Series,
    y_prediction: pd.Series,
    output_path: str | Path,
) -> Path:
    """Create a standalone Evidently HTML report for multiclass errors."""
    evaluation = pd.DataFrame(
        {
            "target": y_true.reset_index(drop=True).astype(str),
            "prediction": pd.Series(y_prediction).reset_index(drop=True).astype(str),
        }
    )
    labels = sorted(set(evaluation["target"]) | set(evaluation["prediction"]))
    mapping = ColumnMapping(
        target="target",
        prediction="prediction",
        task="classification",
        pos_label=labels[-1] if len(labels) == 2 else None,
    )
    report = Report(metrics=[ClassificationPreset()])
    report.run(current_data=evaluation, reference_data=None, column_mapping=mapping)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(destination))
    return destination
