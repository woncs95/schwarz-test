import pandas as pd

from src.evaluate import save_evidently_classification_report


def test_evidently_report_is_created(tmp_path):
    output = tmp_path / "classification.html"
    result = save_evidently_classification_report(
        pd.Series(["A", "A", "B", "B"]),
        pd.Series(["A", "B", "B", "B"]),
        output,
    )
    assert result.exists()
    assert result.stat().st_size > 0

