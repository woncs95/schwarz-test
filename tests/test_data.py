import pandas as pd

from src.data import combine_text_columns


def test_combine_text_columns_handles_missing_values():
    frame = pd.DataFrame({"name": ["Alpha"], "zweck": [None]})
    assert combine_text_columns(frame, ["name", "zweck"]).iloc[0] == "Alpha [SEP] "

