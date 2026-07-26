import pandas as pd
import pytest

from src.recipient_relationships import (
    analyse_recipient_relationships,
    dependency_score,
    normalize_address,
)


def test_normalize_address_merges_common_strasse_variants():
    assert normalize_address("Hauptstr. 10, 10115 Berlin") == normalize_address(
        "hauptstrasse 10,10115 berlin"
    )


def test_dependency_score_uses_modal_value_per_group():
    frame = pd.DataFrame({"id": ["a", "a", "a", "b"], "value": ["x", "x", "y", "z"]})
    assert dependency_score(frame, ["id"], "value") == pytest.approx(0.75)


def test_year_explains_recipient_address_change():
    frame = pd.DataFrame(
        {
            "name": ["Organisation", "Organisation", "Organisation"],
            "anschrift": ["Alte Str. 1", "Alte Strasse 1", "Neue Str. 2"],
            "empfaengerid": ["vr_1", "vr_1", "vr_1"],
            "jahr": [2022, 2022, 2023],
        }
    )

    summary, details = analyse_recipient_relationships(frame)
    metrics = summary.set_index("metric")["value"]

    assert metrics["id_to_address_row_agreement"] == pytest.approx(2 / 3)
    assert metrics["id_and_year_to_address_row_agreement"] == pytest.approx(1.0)
    assert metrics["ids_with_address_change_across_years"] == 1
    assert details["address_changes_by_year"]["empfaengerid"].tolist() == ["vr_1"]
