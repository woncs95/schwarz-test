"""Analyse the relationship between recipient name, address, ID, and year."""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.data import load_dataset

REQUIRED_COLUMNS = {"name", "anschrift", "empfaengerid", "jahr"}


def normalize_text(value: object) -> object:
    """Normalize spelling differences without inventing missing values."""
    if pd.isna(value):
        return pd.NA
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = text.replace("ß", "ss")
    text = re.sub(r"[“”„\"'´`]", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;")
    return text or pd.NA


def normalize_address(value: object) -> object:
    """Normalize common German address spelling and punctuation variants."""
    text = normalize_text(value)
    if pd.isna(text):
        return pd.NA
    text = re.sub(r"str\.?(?=\s*\d)", "strasse", str(text))
    text = re.sub(r"\bstr\.?\b", "strasse", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;")
    return text or pd.NA


def _mode(series: pd.Series) -> object:
    modes = series.dropna().mode()
    return modes.iloc[0] if not modes.empty else pd.NA


def dependency_score(frame: pd.DataFrame, determinants: list[str], dependent: str) -> float:
    """Return the share of rows agreeing with the modal dependent value per group."""
    usable = frame.dropna(subset=determinants + [dependent])
    if usable.empty:
        return float("nan")
    counts = usable.groupby(determinants + [dependent], dropna=False).size()
    agreeing_rows = counts.groupby(level=list(range(len(determinants)))).max().sum()
    return float(agreeing_rows / len(usable))


def analyse_recipient_relationships(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Calculate evidence for recipient-field dependencies and address changes."""
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = frame.loc[:, ["name", "anschrift", "empfaengerid", "jahr"]].copy()
    data["name_norm"] = data["name"].map(normalize_text)
    data["anschrift_norm"] = data["anschrift"].map(normalize_address)
    data["empfaengerid"] = data["empfaengerid"].map(normalize_text)

    by_id = data.dropna(subset=["empfaengerid"]).groupby("empfaengerid")
    id_profile = by_id.agg(
        rows=("empfaengerid", "size"),
        years=("jahr", "nunique"),
        names=("name_norm", "nunique"),
        addresses=("anschrift_norm", "nunique"),
    ).reset_index()

    id_year_profile = (
        data.dropna(subset=["empfaengerid", "jahr"])
        .groupby(["empfaengerid", "jahr"])
        .agg(
            rows=("empfaengerid", "size"),
            names=("name_norm", "nunique"),
            addresses=("anschrift_norm", "nunique"),
            representative_name=("name_norm", _mode),
            representative_address=("anschrift_norm", _mode),
        )
        .reset_index()
    )

    yearly_modes = id_year_profile.pivot(
        index="empfaengerid", columns="jahr", values="representative_address"
    )
    changed_mask = yearly_modes.nunique(axis=1, dropna=True) > 1
    changed_addresses = yearly_modes.loc[changed_mask].reset_index()

    pair_to_id = (
        data.dropna(subset=["name_norm", "anschrift_norm", "empfaengerid"])
        .groupby(["name_norm", "anschrift_norm"])
        .agg(
            ids=("empfaengerid", "nunique"),
            rows=("empfaengerid", "size"),
            empfaengerids=("empfaengerid", lambda values: " | ".join(sorted(values.unique()))),
        )
        .reset_index()
    )
    pair_collisions = pair_to_id[pair_to_id["ids"] > 1].sort_values(
        ["ids", "rows"], ascending=False
    )

    ids_with_multiple_years = id_profile["years"] > 1
    metrics = [
        ("rows_total", len(data)),
        ("rows_with_empfaengerid", data["empfaengerid"].notna().sum()),
        ("unique_empfaengerid", data["empfaengerid"].nunique()),
        ("id_to_name_row_agreement", dependency_score(data, ["empfaengerid"], "name_norm")),
        (
            "id_to_address_row_agreement",
            dependency_score(data, ["empfaengerid"], "anschrift_norm"),
        ),
        (
            "id_and_year_to_address_row_agreement",
            dependency_score(data, ["empfaengerid", "jahr"], "anschrift_norm"),
        ),
        (
            "name_and_address_to_id_row_agreement",
            dependency_score(data, ["name_norm", "anschrift_norm"], "empfaengerid"),
        ),
        ("ids_with_multiple_names", (id_profile["names"] > 1).sum()),
        ("ids_with_multiple_addresses", (id_profile["addresses"] > 1).sum()),
        ("ids_observed_in_multiple_years", ids_with_multiple_years.sum()),
        ("ids_with_address_change_across_years", len(changed_addresses)),
        ("name_address_pairs_linked_to_multiple_ids", len(pair_collisions)),
    ]
    summary = pd.DataFrame(metrics, columns=["metric", "value"])

    details = {
        "id_profile": id_profile.sort_values(["addresses", "names", "rows"], ascending=False),
        "id_year_profile": id_year_profile.sort_values(
            ["addresses", "names", "rows"], ascending=False
        ),
        "address_changes_by_year": changed_addresses,
        "name_address_id_collisions": pair_collisions,
    }
    return summary, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/index.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/generated/recipient_relations"))
    parser.add_argument("--separator", default=";")
    parser.add_argument("--encoding", default="utf-8")
    args = parser.parse_args()

    frame = load_dataset(args.input, separator=args.separator, encoding=args.encoding)
    summary, details = analyse_recipient_relationships(frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    for name, table in details.items():
        table.to_csv(args.output_dir / f"{name}.csv", index=False)

    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(summary.to_string(index=False))
    print(f"\nDetail tables written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
