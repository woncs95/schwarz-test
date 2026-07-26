"""Download the public organisation index from Berlin's Transparenzdatenbank."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://transparenzdatenbank.berlin.de/oberflaeche/index.cfm"
# Registrierungsnummern enthalten auch Rechtsform-Präfixe wie "aör".
# Deshalb darf die Extraktion nicht auf eine feste Präfixliste begrenzt sein.
REGISTRATION_PATTERN = re.compile(r"\b[a-zäöüß]+_[0-9]+\b", re.I)
DATE_PATTERN = re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b")


def build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers["User-Agent"] = (
        "politikbereich-classification/0.1 "
        "(public-data research; contact via repository owner)"
    )
    return session


def fetch_page(session: requests.Session, page: int, timeout: int) -> BeautifulSoup:
    response = session.get(
        BASE_URL,
        params={
            "dateiname": "organisation_suche_transparenz.cfm",
            "anwender_id": "5",
            "seite": page,
            "sortorder_transparenz": "organisation",
            "sortdirection": "desc",
            "name_der_organisation": "",
            "registrierungsnr": "",
            "transparenzlogo": "0",
            "plz": "",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def parse_organisations(soup: BeautifulSoup, page: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen_detail_ids: set[str] = set()

    for heading in soup.select("h3.title"):
        link = heading.find("a", href=True)
        if link is None or "organisation_beschreibung_transparenz.cfm" not in link["href"]:
            continue

        query = parse_qs(urlparse(link["href"]).query)
        detail_id = query.get("id", [""])[0]
        if not detail_id or detail_id in seen_detail_ids:
            continue
        seen_detail_ids.add(detail_id)

        container = heading.parent.parent
        text = " ".join(container.get_text(" ", strip=True).split())
        registration_match = REGISTRATION_PATTERN.search(text)
        date_match = DATE_PATTERN.search(text)

        records.append(
            {
                "name": " ".join(link.get_text(" ", strip=True).split()),
                "empfaengerid": (
                    registration_match.group(0).casefold() if registration_match else pd.NA
                ),
                "letzte_datenaktualisierung": date_match.group(0) if date_match else pd.NA,
                "transparenz_detail_id": detail_id,
                "detail_url": urljoin(BASE_URL, link["href"]),
                "source_page": page,
            }
        )

    return records


def find_last_page(soup: BeautifulSoup) -> int:
    pages = []
    for link in soup.find_all("a", href=True):
        query = parse_qs(urlparse(link["href"]).query)
        if "seite" in query:
            try:
                pages.append(int(query["seite"][0]))
            except ValueError:
                continue
    if not pages:
        raise RuntimeError("Could not determine the number of result pages.")
    return max(pages)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/transparenzdatenbank_organisationen.csv"),
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=Path("data/raw/transparenzdatenbank_download_progress.json"),
    )
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.progress.parent.mkdir(parents=True, exist_ok=True)

    session = build_session()
    first_page = fetch_page(session, page=1, timeout=args.timeout)
    last_page = find_last_page(first_page)
    records = parse_organisations(first_page, page=1)

    for page in range(2, last_page + 1):
        soup = fetch_page(session, page=page, timeout=args.timeout)
        records.extend(parse_organisations(soup, page=page))

        if page % 25 == 0 or page == last_page:
            frame = pd.DataFrame(records).drop_duplicates(
                subset=["transparenz_detail_id"], keep="last"
            )
            frame.to_csv(args.output, index=False, encoding="utf-8")
            args.progress.write_text(
                json.dumps(
                    {
                        "current_page": page,
                        "total_pages": last_page,
                        "rows": len(frame),
                        "complete": page == last_page,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        time.sleep(args.delay + random.uniform(0, 0.1))

    frame = (
        pd.DataFrame(records)
        .drop_duplicates(subset=["transparenz_detail_id"], keep="last")
        .sort_values(["name", "empfaengerid"], na_position="last")
        .reset_index(drop=True)
    )
    frame.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Saved {len(frame)} organisations to {args.output}")


if __name__ == "__main__":
    main()
