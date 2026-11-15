from __future__ import annotations

import csv
import io
from pathlib import Path

from config import REQUIRED_COLUMNS


def _clean_record(raw: dict[str, object], row_number: int) -> dict[str, str]:
    normalized = {str(key).strip().lower(): value for key, value in raw.items() if key}
    missing = [column for column in REQUIRED_COLUMNS if column not in normalized]
    if missing:
        raise ValueError(
            f"CSV row {row_number} is missing required columns: {', '.join(missing)}"
        )
    record = {
        column: str(normalized[column] or "").strip() for column in REQUIRED_COLUMNS
    }
    if not all(record[key] for key in ("company", "job_role", "interview_question")):
        raise ValueError(
            f"CSV row {row_number} must include company, job_role, and interview_question."
        )
    return record


def load_records(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Interview dataset not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("The interview CSV has no header row.")
        return [_clean_record(row, index) for index, row in enumerate(reader, start=2)]


def load_csv_bytes(contents: bytes) -> list[dict[str, str]]:
    try:
        text = contents.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("The archive must be a UTF-8 encoded CSV file.") from exc
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("The uploaded CSV has no header row.")
    normalized_headers = {header.strip().lower() for header in reader.fieldnames if header}
    missing = sorted(set(REQUIRED_COLUMNS) - normalized_headers)
    if missing:
        raise ValueError(f"Uploaded CSV is missing required columns: {', '.join(missing)}")
    rows = [_clean_record(row, index) for index, row in enumerate(reader, start=2)]
    if not rows:
        raise ValueError("The uploaded CSV does not contain any interview records.")
    return rows
