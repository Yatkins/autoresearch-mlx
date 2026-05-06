#!/usr/bin/env python3
"""Fixed invoice extraction scorer for the autoresearch loop."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any


HEADER_TARGET_FIELDS = (
    "vendor",
    "vendor_physical_address",
    "vendor_phone_number",
    "vendor_website",
    "vendor_email",
    "invoice_no",
    "invoice_date",
    "adjustment",
    "bottle_deposit",
    "invoice_amount",
    "document_type",
    "holiday",
)

LINE_TARGET_FIELDS = (
    "case",
    "pieces",
    "quantity",
    "units_per_case",
    "upc",
    "vic",
    "description",
    "unit_price",
    "discount",
    "deposit",
    "line_amount",
)

COMPARISON_COLUMNS = (
    "document_id",
    "section",
    "true_row",
    "predicted_row",
    "field",
    "true_value",
    "predicted_value",
    "score",
    "status",
)

EXCLUDED_TOP_LEVEL_FIELDS = {
    "Batch ID",
    "Link",
    "Error Message",
    "Error Fields",
    "File Name",
}

EXCLUDED_LINE_ITEM_FIELDS = {
    "Error Fields",
    "Deposit Qty",
    "Deposit Quantity",
    "deposit_qty",
}

NUMERIC_FIELDS = {
    "adjustment",
    "bottle_deposit",
    "case",
    "deposit",
    "discount",
    "invoice_amount",
    "line_amount",
    "pieces",
    "quantity",
    "total_quantity",
    "unit_price",
    "units_per_case",
}

IDENTITY_FIELDS = {"invoice_no", "upc", "vic"}

ZERO_EQUIVALENT_FIELDS = {
    "adjustment",
    "bottle_deposit",
    "case",
    "deposit",
    "discount",
    "holiday",
    "pieces",
    "units_per_case",
}

BOOLEAN_HINTS = (
    "invalid",
    "cog",
    "holiday",
)

DATE_HINTS = ("date",)

HEADER_ALIASES = {
    "vendor": "vendor",
    "vendorname": "vendor",
    "supplier": "vendor",
    "suppliername": "vendor",
    "vendorphysicaladdress": "vendor_physical_address",
    "vendoraddress": "vendor_physical_address",
    "physicaladdress": "vendor_physical_address",
    "address": "vendor_physical_address",
    "vendorphonenumber": "vendor_phone_number",
    "vendorphone": "vendor_phone_number",
    "phone": "vendor_phone_number",
    "phonenumber": "vendor_phone_number",
    "telephone": "vendor_phone_number",
    "vendorwebsite": "vendor_website",
    "vendorwebside": "vendor_website",
    "website": "vendor_website",
    "webside": "vendor_website",
    "web": "vendor_website",
    "vendoremail": "vendor_email",
    "vendoremailaddress": "vendor_email",
    "email": "vendor_email",
    "emailaddress": "vendor_email",
    "invoiceno": "invoice_no",
    "invoicenumber": "invoice_no",
    "invoiceid": "invoice_no",
    "creditnumber": "invoice_no",
    "creditno": "invoice_no",
    "invoice": "invoice_no",
    "invoicedate": "invoice_date",
    "date": "invoice_date",
    "totalquantity": "total_quantity",
    "totalqty": "total_quantity",
    "bottledeposit": "bottle_deposit",
    "depositamount": "bottle_deposit",
    "invoiceamount": "invoice_amount",
    "invoicetotal": "invoice_amount",
    "amountdue": "invoice_amount",
    "total": "invoice_amount",
    "documenttype": "document_type",
    "doctype": "document_type",
    "type": "document_type",
    "adjustment": "adjustment",
    "adjustments": "adjustment",
    "holiday": "holiday",
}

LINE_ALIASES = {
    "case": "case",
    "cases": "case",
    "cs": "case",
    "pieces": "pieces",
    "piece": "pieces",
    "pcs": "pieces",
    "unitpercase": "units_per_case",
    "unitspercase": "units_per_case",
    "unitspcase": "units_per_case",
    "casepack": "units_per_case",
    "pack": "units_per_case",
    "itemcode": "vic",
    "productcode": "vic",
    "sku": "upc",
    "upc": "upc",
    "barcode": "upc",
    "vic": "vic",
    "vendoritemcode": "vic",
    "item": "vic",
    "csqty": "units_per_case",
    "caseqty": "units_per_case",
    "casequantity": "units_per_case",
    "packsize": "units_per_case",
    "pack_size": "units_per_case",
    "description": "description",
    "itemdescription": "description",
    "productdescription": "description",
    "quantity": "quantity",
    "qty": "quantity",
    "unitprice": "unit_price",
    "price": "unit_price",
    "lineamount": "line_amount",
    "amount": "line_amount",
    "linetotal": "line_amount",
    "discount": "discount",
    "deposit": "deposit",
    "depositqty": "deposit_qty",
    "depositquantity": "deposit_qty",
}

UPC_SOURCE_PRIORITY = {
    "upc": 5,
    "sku": 4,
    "barcode": 4,
}


@dataclass
class DocumentMetrics:
    header_matches: float = 0.0
    header_total: int = 0
    line_matches: float = 0.0
    line_total: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    crashed: bool = False
    comparisons: list[dict[str, Any]] | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-dir", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--report-json")
    parser.add_argument("--comparison-tsv", help="Write field-level true/predicted/score rows as TSV.")
    parser.add_argument("--document-id", action="append", help="Score only this document id. May be repeated.")
    args = parser.parse_args()

    summary = score_predictions(
        Path(args.ground_truth_dir),
        Path(args.predictions),
        Path(args.report_json) if args.report_json else None,
        Path(args.comparison_tsv) if args.comparison_tsv else None,
        document_ids=args.document_id,
    )
    print_summary(summary)


def score_predictions(
    ground_truth_dir: Path,
    predictions_path: Path,
    report_path: Path | None = None,
    comparison_path: Path | None = None,
    document_ids: list[str] | None = None,
) -> dict[str, Any]:
    gt_paths = sorted(ground_truth_dir.glob("*.json"))
    if document_ids is not None:
        allowed_ids = {str(document_id) for document_id in document_ids}
        gt_paths = [path for path in gt_paths if path.stem in allowed_ids]
    predictions = load_predictions(predictions_path)
    documents: dict[str, DocumentMetrics] = {}

    for gt_path in gt_paths:
        document_id = gt_path.stem
        expected_fields, expected_line_items = parse_ground_truth(gt_path)
        predicted = predictions.get(document_id)
        metrics = DocumentMetrics()
        if predicted is None:
            metrics.crashed = True
            metrics.header_total = len(expected_fields)
            metrics.line_total = sum(len(row) for row in expected_line_items)
            metrics.comparisons = missing_prediction_comparisons(document_id, expected_fields, expected_line_items)
            documents[document_id] = metrics
            continue

        header_matches, header_total, header_comparisons = compare_fields_detailed(
            document_id,
            expected_fields,
            predicted["fields"],
        )
        line_matches, line_total, line_comparisons = compare_line_items_detailed(
            document_id,
            expected_line_items,
            predicted["line_items"],
        )
        metrics.header_matches = header_matches
        metrics.header_total = header_total
        metrics.line_matches = line_matches
        metrics.line_total = line_total
        metrics.cost_usd = predicted["cost_usd"]
        metrics.latency_seconds = predicted["latency_seconds"]
        metrics.crashed = bool(predicted["crashed"])
        metrics.comparisons = header_comparisons + line_comparisons
        documents[document_id] = metrics

    ignored_columns = remove_ignored_empty_columns(documents)
    summary = summarize_documents(documents)
    summary["ignored_empty_columns"] = [
        {"section": section, "field": field}
        for section, field in sorted(ignored_columns)
    ]
    if report_path is not None:
        report_path.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "documents": serialize_documents(documents),
                    "comparisons": comparison_rows(documents),
                },
                indent=2,
                sort_keys=True,
            )
        )
    if comparison_path is not None:
        write_comparison_tsv(comparison_path, comparison_rows(documents))
    return summary


def parse_ground_truth(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and isinstance(payload.get("Ch"), list):
        return parse_grooper_payload(payload)
    if isinstance(payload, dict):
        return parse_plain_invoice_payload(payload)
    return {}, []


def parse_plain_invoice_payload(payload: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, str]]]:
    fields = {key: value for key, value in payload.items() if key not in {"Rows", "rows", "line_items"}}
    rows = payload.get("Rows") or payload.get("rows") or payload.get("line_items") or []
    if not isinstance(rows, list):
        rows = []
    return normalize_expected_header_fields(fields), normalize_expected_line_items(rows)


def parse_grooper_payload(payload: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, str]]]:
    fields: dict[str, Any] = {}
    line_items: list[dict[str, Any]] = []
    for child in payload.get("Ch", []):
        child_type = child.get("__type", "")
        name = child.get("Name")
        if child_type.startswith("FieldInstance") and name:
            value = extract_field_value(child)
            if value not in (None, ""):
                fields[name] = value
        elif child_type.startswith("TableInstance") and name == "Line Items":
            for row in child.get("Ch", []):
                if not row.get("__type", "").startswith("TableRowInstance"):
                    continue
                row_fields: dict[str, Any] = {}
                for cell in row.get("Cells", []):
                    cell_name = cell.get("Name")
                    if not cell_name:
                        continue
                    value = cell.get("Val")
                    if value not in (None, ""):
                        row_fields[cell_name] = value
                if row_fields:
                    line_items.append(row_fields)
    return normalize_expected_header_fields(fields), normalize_expected_line_items(line_items)


def extract_field_value(node: dict[str, Any]) -> Any:
    if "Val" in node and node["Val"] not in (None, ""):
        return node["Val"]
    for alt in node.get("AE", []):
        if alt.get("Val") not in (None, ""):
            return alt["Val"]
    return None


def parse_prediction_record(record: dict[str, Any]) -> tuple[str | None, dict[str, str], list[dict[str, str]], float, float, bool]:
    document_id = record.get("document_id") or record.get("id")
    if not document_id:
        source_path = record.get("source_path") or record.get("file_name")
        if source_path:
            document_id = Path(source_path).stem
    if document_id is not None:
        document_id = str(document_id)

    fields = record.get("fields", {})
    if not isinstance(fields, dict):
        fields = {}
    line_items = record.get("line_items") or record.get("Rows") or record.get("rows") or []
    if not isinstance(line_items, list):
        line_items = []

    cost_usd = safe_float(record.get("cost_usd", 0.0))
    latency_seconds = safe_float(record.get("latency_seconds", 0.0))
    status = str(record.get("status", "ok")).lower()
    crashed = status not in {"ok", "success"}

    return (
        document_id,
        normalize_fields(fields, EXCLUDED_TOP_LEVEL_FIELDS, HEADER_ALIASES),
        normalize_line_items(line_items),
        cost_usd,
        latency_seconds,
        crashed,
    )


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    text = path.read_text().strip()
    if not text:
        return {}
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        parsed = json.loads(text)
        records = parsed if isinstance(parsed, list) else [parsed]

    predictions: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        document_id, fields, line_items, cost_usd, latency_seconds, crashed = parse_prediction_record(record)
        if document_id is None:
            continue
        predictions[document_id] = {
            "fields": fields,
            "line_items": line_items,
            "cost_usd": cost_usd,
            "latency_seconds": latency_seconds,
            "crashed": crashed,
        }
    return predictions


def normalize_expected_header_fields(fields: dict[str, Any]) -> dict[str, str]:
    normalized = normalize_fields(fields, EXCLUDED_TOP_LEVEL_FIELDS, HEADER_ALIASES)
    return {field: normalized.get(field, "") for field in HEADER_TARGET_FIELDS}


def normalize_expected_line_items(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows = normalize_line_items(items)
    return [{field: row.get(field, "") for field in LINE_TARGET_FIELDS} for row in rows]


def normalize_fields(fields: dict[str, Any], excluded: set[str], aliases: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, raw_value in fields.items():
        canonical_key = canonical_key_name(key, aliases)
        if key in excluded or canonical_key in excluded or canonical_key not in HEADER_TARGET_FIELDS:
            continue
        value = canonical_value(canonical_key, raw_value)
        if value not in (None, ""):
            normalized[canonical_key] = value
    return normalized


def normalize_line_items(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized = normalize_line_item(item)
        if normalized:
            rows.append(normalized)
    return rows


def normalize_line_item(item: dict[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    upc_priority = -1
    for key, raw_value in item.items():
        compact = compact_key(key)
        canonical_key = LINE_ALIASES.get(compact, compact)
        if key in EXCLUDED_LINE_ITEM_FIELDS or canonical_key in EXCLUDED_LINE_ITEM_FIELDS or canonical_key not in LINE_TARGET_FIELDS:
            continue
        value = canonical_value(canonical_key, raw_value)
        if value in (None, ""):
            continue
        if canonical_key == "upc":
            priority = UPC_SOURCE_PRIORITY.get(compact, 0)
            if priority < upc_priority:
                continue
            upc_priority = priority
        elif canonical_key in normalized:
            continue
        normalized[canonical_key] = value
    return normalized


def canonical_key_name(key: str, aliases: dict[str, str]) -> str:
    compact = compact_key(key)
    return aliases.get(compact, compact)


def compact_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def canonical_value(field_name: str, value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return canonical_number(str(value))
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True)
    cleaned = collapse_ws(value)
    if cleaned == "":
        return None

    lowered_name = field_name.lower()
    if field_name == "vendor_phone_number":
        return canonical_phone(cleaned)
    if field_name == "vendor_website":
        return canonical_website(cleaned)
    if field_name == "vendor_email":
        return canonical_email(cleaned)
    if lowered_name in IDENTITY_FIELDS:
        return canonical_identifier(cleaned)
    if lowered_name == "description":
        return canonical_string(cleaned)
    if lowered_name == "document_type":
        return canonical_document_type(cleaned)
    if any(hint in lowered_name for hint in BOOLEAN_HINTS):
        normalized = canonical_bool(cleaned)
        if normalized is not None:
            return normalized
    if any(hint in lowered_name for hint in DATE_HINTS):
        normalized = canonical_date(cleaned)
        if normalized is not None:
            return normalized
    if lowered_name in NUMERIC_FIELDS:
        normalized = canonical_number(cleaned)
        if normalized is not None:
            return normalized

    normalized = canonical_date(cleaned)
    if normalized is not None and any(char.isdigit() for char in cleaned):
        return normalized
    normalized = canonical_number(cleaned)
    if normalized is not None and re.fullmatch(r"[$,0-9.\-()% ]+", cleaned):
        return normalized
    return canonical_string(cleaned)


def canonical_identifier(value: str) -> str:
    return canonical_string(value)


def canonical_string(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", collapse_ws(value).upper())


def canonical_phone(value: str) -> str:
    return re.sub(r"\D", "", value)


def canonical_website(value: str) -> str:
    cleaned = collapse_ws(value).lower()
    cleaned = re.sub(r"^https?://", "", cleaned)
    cleaned = re.sub(r"^www\.", "", cleaned)
    cleaned = cleaned.rstrip("/")
    return re.sub(r"[^a-z0-9]", "", cleaned).upper()


def canonical_email(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", collapse_ws(value).upper())


def canonical_document_type(value: str) -> str:
    cleaned = collapse_ws(value).upper()
    if "CREDIT" in cleaned:
        return "CREDIT"
    if cleaned in {"BILL", "INVOICE"}:
        return "BILL"
    return cleaned


def collapse_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def canonical_bool(value: str) -> str | None:
    lowered = collapse_ws(value).lower()
    if lowered in {"true", "yes", "y", "1"}:
        return "1"
    if lowered in {"false", "no", "n", "0"}:
        return "0"
    return None


def canonical_number(value: str) -> str | None:
    cleaned = collapse_ws(value).replace("$", "").replace(",", "").replace("%", "")
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]
    if not cleaned:
        return None
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None
    number = number.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def canonical_date(value: str) -> str | None:
    cleaned = collapse_ws(value)
    if not cleaned:
        return None
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%m-%d-%Y", "%m.%d.%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def missing_prediction_comparisons(
    document_id: str,
    expected_fields: dict[str, str],
    expected_line_items: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows = [
        comparison_record(document_id, "header", "", "", field, expected, "", 0.0)
        for field, expected in expected_fields.items()
    ]
    for row_index, expected_row in enumerate(expected_line_items, start=1):
        for field, expected in expected_row.items():
            rows.append(comparison_record(document_id, "line_item", row_index, "", field, expected, "", 0.0))
    return rows


def compare_fields_detailed(
    document_id: str,
    expected: dict[str, str],
    actual: dict[str, str],
) -> tuple[float, int, list[dict[str, Any]]]:
    matches = 0.0
    comparisons = []
    for field, expected_value in expected.items():
        score = field_score(field, expected_value, actual.get(field), actual)
        matches += score
        comparisons.append(
            comparison_record(document_id, "header", "", "", field, expected_value, actual.get(field, ""), score)
        )
    return matches, len(expected), comparisons


def compare_fields(expected: dict[str, str], actual: dict[str, str]) -> tuple[float, int]:
    matches, total, _comparisons = compare_fields_detailed("", expected, actual)
    return matches, total


def compare_line_items_detailed(
    document_id: str,
    expected: list[dict[str, str]],
    actual: list[dict[str, str]],
) -> tuple[float, int, list[dict[str, Any]]]:
    matches = 0.0
    total = sum(len(row) for row in expected)
    comparisons: list[dict[str, Any]] = []
    unused_actual = set(range(len(actual)))
    for expected_index, expected_row in enumerate(expected, start=1):
        best_index = None
        best_matches = -1.0
        for actual_index in unused_actual:
            row_matches = count_field_matches(expected_row, actual[actual_index])
            if row_matches > best_matches:
                best_matches = row_matches
                best_index = actual_index
        actual_row: dict[str, str] = {}
        predicted_index: int | str = ""
        if best_index is not None:
            unused_actual.remove(best_index)
            actual_row = actual[best_index]
            predicted_index = best_index + 1
            matches += max(0.0, best_matches)
        for field, expected_value in expected_row.items():
            comparisons.append(
                comparison_record(
                    document_id,
                    "line_item",
                    expected_index,
                    predicted_index,
                    field,
                    expected_value,
                    actual_row.get(field, ""),
                    field_score(field, expected_value, actual_row.get(field), actual_row),
                )
            )

    for actual_index in sorted(unused_actual):
        actual_row = actual[actual_index]
        fields = {field: value for field, value in actual_row.items() if value not in ("", None)}
        if not fields:
            continue
        for field, value in fields.items():
            comparisons.append(comparison_record(document_id, "extra_predicted_line", "", actual_index + 1, field, "", value, None))
    return matches, total, comparisons


def compare_line_items(expected: list[dict[str, str]], actual: list[dict[str, str]]) -> tuple[float, int]:
    matches, total, _comparisons = compare_line_items_detailed("", expected, actual)
    return matches, total


def comparison_record(
    document_id: str,
    section: str,
    true_row: int | str,
    predicted_row: int | str,
    field: str,
    true_value: str,
    predicted_value: str,
    score: float | None,
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "section": section,
        "true_row": true_row,
        "predicted_row": predicted_row,
        "field": field,
        "true_value": true_value,
        "predicted_value": predicted_value,
        "score": "" if score is None else round(score, 6),
        "_score_value": score,
        "status": score_status(score),
    }


def score_status(score: float | None) -> str:
    if score is None:
        return "unscored_extra"
    if score >= 1.0:
        return "match"
    if score <= 0.0:
        return "miss"
    return "partial"


def count_field_matches(expected: dict[str, str], actual: dict[str, str]) -> float:
    return sum(field_score(key, value, actual.get(key), actual) for key, value in expected.items())


def field_score(field_name: str, expected: str, actual: str | None, actual_row: dict[str, str]) -> float:
    actual = actual or ""
    if values_blank(expected, actual):
        return 1.0
    if actual == expected:
        return 1.0
    if field_name in ZERO_EQUIVALENT_FIELDS and zeroish(expected) and zeroish(actual):
        return 1.0
    if field_name == "quantity":
        derived_quantity = quantity_from_amount_price(actual_row)
        if derived_quantity is not None and numeric_values_match(expected, derived_quantity):
            return 1.0
        if numeric_values_match(expected, actual_row.get("pieces")):
            return 1.0
        if numeric_values_match(expected, actual_row.get("case")):
            return 1.0
        case_quantity = quantity_from_case_and_units(actual_row.get("case"), actual_row.get("units_per_case"))
        if case_quantity is not None and numeric_values_match(expected, case_quantity):
            return 1.0
    if field_name in NUMERIC_FIELDS or any(hint in field_name for hint in DATE_HINTS):
        return 0.0
    return fuzzy_string_score(expected, actual)


def values_blank(left: str | None, right: str | None) -> bool:
    left = left or ""
    right = right or ""
    return left == "" and right == ""


def numeric_values_match(left: str | None, right: str | None) -> bool:
    left_number = canonical_number(left or "")
    right_number = canonical_number(right or "")
    if left_number is None or right_number is None:
        return values_blank(left, right)
    return left_number == right_number


def fuzzy_string_score(expected: str, actual: str) -> float:
    if expected == actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    distance = edit_distance(expected, actual)
    return max(0.0, 1.0 - distance / max(len(expected), len(actual)))


def edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            substitution = previous[right_index - 1] + (left_char != right_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def quantity_from_amount_price(row: dict[str, str]) -> str | None:
    amount = row.get("line_amount")
    unit_price = row.get("unit_price")
    if amount in (None, "0") or unit_price in (None, "0"):
        return None
    try:
        quantity = Decimal(amount) / Decimal(unit_price)
    except (InvalidOperation, ZeroDivisionError):
        return None
    return canonical_number(str(quantity))


def quantity_from_case_and_units(case_count: str | None, units_per_case: str | None) -> str | None:
    if case_count in (None, "0") or units_per_case in (None, "0"):
        return None
    try:
        quantity = Decimal(case_count) * Decimal(units_per_case)
    except (InvalidOperation, ZeroDivisionError):
        return None
    return canonical_number(str(quantity))


def remove_ignored_empty_columns(documents: dict[str, DocumentMetrics]) -> set[tuple[str, str]]:
    ignored = empty_zero_columns(documents)
    for metric in documents.values():
        rows = [
            row
            for row in (metric.comparisons or [])
            if (str(row.get("section")), str(row.get("field"))) not in ignored
        ]
        metric.comparisons = rows
        metric.header_matches = sum(score_value(row) for row in rows if row.get("section") == "header")
        metric.header_total = sum(1 for row in rows if row.get("section") == "header" and row.get("_score_value") is not None)
        metric.line_matches = sum(score_value(row) for row in rows if row.get("section") == "line_item")
        metric.line_total = sum(1 for row in rows if row.get("section") == "line_item" and row.get("_score_value") is not None)
    return ignored


def empty_zero_columns(documents: dict[str, DocumentMetrics]) -> set[tuple[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for metric in documents.values():
        for row in metric.comparisons or []:
            if row.get("section") not in {"header", "line_item"} or row.get("_score_value") is None:
                continue
            grouped.setdefault((str(row["section"]), str(row["field"])), []).append(row)
    ignored: set[tuple[str, str]] = set()
    for key, rows in grouped.items():
        if rows and all(zeroish(row.get("true_value")) and zeroish(row.get("predicted_value")) for row in rows):
            ignored.add(key)
    return ignored


def zeroish(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if text == "":
        return True
    number = canonical_number(text)
    return number == "0"


def score_value(row: dict[str, Any]) -> float:
    value = row.get("_score_value")
    return float(value) if value is not None else 0.0


def summarize_documents(documents: dict[str, DocumentMetrics]) -> dict[str, Any]:
    total_header_matches = sum(metric.header_matches for metric in documents.values())
    total_header = sum(metric.header_total for metric in documents.values())
    total_line_matches = sum(metric.line_matches for metric in documents.values())
    total_line = sum(metric.line_total for metric in documents.values())
    total_matches = total_header_matches + total_line_matches
    total_fields = total_header + total_line

    accuracy = total_matches / total_fields if total_fields else 0.0
    header_accuracy = total_header_matches / total_header if total_header else 0.0
    line_item_accuracy = total_line_matches / total_line if total_line else 0.0

    doc_count = len(documents)
    total_cost = sum(metric.cost_usd for metric in documents.values())
    total_latency = sum(metric.latency_seconds for metric in documents.values())
    avg_cost_usd = total_cost / doc_count if doc_count else 0.0
    avg_latency_seconds = total_latency / doc_count if doc_count else 0.0
    docs_per_minute = 60.0 / avg_latency_seconds if avg_latency_seconds > 0 else float("inf")
    crash_count = sum(1 for metric in documents.values() if metric.crashed)
    crash_rate = crash_count / doc_count if doc_count else 0.0
    adjusted_score = compute_adjusted_score(accuracy, avg_cost_usd, avg_latency_seconds, docs_per_minute, crash_rate)

    return {
        "accuracy": accuracy,
        "header_accuracy": header_accuracy,
        "line_item_accuracy": line_item_accuracy,
        "adjusted_score": adjusted_score,
        "docs": doc_count,
        "avg_cost_usd": avg_cost_usd,
        "avg_latency_seconds": avg_latency_seconds,
        "docs_per_minute": docs_per_minute,
        "crash_rate": crash_rate,
        "field_matches": total_matches,
        "field_total": total_fields,
    }


def compute_adjusted_score(
    accuracy: float,
    avg_cost_usd: float,
    avg_latency_seconds: float,
    docs_per_minute: float,
    crash_rate: float,
) -> float:
    cost_penalty = min(0.12, avg_cost_usd * 4.0)
    latency_penalty = min(0.08, avg_latency_seconds / 120.0)
    throughput_penalty = min(0.05, max(0.0, 20.0 - docs_per_minute) / 200.0)
    crash_penalty = min(0.20, crash_rate * 0.20)
    return max(0.0, accuracy - cost_penalty - latency_penalty - throughput_penalty - crash_penalty)


def serialize_documents(documents: dict[str, DocumentMetrics]) -> dict[str, Any]:
    return {
        document_id: {
            "header_matches": metric.header_matches,
            "header_total": metric.header_total,
            "line_matches": metric.line_matches,
            "line_total": metric.line_total,
            "cost_usd": metric.cost_usd,
            "latency_seconds": metric.latency_seconds,
            "crashed": metric.crashed,
        }
        for document_id, metric in documents.items()
    }


def comparison_rows(documents: dict[str, DocumentMetrics]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document_id in sorted(documents, key=natural_sort_key):
        rows.extend(public_comparison_row(row) for row in (documents[document_id].comparisons or []))
    return rows


def public_comparison_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not str(key).startswith("_")}


def write_comparison_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COMPARISON_COLUMNS), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def natural_sort_key(value: str) -> list[int | str]:
    parts: list[int | str] = []
    for part in re.split(r"(\d+)", value):
        if part.isdigit():
            parts.append(int(part))
        elif part:
            parts.append(part)
    return parts


def print_summary(summary: dict[str, Any]) -> None:
    print("---")
    print(f"accuracy:             {summary['accuracy']:.6f}")
    print(f"header_accuracy:      {summary['header_accuracy']:.6f}")
    print(f"line_item_accuracy:   {summary['line_item_accuracy']:.6f}")
    print(f"adjusted_score:       {summary['adjusted_score']:.6f}")
    print(f"docs:                 {summary['docs']}")
    print(f"avg_cost_usd:         {summary['avg_cost_usd']:.6f}")
    print(f"avg_latency_seconds:  {summary['avg_latency_seconds']:.3f}")
    print(f"docs_per_minute:      {summary['docs_per_minute']:.3f}")
    print(f"crash_rate:           {summary['crash_rate']:.6f}")
    print(f"field_matches:        {summary['field_matches']}")
    print(f"field_total:          {summary['field_total']}")


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
