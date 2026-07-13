# score.py — READ ONLY, agent must never modify this file

import re
from typing import Any

def levenshtein(a: str, b: str) -> int:
    if not a: return len(b)
    if not b: return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], prev if ca == cb else 1 + min(prev, dp[j], dp[j-1])
    return dp[-1]

def normalize(value: Any) -> str:
    """
    Normalize before comparison.
    Handles: dates, currency, phone numbers, whitespace, case.
    The agent can extend EXTRA_INSTRUCTIONS to guide model output format,
    but this function is the canonical normalization — do not modify it.
    """
    if value is None:
        return ""
    s = str(value).strip()

    # Dates: normalize M/D/YYYY, MM-DD-YYYY, YYYY-MM-DD etc. → YYYY-MM-DD
    date_match = re.match(
        r'^(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})$', s
    )
    if date_match:
        m, d, y = date_match.groups()
        if len(y) == 2:
            y = "20" + y
        return f"{y}-{int(m):02d}-{int(d):02d}"

    # Currency: strip symbols and thousands separators, keep decimal
    s = re.sub(r'[$€£]', '', s)
    s = re.sub(r',(?=\d{3})', '', s)  # remove thousands commas only

    # Numeric: drop trailing zeros after a decimal point so 64.00 == 64 and
    # 46.80 == 46.8 (decimal-place differences are not penalized). Integers and
    # alphanumeric codes (e.g. barcodes/SKUs) are left untouched.
    if re.fullmatch(r'-?\d+\.\d+', s):
        s = s.rstrip('0').rstrip('.')

    # Phone/fax: strip all non-digit characters for comparison
    if re.match(r'^[\d\s\(\)\-\+\.ext]+$', s) and len(re.sub(r'\D', '', s)) >= 7:
        return re.sub(r'\D', '', s)

    # General: lowercase, drop commas (missing commas in addresses are not
    # penalized), collapse whitespace
    s = s.lower().replace(',', ' ')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def edit_sim(pred: Any, truth: Any) -> float:
    """Character-level similarity. 1 char off on a 10-char string = 0.9, not 0.0."""
    p, t = normalize(pred), normalize(truth)
    if not t:
        # Ground truth is empty/null — field not present on this invoice.
        # Don't penalize model for also returning empty; don't reward hallucination.
        return 1.0 if not p else 0.5
    max_len = max(len(p), len(t))
    if max_len == 0:
        return 1.0
    return 1.0 - levenshtein(p, t) / max_len

def score_rows(pred_rows: list, truth_rows: list) -> float:
    """
    Score Rows field: align by position, score each row's fields.
    Only evaluate sub-fields that appear in the ground truth row.
    Extra fields in pred (e.g. 'Cases') are ignored — not penalized.
    Missing pred rows count as 0.
    """
    if not truth_rows:
        return 1.0 if not pred_rows else 0.5
    
    row_scores = []
    for i, truth_row in enumerate(truth_rows):
        if i >= len(pred_rows):
            row_scores.append(0.0)
            continue
        pred_row = pred_rows[i] if isinstance(pred_rows[i], dict) else {}
        
        field_scores = []
        for field, truth_val in truth_row.items():
            pred_val = pred_row.get(field)
            field_scores.append(edit_sim(pred_val, truth_val))
        
        row_scores.append(sum(field_scores) / len(field_scores) if field_scores else 0.0)
    
    return sum(row_scores) / len(row_scores)

def _invoice_cells(predicted: dict, ground_truth: dict) -> tuple[list, dict]:
    """
    FLATTENED scoring for one invoice. Returns (cells, field_cells):
      - cells: flat list of every scored cell (one per header field, plus one per
        Rows sub-field per line-item present in ground truth).
      - field_cells: field name -> list of that field's cells (Rows keeps all its
        row cells; header fields have a single-element list).
    Fields absent from ground_truth are skipped (not every invoice has every field).
    This is the single source of truth for cells; both per-invoice and corpus
    aggregation build on it so weighting is consistent.
    """
    cells: list = []
    field_cells: dict[str, list] = {}
    if not ground_truth:
        return cells, field_cells
    for field, truth_val in ground_truth.items():
        pred_val = predicted.get(field)
        if field == "Rows":
            truth_rows = truth_val if isinstance(truth_val, list) else []
            pred_rows = pred_val if isinstance(pred_val, list) else []
            row_cells = []
            for i, truth_row in enumerate(truth_rows):
                if not isinstance(truth_row, dict):
                    continue
                pred_row = pred_rows[i] if i < len(pred_rows) and isinstance(pred_rows[i], dict) else {}
                for sub, tv in truth_row.items():
                    row_cells.append(edit_sim(pred_row.get(sub), tv))
            cells.extend(row_cells)
            field_cells[field] = row_cells
        else:
            s = edit_sim(pred_val, truth_val)
            cells.append(s)
            field_cells[field] = [s]
    return cells, field_cells

def score_invoice(predicted: dict, ground_truth: dict) -> tuple[float, dict]:
    """
    Score one invoice (for per-invoice display/reporting). Every field-cell is
    worth the same: each header field = 1 cell; EACH Rows sub-field cell = 1 cell,
    so a 16-row invoice's line items dominate its own score.
    Returns (overall_score, per_field_scores). per_field["Rows"] is the mean of
    that invoice's row cells; overall is the mean of ALL of this invoice's cells.
    NOTE: the corpus `overall` is NOT the mean of these per-invoice scores — see
    score_corpus, which pools every cell across all invoices (global per-cell).
    """
    if not ground_truth:
        return 0.0, {}
    cells, field_cells = _invoice_cells(predicted, ground_truth)
    field_scores = {f: (sum(c) / len(c) if c else 1.0) for f, c in field_cells.items()}
    overall = sum(cells) / len(cells) if cells else 0.0
    return overall, field_scores

def score_corpus(results: list) -> dict:
    """
    results: list of (predicted_dict, ground_truth_dict)

    Computes TWO aggregations of the same per-cell scores (user-directed 2026-07-13):

      - overall           = "all extractions equal" (GLOBAL per-cell). Every cell in the
                            whole corpus pooled and averaged ONCE — one extraction weighs
                            the same regardless of invoice or invoice size. PRIMARY /
                            optimization target. Many-row invoices contribute more cells.
      - overall_invoice   = "all invoices equal". Each invoice's own cell-mean, then those
                            per-invoice means averaged equally. Small invoices' cells weigh
                            more. (This was the pre-2026-07-13 `overall`.)

    per_field is pooled globally: each field's score is the mean over ALL of its cells
    across the corpus (for Rows, all row cells corpus-wide) — for display only.
    """
    if not results:
        return {"overall": 0.0, "overall_invoice": 0.0,
                "per_field": {}, "n_invoices": 0, "n_cells": 0}

    global_cells: list = []
    field_cells_all: dict[str, list] = {}
    invoice_means: list = []

    for predicted, ground_truth in results:
        cells, field_cells = _invoice_cells(predicted, ground_truth)
        global_cells.extend(cells)
        invoice_means.append(sum(cells) / len(cells) if cells else 0.0)
        for field, c in field_cells.items():
            field_cells_all.setdefault(field, []).extend(c)

    per_field = {f: sum(c) / len(c) for f, c in field_cells_all.items() if c}
    overall = sum(global_cells) / len(global_cells) if global_cells else 0.0
    overall_invoice = sum(invoice_means) / len(invoice_means) if invoice_means else 0.0
    return {
        "overall": overall,
        "overall_invoice": overall_invoice,
        "per_field": per_field,
        "n_invoices": len(results),
        "n_cells": len(global_cells),
    }