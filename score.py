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

    # Phone/fax: strip all non-digit characters for comparison
    if re.match(r'^[\d\s\(\)\-\+\.ext]+$', s) and len(re.sub(r'\D', '', s)) >= 7:
        return re.sub(r'\D', '', s)

    # General: lowercase, collapse whitespace
    s = s.lower()
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

def score_invoice(predicted: dict, ground_truth: dict) -> tuple[float, dict]:
    """
    Score one invoice. FLATTENED weighting: every field-cell is worth the same.
    Each header field counts as one cell, and EACH Rows sub-field cell (one per
    line-item per sub-field present in ground truth) also counts as one cell —
    so a 16-row invoice's line items dominate its score rather than the whole
    Rows block collectively counting as a single header-equivalent field.

    Fields absent from ground_truth are skipped entirely (not every invoice
    has every field — e.g. no Fax Number, no Bottle Deposit).
    Returns (overall_score, per_field_scores). per_field["Rows"] is the mean of
    that invoice's row cells (for display); overall is the mean of ALL cells.
    """
    if not ground_truth:
        return 0.0, {}

    field_scores = {}   # for per-field display
    cells = []          # flattened list — every cell equal weight in overall
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
            field_scores[field] = sum(row_cells) / len(row_cells) if row_cells else 1.0
        else:
            s = edit_sim(pred_val, truth_val)
            field_scores[field] = s
            cells.append(s)

    overall = sum(cells) / len(cells) if cells else 0.0
    return overall, field_scores

def score_corpus(results: list) -> dict:
    """
    results: list of (predicted_dict, ground_truth_dict)
    Returns overall score + per-field averages across corpus.
    """
    if not results:
        return {"overall": 0.0, "per_field": {}, "n_invoices": 0}

    all_field_scores: dict[str, list[float]] = {}
    invoice_scores = []

    for predicted, ground_truth in results:
        inv_score, field_scores = score_invoice(predicted, ground_truth)
        invoice_scores.append(inv_score)
        for field, s in field_scores.items():
            all_field_scores.setdefault(field, []).append(s)

    return {
        "overall": sum(invoice_scores) / len(invoice_scores),
        "per_field": {f: sum(v) / len(v) for f, v in all_field_scores.items()},
        "n_invoices": len(invoice_scores),
    }