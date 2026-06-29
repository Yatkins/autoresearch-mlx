# evaluate.py — AGENT MODIFIES THIS FILE each experiment
# Rules:
#   - Only modify the AGENT CONFIGURATION block
#   - Must complete in under 8 minutes
#   - Must print SCORE: X.XXXX as the first output line
#   - Must append one row to results.tsv
#   - Never modify score.py or any file in Training Invoices/

import json, time, base64, os, re
from pathlib import Path
from score import score_corpus, score_invoice, edit_sim  # do not change score.py's logic via here

# ============================================================
# AGENT CONFIGURATION BLOCK — modify only within this section
# ============================================================

MODEL_BACKEND = "mistral"
# Options: "mistral" | "openrouter" | "azure" | "ollama"

MODEL_NAME = "mistral-small-latest"
# NOTE: mistral-ocr-latest is NOT a chat model (only valid on the OCR endpoint),
# so it errors via chat.complete. mistral-small-latest is vision-capable and is
# the working mistral baseline. PDFs are rendered to PNG by render_to_images().
# mistral:     mistral-small-latest | mistral-medium-latest
# openrouter:  mistralai/mistral-small-3.2-24b-instruct
#              qwen/qwen2.5-vl-72b-instruct
#              deepseek/deepseek-chat
# azure:       (ignored — uses prebuilt-invoice)
# ollama:      hunyuan-vision | llava | minicpm-v

MAX_INVOICES = 20
# Keep experiments under 8 min. Reduce if timing out.

EXTRACTION_PROMPT = """
You are an invoice data extraction system.

Extract every field you can find from this invoice image and return a single JSON object.
Return ONLY valid JSON — no explanation, no markdown fences.

Use these exact field names as keys. If a field is not present on the invoice, omit it entirely (do not return null or empty string).

Top-level fields:
- Vendor
- Vendor Physical Address
- Vendor Phone Number
- Vendor Fax Number
- Vendor Email
- Vendor Website
- Invoice No
- Invoice Date
- Total Quantity
- Adjustment
- Bottle Deposit
- Invoice Amount
- Document Type
- Holiday Invoice
- Consession Vendor

Line items must be returned as a JSON array under the key "Rows". Each element in Rows should include whatever of these sub-fields are present:
- Item Code
- SKU
- Unit Per Case
- Description
- Quantity
- Unit Price
- Line Amount
- Discount
- Deposit

Example output structure (values are illustrative only):
{
  "Vendor": "Acme Supply Co",
  "Invoice No": "INV-1234",
  "Invoice Date": "3/24/2026",
  "Invoice Amount": "1250.00",
  "Rows": [
    {"Item Code": "A001", "Description": "Widget", "Quantity": "10", "Unit Price": "5.00", "Line Amount": "50.00"}
  ]
}
"""

EXTRA_INSTRUCTIONS = """
The "Document Type" field must be exactly "Bill" for a standard invoice or bill, or exactly "Credit" for a credit memo / return / credit note. Do not output "Invoice", "INVOICE", or any other value.

For the "Adjustment" and "Bottle Deposit" fields: if no explicit adjustment or bottle deposit amount is shown on the invoice, output "0.00" (these default to zero rather than being omitted). If a value is shown, use it.

Every element of "Rows" must ALWAYS include all of these sub-fields, even when a column is blank: Item Code, SKU, Unit Per Case, Description, Quantity, Cases, Pieces, Unit Price, Line Amount, Discount, Deposit, Deposit Qty. For "Cases", "Pieces", "Discount", and "Deposit", output "0" when that column is blank or absent (do not omit them). "Deposit Qty" is the deposit/unit count for the line — output the number shown (commonly "1"). Quantity is the number of units ordered for the line; if the layout has only a "Cases" column and no separate quantity column, use that column's value for Quantity.
"""
# Agent: add format hints here when per-field scores reveal systematic errors.
# Examples:
#   "Invoice Date must be returned exactly as printed on the invoice, e.g. 3/24/2026"
#   "Invoice Amount must be a plain decimal number without currency symbols, e.g. 82.80"
#   "Vendor Phone Number must be digits only, no dashes or parentheses"

# --- Cost / latency tracking (tunable; NOT optimization targets) ---
# Approximate USD per 1M tokens (input, output). Adjust to your provider's rates.
PRICE_PER_M = {
    "mistral-small-latest":  (0.10, 0.30),
    "mistral-medium-latest": (0.40, 2.00),
    "mistralai/mistral-small-3.2-24b-instruct": (0.05, 0.10),
    "qwen/qwen2.5-vl-72b-instruct": (0.25, 0.75),
    "deepseek/deepseek-chat": (0.27, 1.10),
}
PRICE_DEFAULT = (0.20, 0.60)  # fallback when model not in table

# "Adjusted score" = overall minus small, capped penalties for slow/expensive runs.
# It is recorded only as a DEAL-BREAKER signal — never use it to pick experiments.
LAT_FREE_S   = 8.0     # no latency penalty up to this many seconds/invoice
LAT_RATE     = 0.005   # penalty per second/invoice beyond the free allowance
LAT_CAP      = 0.10    # max latency penalty
COST_FREE    = 0.005   # no cost penalty up to this many USD/invoice
COST_RATE    = 4.0     # penalty per USD/invoice beyond the free allowance
COST_CAP     = 0.10    # max cost penalty

# ============================================================
# FIXED INFRASTRUCTURE — agent must not modify below this line
# ============================================================

# Token usage accumulator — backends append (prompt_tokens, completion_tokens)
# per API call so main() can estimate cost. Reset at the start of each run.
_USAGE: list = []

def _record_usage(prompt_tokens, completion_tokens):
    _USAGE.append((int(prompt_tokens or 0), int(completion_tokens or 0)))

def estimate_cost(model: str) -> float:
    pin, pout = PRICE_PER_M.get(model, PRICE_DEFAULT)
    tin = sum(u[0] for u in _USAGE)
    tout = sum(u[1] for u in _USAGE)
    return (tin / 1e6) * pin + (tout / 1e6) * pout

def load_env():
    env_file = Path(".env.local")
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v:
                os.environ[k] = v

def load_corpus(limit: int) -> list:
    """
    Expects Training Invoices/ to contain paired files:
        SomeInvoice.pdf  (or .png / .jpg)
        SomeInvoice.json  (ground truth)
    Returns list of (image_path, ground_truth_dict).
    """
    invoice_dir = Path("Training Invoices")
    pairs = []
    for gt_file in sorted(invoice_dir.glob("*.json"))[:limit]:
        for ext in [".pdf", ".png", ".jpg", ".jpeg", ".tiff"]:
            img = gt_file.with_suffix(ext)
            if img.exists():
                with open(gt_file) as f:
                    pairs.append((img, json.load(f)))
                break
    return pairs

def encode_file(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    media_map = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tiff": "image/tiff",
    }
    media = media_map.get(suffix, "image/png")
    return base64.b64encode(path.read_bytes()).decode(), media

# Chat/vision backends (mistral, openrouter, ollama) cannot read PDFs directly.
# Render every page of a PDF to PNG so multipage invoices are fully covered;
# pass through native image files unchanged. Returns a list of (base64, media).
PDF_RENDER_SCALE = 2.0   # ~144 DPI relative to PDF points; readable without bloat
MAX_IMG_WIDTH = 2000     # downscale wider renders to cap payload/latency

def render_to_images(path: Path) -> list[tuple[str, str]]:
    suffix = path.suffix.lower()
    if suffix != ".pdf":
        data, media = encode_file(path)
        return [(data, media)]

    import io
    import pypdfium2 as pdfium
    images = []
    pdf = pdfium.PdfDocument(str(path))
    try:
        for page in pdf:
            bitmap = page.render(scale=PDF_RENDER_SCALE)
            pil = bitmap.to_pil().convert("RGB")
            if pil.width > MAX_IMG_WIDTH:
                h = int(pil.height * MAX_IMG_WIDTH / pil.width)
                pil = pil.resize((MAX_IMG_WIDTH, h))
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            images.append((base64.b64encode(buf.getvalue()).decode(), "image/png"))
    finally:
        pdf.close()
    return images

def parse_json(text: str) -> dict:
    text = text.strip()
    # Strip markdown fences if present
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        # Attempt to extract JSON object from within the response
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}

def full_prompt() -> str:
    parts = [EXTRACTION_PROMPT.strip()]
    if EXTRA_INSTRUCTIONS.strip():
        parts.append(EXTRA_INSTRUCTIONS.strip())
    return "\n\n".join(parts)

# --- Backends ---

def run_mistral(path: Path) -> dict:
    from mistralai import Mistral
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    content = [
        {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}}
        for data, media in render_to_images(path)
    ]
    content.append({"type": "text", "text": full_prompt()})
    resp = client.chat.complete(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": content}],
        temperature=0,  # deterministic output so experiments are comparable
    )
    u = getattr(resp, "usage", None)
    if u is not None:
        _record_usage(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))
    return parse_json(resp.choices[0].message.content)

def run_openrouter(path: Path) -> dict:
    import httpx
    content = [
        {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}}
        for data, media in render_to_images(path)
    ]
    content.append({"type": "text", "text": full_prompt()})
    resp = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        json={"model": MODEL_NAME, "messages": [{"role": "user", "content": content}]},
        timeout=90,
    )
    body = resp.json()
    u = body.get("usage") or {}
    _record_usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
    return parse_json(body["choices"][0]["message"]["content"])

def run_azure(path: Path) -> dict:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential
    client = DocumentIntelligenceClient(
        os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"],
        AzureKeyCredential(os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"])
    )
    with open(path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-invoice", f, content_type="application/octet-stream"
        )
    result = poller.result()
    if not result.documents:
        return {}
    doc = result.documents[0]
    # Azure field names → your schema. Agent can refine EXTRA_INSTRUCTIONS
    # to improve this mapping rather than editing this function.
    azure_map = {
        "VendorName":           "Vendor",
        "VendorAddress":        "Vendor Physical Address",
        "VendorAddressRecipient": "Vendor",
        "InvoiceId":            "Invoice No",
        "InvoiceDate":          "Invoice Date",
        "InvoiceTotal":         "Invoice Amount",
        "Items":                "Rows",
    }
    out = {}
    for az_key, our_key in azure_map.items():
        field = (doc.fields or {}).get(az_key)
        if field:
            out[our_key] = field.content if hasattr(field, "content") else str(field.value or "")
    return out

def run_ollama(path: Path) -> dict:
    import httpx
    images = [data for data, _ in render_to_images(path)]
    resp = httpx.post(
        "http://localhost:11434/api/generate",
        json={"model": MODEL_NAME, "prompt": full_prompt(), "images": images, "stream": False},
        timeout=120,
    )
    return parse_json(resp.json().get("response", ""))

def run_ocr(path: Path) -> dict:
    dispatch = {
        "mistral":    run_mistral,
        "openrouter": run_openrouter,
        "azure":      run_azure,
        "ollama":     run_ollama,
    }
    fn = dispatch.get(MODEL_BACKEND)
    if not fn:
        raise ValueError(f"Unknown MODEL_BACKEND: {MODEL_BACKEND}")
    return fn(path)

# --- Reporting ---

def _fmt(v) -> str:
    if v is None:
        return "∅"
    s = str(v).replace("\n", " ")
    return s if len(s) <= 60 else s[:57] + "..."

def build_report(results, names, scores, meta: dict) -> str:
    """Markdown report: per invoice, every field extracted-vs-GT with similarity,
    and Rows expanded row-by-row so line-item extraction can be eyeballed."""
    lines = []
    lines.append(f"# Run report — {meta['timestamp']}")
    lines.append("")
    lines.append(f"- backend/model: `{meta['backend']}` / `{meta['model']}`")
    lines.append(f"- overall (flattened): **{meta['overall']:.4f}**   adjusted: {meta['adjusted']:.4f}")
    lines.append(f"- latency: {meta['latency']:.1f}s total ({meta['lat_per']:.1f}s/invoice)   "
                 f"cost: ${meta['cost']:.4f} (${meta['cost_per']:.5f}/invoice, est.)   "
                 f"errors: {meta['errors']}   invoices: {meta['n']}")
    lines.append("")
    for (pred, gt), name in zip(results, names):
        inv_score, field_scores = score_invoice(pred, gt)
        lines.append(f"## {name}  —  {inv_score:.3f}")
        lines.append("")
        # Header (non-Rows) fields
        lines.append("| field | sim | ground truth | extracted |")
        lines.append("|---|---|---|---|")
        for field, tv in gt.items():
            if field == "Rows":
                continue
            sim = edit_sim(pred.get(field), tv)
            lines.append(f"| {field} | {sim:.2f} | {_fmt(tv)} | {_fmt(pred.get(field))} |")
        lines.append("")
        # Rows expanded
        grows = gt.get("Rows", []) if isinstance(gt.get("Rows"), list) else []
        prows = pred.get("Rows", []) if isinstance(pred.get("Rows"), list) else []
        if grows:
            lines.append(f"**Rows** (gt {len(grows)} / pred {len(prows)}, sim {field_scores.get('Rows', 0):.2f})")
            lines.append("")
            for i, grow in enumerate(grows):
                if not isinstance(grow, dict):
                    continue
                prow = prows[i] if i < len(prows) and isinstance(prows[i], dict) else {}
                lines.append(f"- row {i+1}:")
                lines.append("")
                lines.append("  | sub-field | sim | gt | extracted |")
                lines.append("  |---|---|---|---|")
                for sub, tv in grow.items():
                    sim = edit_sim(prow.get(sub), tv)
                    lines.append(f"  | {sub} | {sim:.2f} | {_fmt(tv)} | {_fmt(prow.get(sub))} |")
                lines.append("")
        lines.append("")
    return "\n".join(lines)

# --- Main ---

def main():
    load_env()
    _USAGE.clear()
    start = time.time()
    corpus = load_corpus(MAX_INVOICES)

    if not corpus:
        print("SCORE: 0.0000  (0.0s, 0 invoices found — check Training Invoices/ path)")
        return

    results, names, errors = [], [], 0
    for img_path, ground_truth in corpus:
        names.append(img_path.stem)
        try:
            extracted = run_ocr(img_path)
            results.append((extracted, ground_truth))
        except Exception as e:
            print(f"  ERROR {img_path.name}: {e}")
            errors += 1
            results.append(({}, ground_truth))

    scores = score_corpus(results)
    elapsed = time.time() - start

    overall = scores["overall"]
    n = scores["n_invoices"]
    cost = estimate_cost(MODEL_NAME)
    lat_per = elapsed / n if n else 0.0
    cost_per = cost / n if n else 0.0

    # Adjusted score: deal-breaker signal only (penalize slow / expensive runs).
    lat_pen = min(LAT_CAP, max(0.0, (lat_per - LAT_FREE_S) * LAT_RATE))
    cost_pen = min(COST_CAP, max(0.0, (cost_per - COST_FREE) * COST_RATE))
    adjusted = overall - lat_pen - cost_pen

    print(f"SCORE: {overall:.4f}  ({elapsed:.1f}s, {errors} errors, {n} invoices)")
    print(f"Adjusted: {adjusted:.4f} (lat -{lat_pen:.3f}, cost -{cost_pen:.3f})  "
          f"Latency: {elapsed:.1f}s ({lat_per:.1f}s/inv)  Cost: ${cost:.4f} (${cost_per:.5f}/inv, est.)")
    print("Per-field:")
    for field, s in sorted(scores["per_field"].items(), key=lambda x: x[1]):
        bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
        print(f"  {field:35s} {s:.3f}  {bar}")

    # Timestamped per-run report
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(start))
    meta = {
        "timestamp": timestamp, "backend": MODEL_BACKEND, "model": MODEL_NAME,
        "overall": overall, "adjusted": adjusted, "latency": elapsed, "lat_per": lat_per,
        "cost": cost, "cost_per": cost_per, "errors": errors, "n": n,
    }
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_file = reports_dir / f"run_{timestamp}.md"
    report_file.write_text(build_report(results, names, scores, meta))
    print(f"Report: {report_file}")

    # Append to results.tsv (columns: score, adjusted, latency_s, cost_usd,
    # backend, model, errors, invoices, report, per_field_json, description)
    with open("results.tsv", "a") as f:
        desc = f"backend={MODEL_BACKEND} model={MODEL_NAME}"
        per_field_json = json.dumps(scores["per_field"])
        f.write(
            f"{overall:.4f}\t{adjusted:.4f}\t{elapsed:.1f}\t{cost:.4f}\t"
            f"{MODEL_BACKEND}\t{MODEL_NAME}\t{errors}\t{n}\t"
            f"{report_file.name}\t{per_field_json}\t{desc}\n"
        )

if __name__ == "__main__":
    main()