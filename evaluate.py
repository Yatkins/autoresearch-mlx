# evaluate.py — AGENT MODIFIES THIS FILE each experiment
# Rules:
#   - Only modify the AGENT CONFIGURATION block
#   - Must complete in under 8 minutes
#   - Must print SCORE: X.XXXX as the first output line
#   - Must append one row to results.tsv
#   - Never modify score.py or any file in Training Invoices/

import json, time, base64, os, re
from pathlib import Path
from score import score_corpus  # do not change this import

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
"""
# Agent: add format hints here when per-field scores reveal systematic errors.
# Examples:
#   "Invoice Date must be returned exactly as printed on the invoice, e.g. 3/24/2026"
#   "Invoice Amount must be a plain decimal number without currency symbols, e.g. 82.80"
#   "Vendor Phone Number must be digits only, no dashes or parentheses"

# ============================================================
# FIXED INFRASTRUCTURE — agent must not modify below this line
# ============================================================

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
    return parse_json(resp.json()["choices"][0]["message"]["content"])

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

# --- Main ---

def main():
    load_env()
    start = time.time()
    corpus = load_corpus(MAX_INVOICES)

    if not corpus:
        print("SCORE: 0.0000  (0.0s, 0 invoices found — check Training Invoices/ path)")
        return

    results, errors = [], 0
    for img_path, ground_truth in corpus:
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
    print(f"SCORE: {overall:.4f}  ({elapsed:.1f}s, {errors} errors, {len(corpus)} invoices)")
    print("Per-field:")
    for field, s in sorted(scores["per_field"].items(), key=lambda x: x[1]):
        bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
        print(f"  {field:35s} {s:.3f}  {bar}")

    # Append to results.tsv
    with open("results.tsv", "a") as f:
        desc = f"backend={MODEL_BACKEND} model={MODEL_NAME}"
        per_field_json = json.dumps(scores["per_field"])
        f.write(
            f"{overall:.4f}\t{MODEL_BACKEND}\t{MODEL_NAME}\t"
            f"{elapsed:.1f}s\t{errors}err\t{scores['n_invoices']}inv\t"
            f"{per_field_json}\t{desc}\n"
        )

if __name__ == "__main__":
    main()