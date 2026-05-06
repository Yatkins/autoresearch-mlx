# invoice-ocr-autoresearch

Autonomous experiment loop for invoice OCR and structured data extraction.

The goal is to maximize extraction `accuracy` on a labeled invoice set. The runner also reports an `adjusted_score` that penalizes cost, latency, low throughput, and crashes, but accuracy remains the primary metric.

This repo is designed for “bring your own keys”: no credentials, invoices, predictions, logs, or generated reports are committed.

## What Is Included

- `train.py` - mutable experiment runner. Autoresearch edits this file.
- `score_invoices.py` - fixed evaluator. Treat this as read-only during a run.
- `program.md` - protocol for the autonomous keep/discard loop.
- `results.tsv` - run history.
- `pyproject.toml` / `uv.lock` - reproducible Python environment.

## Data Layout

By default, the runner expects invoices one folder above the repo:

```text
../Training_Invoices/
  1.pdf
  1.json
  2.pdf
  2.json
```

Each document file should have a matching label JSON with the same stem. The current evaluator supports simple labels with header fields plus `Rows`, for example:

```json
{
  "Vendor": "Acme",
  "Invoice No": "123",
  "Invoice Date": "4/13/2026",
  "Invoice Amount": "748.05",
  "Rows": [
    {"Item Code": "1103", "Description": "Water", "Quantity": "10"}
  ]
}
```

Override the location with:

```bash
export INVOICE_DATA_DIR=/path/to/Training_Invoices
```

## Setup

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

For local OCR experiments:

```bash
uv sync --extra local-ocr
```

For Hugging Face document models such as Donut, LayoutLMv3, HunyuanOCR, and DeepSeek-OCR:

```bash
uv sync --extra local-ocr --extra hf-doc
```

The `hf-doc` extra pins the source Transformers commit required by HunyuanOCR.

## Credentials

Export only the keys you intend to use. Do not commit keys.

```bash
export MISTRAL_API_KEY=...
export OPENROUTER_API_KEY=...
export AZURE_FORM_RECOGNIZER_ENDPOINT=...
export AZURE_FORM_RECOGNIZER_KEY=...
export HF_TOKEN=...
export AZURE_CUSTOM_MODEL_ID=...
```

On macOS with zsh, put personal exports in your home config, not in this repo:

```bash
/Users/<you>/.zshrc
```

Then reload:

```bash
source ~/.zshrc
```

## Run One Experiment

Start with one invoice while developing:

```bash
INVOICE_DOC_LIMIT=1 \
INVOICE_EXPERIMENT=mistral_ocr_small4_v1 \
RUN_DESCRIPTION="Mistral OCR plus structured extraction" \
uv run train.py > run.log 2>&1
```

Useful experiments:

```bash
INVOICE_EXPERIMENT=mistral_ocr_small4_v1
INVOICE_EXPERIMENT=mistral_ocr_small4_table_html
INVOICE_EXPERIMENT=paddleocr_v4_mistral
INVOICE_EXPERIMENT=azure_prebuilt_invoice
INVOICE_EXPERIMENT=openrouter_vision
INVOICE_EXPERIMENT=paddleocr_v4_regex
INVOICE_EXPERIMENT=donut_cord_regex
INVOICE_EXPERIMENT=layoutlmv3_invoice_token
INVOICE_EXPERIMENT=hunyuanocr_direct
INVOICE_EXPERIMENT=deepseek_ocr_regex
```

## Overnight API Queue

Codex cannot launch external API runs that upload invoice contents, but you can
start the API lane yourself and let it run unattended. The queue runner executes
the experiments in `api_experiments.json`, stores per-run artifacts under
`api_runs/`, and appends keep/discard/crash rows to `results.tsv`.

Review `api_experiments.json`, export the provider keys you want to use, then run:

```bash
python3 api_experiment_queue.py --queue api_experiments.json
```

Useful controls:

```bash
python3 api_experiment_queue.py --dry-run
python3 api_experiment_queue.py --max-runs 1
python3 api_experiment_queue.py --include-disabled
python3 api_experiment_queue.py --baseline-accuracy 0.562167 --baseline-adjusted-score 0.464865
```

This command may send invoice PDFs/images or OCR text to the external providers
listed in the queue.

For HunyuanOCR on Apple Silicon, use lower render DPI to avoid MPS memory errors:

```bash
PDF_RENDER_DPI=96 HUNYUAN_MAX_NEW_TOKENS=512
```

## Tracking Runs

`train.py` appends every run to `results.tsv` unless disabled:

```bash
AUTO_LOG_RESULTS=0 uv run train.py
```

After a run:

```bash
tail -n 5 results.tsv
tail -n 80 run.log
```

The TSV columns are:

```tsv
commit	accuracy	adjusted_score	cost_per_doc	latency_s	status	description
```

Use `keep` for improvements, `discard` for losing ideas, and `crash` for failed runs. The loop should prefer higher `accuracy`; use `adjusted_score` as a tie-breaker.

The scorer is tuned for business-impact extraction accuracy. Model outputs are saved raw; normalization is only used inside the scorer and comparison report.

- Critical line-item fields are VIC/vendor item code, UPC/SKU/barcode when visible, description, quantity, unit price, line amount, discount, and deposit.
- `Total Quantity` is optional and is not scored.
- `Units Per Case` is synonymous with `CS Qty`, case quantity, pack size, and case pack.
- Predictions are not rewritten to default missing values to `0`.
- For zero-style fields such as adjustment, bottle deposit, cases, pieces, discount, and deposit, missing and zero are equivalent in the scorer.
- Quantity can match when it is derivable from `Line Amount / Unit Price`, present as `Pieces` or `Cases`, or computed from `Cases * Units Per Case`.
- Invoice number and credit number both score as `invoice_no`.
- Document type is treated as model inference for Bill vs Credit.
- Columns that are blank/zero in both truth and predictions for the whole evaluated set are ignored.

## Current Practical Notes

- The finalized local baseline is PaddleOCR v4 regex extraction at `accuracy=0.708791`, `adjusted_score=0.608084`.
- Earlier rows from older metric definitions are not comparable to the finalized scorer.
- Mistral OCR plus structured extraction, Mistral table HTML, Azure, and OpenRouter are wired for API comparison once credentials are loaded.
- HunyuanOCR and DeepSeek-OCR are available as local/self-hosted experiments, but may require dependency or hardware-specific tuning.

## License

MIT. See [LICENSE](LICENSE).
