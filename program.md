# Invoice OCR Extraction — Autoresearch Program

## Setup

Do these steps exactly once per session before starting the loop:

1. Activate the environment: `source ocr-research/bin/activate`
2. Load env vars: they are read automatically from `.env.local` by `evaluate.py` — no export needed
3. Check which backends are available by verifying `.env.local` has non-empty values for:
   - `MISTRAL_API_KEY` → mistral backend available
   - `OPENROUTER_API_KEY` → openrouter backend available
   - `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` + `AZURE_DOCUMENT_INTELLIGENCE_KEY` → azure available
   - Ollama running locally → run `curl -s http://localhost:11434/api/tags` to check
4. Run the baseline and record results:
   ```
   python evaluate.py
   ```
5. Fill in before starting the loop:

   **Baseline overall score:** `[fill in]`
   **Weakest fields (score < 0.80):** `[fill in from per-field output]`
   **Errors:** `[fill in]`

6. Commit the baseline: `git add evaluate.py score.py results.tsv && git commit -m "baseline"`

---

## Experimentation

### Goal

Maximize `overall` in `results.tsv`. Score is 0.0–1.0: character-level field accuracy averaged equally across all fields present in each ground truth, averaged across all invoices. Higher is always better. Git commits are the ground truth of what actually improved — `results.tsv` logs everything including failures.

---

### The optimization hierarchy

Work through phases in order. Do not skip to model-swapping — prompt engineering and field mapping almost always produce larger gains with zero API cost increase.

---

**Phase 1 — Prompt-driven field mapping (experiments 1–10)**

The model will return field names that may not exactly match the ground truth keys. Do not hardcode aliases — instead, fix this through the prompt. The `EXTRACTION_PROMPT` in `evaluate.py` already lists the exact required field names. If scores are low, the model is likely:

- Using different casing (e.g. `"vendor"` vs `"Vendor"`)
- Using abbreviated names (e.g. `"inv_no"` vs `"Invoice No"`)
- Nesting fields differently (e.g. returning address sub-fields instead of one string)
- Returning `Rows` under a different key (e.g. `"items"`, `"line_items"`)

To diagnose: add a temporary `print(json.dumps(extracted, indent=2))` for the first invoice in `evaluate.py`'s main loop, run once, read the raw output, then remove the debug line. Do not commit the debug print.

Fix mapping failures by improving `EXTRACTION_PROMPT` — make the field name requirements more explicit, add a negative example showing what NOT to do, or add the exact key name in quotes with emphasis. Fix format failures (dates, amounts, phone numbers) through `EXTRA_INSTRUCTIONS`.

The `Rows` sub-fields need particular attention: the ground truth uses `"Unit Per Case"`, `"Item Code"`, `"SKU"`, etc. The model may return these under different names or flatten them. Use `EXTRA_INSTRUCTIONS` to add per-subfield name enforcement if needed.

---

**Phase 2 — Prompt strategy exploration (experiments 11–25)**

With field naming clean, explore prompt strategies:

- **Explicit format constraints** — add to `EXTRA_INSTRUCTIONS`: date format, amount format (no currency symbols), phone number format (digits only). The scorer normalizes, but exact format match avoids edit-distance penalties entirely.
- **Chain-of-thought** — ask the model to list fields it can see before emitting JSON. Prefix: "First, list every field you can identify on this invoice. Then return the JSON."
- **Confidence filtering** — ask the model to omit fields it is uncertain about rather than guessing. Reduces hallucination on fields like `Holiday Invoice` or `Consession Vendor` that rarely appear.
- **Two-pass extraction** — first pass extracts header fields (Vendor, Invoice No, Invoice Date, Invoice Amount). Second pass extracts Rows only. Merge results. Implement as two sequential API calls in `run_mistral` / `run_openrouter`.
- **Few-shot example in prompt** — add one complete example extraction to `EXTRACTION_PROMPT` using a fake invoice. Show exact field names, date format, Rows structure. Keep the example compact to avoid token bloat.
- **Row-by-row instruction** — instead of listing Rows sub-fields once, describe what each sub-field means (e.g. "Unit Per Case: how many individual units are in one case/box"). Helps with domain-specific fields the model may not recognize.

One change per experiment. Commit if score improves, revert if not.

---

**Phase 3 — Model exploration (experiments 26–45)**

Change `MODEL_BACKEND` and `MODEL_NAME`. Always carry forward the best prompt from Phase 2 — test each new model with that prompt, not the original baseline prompt. Models to try:

| Backend | Model string | Notes |
|---|---|---|
| `mistral` | `mistral-ocr-latest` | baseline |
| `openrouter` | `mistralai/mistral-small-3.2-24b-instruct` | fast, cheaper |
| `openrouter` | `qwen/qwen2.5-vl-72b-instruct` | strong on structured docs |
| `openrouter` | `deepseek/deepseek-chat` | good on tables/rows |
| `azure` | _(no model string)_ | prebuilt-invoice, strong on standard fields |
| `ollama` | `hunyuan-vision` | local, no data sovereignty concern |
| `ollama` | `minicpm-v` | fast local option |

When changing models, only change `MODEL_BACKEND` and `MODEL_NAME`. Do not change the prompt simultaneously.

---

**Phase 4 — Compound and hybrid strategies (experiments 46+)**

Once best model + best prompt are known independently:

- Combine them and measure whether the improvement stacks
- **Hybrid routing**: use Azure prebuilt for header fields (it excels at structured invoice fields like vendor, date, total) and the best LLM for `Rows` (Azure often misses line item sub-fields). Merge the two outputs. Implement in `run_ocr()`.
- **Confidence-based field filling**: run the primary model, identify null or suspiciously short fields, re-query with a targeted prompt for just those fields
- Increase `MAX_INVOICES` to 40+ to validate that improvements hold across the full corpus, not just the first 20

---

### Rules — follow all of these

- **Only modify `evaluate.py`** — never touch `score.py`, `InvoiceFormat.json`, or any file in `Training Invoices/`
- **One logical change per experiment** — one prompt edit, or one model swap, or one structural change. Not combinations.
- **8-minute hard limit** — if a run exceeds 8 minutes, reduce `MAX_INVOICES` by 5 and rerun. Log the timeout in `results.tsv`.
- **On crash or all-zero scores** — revert immediately with `git checkout -- evaluate.py`. Do not spend more than 5 minutes debugging a single failed experiment.
- **On improvement** — `git add evaluate.py && git commit -m "exp-N: one-line description, score X.XXXX"`
- **On no improvement** — `git checkout -- evaluate.py`
- **Log every attempt** — write to `results.tsv` whether the experiment succeeded or failed. Include a short description of what changed in the final column.
- **Never invent field content** — do not add fake example data to the prompt that could be confused with real invoice data.
- **Do not chase individual invoice failures** — optimize for corpus average, not perfect scores on outliers.

---

## Output format

Every `python evaluate.py` run must produce this exact stdout format (the loop reads the SCORE line):

```
SCORE: 0.8234  (47.3s, 1 errors, 20 invoices)
Per-field:
  Vendor                              0.961  ████████████████████
  Invoice No                          0.921  ██████████████████░░
  Invoice Date                        0.887  █████████████████░░░
  Invoice Amount                      0.843  ████████████████░░░░
  Rows                                0.721  ██████████████░░░░░░
  Vendor Phone Number                 0.503  ██████████░░░░░░░░░░
  Holiday Invoice                     0.201  ████░░░░░░░░░░░░░░░░
```

`results.tsv` row format (tab-separated, one row per run):

```
<score>\t<backend>\t<model>\t<elapsed>\t<errors>err\t<n>inv\t<per_field_json>\t<description>
```

---

## Loop procedure

At the start of every session:

1. Read this file (`program.md`)
2. Read `results.tsv` to find current best score and experiment count
3. Run `git log --oneline -15` to see what has been committed
4. Determine which phase applies based on experiment count and score plateau
5. State your hypothesis for the next experiment in one sentence
6. Make exactly one change to `evaluate.py`
7. Run `python evaluate.py` and wait for completion
8. Compare `SCORE` to the best committed score
9. Commit if improved, revert if not, append to `results.tsv`
10. Repeat from step 4

Pause and summarize if 6 consecutive experiments produce no improvement, or if the session has run for 6 hours.

---

## Interpretation guide (for the human — read after overnight runs)

- **Low `Rows` score across all models** → the prompt's Rows sub-field description needs more specificity; try two-pass extraction (Phase 2)
- **Low `Vendor Phone Number` / `Vendor Fax Number`** → normalization is already handling digit-only comparison; if still low, the model is missing the field entirely, not formatting it wrong — add emphasis in the prompt
- **Low `Holiday Invoice` / `Consession Vendor`** → these are rare fields; if ground truth is mostly empty and model returns empty, score should be ~1.0. If score is low, check whether ground truth has values the model is missing entirely
- **Azure outperforming on header fields but losing on Rows** → implement hybrid routing in Phase 4
- **Score plateau after Phase 2** → the limiting factor is the model's vision quality on your invoice layouts, not the prompt; move to Phase 3
- **Git log vs results.tsv discrepancy** → normal; results.tsv has all attempts, git has only improvements. Count of git commits = number of genuine improvements