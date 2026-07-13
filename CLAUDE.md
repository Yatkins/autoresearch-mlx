# CLAUDE.md — Invoice OCR Autoresearch (session-recovery brief)

**Purpose of this file:** if a chat session is lost, read this first. It captures everything
learned about this project so we don't have to re-derive it. It complements (does not replace)
the persistent memory in
`~/.claude/projects/-Users-compudime-Documents-Scan-N-Save-autoresearch-mlx/memory/`
(see MEMORY.md there). When something important changes, update BOTH this file and memory.

This is Karpathy's "autoresearch" loop, MLX/Mac port, applied to **invoice data extraction**:
an agent repeatedly edits `evaluate.py`, runs it, and keeps changes that raise the score.

---

## What we're optimizing

Extract structured fields from scanned invoices (vendor, invoice no/date/amount, and a `Rows`
array of line items) and match them against hand-labeled ground-truth JSON. The metric is
character-level field accuracy, 0.0–1.0, **higher is always better**. Optimize the raw
`overall` on the TRAIN set; validate winners on the held-out TEST set.

---

## Files (what each is and whether the agent may edit it)

| File | Editable? | What it is |
|---|---|---|
| `evaluate.py` | **YES — agent edits this each experiment** | Backends, prompt, config. Only the AGENT CONFIGURATION BLOCK. |
| `score.py` | **NO (locked)** — except by explicit user request | Canonical scorer + `normalize()`. |
| `program.md` | reference | The autoresearch loop spec / phase plan. |
| `leaderboard.py` | helper (read-only on results.tsv) | Best round **per model**, not just overall. |
| `Training Invoices/*.pdf, *.png` | **NO** | Invoice images (the inputs). |
| `Training Invoices/*.json` | **NO** | Ground truth. Paired by stem with the image. |
| `results.tsv` | append-only log | One row per run (all attempts, incl. failures). |
| `reports/run_<ts>.md` | generated (gitignored) | Per-invoice extracted-vs-GT diff, Rows expanded. |
| `.env.local` | secrets (gitignored) | API keys. |
| `InvoiceFormat.json` | reference | The target schema. |

**Corpus:** 49 paired invoices (41 PDF + 8 PNG) as of 2026-07-13. Ground truth lives in the
matching `.json`. `load_corpus()` sorts by filename, holds out the LAST `TEST_COUNT` (=10) as
TEST; the rest are TRAIN capped at `MAX_INVOICES` (=39). Current split = **39 train / 10 test
(~80/20), full corpus used.** For fast prompt iteration on slow backends you may temporarily
lower MAX_INVOICES; keep TEST_COUNT fixed so the test set stays stable.

---

## How to run

```bash
source ocr-research/bin/activate          # the venv (NOT ./run.sh's `claude` launcher)
python evaluate.py                         # TRAIN set, default backend/model
MODEL_BACKEND=openrouter MODEL_NAME=google/gemini-2.5-pro python evaluate.py   # sweep via env
EVAL_SET=test python evaluate.py           # validate on held-out set
python leaderboard.py                      # per-model best round + overall best
python leaderboard.py --set test           # only test-set rounds
```

Backends/models are chosen via `MODEL_BACKEND` / `MODEL_NAME` env vars (override the in-file
defaults) so a sweep runs many models without editing the file. `evaluate.py` auto-loads
`.env.local`. First stdout line is always `SCORE: X.XXXX`; every run appends a row to
`results.tsv` and writes `reports/run_<ts>.md`.

---

## The scorer (locked) — key behaviors

- **Flattened / per-cell equal weight.** Each header field = 1 cell; EACH `Rows` sub-field
  cell (per line-item × per sub-field present in GT) = 1 cell. So line items DOMINATE each
  invoice's score → optimization should focus on `Rows` accuracy.
- **TWO corpus metrics reported (user-directed 2026-07-13); `score_corpus` returns both:**
  - `overall` = **extractions-equal** (GLOBAL per-cell) — pools EVERY cell across ALL
    invoices and averages ONCE. "1 extraction = 1 extraction" corpus-wide; a cell weighs the
    same regardless of invoice or invoice size. Many-row invoices contribute more cells.
    **This is the PRIMARY / optimization target.**
  - `overall_invoice` = **invoices-equal** — each invoice's own cell-mean, then those means
    averaged equally (each invoice counts once; this was the pre-2026-07-13 `overall`).
    Reported for comparison only — do NOT optimize it.
  - (The user explicitly rejected a third "all-fields-equal" metric — do not add it back.)
  - `per_field` is pooled globally (display only). `score_invoice` still returns a per-invoice
    mean (used only by the report). All build on the shared `_invoice_cells()` helper.
  - **results.tsv columns changed** to: score, score_invoice, adjusted, latency_s, cost_usd,
    backend, model, errors, invoices, report, per_field_json, description. `leaderboard.py`
    ranks by `score` and shows `score_invoice` alongside.
  - **This changed all scores — old baselines are NOT comparable; re-baseline.** (Metric
    history: whole-Rows≈0.8233 → flattened per-invoice≈0.7183 → now global per-cell primary.)
- **Character-level** similarity (Levenshtein), not exact match: 1 char off on 10 chars = 0.9.
- **`normalize()` tolerances** (in score.py): dates → YYYY-MM-DD; currency symbols + thousands
  commas stripped; **trailing decimal zeros dropped** (`64.00`==`64`, `46.80`==`46.8`; integers
  / barcodes untouched); phones/fax → digits only; general text lowercased, commas dropped
  (missing commas in addresses not penalized), whitespace collapsed. These tolerances RAISE
  scores vs. the earliest baselines — re-baseline before comparing across that change.
- **Empty GT field:** model returning empty = 1.0 (not penalized); hallucinating a value = 0.5.
- **Rows aligned by position;** missing predicted rows score 0; extra pred sub-fields ignored.
- **Adjusted score** = overall minus small capped penalties for slow/expensive runs. It is a
  **deal-breaker signal ONLY — never the optimization target.** Target stays raw `overall`.

---

## Domain knowledge that matters

- **Quantity is the priority `Rows` sub-field.** On some invoices the printed **"Cases"
  column is actually the Quantity** (no separate quantity column). Describe Quantity
  *semantically* in `EXTRA_INSTRUCTIONS` (units/cases ordered) so the model maps the right
  column per layout — do NOT hardcode column-name aliases.
- Ground truth previously had stray keys `Store` (8 invoices) and `demo` (1) that the user
  removed (2026-06-25) — they shouldn't be scored.
- **Email field name (fixed 2026-07-13):** GT uses `Vendor Email Address` (12 files); the
  prompt now requests that name. **2 GT files still use the old `Vendor Email` key** — fix
  those stragglers or they score 0 (search: `grep -l '"Vendor Email"' "Training Invoices"/*.json`).
- **`Vendor Contact` stray key:** exactly 1 GT file has `"Vendor Contact": "Denise Bruno"`;
  the key is not in the schema/prompt, so that single cell scores 0. Blank cells (GT `""` +
  empty pred) do NOT hurt — scorer returns 1.0. This is one non-blank stray (like the old
  `Store`/`demo` keys); blank/remove it in GT if you want that cell to stop costing ~0.

---

## Model sweep — goal and status

**User's priority (2026-06-29):** breadth-first. Get EVERY target model running and record its
baseline cost/latency/accuracy BEFORE optimizing any single one. And when optimizing, optimize
**every** model — including the non-prompt ones (Azure `prebuilt-invoice`, `mistral-ocr`), which
are tuned via CODE/SCHEMA (field mapping, `_OCR_SCHEMA` enums/descriptions, post-processing),
not prompts. The true accuracy ranking is unknown until all are optimized.

**Best rounds so far (TRAIN, run `python leaderboard.py` for live numbers):**

| best | backend/model | notes |
|---|---|---|
| 0.8703 | openrouter `google/gemini-2.5-pro` | current leader |
| 0.8094 | gemini `gemini-2.5-flash` (native Google API) | |
| 0.7651 | openrouter `qwen/qwen2.5-vl-72b-instruct` | |
| 0.7183 | mistral `mistral-small-latest` | the working mistral baseline / reference |
| 0.6105 | mistral `mistral-ocr-4` (OCR endpoint, no prompt) | tune via `_OCR_SCHEMA` |
| 0.1146 | azure `prebuilt-invoice` | low on flattened scorer (few fields); expand field map |
| 0.0284 | gemini `gemini-2.5-pro` (native) | **FAILED — HTTP 429**, needs billing-enabled tier; use OpenRouter route instead |

Reachable with current keys: mistral chat, Mistral OCR endpoint, Azure prebuilt-invoice, and via
OpenRouter: gemini-2.5-pro/flash, qwen2.5-vl-72b, mistral-small-3.2. Need self-serving (not on
OpenRouter): Qwen2.5-VL-7B, HunyuanOCR, PaddleOCR-VL, LayoutLMv3 (encoder, not generative).

**PaddleOCR-VL native transformers-on-Mac is BLOCKED** (rope/causal_mask version whack-a-mole).
Don't retry that path — use GGUF via llama.cpp/Docker Model Runner (Metal), or vLLM on a
Linux/NVIDIA box. See memory `paddleocr-vl-native-blocked.md`.

---

## Backend gotchas (in evaluate.py)

- **Vision/chat backends** (mistral chat, openrouter, gemini, ollama) can't read PDFs directly —
  `render_to_images()` rasterizes every PDF page to PNG first (scale 2.0 ≈ 144 DPI, downscaled
  to ≤2000px wide). So the PDF text layer, if any, is invisible to these backends.
- **Mistral OCR endpoint** (`mistral-ocr*`) and **Azure** (`prebuilt-invoice`) receive the RAW
  PDF bytes — so an embedded text layer WOULD let them skip real OCR. To keep the benchmark
  honest, **all invoice PDFs must be image-only (no text layer).** Verified 2026-07-13: all 41
  PDFs are clean. Re-check after adding invoices:
  ```python
  import pypdfium2 as pdfium; from pathlib import Path
  for p in sorted(Path("Training Invoices").glob("*.pdf")):
      pdf=pdfium.PdfDocument(str(p))
      n=sum(len(pg.get_textpage().get_text_range().strip()) for pg in pdf); pdf.close()
      if n>20: print("TEXT LAYER:", p.name, n)
  ```
  To flatten an offender: render pages with pypdfium2 → PIL, re-save with
  `img.save(path, format="PDF", save_all=True, append_images=rest)` (strips text). Back up first.
- `mistral-ocr-latest` is NOT a chat model — only valid on the OCR endpoint (errors via
  chat.complete). `mistral-small-latest` is the working vision chat baseline.
- OpenRouter/Mistral/Gemini calls use `temperature=0` for deterministic, comparable runs.

---

## Loop discipline (from program.md)

- One logical change per experiment (one prompt edit OR one model swap OR one structural change).
- 8-minute hard limit per run; reduce `MAX_INVOICES` if timing out.
- On improvement: `git commit` (git = ground truth of what actually helped).
- On no improvement / crash / all-zeros: `git checkout -- evaluate.py`, don't debug >5 min.
- Log EVERY attempt to results.tsv (success or fail).
- Don't chase outlier invoices — optimize the corpus average.
- Pause & summarize after 6 no-improvement experiments in a row.

---

## Open threads / next steps

1. **Track optimal round per model** — done via `leaderboard.py` (best per backend/model/set +
   overall). Run it any time; it reads results.tsv, no re-runs needed.
2. **Full corpus in use** — split raised to 39 train / 10 test (2026-07-13). DONE.
3. **Email field name fixed** in the prompt (`Vendor Email Address`). Remaining: 2 GT files
   still on the old `Vendor Email` key; 1 GT file has a stray non-blank `Vendor Contact`.
4. **Re-baseline all models** under the new 39/10 split — the old results.tsv rows were on the
   20/7 split (different invoices), so they're not directly comparable. Re-run the sweep.
5. Continue the breadth-first sweep, then optimize each model (prompts for chat models; schema/
   code for Azure + mistral-ocr).
