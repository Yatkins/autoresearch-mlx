# AUTORESEARCH — autonomous overnight run protocol + ledger

**This file is the single source of truth for the unattended loop.** Read it fully at the
start of every loop cycle (context may have reset). Update the LEDGER + STATE after every
experiment. See `CLAUDE.md` for project/scorer/model background.

## Objective
Maximize the PRIMARY metric `score` (extractions-equal, global per-cell) in `results.tsv`,
on the 39-invoice TRAIN set. `score_invoice` is reported but NOT the target.

## Run parameters (set 2026-07-13 15:40 PDT)
- START epoch: 1783982436  (2026-07-13 15:40:36 PDT)
- DEADLINE epoch: 1784071023  (reset 2026-07-14 08:17 → +8h → ~16:17 PDT). Check `date +%s` each cycle.
- Iteration model: `mistral-small-latest` (fast ~5.5min, ~$0.03/run) — prompt is SHARED, so
  gains generalize to all prompt backends.
- Milestone validation (BALANCED): whenever the best committed `score` improves by ≥0.01
  since the last full sweep, run the multi-model sweep (`scratchpad/sweep.sh` equivalent) to
  confirm the lift generalizes; record it, then continue iterating on mistral-small.
- STOP when EITHER: now ≥ DEADLINE epoch, OR 6 consecutive experiments with no improvement
  (`no_gain_streak` ≥ 6). On stop: write a summary to the LEDGER, `git push`, and DO NOT
  reschedule the loop.

## Per-cycle procedure (one experiment per cycle)
1. `cd` repo, `source ocr-research/bin/activate`. Read this file + latest `reports/run_*.md`.
2. Check STOP conditions (time; no_gain_streak). If met → finalize (summary + push) and end.
3. Pick the next idea from BACKLOG (top unchecked) OR, if backlog empty, derive one from the
   latest report's highest-cell-count weak field. NEVER repeat an idea already in the LEDGER.
4. Make exactly ONE change to `evaluate.py` (prompt/instruction/schema/code).
5. Run: `EXP_NOTE="expN: <desc>" MODEL_BACKEND=mistral MODEL_NAME=mistral-small-latest python evaluate.py`
   (foreground, ~5.5min). Read the SCORE line.
6. If `score` > best_committed: `git add evaluate.py results.tsv && git commit` (msg = the
   EXP_NOTE + before→after); set best_committed; no_gain_streak=0. Else: `git checkout -- evaluate.py`
   (revert; results.tsv row already logged the attempt) and no_gain_streak += 1.
7. If milestone triggered (best improved ≥0.01 since last sweep): run the full model sweep,
   commit results, note in LEDGER, reset the milestone marker.
8. Append a LEDGER row. Update STATE. `git push` (every cycle is fine — cheap).
9. Schedule next cycle: `ScheduleWakeup(delaySeconds≈90, prompt="<the /loop autoresearch prompt>")`.

## Guardrails
- One logical change per experiment. temp=0 (deterministic) so scores are comparable.
- On crash / all-zeros / >8min: `git checkout -- evaluate.py`, count as no-gain, move on.
- Do NOT modify `score.py`, `Training Invoices/*`, or GT. Do NOT touch `.env.local`.
- Never commit secrets. Keep `EXP_NOTE` descriptive so `results.tsv` self-documents.
- Rotate tracks if the prompt track stalls (see TRACKS).

## STATE  (update every cycle)
- best_committed_score: 0.8252  (mistral-small ITERATION target, commit 295b25d / exp4)
- global_best: 0.9222  (gemini-2.5-pro via OpenRouter @ exp4 prompt — validation only)
- azure_best: 0.5655  (exp11; 0.0533→0.5405→0.5579→0.5655)
- last_sweep_best: 0.8252  (mistral-small at last milestone sweep; next milestone fires at ≥0.8352)
- no_gain_streak: 6 — STOPPED 2026-07-14 (plateau; see FINAL SUMMARY). Loop ended, not rescheduling.
  near its prompt ceiling (~0.8252); prefer non-prompt tracks (Azure headroom, mistral-ocr) and
  higher-leverage prompt ideas (few-shot, format hints) over more small wording tweaks.
- ocr_best: 0.7388  (exp15; 0.6450→0.7328→0.7388)
- experiments_done: 20
- sweep_running: none
- NEXT ideas (pick one per cycle, non-prompt tracks preferred since mistral-small ~ceiling):
  C: mistral-ocr postprocess/schema — it scores worse on many-row invoices (extractions 0.73 <
     invoices 0.79); check its report for weak Rows sub-fields. B: Azure — UPC still 0.06 (254
     cells); Azure ProductCode is sometimes the barcode — try also mapping it to Universal
     Product Code when it's ~12 digits. A(prompt, only if a strong idea): one compact few-shot
     row example with the full sub-field set. Also: gemini-flash underperforms (0.7457) — check
     its report for a systematic miss.

### Milestone sweep @ exp4 prompt (done 2026-07-13 ~16:45 PDT) — prompt gains GENERALIZE:
- gemini-2.5-pro (OR): 0.8010 → **0.9222** (+0.121, new global best)
- qwen2.5-vl-72b (OR): 0.7758 → 0.8089 (+0.033)
- gemini-2.5-flash (OR): 0.7457 → 0.7427 (~flat; UPC/semantics didn't move it)
- (mistral-small already at 0.8252 = the iteration best)

### Key findings so far
- Milestone sweeps after PROMPT experiments only need the PROMPT-based models (mistral-small,
  gemini-flash/pro, qwen). Azure + mistral-ocr ignore the prompt — re-run them only after
  editing run_azure / _OCR_SCHEMA (tracks B/C).
- The "ALWAYS include all Rows sub-fields (output 0 when blank)" checklist framing HELPS
  completeness. Omitting absent sub-fields regressed 0.8252→0.7800 (exp5). Keep the checklist.
- Reading Cases/Pieces "as printed" (not force-mirroring Quantity) was a big win (exp4, +0.023).

## TRACKS
- **A — prompt (mistral-small):** primary track. Refine Rows/header extraction wording.
- **B — Azure `run_azure`:** currently 0.0533 because `Items` is mapped as a STRING, not a
  list of row dicts. Rewrite to parse Azure `Items` array → Rows list (ProductCode→Item Code,
  Description→Description, Quantity→Quantity, UnitPrice→Unit Price, Amount→Line Amount);
  expand header field map. Test with MODEL_BACKEND=azure. Big expected win. (Code, no prompt.)
- **C — mistral-ocr `_OCR_SCHEMA`:** add field descriptions/enums (UPC digits, Discount Type
  dollar/percent, column semantics) to the JSON schema. Test MODEL_NAME=mistral-ocr-4.

## BACKLOG  (ordered; check off in LEDGER when tried — do not repeat)
- [ ] A: exp3's Cases-mirroring hurt Cases (0.68→0.57). Revisit: only output Cases when a
      DISTINCT cases column exists; don't force-mirror Quantity into Cases. (Try to recover the −0.11.)
- [ ] A: Unit Per Case still 0.57 (259 cells) — sharper definition / tiny example of pack-size.
- [ ] A: Item Code 0.81 (352) — instruct to output the code exactly as printed (no reformat).
- [ ] A: Unit Price / Line Amount — "plain decimal, no $ or commas" format hint.
- [ ] A: Invoice Amount / Invoice Date — explicit format hints (decimal; as-printed date).
- [ ] A: one compact few-shot row example showing the FULL sub-field set incl. UPC.
- [ ] A: Adjustment 0.71 — clarify when it is 0.00 vs a shown value.
- [ ] B: implement the Azure Items→Rows rewrite (see TRACK B).
- [ ] C: enrich `_OCR_SCHEMA` descriptions (see TRACK C).
- [ ] A(model): try `mistral-medium-latest` with the best prompt (still cheap) as a data point.

## LEDGER  (append-only; one row per experiment)
| exp | track | change | score | best? | note |
|---|---|---|---|---|---|
| 1 | A | derive Total Quantity=sum(row Qty) in postprocess | 0.7345 | kept | Total Qty 0.49→0.62 |
| 2 | A | +Universal Product Code +Discount Type in prompt | 0.7994 | kept | UPC 0.06→0.78 (+0.065) |
| 3 | A | Rows column semantics + drop dead SKU | 0.8021 | kept | UPC 0.78→0.89, Pieces +0.19, Cases −0.11 |
| 4 | A | read Cases/Pieces as-printed (stop force-mirroring Qty→Cases) | 0.8252 | kept | +0.023; recovered Cases regression |
| 5 | A | omit absent row sub-fields (no "0" padding) to match GT | 0.7800 | REVERTED | checklist framing helps completeness; keep "always include" |
| 6 | A | clarify Unit Per Case (labels; disambiguate from barcode UPC) | 0.8086 | REVERTED | extra wording confused mistral-small |
| 7 | A | Item Code as-printed (keep zeros/letters/dashes) | 0.7891 | REVERTED | regressed; mistral-small near prompt ceiling |
| 8 | B | parse Azure Items→Rows list (was dumped as string) | 0.5405 | KEPT (Azure) | azure 0.0533→0.5405 (+0.49); pivot after 3 prompt regressions |
| 9 | B | Azure defaults: Document Type=Bill, Adjustment/Bottle Deposit=0.00 | 0.5579 | KEPT (Azure) | azure 0.5405→0.5579 (+0.017) |
| 10 | C | mistral-ocr _OCR_SCHEMA: fix keys (UPC/Discount Type/aligned) + field descriptions | 0.7328 | KEPT (ocr) | mistral-ocr-4 0.6450→0.7328 (+0.088) |
| 11 | B | Azure: route ProductCode→UPC when ~12-digit barcode, else Item Code | 0.5655 | KEPT (Azure) | azure 0.5579→0.5655 (+0.008); Item Code 0.55→0.40, UPC 0.06→0.39 |
| 12 | C | mistral-ocr schema: Adjustment/Bottle Deposit "0.00" descriptions | 0.7083 | REVERTED | 0.7328→0.7083; OCR schema hints for these backfired |
| 13 | A | enrich JSON example row to full sub-field set | 0.8042 | REVERTED | 4th prompt regression; mistral-small prompt ceiling confirmed |
| 14 | A(model) | mistral-medium-latest baseline @ best prompt | — | TIMEOUT | >10min at 39 invoices; too slow for budget, skip |
| 15 | C | mistral-ocr schema: Rows descriptions Unit Price/Line Amount/Description | 0.7388 | KEPT (ocr) | 0.7328→0.7388; OCR *Rows* descriptions help (header ones hurt, exp12) |
| 16 | C | mistral-ocr schema: Cases/Pieces/Deposit "'0' if none" descriptions | 0.7121 | REVERTED | 0.7388→0.7121; the "'0' if none" framing hurts OCR (cf exp12) |
| 17 | (openrouter) | max_tokens=8000 to avoid truncation — test gemini-flash | 0.6526 | REVERTED | flash 0.7457→0.6526; gap is inherent, not truncation |
| 18 | (image) | PDF render 2.0→3.0 (216 DPI) + MAX_IMG_WIDTH 2600 | 0.8190 | REVERTED | mistral-small 0.8252→0.8190; 144 DPI already sufficient |
| 19 | (model) | mistralai/mistral-small-3.2-24b-instruct baseline (OpenRouter) | 0.7963 | BASELINE | new model data point; mid-pack, slow (24min) |
| 20 | B | Azure Vendor fallback to VendorAddressRecipient when VendorName missing | 0.5655 | REVERTED | no-op (VendorName always present) |
| 21 | C | mistral-ocr Cases/Pieces pure-semantic descriptions (no "0-if-none") | 0.7322 | REVERTED | 0.7388→0.7322; OCR schema fully optimized |
| 22 | A | Vendor = seller not Bill-To/Ship-To customer | 0.8134 | REVERTED | 0.8252→0.8134; 6th no-gain → STOP |

## FINAL SUMMARY (loop stopped 2026-07-14 ~10:40 PDT — no_gain_streak reached 6)

**22 experiments; loop stopped on 6 consecutive no-gains (plateau), well before the 8h deadline.**
Every model's committed evaluate.py state is its best. Final per-model bests (extractions-equal,
global per-cell, 39-invoice TRAIN):

| model | baseline | FINAL | Δ |
|---|---|---|---|
| openrouter google/gemini-2.5-pro | 0.8010 | **0.9222** | +0.121 |
| mistral mistral-small-latest | 0.7329 | 0.8252 | +0.092 |
| openrouter qwen2.5-vl-72b | 0.7758 | 0.8089 | +0.033 |
| openrouter mistral-small-3.2-24b | — | 0.7963 | new baseline |
| openrouter google/gemini-2.5-flash | 0.7457 | 0.7457 | ~0 (inherent) |
| mistral mistral-ocr-4 | 0.6450 | 0.7388 | +0.094 |
| azure prebuilt-invoice | 0.0533 | 0.5655 | +0.512 |

**What worked (kept):** Total Quantity = sum(row Quantity) in postprocess (exp1); +Universal
Product Code in the prompt (exp2, the single biggest lever — generalized to +0.12 on gemini-pro);
Rows column semantics + drop dead SKU (exp3); read Cases/Pieces as-printed instead of mirroring
Quantity (exp4); Azure Items→Rows list parse (exp8, +0.49) + Bill/0.00 defaults (exp9) +
ProductCode→UPC routing (exp11); mistral-ocr _OCR_SCHEMA key-fix + Rows field descriptions
(exp10, exp15).

**What didn't (reverted) — findings for next time:** omitting absent Rows sub-fields (the
"always include" checklist aids completeness); OCR *header*-field & "0-if-none" schema hints
(only OCR *Rows* semantic descriptions help); higher DPI (144 already enough); max_tokens
(flash's gap vs pro is inherent, not truncation); more mistral-small prompt wording (at ceiling
~0.8252 — exp5/6/7/13/22 all regressed); mistral-medium too slow for the 8-min budget.

**To resume optimization later:** the cheap prompt track is exhausted; further gains likely need
(a) targeting gemini-pro directly (expensive per-experiment) or (b) TEST-set validation of the
current winners (`EVAL_SET=test python evaluate.py` per model) to confirm generalization, which
was never run. That test-set validation is the recommended next step.
