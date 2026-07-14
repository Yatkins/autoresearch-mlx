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
- azure_best: 0.5579  (commit exp9; was 0.0533 → 0.5405 → 0.5579)
- last_sweep_best: 0.8252  (mistral-small at last milestone sweep; next milestone fires at ≥0.8352)
- no_gain_streak: 0  (reset — exp8 Azure win). exp5/6/7 (prompt) all regressed → mistral-small
  near its prompt ceiling (~0.8252); prefer non-prompt tracks (Azure headroom, mistral-ocr) and
  higher-leverage prompt ideas (few-shot, format hints) over more small wording tweaks.
- experiments_done: 8
- sweep_running: none
- NEXT: exp9 = extend Azure header/item mapping (Unit Per Case, UPC, VendorPhone/Email/Website,
  DueDate→? ) to push azure past 0.5405; then mistral-ocr _OCR_SCHEMA enrich (track C).

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
