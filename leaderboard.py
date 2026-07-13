# leaderboard.py — READ-ONLY analysis of results.tsv (does not touch evaluate.py/score.py).
# Shows the BEST round per model (its optimal config so far), not just the single
# overall winner. Run:  python leaderboard.py   [--set train|test|all]
#
# results.tsv columns (tab-separated, header on line 1):
#   score  score_invoice  adjusted  latency_s  cost_usd  backend  model  errors
#   invoices  report  per_field_json  description
# `score` = extractions-equal (PRIMARY, global per-cell); score_invoice = invoices-equal.
# Ranking uses `score`. eval_set parsed from the description prefix "[train]"/"[test]".

import sys, json
from pathlib import Path

RESULTS = Path("results.tsv")

def load_rows():
    rows = []
    if not RESULTS.exists():
        return rows
    lines = RESULTS.read_text().splitlines()
    for line in lines[1:]:  # skip header
        if not line.strip():
            continue
        c = line.split("\t")
        if len(c) < 12:
            continue
        desc = c[11]
        eval_set = "train"
        if desc.startswith("["):
            eval_set = desc[1:desc.index("]")] if "]" in desc else "train"
        try:
            rows.append({
                "score": float(c[0]), "score_invoice": float(c[1]),
                "adjusted": float(c[2]), "latency_s": float(c[3]),
                "cost_usd": float(c[4]), "backend": c[5], "model": c[6],
                "errors": int(c[7]), "invoices": int(c[8]),
                "report": c[9], "per_field": json.loads(c[10]),
                "desc": desc, "set": eval_set,
            })
        except (ValueError, json.JSONDecodeError):
            continue
    return rows

def weakest_fields(per_field, k=3):
    items = sorted(per_field.items(), key=lambda x: x[1])
    return ", ".join(f"{f}={s:.2f}" for f, s in items[:k])

def main():
    want = "all"
    if "--set" in sys.argv:
        want = sys.argv[sys.argv.index("--set") + 1]

    rows = load_rows()
    if want != "all":
        rows = [r for r in rows if r["set"] == want]
    if not rows:
        print("No matching rows in results.tsv.")
        return

    # Best round per (backend, model, set), ranked by the PRIMARY `score`.
    best, counts = {}, {}
    for r in rows:
        key = (r["backend"], r["model"], r["set"])
        counts[key] = counts.get(key, 0) + 1
        if key not in best or r["score"] > best[key]["score"]:
            best[key] = r

    ranked = sorted(best.values(), key=lambda r: r["score"], reverse=True)

    print(f"{'='*102}")
    print(f"PER-MODEL BEST ROUND   ({len(ranked)} model/set combos, {len(rows)} total rounds)")
    print(f"score = extractions-equal (PRIMARY) | inv = invoices-equal")
    print(f"{'='*102}")
    print(f"{'score':>7} {'inv':>6} {'adj':>6} {'lat/inv':>8} {'$/inv':>8} {'set':>5} {'rnds':>4}  backend/model")
    print("-" * 102)
    for r in ranked:
        key = (r["backend"], r["model"], r["set"])
        n = r["invoices"] or 1
        print(f"{r['score']:>7.4f} {r['score_invoice']:>6.4f} {r['adjusted']:>6.3f} "
              f"{r['latency_s']/n:>7.1f}s {r['cost_usd']/n:>7.4f} {r['set']:>5} {counts[key]:>4}  "
              f"{r['backend']}/{r['model']}")
        print(f"{'':>36}weakest: {weakest_fields(r['per_field'])}")
        print(f"{'':>36}report: {r['report']}")

    top = ranked[0]
    print("-" * 102)
    print(f"OVERALL BEST (extractions-equal): {top['score']:.4f}  "
          f"(invoices-equal {top['score_invoice']:.4f})  {top['backend']}/{top['model']} "
          f"[{top['set']}]  report {top['report']}")

if __name__ == "__main__":
    main()
