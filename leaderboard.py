# leaderboard.py — READ-ONLY analysis of results.tsv (does not touch evaluate.py/score.py).
# Shows the BEST round per model (its optimal config so far), not just the single
# overall winner. Run:  python leaderboard.py   [--set train|test|all]
#
# results.tsv columns (tab-separated, header on line 1):
#   score  adjusted  latency_s  cost_usd  backend  model  errors  invoices
#   report  per_field_json  description
# eval_set is parsed from the description prefix "[train]"/"[test]" when present
# (older baseline rows predate the split and are treated as "train").

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
        if len(c) < 11:
            continue
        desc = c[10]
        eval_set = "train"
        if desc.startswith("["):
            eval_set = desc[1:desc.index("]")] if "]" in desc else "train"
        try:
            rows.append({
                "score": float(c[0]), "adjusted": float(c[1]),
                "latency_s": float(c[2]), "cost_usd": float(c[3]),
                "backend": c[4], "model": c[5],
                "errors": int(c[6]), "invoices": int(c[7]),
                "report": c[8], "per_field": json.loads(c[9]),
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

    # Best round per (backend, model, set)
    best = {}
    counts = {}
    for r in rows:
        key = (r["backend"], r["model"], r["set"])
        counts[key] = counts.get(key, 0) + 1
        if key not in best or r["score"] > best[key]["score"]:
            best[key] = r

    ranked = sorted(best.values(), key=lambda r: r["score"], reverse=True)

    print(f"{'='*100}")
    print(f"PER-MODEL BEST ROUND   ({len(ranked)} model/set combos, {len(rows)} total rounds)")
    print(f"{'='*100}")
    hdr = f"{'best':>7} {'adj':>6} {'lat/inv':>8} {'$/inv':>8} {'set':>5} {'rnds':>4}  backend/model"
    print(hdr)
    print("-" * 100)
    for r in ranked:
        key = (r["backend"], r["model"], r["set"])
        n = r["invoices"] or 1
        print(f"{r['score']:>7.4f} {r['adjusted']:>6.3f} {r['latency_s']/n:>7.1f}s "
              f"{r['cost_usd']/n:>7.4f} {r['set']:>5} {counts[key]:>4}  {r['backend']}/{r['model']}")
        print(f"{'':>36}weakest: {weakest_fields(r['per_field'])}")
        print(f"{'':>36}report: {r['report']}")

    top = ranked[0]
    print("-" * 100)
    print(f"OVERALL BEST: {top['score']:.4f}  {top['backend']}/{top['model']} "
          f"[{top['set']}]  (report {top['report']})")

if __name__ == "__main__":
    main()
