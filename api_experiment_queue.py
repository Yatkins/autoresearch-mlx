#!/usr/bin/env python3
"""Run external-API invoice experiments from a queue.

This script is intentionally launched by the user. It may send invoice document
contents or OCR text to the providers configured in the queue.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RESULTS_HEADER = ["commit", "accuracy", "adjusted_score", "cost_per_doc", "latency_s", "status", "description"]


@dataclass
class ResultRow:
    commit: str
    accuracy: float
    adjusted_score: float
    cost_per_doc: float
    latency_s: float
    status: str
    description: str

    @classmethod
    def from_values(cls, values: list[str]) -> "ResultRow":
        padded = values + [""] * (len(RESULTS_HEADER) - len(values))
        return cls(
            commit=padded[0],
            accuracy=safe_float(padded[1]),
            adjusted_score=safe_float(padded[2]),
            cost_per_doc=safe_float(padded[3]),
            latency_s=safe_float(padded[4]),
            status=padded[5],
            description=padded[6],
        )

    def values(self) -> list[str]:
        return [
            self.commit,
            f"{self.accuracy:.6f}",
            f"{self.adjusted_score:.6f}",
            f"{self.cost_per_doc:.6f}",
            f"{self.latency_s:.3f}",
            sanitize_tsv(self.status),
            sanitize_tsv(self.description),
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", default="api_experiments.json", help="JSON queue file to run.")
    parser.add_argument("--artifacts-dir", default="api_runs", help="Directory for logs, predictions, and reports.")
    parser.add_argument("--results", default="results.tsv", help="Canonical results TSV to append/update.")
    parser.add_argument("--min-accuracy-improvement", type=float, default=0.0005)
    parser.add_argument("--min-adjusted-improvement", type=float, default=0.002)
    parser.add_argument("--baseline-accuracy", type=float, help="Override the initial keep/discard accuracy threshold.")
    parser.add_argument("--baseline-adjusted-score", type=float, help="Override the initial adjusted-score tie-breaker.")
    parser.add_argument("--include-disabled", action="store_true", help="Run queue entries marked enabled=false.")
    parser.add_argument("--max-runs", type=int, default=0, help="Stop after this many executed experiments. 0 means no limit.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without executing train.py.")
    args = parser.parse_args()

    queue_path = Path(args.queue)
    queue = load_queue(queue_path)
    results_path = Path(args.results)
    rows = read_results(results_path)

    commit = git_commit()
    run_root = Path(args.artifacts_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    baseline = initial_baseline(rows, queue, args)
    best_accuracy, best_adjusted = baseline

    print(f"queue: {queue_path}")
    print(f"artifacts: {run_root}")
    print(f"commit: {commit}")
    print(f"initial_best_accuracy: {best_accuracy:.6f}")
    print(f"initial_best_adjusted_score: {best_adjusted:.6f}")

    executed = 0
    for index, experiment in enumerate(queue.get("experiments", []), start=1):
        if not experiment.get("enabled", True) and not args.include_disabled:
            print(f"skip disabled: {experiment.get('name', f'experiment-{index}')}")
            continue
        missing = [name for name in experiment.get("required_env", []) if not os.getenv(name)]
        if missing:
            print(f"skip missing env {missing}: {experiment.get('name', f'experiment-{index}')}")
            continue
        if args.max_runs and executed >= args.max_runs:
            break

        executed += 1
        result = run_experiment(
            experiment=experiment,
            index=index,
            queue_path=queue_path,
            run_root=run_root,
            commit=commit,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            continue

        row = result.row
        if row is None:
            row = ResultRow(
                commit=commit,
                accuracy=0.0,
                adjusted_score=0.0,
                cost_per_doc=0.0,
                latency_s=0.0,
                status="crash",
                description=result.description,
            )
        elif result.returncode != 0:
            row.status = "crash"
        else:
            row.status = decide_status(
                row,
                best_accuracy,
                best_adjusted,
                args.min_accuracy_improvement,
                args.min_adjusted_improvement,
            )
        append_result(results_path, row)

        if row.status == "keep":
            best_accuracy = max(best_accuracy, row.accuracy)
            best_adjusted = max(best_adjusted, row.adjusted_score)

        print(
            f"{row.status:7s} accuracy={row.accuracy:.6f} "
            f"adjusted={row.adjusted_score:.6f} artifact={result.artifact_dir}"
        )

    print(f"executed: {executed}")
    print(f"final_best_accuracy: {best_accuracy:.6f}")
    print(f"final_best_adjusted_score: {best_adjusted:.6f}")


@dataclass
class ExperimentResult:
    row: ResultRow | None
    returncode: int
    description: str
    artifact_dir: Path


def load_queue(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Queue file not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("experiments"), list):
        raise SystemExit("Queue file must contain an object with an experiments list.")
    return payload


def run_experiment(
    experiment: dict[str, Any],
    index: int,
    queue_path: Path,
    run_root: Path,
    commit: str,
    dry_run: bool,
) -> ExperimentResult:
    name = sanitize_slug(str(experiment.get("name") or f"experiment-{index}"))
    artifact_dir = run_root / f"{index:02d}-{name}"
    result_path = artifact_dir / "result.tsv"
    log_path = artifact_dir / "run.log"
    prediction_path = artifact_dir / "predictions.jsonl"
    report_path = artifact_dir / "invoice_report.json"
    description = scoped_description(experiment)

    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in experiment.get("env", {}).items()})
    if experiment.get("doc_limit") not in (None, "", 0, "0"):
        env["INVOICE_DOC_LIMIT"] = str(experiment["doc_limit"])
    env.update(
        {
            "AUTO_LOG_RESULTS": "1",
            "RUN_DESCRIPTION": description,
            "RUN_COMMIT": commit,
            "RESULTS_PATH": str(result_path),
            "PREDICTIONS_PATH": str(prediction_path),
            "REPORT_PATH": str(report_path),
        }
    )

    command = ["uv", "run", "train.py"]
    print(f"run {index}: {description}")
    print(f"  command: {' '.join(command)}")
    print(f"  artifact_dir: {artifact_dir}")
    if dry_run:
        return ExperimentResult(None, 0, description, artifact_dir)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "queue.json").write_text(json.dumps(experiment, indent=2, sort_keys=True) + "\n")
    shutil.copy2(queue_path, artifact_dir / "queue.snapshot.json")
    with log_path.open("w") as log_file:
        completed = subprocess.run(command, env=env, stdout=log_file, stderr=subprocess.STDOUT, check=False)

    row = read_single_result(result_path)
    return ExperimentResult(row, completed.returncode, description, artifact_dir)


def scoped_description(experiment: dict[str, Any]) -> str:
    description = str(experiment.get("description") or experiment.get("name") or "api experiment")
    doc_limit = str(experiment.get("doc_limit") or experiment.get("env", {}).get("INVOICE_DOC_LIMIT") or "")
    scope = "all-doc" if doc_limit in {"", "0"} else f"{doc_limit}-doc"
    if scope not in description.lower():
        description = f"{scope} {description}"
    return description


def read_single_result(path: Path) -> ResultRow | None:
    rows = read_results(path)
    return rows[-1] if rows else None


def read_results(path: Path) -> list[ResultRow]:
    if not path.exists() or not path.read_text().strip():
        return []
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        raw_rows = list(reader)
    if raw_rows and raw_rows[0] == RESULTS_HEADER:
        raw_rows = raw_rows[1:]
    return [ResultRow.from_values(row) for row in raw_rows if row]


def append_result(path: Path, row: ResultRow) -> None:
    path_exists = path.exists() and path.read_text().strip()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        if not path_exists:
            writer.writerow(RESULTS_HEADER)
        writer.writerow(row.values())


def initial_baseline(rows: list[ResultRow], queue: dict[str, Any], args: argparse.Namespace) -> tuple[float, float]:
    if args.baseline_accuracy is not None:
        return args.baseline_accuracy, args.baseline_adjusted_score or 0.0
    full_doc_queue = all(
        str(experiment.get("doc_limit") or experiment.get("env", {}).get("INVOICE_DOC_LIMIT") or "") in {"", "0"}
        for experiment in queue.get("experiments", [])
        if experiment.get("enabled", True) or args.include_disabled
    )
    kept = [row for row in rows if row.status == "keep"]
    if full_doc_queue:
        scoped = [row for row in kept if "all-doc" in row.description.lower()]
        if scoped:
            kept = scoped
    if not kept:
        return 0.0, 0.0
    best = max(kept, key=lambda row: (row.accuracy, row.adjusted_score))
    return best.accuracy, best.adjusted_score


def decide_status(
    row: ResultRow,
    best_accuracy: float,
    best_adjusted: float,
    min_accuracy_improvement: float,
    min_adjusted_improvement: float,
) -> str:
    if row.status == "crash":
        return "crash"
    if row.accuracy > best_accuracy + min_accuracy_improvement:
        return "keep"
    tied_accuracy = abs(row.accuracy - best_accuracy) <= min_accuracy_improvement
    if tied_accuracy and row.adjusted_score > best_adjusted + min_adjusted_improvement:
        return "keep"
    return "discard"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def sanitize_slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return value or "experiment"


def sanitize_tsv(value: str) -> str:
    return str(value).replace("\t", " ").replace("\n", " ").strip()


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
