"""September 2026 offline audit: reaggregate saved scores; never run a model."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite metric")
    return number


def inspect_metrics(summary_path):
    """Check saved summary denominators and arithmetic means where rows survive."""
    summary_path = Path(summary_path)
    summary = read_csv(summary_path)
    raw_path = summary_path.with_name("per_example_scores.csv")
    raw = read_csv(raw_path) if raw_path.exists() else None
    errors, warnings, cohorts = [], [], []
    if not summary:
        errors.append("empty summary")
    if len({row.get("stage") for row in summary}) != len(summary):
        errors.append("duplicate summary stages")
    if raw is not None and set(row.get("stage") for row in raw) != set(row.get("stage") for row in summary):
        errors.append("raw and summary stage sets differ")
    metrics = ("rouge1", "rouge2", "rougeL", "rougeLsum", "sent_bleu", "bertscore_f1", "pred_chars")
    for row in summary:
        stage = row["stage"]
        used, total = int(row["n_used"]), int(row["n_total"])
        if not 0 < used <= total:
            errors.append(f"{stage}: invalid used/total denominator")
        for metric in metrics:
            val = finite(row[metric + "_mean"])
            if val < 0 or (metric.startswith("rouge") or metric == "bertscore_f1") and val > 1:
                errors.append(f"{stage}: invalid {metric} range")
        if not 0 <= finite(row["corpus_bleu"]) <= 100:
            errors.append(f"{stage}: invalid corpus BLEU range")
        cohort = {"stage": stage, "n_used": used, "n_total": total, "raw_scores_available": raw is not None}
        if raw is not None:
            rows = [item for item in raw if item["stage"] == stage]
            if len(rows) != used:
                errors.append(f"{stage}: row count differs from n_used")
            duplicates = len(rows) - len({item["key"] for item in rows})
            cohort["duplicate_keys"] = duplicates
            if duplicates:
                warnings.append(f"{stage}: {duplicates} repeated question/reference keys; original weighting retained")
            if not rows:
                errors.append(f"{stage}: missing raw rows")
            else:
                for metric in metrics:
                    mean = statistics.mean(finite(item[metric]) for item in rows)
                    if not math.isclose(mean, finite(row[metric + "_mean"]), rel_tol=1e-9, abs_tol=1e-9):
                        errors.append(f"{stage}: {metric} mean mismatch")
        cohorts.append(cohort)
    return {"cohorts": cohorts, "errors": errors, "warnings": warnings,
            "scope": "arithmetic means only; corpus BLEU cannot be recovered by averaging sentence BLEU"}


def inspect_decisions(summary_path):
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    decisions = summary_path.with_name("decisions.csv")
    rows = read_csv(decisions)
    counts = dict(Counter(row["winner"] for row in rows))
    expected = summary.get("wins", summary.get("counts", {}))
    errors = []
    if not rows or len(rows) != summary["judged"]:
        errors.append("decision count differs from judged denominator")
    if len({row["key"] for row in rows}) != len(rows):
        errors.append("duplicate decision keys")
    if set(counts) - set(expected):
        errors.append("unrecognized winner label")
    for label, count in expected.items():
        if counts.get(label, 0) != count:
            errors.append("winner counts differ from summary")
        if not math.isclose(finite(summary["winrate"][label]), count / len(rows) if rows else 0,
                            rel_tol=1e-9, abs_tol=1e-9):
            errors.append("winrate denominator mismatch")
    return {"rows": len(rows), "counts": counts, "errors": errors,
            "scope": "saved judge decisions only; no API request, re-judging or clinical validation"}


def verify_repository(root):
    root = Path(root)
    results = {"scope": "offline artifact integrity and saved-score reaggregation; no training/inference",
               "metrics": {}, "decisions": {}, "errors": [], "provenance_files_verified": 0}
    for p in sorted((root / "historical").rglob("summary_scores.csv")):
        result = inspect_metrics(p)
        name = p.relative_to(root).as_posix()
        results["metrics"][name] = result
        results["errors"].extend(f"{name}: {error}" for error in result["errors"])
    for p in sorted((root / "historical").rglob("decisions.csv")):
        summary = p.with_name("pairwise.summary.json")
        result = inspect_decisions(summary)
        name = p.relative_to(root).as_posix()
        results["decisions"][name] = result
        results["errors"].extend(f"{name}: {error}" for error in result["errors"])
    if not results["metrics"] or not results["decisions"]:
        results["errors"].append("historical score/decision artifacts missing")
    manifest = json.loads((root / "docs/provenance.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = (root / item["file"]).resolve()
        if not path.is_relative_to(root.resolve()):
            results["errors"].append("provenance path outside repository")
            continue
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["published_sha256"]:
            results["errors"].append(f"{item['file']}: provenance hash mismatch")
        else:
            results["provenance_files_verified"] += 1
    results["passed"] = not results["errors"]
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = verify_repository(args.root)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        result = {"passed": False, "errors": [f"invalid or missing artifact ({type(exc).__name__})"]}
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
