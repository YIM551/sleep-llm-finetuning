#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RESPONSE_SPLIT_RE = re.compile(r"### Response:\s*", re.IGNORECASE)

def extract_response(text: str) -> str:
    if not text:
        return ""
    parts = RESPONSE_SPLIT_RE.split(text, maxsplit=1)
    return (parts[1] if len(parts) > 1 else text).strip()

def read_jsonl(p: Path) -> List[Dict]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def load_preds_refs(p: Path) -> Tuple[List[str], List[str], Dict[str, float]]:
    rows = read_jsonl(p)

    preds: List[str] = []
    refs: List[str] = []
    empty = 0
    pred_chars = 0

    for r in rows:
        # priority: response_only format
        if "response" in r and ("reference" in r or "output" in r):
            pred = (r.get("response") or "").strip()
            ref = (r.get("reference") or r.get("output") or "").strip()
        else:
            gen = (r.get("generated") or r.get("generation") or r.get("pred") or r.get("text") or "").strip()
            pred = extract_response(gen)
            ref = (r.get("reference") or r.get("output") or "").strip()

        if not pred:
            empty += 1
        pred_chars += len(pred)
        preds.append(pred)
        refs.append(ref)

    n = max(1, len(preds))
    stats = {
        "n": float(len(preds)),
        "empty_rate": empty / n,
        "avg_pred_chars": pred_chars / n,
    }
    return preds, refs, stats

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return x

def compute_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    # Uses `evaluate` (like you already used). If unavailable, returns empty dict.
    try:
        import evaluate
        rouge = evaluate.load("rouge")
        bleu = evaluate.load("sacrebleu")
        r = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
        b = bleu.compute(predictions=preds, references=[[x] for x in refs])
        out = {
            "rouge1": safe_float(r.get("rouge1")),
            "rouge2": safe_float(r.get("rouge2")),
            "rougeL": safe_float(r.get("rougeL")),
            "bleu": safe_float(b.get("score")),
        }
        return out
    except Exception as e:
        return {"error": str(e)}

def find_eval_file(run_dir: Path, stage_tag: str) -> Optional[Path]:
    """
    Prefer:
      eval_fixed_<stage_tag>_200.response_only.jsonl
      eval_fixed_<stage_tag>_200.jsonl
    Then fallback to glob search.
    """
    cands = [
        run_dir / f"eval_fixed_{stage_tag}_200.response_only.jsonl",
        run_dir / f"eval_fixed_{stage_tag}_200.jsonl",
    ]
    for c in cands:
        if c.exists():
            return c

    # glob fallback
    globs = [
        f"eval_*{stage_tag}*response_only*.jsonl",
        f"eval_*{stage_tag}*.jsonl",
    ]
    for g in globs:
        hits = sorted(run_dir.glob(g))
        for h in hits:
            if h.exists() and h.is_file():
                return h
    return None

@dataclass
class RunSpec:
    name: str
    run_dir: Path
    stage_tag: str

def md_escape(s: str) -> str:
    return s.replace("|", "\\|")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results.md", help="Output markdown path")
    ap.add_argument("--root", type=str, default=".", help="Project root")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_path = (root / args.out).resolve()

    runs = [
        RunSpec("BASE(stage1)", root / "outputs/base_mistral7b", "stage1"),
        RunSpec("LoRA(stage1)", root / "outputs/exp_factual_only/stage1_factual", "stage1"),
        RunSpec("BASE(stage2)", root / "outputs/base_mistral7b", "stage2"),
        RunSpec("LoRA(stage2)", root / "outputs/m2_counsel_only/stage2_counseling", "stage2"),
        RunSpec("stage3(warmstart)", root / "outputs/exp_stage3_full_warmstart/stage3_full", "stage3"),
        RunSpec("stage3(scratch)", root / "outputs/exp_stage3_full_scratch/stage3_full", "stage3"),
    ]

    rows_md = []
    missing = []

    for rs in runs:
        if not rs.run_dir.exists():
            missing.append((rs.name, f"run_dir not found: {rs.run_dir}"))
            rows_md.append((rs.name, "MISSING", "-", "-", "-", "-", "-", "-", "-"))
            continue

        eval_file = find_eval_file(rs.run_dir, rs.stage_tag)
        if not eval_file:
            missing.append((rs.name, f"eval file not found under: {rs.run_dir} (tag={rs.stage_tag})"))
            rows_md.append((rs.name, "MISSING", "-", "-", "-", "-", "-", "-", "-"))
            continue

        preds, refs, stats = load_preds_refs(eval_file)
        metrics = compute_metrics(preds, refs)

        if "error" in metrics:
            # metrics failed; still write stats
            rows_md.append((
                rs.name,
                str(eval_file.relative_to(root)),
                int(stats["n"]),
                "ERR",
                "ERR",
                "ERR",
                "ERR",
                f'{stats["avg_pred_chars"]:.2f}',
                f'{stats["empty_rate"]:.3f}',
            ))
            missing.append((rs.name, f"metric compute failed: {metrics['error']}"))
        else:
            rows_md.append((
                rs.name,
                str(eval_file.relative_to(root)),
                int(stats["n"]),
                f'{metrics["rouge1"]:.6f}',
                f'{metrics["rouge2"]:.6f}',
                f'{metrics["rougeL"]:.6f}',
                f'{metrics["bleu"]:.6f}',
                f'{stats["avg_pred_chars"]:.2f}',
                f'{stats["empty_rate"]:.3f}',
            ))

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = []
    md.append(f"# Evaluation Summary\n")
    md.append(f"- Generated at: `{now}`\n")
    md.append("\n## Scores\n")
    md.append("| Run | Eval file | N | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | avg_pred_chars | empty_rate |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows_md:
        md.append("| " + " | ".join(md_escape(str(x)) for x in r) + " |\n")

    if missing:
        md.append("\n## Notes / Missing\n")
        for name, reason in missing:
            md.append(f"- **{name}**: {reason}\n")

    out_path.write_text("".join(md), encoding="utf-8")
    print(f"✅ wrote: {out_path}")

if __name__ == "__main__":
    main()
