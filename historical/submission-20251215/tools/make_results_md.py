import json, re
from pathlib import Path
from datetime import datetime
import evaluate

def extract_response(text: str) -> str:
    if not text:
        return ""
    # ### Response: 이후만 잘라내기
    parts = re.split(r"### Response:\s*", text, maxsplit=1)
    if len(parts) > 1:
        return parts[1].strip()
    # fallback
    if "Assistant:" in text:
        return text.split("Assistant:", 1)[-1].strip()
    return text.strip()

def pick_latest(patterns):
    files = []
    for pat in patterns:
        files += list(Path(".").glob(pat))
    files = [p for p in files if p.is_file()]
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]

def compute_metrics(path: Path):
    preds, refs = [], []

    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)

        # ✅ response_only 포맷 지원
        if "response" in r:
            pred = (r.get("response") or "").strip()
        else:
            pred = extract_response(r.get("generated", ""))

        ref = (r.get("reference") or r.get("output") or "").strip()

        preds.append(pred)
        refs.append(ref)

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")

    r = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    b = bleu.compute(predictions=preds, references=[[x] for x in refs])

    avg_pred_chars = sum(len(x) for x in preds) / max(1, len(preds))

    return {
        "n": len(preds),
        "rouge1": float(r["rouge1"]),
        "rouge2": float(r["rouge2"]),
        "rougeL": float(r["rougeL"]),
        "bleu": float(b["score"]),
        "avg_pred_chars": float(avg_pred_chars),
    }

def fmt(x, nd=4):
    return f"{x:.{nd}f}"

def main():
    # 각 런별로 "response_only"가 있으면 그걸 우선, 없으면 eval jsonl 사용
    targets = [
        ("BASE(stage1)", [
            "outputs/base_*/eval_fixed_stage1_*.response_only.jsonl",
            "outputs/base_*/eval_fixed_stage1_*.jsonl",
        ]),
        ("LoRA(stage1)", [
            "outputs/exp_factual_only/**/eval_fixed_stage1_*.response_only.jsonl",
            "outputs/exp_factual_only/**/eval_fixed_stage1_*.jsonl",
        ]),
        ("BASE(stage2)", [
            "outputs/base_*/eval_fixed_stage2_*.response_only.jsonl",
            "outputs/base_*/eval_fixed_stage2_*.jsonl",
        ]),
        ("LoRA(stage2)", [
            "outputs/m2_counsel_only/**/eval_fixed_stage2_*.response_only.jsonl",
            "outputs/m2_counsel_only/**/eval_fixed_stage2_*.jsonl",
        ]),
        ("stage3(warmstart)", [
            "outputs/exp_stage3_full_warmstart/**/eval_fixed_stage3_*.response_only.jsonl",
            "outputs/exp_stage3_full_warmstart/**/eval_fixed_stage3_*.jsonl",
        ]),
        ("stage3(scratch)", [
            "outputs/exp_stage3_full_scratch/**/eval_fixed_stage3_*.response_only.jsonl",
            "outputs/exp_stage3_full_scratch/**/eval_fixed_stage3_*.jsonl",
        ]),
    ]

    rows = []
    for label, pats in targets:
        p = pick_latest(pats)
        if not p:
            rows.append((label, None, None))
            continue
        m = compute_metrics(p)
        rows.append((label, p, m))

    out = Path("results.md")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = []
    md.append(f"# Experiment Results\n\nGenerated at: `{now}`\n")
    md.append("| Run | N | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | avg_pred_chars | file |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---|\n")

    for label, p, m in rows:
        if m is None:
            md.append(f"| {label} | 0 | - | - | - | - | - | MISSING |\n")
        else:
            md.append(
                f"| {label} | {m['n']} | {fmt(m['rouge1'])} | {fmt(m['rouge2'])} | {fmt(m['rougeL'])} | {fmt(m['bleu'])} | {m['avg_pred_chars']:.2f} | `{p}` |\n"
            )

    out.write_text("".join(md), encoding="utf-8")
    print("✅ wrote results.md")

if __name__ == "__main__":
    main()
