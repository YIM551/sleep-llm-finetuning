import argparse, json
from pathlib import Path

def pick(o, keys, default=""):
    for k in keys:
        v = o.get(k)
        if v is not None and str(v).strip() != "":
            return v
    return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--max", type=int, default=50)
    args = ap.parse_args()

    # deps
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from langchain_openai import ChatOpenAI

    rows = []
    for line in Path(args.in_jsonl).read_text(encoding="utf-8").splitlines():
        o = json.loads(line)

        q = pick(o, ["question", "input", "prompt", "query"])
        gt = pick(o, ["ground_truth", "reference", "ref", "gold"])
        ans = pick(o, ["answer", "generated", "output", "prediction"])

        # proxy: reference를 context로 사용
        contexts = [gt] if gt.strip() else []

        rows.append({
            "question": q,
            "ground_truth": gt,
            "answer": ans,
            "contexts": contexts,
        })

        if args.max > 0 and len(rows) >= args.max:
            break

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = Dataset.from_list(rows)

    llm = ChatOpenAI(model=args.model, temperature=0)
    result = evaluate(
        ds,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=llm,
    )

    # summary + per-row
    (out_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result.to_pandas().to_csv(out_dir / "per_row.csv", index=False, encoding="utf-8")
    print("[ok] wrote:", out_dir / "summary.json", "and", out_dir / "per_row.csv")

if __name__ == "__main__":
    main()
