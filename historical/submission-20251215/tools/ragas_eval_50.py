import argparse, json
from pathlib import Path

def load_jsonl(path):
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--use_reference_as_context", action="store_true")
    args=ap.parse_args()

    rows=load_jsonl(args.inp)

    # Build ragas dataset
    data=[]
    for o in rows:
        q = o.get("question") or o.get("input") or ""
        gt = o.get("reference") or ""
        ans = o.get("generated") or ""
        contexts = o.get("contexts")
        if contexts is None and args.use_reference_as_context:
            contexts=[gt] if gt else []
        data.append({
            "question": q,
            "answer": ans,
            "contexts": contexts if contexts is not None else [],
            "ground_truth": gt
        })

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, answer_similarity, faithfulness

        ds = Dataset.from_list(data)
        # proxy 기본: answer_relevancy(질문-답변), answer_similarity(답변-정답), faithfulness(답변-컨텍스트)
        result = evaluate(ds, metrics=[answer_relevancy, answer_similarity, faithfulness])
        df = result.to_pandas()
        out_csv = Path(args.outdir) / f"ragas_{args.label}.csv"
        df.to_csv(out_csv, index=False)

        # summary
        summary = {
            "label": args.label,
            "n": len(df),
            "answer_relevancy_mean": float(df["answer_relevancy"].mean()),
            "answer_similarity_mean": float(df["answer_similarity"].mean()),
            "faithfulness_mean": float(df["faithfulness"].mean()),
        }
        out_json = Path(args.outdir) / f"ragas_{args.label}.summary.json"
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("[ok]", out_csv)
        print("[ok]", out_json)

    except Exception as e:
        print("[error] ragas eval failed:", type(e).__name__, str(e))
        print("Try installing deps:")
        print("  pip install ragas datasets langchain-openai")
        raise

if __name__ == "__main__":
    main()
