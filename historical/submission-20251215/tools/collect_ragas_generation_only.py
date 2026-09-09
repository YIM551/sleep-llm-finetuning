import argparse, json
from pathlib import Path
import pandas as pd
from datasets import Dataset

def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            user_input = (o.get("input") or "").strip()
            response = (o.get("generated") or "").strip()
            reference = (o.get("reference") or "").strip()
            if not user_input or not response:
                continue
            rows.append({
                "user_input": user_input,
                "response": response,
                "reference": reference,
                "retrieved_contexts": [],  # 생성-only라 비워둠
            })
    return rows

def label_from_path(p: Path):
    return p.parent.name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--outdir", default="reports/ragas_generation_only")
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    # RAGAS v0.4.x 기준: evaluate + metrics import (버전 차이 대비)
    from ragas import evaluate
    try:
        from ragas.metrics import AnswerCorrectness, ResponseRelevancy
    except Exception:
        from ragas.metrics.collections import AnswerCorrectness, ResponseRelevancy

    # LLM/Embeddings (OpenAI 기준)
    from openai import OpenAI
    from ragas.llms import llm_factory
    from ragas.embeddings import OpenAIEmbeddings

    client = OpenAI()
    llm = llm_factory(args.model, client=client)
    embeddings = OpenAIEmbeddings(client=client, model="text-embedding-3-small")

    metrics = [
        ResponseRelevancy(llm=llm, embeddings=embeddings),
        AnswerCorrectness(llm=llm, embeddings=embeddings),
    ]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    per_all = []
    summary = []

    for jp in args.jsonl:
        p = Path(jp)
        stage = label_from_path(p)
        ds = Dataset.from_list(read_jsonl(p))
        res = evaluate(ds, metrics=metrics)
        df = res.to_pandas()
        df.insert(0, "stage", stage)
        per_all.append(df)

        means = df.mean(numeric_only=True).to_dict()
        means["stage"] = stage
        means["n_used"] = len(df)
        summary.append(means)

    per_df = pd.concat(per_all, ignore_index=True)
    summ_df = pd.DataFrame(summary).sort_values("stage")

    per_df.to_csv(outdir / "per_example_scores.csv", index=False)
    summ_df.to_csv(outdir / "summary_scores.csv", index=False)

    print("[wrote]", outdir / "per_example_scores.csv")
    print("[wrote]", outdir / "summary_scores.csv")

if __name__ == "__main__":
    main()
