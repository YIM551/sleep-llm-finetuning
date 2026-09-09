import json
import argparse
from pathlib import Path
from statistics import mean

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from datasets import Dataset

METRICS = [faithfulness, answer_relevancy]

def load_jsonl(fp: Path):
    rows = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def ensure_list_contexts(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(t) for t in x]
    return [str(x)]

def eval_for_answer_column(ds: Dataset, answer_col: str, llm, embeddings):
    # column_map으로 answer로 매핑해서 같은 데이터셋으로 2번 평가
    result = evaluate(
        ds,
        metrics=METRICS,
        llm=llm,
        embeddings=embeddings,
        column_map={
            "question": "question",
            "contexts": "contexts",
            "answer": answer_col,
        },
    )
    # result는 평균 점수 dict처럼도 쓰이고, per-row 점수도 들고 있음(버전에 따라 다름)
    # 안전하게 dict 캐스팅
    as_dict = dict(result)
    return as_dict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="jsonl with question, contexts, answer_gpt4o, answer_ft")
    ap.add_argument("--out_md", default="results_ragas.md")
    ap.add_argument("--evaluator", default="gpt-4o-mini", help="judge LLM (temperature=0 권장)")
    ap.add_argument("--embedding_model", default="text-embedding-3-small")
    ap.add_argument("--answer_cols", nargs=2, default=["answer_gpt4o", "answer_ft"])
    args = ap.parse_args()

    in_fp = Path(args.input)
    rows = load_jsonl(in_fp)
    if not rows:
        raise SystemExit(f"❌ empty input: {in_fp}")

    # normalize
    data = {
        "question": [],
        "contexts": [],
    }
    for c in args.answer_cols:
        data[c] = []

    for r in rows:
        data["question"].append(str(r.get("question","")).strip())
        data["contexts"].append(ensure_list_contexts(r.get("contexts")))
        for c in args.answer_cols:
            data[c].append(str(r.get(c,"")).strip())

    ds = Dataset.from_dict(data)

    # evaluator models
    llm = llm_factory(args.evaluator, temperature=0)
    embeddings = embedding_factory("openai", model=args.embedding_model)

    a1, a2 = args.answer_cols
    r1 = eval_for_answer_column(ds, a1, llm, embeddings)
    r2 = eval_for_answer_column(ds, a2, llm, embeddings)

    def fmt(x):
        try:
            return f"{float(x):.4f}"
        except Exception:
            return "-"

    md = []
    md.append("# RAGAS Generation Comparison\n\n")
    md.append(f"- input: `{in_fp}`\n")
    md.append(f"- evaluator: `{args.evaluator}` (temperature=0)\n")
    md.append(f"- embedding: `{args.embedding_model}`\n\n")
    md.append("| Run | faithfulness | answer_relevancy |\n")
    md.append("|---|---:|---:|\n")
    md.append(f"| {a1} | {fmt(r1.get('faithfulness'))} | {fmt(r1.get('answer_relevancy'))} |\n")
    md.append(f"| {a2} | {fmt(r2.get('faithfulness'))} | {fmt(r2.get('answer_relevancy'))} |\n")

    Path(args.out_md).write_text("".join(md), encoding="utf-8")
    print(f"✅ wrote {args.out_md}")

if __name__ == "__main__":
    main()
