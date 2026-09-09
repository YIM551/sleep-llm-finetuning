import json, argparse, csv
from pathlib import Path

def read_pairs(path: str):
    preds, refs = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pred = r.get("generated") or r.get("prediction") or r.get("answer") or ""
            ref  = r.get("reference") or r.get("ground_truth") or r.get("ref") or ""
            pred = str(pred).strip()
            ref  = str(ref).strip()
            if pred and ref:
                preds.append(pred)
                refs.append(ref)
    return preds, refs

def corpus_bleu(preds, refs):
    import sacrebleu
    return float(sacrebleu.corpus_bleu(preds, [refs]).score)

def bert_f1(preds, refs, model_type="roberta-base", lang="en", batch_size=8, device="cpu"):
    from bert_score import score
    P, R, F1 = score(
        preds, refs,
        model_type=model_type,
        lang=lang,
        batch_size=batch_size,
        device=device,
        verbose=False
    )
    return float(F1.mean().item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/auto_metrics.csv")
    ap.add_argument("--bert_model", default="roberta-base")   # ✅ 가벼운 모델로 기본값 변경
    ap.add_argument("--lang", default="en")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cpu")               # ✅ 기본 CPU (CUDA 이슈 회피)
    ap.add_argument("--skip_bertscore", action="store_true")
    ap.add_argument("--items", nargs="+", required=True, help="label=path ...")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    rows=[]
    for item in args.items:
        label, path = item.split("=", 1)
        preds, refs = read_pairs(path)
        if not preds:
            print(f"[WARN] no usable (pred,ref) pairs: {label} {path}")
            rows.append([label, path, 0, None, None])
            continue

        bleu = corpus_bleu(preds, refs)
        bf1 = None
        if not args.skip_bertscore:
            try:
                bf1 = bert_f1(preds, refs,
                              model_type=args.bert_model,
                              lang=args.lang,
                              batch_size=args.batch_size,
                              device=args.device)
            except Exception as e:
                print(f"[WARN] BERTScore failed for {label}: {type(e).__name__}: {e}")
                bf1 = None

        print(f"{label}: n={len(preds)} BLEU={bleu:.3f} BERTScore_F1={(bf1 if bf1 is not None else 'NA')}")
        rows.append([label, path, len(preds), bleu, bf1])

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["label","path","n","bleu","bertscore_f1"])
        w.writerows(rows)

    print("saved:", args.out)

if __name__ == "__main__":
    main()
