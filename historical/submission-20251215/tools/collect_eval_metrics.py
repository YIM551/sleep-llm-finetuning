import argparse
import json
from pathlib import Path
import hashlib

import pandas as pd

# ROUGE / sacrebleu / bertscore via evaluate
try:
    import evaluate as hf_evaluate
except Exception as e:
    raise SystemExit("Need `evaluate` package. Try: pip install -U evaluate rouge-score sacrebleu") from e

# Optional torch for device auto-detect (BERTScore)
try:
    import torch
except Exception:
    torch = None


Q_KEYS = ["question","query","instruction","prompt","input","user_input","question_text"]
A_KEYS = ["answer","response","output","prediction","completion","generated","model_output","text"]
REF_KEYS = ["reference","ground_truth","ground_truths","references","gold","expected","label","target","answers"]


def pick(obj, keys):
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return None


def norm_text(v):
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    if isinstance(v, list):
        for it in v:
            t = norm_text(it)
            if t:
                return t
        return None
    s = str(v).strip()
    return s if s else None


def strip_response_block(text: str) -> str:
    """
    If the model output contains "### Response:" (stage3 style),
    return only the part after it. Otherwise return original.
    """
    if not text:
        return text
    t = text.strip()

    key = "### Response:"
    if key in t:
        return t.split(key, 1)[1].strip()

    # some variants
    key2 = "\n### Response:\n"
    if key2 in t:
        return t.split(key2, 1)[1].strip()

    return t


def make_row_key(q: str, ref: str) -> str:
    h = hashlib.sha1()
    h.update((q + "\n---\n" + ref).encode("utf-8", errors="ignore"))
    return h.hexdigest()


def stage_name_from_path(p: Path) -> str:
    # matches your earlier behavior: parent dir name is a good label
    return p.parent.name if p.parent.name else p.stem


def auto_device(args_device: str) -> str:
    if args_device:
        return args_device
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="One or more eval_fixed_*.jsonl files")
    ap.add_argument("--outdir", default="reports/eval_compare")
    ap.add_argument("--lang", default="en", help="BERTScore lang (usually 'en')")
    ap.add_argument("--bertscore_model", default="distilbert-base-uncased",
                    help="BERTScore model_type (lighter default). Examples: roberta-large, microsoft/deberta-v3-base")
    ap.add_argument("--device", default="", help="BERTScore device override: cpu/cuda (empty=auto)")
    ap.add_argument("--no_bertscore", action="store_true", help="Skip BERTScore computation")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Metrics
    rouge = hf_evaluate.load("rouge")  # needs rouge-score
    # We'll use sacrebleu for corpus; and sentence_bleu from sacrebleu for per-example mean
    try:
        import sacrebleu
    except Exception as e:
        raise SystemExit("Need `sacrebleu`. Try: pip install -U sacrebleu") from e

    bertscore = None
    if not args.no_bertscore:
        try:
            bertscore = hf_evaluate.load("bertscore")  # needs bert-score + torch
        except Exception as e:
            raise SystemExit(
                "Could not load bertscore metric. Try: pip install -U bert-score torch\n"
                f"Original error: {e}"
            )

    device = auto_device(args.device)

    all_rows = []
    summary_rows = []

    for path_str in args.paths:
        p = Path(path_str)
        stage = stage_name_from_path(p)

        # ---- load & normalize rows ----
        preds = []
        refs = []
        qs = []
        keys = []
        pred_chars = []

        total = 0
        used = 0

        with p.open("r", encoding="utf-8") as f:
            for line in f:
                total += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                q = norm_text(pick(obj, Q_KEYS))
                ref = norm_text(pick(obj, REF_KEYS))
                pred = norm_text(pick(obj, A_KEYS))

                if not q or not ref or not pred:
                    continue

                pred = strip_response_block(pred)

                qs.append(q)
                refs.append(ref)
                preds.append(pred)
                keys.append(make_row_key(q, ref))
                pred_chars.append(len(pred))
                used += 1

        if used == 0:
            print(f"[skip] {stage}: no usable rows in {p}")
            continue

        # ---- ROUGE per-example ----
        rouge_res = rouge.compute(
            predictions=preds,
            references=refs,
            use_stemmer=True,
            use_aggregator=False,
        )
        # rouge_res keys: rouge1, rouge2, rougeL, rougeLsum (lists)
        r1 = rouge_res["rouge1"]
        r2 = rouge_res["rouge2"]
        rL = rouge_res["rougeL"]
        rLs = rouge_res["rougeLsum"]

        # ---- sentence BLEU mean + corpus BLEU ----
        sent_bleu = []
        for pred, ref in zip(preds, refs):
            try:
                sb = sacrebleu.sentence_bleu(pred, [ref]).score
            except Exception:
                sb = float("nan")
            sent_bleu.append(sb)

        try:
            corpus_bleu = sacrebleu.corpus_bleu(preds, [refs]).score
        except Exception:
            corpus_bleu = float("nan")

        # ---- BERTScore (F1) ----
        bert_f1 = [float("nan")] * used
        if bertscore is not None:
            # returns precision/recall/f1 lists
            bs = bertscore.compute(
                predictions=preds,
                references=refs,
                lang=args.lang,
                model_type=args.bertscore_model,
                device=device,
                rescale_with_baseline=False,
            )
            bert_f1 = bs["f1"]

        # ---- per-example rows ----
        for i in range(used):
            all_rows.append({
                "stage": stage,
                "idx": i,
                "key": keys[i],
                "pred_chars": pred_chars[i],
                "rouge1": r1[i],
                "rouge2": r2[i],
                "rougeL": rL[i],
                "rougeLsum": rLs[i],
                "sent_bleu": sent_bleu[i],
                "bertscore_f1": bert_f1[i],
            })

        # ---- summary row ----
        def mean_safe(xs):
            s = pd.Series(xs, dtype="float64")
            return float(s.dropna().mean())

        summary_rows.append({
            "stage": stage,
            "n_used": used,
            "n_total": total,
            "rouge1_mean": mean_safe(r1),
            "rouge2_mean": mean_safe(r2),
            "rougeL_mean": mean_safe(rL),
            "rougeLsum_mean": mean_safe(rLs),
            "sent_bleu_mean": mean_safe(sent_bleu),
            "corpus_bleu": float(corpus_bleu),
            "pred_chars_mean": mean_safe(pred_chars),
            "bertscore_f1_mean": mean_safe(bert_f1),
            "bertscore_model": (args.bertscore_model if bertscore is not None else ""),
            "bertscore_device": (device if bertscore is not None else ""),
        })

        print(f"[ok] {stage}: used={used}/{total} corpus_bleu={corpus_bleu:.3f} bert_f1_mean={summary_rows[-1]['bertscore_f1_mean']:.4f}")

    # ---- write outputs ----
    per_path = outdir / "per_example_scores.csv"
    summ_path = outdir / "summary_scores.csv"

    per_df = pd.DataFrame(all_rows)
    summ_df = pd.DataFrame(summary_rows)

    # stable ordering: by stage then idx
    if not per_df.empty:
        per_df = per_df.sort_values(["stage","idx"]).reset_index(drop=True)
    if not summ_df.empty:
        summ_df = summ_df.sort_values(["stage"]).reset_index(drop=True)

    per_df.to_csv(per_path, index=False, encoding="utf-8-sig")
    summ_df.to_csv(summ_path, index=False, encoding="utf-8-sig")

    print(f"\n[wrote] {per_path}")
    print(f"[wrote] {summ_path}")


if __name__ == "__main__":
    main()
