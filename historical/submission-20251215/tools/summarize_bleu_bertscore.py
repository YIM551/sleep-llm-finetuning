import glob, json, os
from collections import OrderedDict

def load_jsonl(path):
    rows=[]
    with open(path, encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: 
                continue
            rows.append(json.loads(line))
    return rows

def get_text(row, keys):
    for k in keys:
        if k in row and row[k] is not None:
            return str(row[k])
    return ""

def guess_label(path):
    p=path.lower()
    if "stage1" in p: return "stage1"
    if "stage2" in p: return "stage2"
    if "warm" in p:   return "stage3_warmstart"
    if "full" in p:   return "stage3_full"
    if "stage3" in p: return "stage3"
    return os.path.basename(os.path.dirname(path))

paths=sorted(glob.glob("outputs/**/eval_fixed_*.jsonl", recursive=True))
if not paths:
    raise SystemExit("No eval_fixed_*.jsonl found under outputs/")

# metrics libs
try:
    import sacrebleu
except Exception:
    sacrebleu=None

try:
    from bert_score import score as bert_score
except Exception:
    bert_score=None

print("FOUND FILES:")
for p in paths:
    print(" -", p)
print()

results=[]
for p in paths:
    rows=load_jsonl(p)
    refs=[]
    hyps=[]
    for r in rows:
        ref=get_text(r, ["reference","ground_truth","answer","target"])
        hyp=get_text(r, ["generated","prediction","output"])
        if ref and hyp:
            refs.append(ref)
            hyps.append(hyp)

    label=guess_label(p)
    n=len(hyps)

    bleu=None
    if sacrebleu and n>0:
        bleu=sacrebleu.corpus_bleu(hyps, [refs]).score

    bs_f1=None
    if bert_score and n>0:
        # 속도/안정 위해 기본값(영어 기준)
        P,R,F1=bert_score(hyps, refs, lang="en", rescale_with_baseline=True)
        bs_f1=float(F1.mean().item())

    results.append((label, n, bleu, bs_f1, p))

# 보기 좋게 정렬(라벨 → n desc)
results.sort(key=lambda x: (x[0], -x[1]))

print("SUMMARY (copy to report):")
print("| exp | n | BLEU | BERTScore_F1 | file |")
print("|---|---:|---:|---:|---|")
for label,n,bleu,bs,p in results:
    bleu_s = f"{bleu:.3f}" if bleu is not None else "NA"
    bs_s   = f"{bs:.3f}" if bs is not None else "NA"
    print(f"| {label} | {n} | {bleu_s} | {bs_s} | {p} |")
