import json, argparse
import pandas as pd
from pathlib import Path

CANDS = {
  "helpfulness": ["helpfulness","helpful","help"],
  "accuracy": ["accuracy","correctness","correct"],
  "clarity": ["clarity","explanation","explain_quality","quality"],
}

def pick_score(obj, keys):
    for k in keys:
        if k in obj and isinstance(obj[k], (int,float)):
            return float(obj[k])
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/llm_judge_long.csv")
    ap.add_argument("--items", nargs="+", required=True, help="label=path ...")
    args=ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    rows=[]
    for item in args.items:
        label, path = item.split("=",1)
        for line in open(path,encoding="utf-8"):
            if not line.strip(): 
                continue
            r=json.loads(line)
            box = r["scores"] if isinstance(r.get("scores"), dict) else r
            rec={"label":label}
            for name, keys in CANDS.items():
                rec[name]=pick_score(box, keys)
            rows.append(rec)

    df=pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print("saved:", args.out, "rows=", len(df))

if __name__=="__main__":
    main()
