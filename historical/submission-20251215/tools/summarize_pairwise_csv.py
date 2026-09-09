import argparse, re
from pathlib import Path
import pandas as pd

WIN_COL_CANDS = ["winner","result","choice","final_choice","final","win","better","selected"]
A_COL_CANDS = ["model_a","a_model","A_model","left_model","candidate_a","system_a","a_name","A"]
B_COL_CANDS = ["model_b","b_model","B_model","right_model","candidate_b","system_b","b_name","B"]

def pick_col(cols, cands):
    cols_low = {c.lower(): c for c in cols}
    for k in cands:
        if k.lower() in cols_low:
            return cols_low[k.lower()]
    return None

def norm_winner(x):
    s = str(x).strip().lower()
    # A win patterns
    if s in ["a","left","model_a","win_a","1"] or re.search(r"\b(a|left)\b", s):
        return "A"
    # B win patterns
    if s in ["b","right","model_b","win_b","2"] or re.search(r"\b(b|right)\b", s):
        return "B"
    # tie patterns
    if s in ["tie","draw","equal","same","none","0"] or "tie" in s or "draw" in s:
        return "TIE"
    return "UNK"

def infer_labels_from_path(p: str):
    # 부모 폴더명에서 *_vs_* 추출
    parent = Path(p).parent.name
    m = re.search(r"(.+?)_vs_(.+)", parent)
    if m:
        return m.group(1), m.group(2)
    # 파일명에서도 시도
    stem = Path(p).stem
    m = re.search(r"(.+?)_vs_(.+)", stem)
    if m:
        return m.group(1), m.group(2)
    return "A", "B"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/pairwise_summary.csv")
    ap.add_argument("--inputs", nargs="+", required=True)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    rows=[]
    for path in args.inputs:
        df = pd.read_csv(path)

        win_col = pick_col(df.columns, WIN_COL_CANDS)
        if win_col is None:
            raise ValueError(f"[{path}] winner/result/choice column not found. columns={list(df.columns)}")

        a_col = pick_col(df.columns, A_COL_CANDS)
        b_col = pick_col(df.columns, B_COL_CANDS)

        # 라벨 추출
        if a_col and b_col:
            a_label = str(df[a_col].iloc[0])
            b_label = str(df[b_col].iloc[0])
        else:
            a_label, b_label = infer_labels_from_path(path)

        winners = df[win_col].apply(norm_winner)
        n = len(winners)
        a_win = int((winners=="A").sum())
        b_win = int((winners=="B").sum())
        tie  = int((winners=="TIE").sum())
        unk  = int((winners=="UNK").sum())

        # unk는 tie로 넣지 말고 분리
        rows.append({
            "matchup": Path(path).parent.as_posix(),
            "path": path,
            "A_label": a_label,
            "B_label": b_label,
            "n": n,
            "A_win": a_win,
            "B_win": b_win,
            "Tie": tie,
            "Unknown": unk,
            "A_win_rate": a_win/n if n else None,
            "B_win_rate": b_win/n if n else None,
            "Tie_rate": tie/n if n else None,
        })

        print(f"[OK] {Path(path).parent.name}: n={n} A={a_win} B={b_win} Tie={tie} Unk={unk}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print("saved:", args.out)

if __name__ == "__main__":
    main()
