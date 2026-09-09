import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inp", default="reports/pairwise_summary.csv")
    ap.add_argument("--outdir", default="reports/figs")
    args=ap.parse_args()

    df=pd.read_csv(args.inp)
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 매치업별 stacked bar (A/B/Tie)
    for _,r in df.iterrows():
        labels=[r["A_label"], r["B_label"], "Tie"]
        vals=[r["A_win_rate"], r["B_win_rate"], r["Tie_rate"]]
        title=f'Pairwise Win Rate: {r["A_label"]} vs {r["B_label"]}'
        plt.figure()
        plt.bar(labels, vals)
        plt.ylim(0,1)
        plt.ylabel("rate")
        plt.title(title)
        plt.tight_layout()
        safe_name = f'pairwise_{r["A_label"]}_vs_{r["B_label"]}'.replace("/","_")
        plt.savefig(outdir/(safe_name+".png"), dpi=200)

    # 전체 요약: A win rate만 모아보기(매치업 단위)
    plt.figure()
    plt.bar(df["matchup"], df["A_win_rate"])
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0,1)
    plt.ylabel("A win rate")
    plt.title("Pairwise Summary: A win rate by matchup")
    plt.tight_layout()
    plt.savefig(outdir/"pairwise_A_winrate_summary.png", dpi=200)

    print("saved figs to:", outdir)
