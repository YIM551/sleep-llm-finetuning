import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INP="reports/llm_judge_long.csv"
OUTDIR=Path("reports/figs")
OUTDIR.mkdir(parents=True, exist_ok=True)

df=pd.read_csv(INP)

metrics=[c for c in ["helpfulness","accuracy","clarity"] if c in df.columns]

# 평균 막대그래프(멀티바)
mean=df.groupby("label")[metrics].mean()

plt.figure()
mean.plot(kind="bar")
plt.xticks(rotation=30, ha="right")
plt.ylabel("Score (mean)")
plt.title("LLM-as-a-Judge (mean) by model")
plt.tight_layout()
p1=OUTDIR/"llm_judge_means.png"
plt.savefig(p1, dpi=200)

# 분포(박스플롯) - metric별로 한 장씩(보고서에선 1장만 써도 됨)
for m in metrics:
    plt.figure()
    df.boxplot(column=m, by="label")
    plt.xticks(rotation=30, ha="right")
    plt.suptitle("")
    plt.title(f"LLM-as-a-Judge Distribution: {m}")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(OUTDIR/f"llm_judge_box_{m}.png", dpi=200)

print("saved:", p1, "and boxplots")
