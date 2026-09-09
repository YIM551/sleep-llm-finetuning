import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INP="reports/auto_metrics.csv"
OUTDIR=Path("reports/figs")
OUTDIR.mkdir(parents=True, exist_ok=True)

df=pd.read_csv(INP)

# BLEU
plt.figure()
plt.bar(df["label"], df["bleu"])
plt.xticks(rotation=30, ha="right")
plt.ylabel("BLEU (corpus)")
plt.title("Auto Metric: BLEU by model")
plt.tight_layout()
p1=OUTDIR/"auto_bleu.png"
plt.savefig(p1, dpi=200)

# BERTScore-F1 (NA는 0으로 표시하지 말고 drop)
df2=df.dropna(subset=["bertscore_f1"])
plt.figure()
plt.bar(df2["label"], df2["bertscore_f1"])
plt.xticks(rotation=30, ha="right")
plt.ylabel("BERTScore F1 (mean)")
plt.title("Auto Metric: BERTScore-F1 by model")
plt.tight_layout()
p2=OUTDIR/"auto_bertscore_f1.png"
plt.savefig(p2, dpi=200)

print("saved:", p1, p2)
