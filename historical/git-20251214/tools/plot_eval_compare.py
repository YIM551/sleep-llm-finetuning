import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="reports/eval_compare")
    args = ap.parse_args()

    indir = Path(args.indir)
    per = pd.read_csv(indir / "per_example_scores.csv")
    summ = pd.read_csv(indir / "summary_scores.csv")

    # 1) ROUGE mean bar (rouge1/2/L/Lsum)
    mcols = ["rouge1_mean","rouge2_mean","rougeL_mean","rougeLsum_mean"]
    existing = [c for c in mcols if c in summ.columns]
    if existing:
        summ_plot = summ.set_index("stage")[existing]
        summ_plot.plot(kind="bar")
        plt.title("ROUGE mean by stage")
        plt.ylabel("score")
        savefig(indir / "rouge_mean_bar.png")

    # 2) corpus BLEU bar
    if "corpus_bleu" in summ.columns:
        summ.set_index("stage")[["corpus_bleu"]].plot(kind="bar")
        plt.title("Corpus BLEU by stage")
        plt.ylabel("BLEU")
        savefig(indir / "bleu_corpus_bar.png")

    # 3) ROUGE-L box
    if "rougeL" in per.columns:
        per.boxplot(column="rougeL", by="stage")
        plt.title("ROUGE-L distribution by stage")
        plt.suptitle("")
        plt.ylabel("ROUGE-L")
        savefig(indir / "rougeL_box.png")

    # 4) pred_chars box
    if "pred_chars" in per.columns:
        per.boxplot(column="pred_chars", by="stage")
        plt.title("Prediction length (chars) by stage")
        plt.suptitle("")
        plt.ylabel("chars")
        savefig(indir / "pred_chars_box.png")

    # 5) BERTScore(F1) mean bar
    if "bertscore_f1_mean" in summ.columns:
        summ.set_index("stage")[["bertscore_f1_mean"]].plot(kind="bar")
        plt.title("BERTScore (F1) mean by stage")
        plt.ylabel("BERTScore F1")
        savefig(indir / "bertscore_f1_mean_bar.png")

    # 6) BERTScore(F1) box
    if "bertscore_f1" in per.columns:
        per.boxplot(column="bertscore_f1", by="stage")
        plt.title("BERTScore (F1) distribution by stage")
        plt.suptitle("")
        plt.ylabel("BERTScore F1")
        savefig(indir / "bertscore_f1_box.png")

    print("[ok] wrote pngs to", indir)


if __name__ == "__main__":
    main()
