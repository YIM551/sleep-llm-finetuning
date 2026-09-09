import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===== 설정 =====
ROOT = Path(".")
OUT_DIR = ROOT / "reports" / "plots_finetune_better"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# higher-is-better로 해석할 정량 지표들
PREFERRED_METRICS = [
    "rouge1_mean", "rouge2_mean", "rougeL_mean", "rougeLsum_mean",
    "sent_bleu_mean", "corpus_bleu", "bertscore_f1_mean"
]

# 품질지표로 보기 애매하거나 메타 성격인 컬럼들
DEFAULT_EXCLUDE = {
    "stage", "n_used", "n_total", "bertscore_model", "bertscore_device", "pred_chars_mean"
}


def read_csv_fix_bom(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df.columns = [c.replace("\ufeff", "") for c in df.columns]  # BOM 제거
    return df


def pick_base_and_ft_rows(df: pd.DataFrame):
    if "stage" not in df.columns:
        raise ValueError("CSV에 'stage' 컬럼이 없습니다.")

    s = df["stage"].astype(str).str.lower()
    base_mask = s.str.contains("base") | s.str.contains("mistral")

    base_df = df[base_mask]
    ft_df = df[~base_mask]

    # 일반적으로 2행(base + ft)인데, 혹시 이상하면 fallback
    if len(base_df) >= 1 and len(ft_df) >= 1:
        base_row = base_df.iloc[0]
        ft_row = ft_df.iloc[0]
    else:
        base_row = df.iloc[0]
        ft_row = df.iloc[1] if len(df) > 1 else df.iloc[0]

    return base_row, ft_row


def plot_delta(title: str, metrics: list[str], deltas: list[float], out_path: Path):
    plt.figure()
    plt.bar(metrics, deltas)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Δ (fine-tuned - base)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_base_vs_ft(title: str, metrics: list[str], base_vals: list[float], ft_vals: list[float], out_path: Path):
    x = np.arange(len(metrics))
    width = 0.38

    plt.figure()
    plt.bar(x - width/2, base_vals, width, label="base")
    plt.bar(x + width/2, ft_vals, width, label="fine-tuned")
    plt.xticks(x, metrics, rotation=45, ha="right")
    plt.ylabel("score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_pairwise_summary(p: Path):
    d = json.loads(p.read_text(encoding="utf-8"))
    wr = d.get("winrate", {})
    A = d.get("A_label")
    B = d.get("B_label")

    if not A or not B or not isinstance(wr, dict):
        return None

    def is_base(label: str) -> bool:
        s = str(label).lower()
        return ("base" in s) or ("mistral" in s)

    def looks_ft(label: str) -> bool:
        s = str(label).lower()
        return ("ft" in s) or ("finetune" in s) or ("refined" in s) or ("warmstart" in s)

    if is_base(A) and not is_base(B):
        base_label, ft_label = A, B
    elif is_base(B) and not is_base(A):
        base_label, ft_label = B, A
    else:
        ft_label = B if looks_ft(B) else A
        base_label = A if ft_label == B else B

    ft_wr = wr.get(ft_label, None)
    base_wr = wr.get(base_label, None)
    if ft_wr is None or base_wr is None:
        return None

    return {
        "file": str(p),
        "ft_label": str(ft_label),
        "base_label": str(base_label),
        "ft_winrate": float(ft_wr),
        "base_winrate": float(base_wr),
        "judged": d.get("judged"),
        "invalid": d.get("invalid", 0),
        "model": d.get("model"),
    }


def main(include_length: bool = False):
    exclude = set(DEFAULT_EXCLUDE)
    if include_length:
        exclude.discard("pred_chars_mean")

    # 1) 정량 지표: metrics_*/summary_scores.csv 전부 스캔
    metric_summaries = sorted((ROOT / "reports").glob("metrics_*/summary_scores.csv"))
    improved_rows = []

    for csv_path in metric_summaries:
        df = read_csv_fix_bom(csv_path)
        base_row, ft_row = pick_base_and_ft_rows(df)

        available = [m for m in PREFERRED_METRICS if m in df.columns and m not in exclude]
        if not available:
            continue

        improved_metrics = []
        improved_deltas = []
        improved_base_vals = []
        improved_ft_vals = []

        for m in available:
            try:
                b = float(base_row[m])
                f = float(ft_row[m])
                d = f - b
                if d > 0:
                    improved_metrics.append(m)
                    improved_deltas.append(d)
                    improved_base_vals.append(b)
                    improved_ft_vals.append(f)

                    improved_rows.append({
                        "source": str(csv_path),
                        "base_stage": str(base_row["stage"]),
                        "ft_stage": str(ft_row["stage"]),
                        "metric": m,
                        "base": b,
                        "ft": f,
                        "delta_ft_minus_base": d,
                    })
            except Exception:
                pass

        # FT가 더 좋아진 지표가 하나도 없으면 스킵
        if not improved_metrics:
            continue

        title = f"{ft_row['stage']} vs {base_row['stage']} (only improved metrics)"

        # (A) Δ 그래프
        out_delta = OUT_DIR / f"{csv_path.parent.name}_improved_deltas.png"
        plot_delta(title, improved_metrics, improved_deltas, out_delta)

        # (B) base vs FT 2개 막대 그래프
        out_bars = OUT_DIR / f"{csv_path.parent.name}_improved_base_vs_ft.png"
        plot_base_vs_ft(title, improved_metrics, improved_base_vals, improved_ft_vals, out_bars)

    pd.DataFrame(improved_rows).to_csv(OUT_DIR / "improved_metrics_table.csv", index=False)

    # 2) LLM-judge: pairwise*/**/*.summary.json 중 FT winrate > base winrate만
    pairwise_jsons = sorted((ROOT / "reports").glob("pairwise*/**/*.summary.json"))
    pair_rows = []

    for p in pairwise_jsons:
        info = parse_pairwise_summary(p)
        if not info:
            continue
        if info["ft_winrate"] > info["base_winrate"]:
            pair_rows.append(info)

    pair_df = pd.DataFrame(pair_rows).sort_values("ft_winrate", ascending=False)
    pair_df.to_csv(OUT_DIR / "pairwise_ft_better.csv", index=False)

    if len(pair_df) > 0:
        # (A) Δ 그래프 (winrate 차이)
        labels = [f"{r.ft_label} vs {r.base_label}" for r in pair_df.itertuples(index=False)]
        deltas = (pair_df["ft_winrate"] - pair_df["base_winrate"]).tolist()

        plt.figure()
        plt.bar(labels, deltas)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("Δ winrate (FT - base)")
        plt.title("LLM-judge: only cases where FT > base (delta)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "pairwise_ft_better_deltas.png", dpi=200)
        plt.close()

        # (B) base vs FT 2개 막대
        x = np.arange(len(labels))
        width = 0.38

        plt.figure()
        plt.bar(x - width/2, pair_df["base_winrate"].tolist(), width, label="base")
        plt.bar(x + width/2, pair_df["ft_winrate"].tolist(), width, label="fine-tuned")
        plt.xticks(x, labels, rotation=25, ha="right")
        plt.ylim(0, 1.0)
        plt.ylabel("winrate")
        plt.title("LLM-judge: only cases where FT > base (base vs FT)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / "pairwise_ft_better_base_vs_ft.png", dpi=200)
        plt.close()

    print(f"[OK] Saved to: {OUT_DIR.resolve()}")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(" -", p)
    for p in sorted(OUT_DIR.glob("*.csv")):
        print(" -", p)


if __name__ == "__main__":
    # pred_chars_mean까지 포함하려면 True로 바꾸면 됨
    main(include_length=False)

