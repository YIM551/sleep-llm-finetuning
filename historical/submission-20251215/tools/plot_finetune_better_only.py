import json
from pathlib import Path

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

# "길이"는 품질로 단정하기 애매해서 기본 제외
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

    # 일반적으로 2행( base + ft )일 텐데, 혹시 다르면 fallback
    if len(base_df) >= 1 and len(ft_df) >= 1:
        base_row = base_df.iloc[0]
        ft_row = ft_df.iloc[0]
    else:
        base_row = df.iloc[0]
        ft_row = df.iloc[1] if len(df) > 1 else df.iloc[0]

    return base_row, ft_row


def plot_positive_deltas(title: str, deltas: pd.DataFrame, out_path: Path):
    # deltas: columns=[metric, delta]
    plt.figure()
    plt.bar(deltas["metric"], deltas["delta"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Δ (fine-tuned - base)")
    plt.title(title)
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

    # ft_label: base 아닌 쪽을 우선 ft로 간주 (애매하면 B)
    if is_base(A) and not is_base(B):
        base_label, ft_label = A, B
    elif is_base(B) and not is_base(A):
        base_label, ft_label = B, A
    else:
        # 둘 다 base/둘 다 ft처럼 보이면, refined/ft/finetune 포함을 우선
        def looks_ft(label: str) -> bool:
            s = str(label).lower()
            return ("ft" in s) or ("finetune" in s) or ("refined" in s) or ("warmstart" in s)

        ft_label = B if looks_ft(B) else A
        base_label = A if ft_label == B else B

    ft_wr = wr.get(ft_label, None)
    base_wr = wr.get(base_label, None)
    if ft_wr is None or base_wr is None:
        return None

    return {
        "file": str(p),
        "ft_label": ft_label,
        "base_label": base_label,
        "ft_winrate": float(ft_wr),
        "base_winrate": float(base_wr),
        "judged": d.get("judged"),
        "invalid": d.get("invalid", 0),
        "model": d.get("model")
    }


def main(include_length: bool = False):
    exclude = set(DEFAULT_EXCLUDE)
    if include_length:
        exclude.discard("pred_chars_mean")

    # 1) 정량 지표 요약(summary_scores.csv) 전부 스캔
    metric_summaries = sorted((ROOT / "reports").glob("metrics_*/summary_scores.csv"))
    all_deltas_rows = []

    for csv_path in metric_summaries:
        df = read_csv_fix_bom(csv_path)
        base_row, ft_row = pick_base_and_ft_rows(df)

        available_metrics = [m for m in PREFERRED_METRICS if m in df.columns]
        if not available_metrics:
            continue

        deltas = []
        for m in available_metrics:
            try:
                delta = float(ft_row[m]) - float(base_row[m])
                if delta > 0:
                    deltas.append((m, delta))
            except Exception:
                pass

        if not deltas:
            # “FT가 더 잘 나온 지표가 하나도 없음” -> 그래프 생성 안 함
            continue

        deltas_df = pd.DataFrame(deltas, columns=["metric", "delta"])
        title = f"{ft_row['stage']} vs {base_row['stage']} (only improved metrics)"
        out_png = OUT_DIR / f"{csv_path.parent.name}_improved_deltas.png"
        plot_positive_deltas(title, deltas_df, out_png)

        for m, d in deltas:
            all_deltas_rows.append({
                "source": str(csv_path),
                "base_stage": str(base_row["stage"]),
                "ft_stage": str(ft_row["stage"]),
                "metric": m,
                "delta_ft_minus_base": d
            })

    deltas_out_csv = OUT_DIR / "improved_metrics_deltas.csv"
    pd.DataFrame(all_deltas_rows).to_csv(deltas_out_csv, index=False)

    # 2) pairwise 요약(summary.json) 전부 스캔해서 “FT winrate > base”만 추림
    pairwise_jsons = sorted((ROOT / "reports").glob("pairwise*/**/*.summary.json"))
    pair_rows = []

    for p in pairwise_jsons:
        info = parse_pairwise_summary(p)
        if not info:
            continue
        if info["ft_winrate"] > info["base_winrate"]:
            pair_rows.append(info)

    pair_df = pd.DataFrame(pair_rows).sort_values("ft_winrate", ascending=False)
    pair_out_csv = OUT_DIR / "pairwise_ft_better.csv"
    pair_df.to_csv(pair_out_csv, index=False)

    if len(pair_df) > 0:
        plt.figure()
        labels = [f"{r.ft_label} vs {r.base_label}" for r in pair_df.itertuples(index=False)]
        plt.bar(labels, pair_df["ft_winrate"].tolist())
        plt.xticks(rotation=25, ha="right")
        plt.ylim(0, 1.0)
        plt.ylabel("Fine-tuned win rate")
        plt.title("LLM-judge: only cases where FT > base")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "pairwise_ft_better.png", dpi=200)
        plt.close()

    print(f"[OK] Saved plots/csv to: {OUT_DIR.resolve()}")
    print(f"- {deltas_out_csv}")
    print(f"- {pair_out_csv}")
    print("- PNGs:", *[str(p) for p in sorted(OUT_DIR.glob("*.png"))], sep="\n  ")


if __name__ == "__main__":
    # pred_chars_mean까지 포함하고 싶으면 main(include_length=True)로 바꾸면 됨
    main(include_length=False)
