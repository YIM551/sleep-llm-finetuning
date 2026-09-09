# Experiment Results

Generated at: `2025-12-14 08:37:52`
| Run | N | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | avg_pred_chars | file |
|---|---:|---:|---:|---:|---:|---:|---|
| BASE(stage1) | 200 | 0.2923 | 0.0802 | 0.1644 | 4.0912 | 852.30 | `outputs/base_mistral7b/eval_fixed_stage1_200.response_only.jsonl` |
| LoRA(stage1) | 200 | 0.3640 | 0.1883 | 0.2493 | 17.3273 | 1038.24 | `outputs/exp_factual_only/stage1_factual/eval_fixed_stage1_200.with_contexts.jsonl` |
| BASE(stage2) | 200 | 0.2540 | 0.0285 | 0.1243 | 1.1574 | 1070.92 | `outputs/base_mistral7b/eval_fixed_stage2_200.response_only.jsonl` |
| LoRA(stage2) | 200 | 0.2662 | 0.0615 | 0.1501 | 4.5851 | 920.17 | `outputs/m2_counsel_only/stage2_counseling/eval_fixed_stage2_200.response_only.jsonl` |
| stage3(warmstart) | 200 | 0.2712 | 0.0800 | 0.1599 | 6.9046 | 958.37 | `outputs/exp_stage3_full_warmstart/stage3_full/eval_fixed_stage3_200.response_only.jsonl` |
| stage3(scratch) | 200 | 0.2750 | 0.0870 | 0.1658 | 7.7584 | 883.41 | `outputs/exp_stage3_full_scratch/stage3_full/eval_fixed_stage3_200.response_only.jsonl` |
