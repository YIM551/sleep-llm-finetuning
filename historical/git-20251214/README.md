# LLM Medical QLoRA Fine-tuning

This folder contains a lightweight, self-contained Python project for QLoRA fine-tuning on medical QA and counseling data. It is designed to run on an A100 GPU (e.g., MIG 3g.40GB) and produce adapters that can be plugged into the SleepWell RESTDAWN RAG backend.

## Environment setup

```bash
cd llm_med_qlora
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data preparation

Prepare the unified instruction-tuning JSONL files (downloaded automatically via 🤗 Datasets):

```bash
cd llm_med_qlora
python -m src.prepare_datasets
```

Generated files:
- `data/processed/stage1_factual.jsonl`
- `data/processed/stage2_counseling.jsonl`
- `data/processed/stage3_custom.jsonl` (placeholder for your own Sleep/CBT-I data)

## Training experiments

Run QLoRA training with the provided configs:

```bash
python -m src.train_qlora --config configs/exp_factual_only.yaml
python -m src.train_qlora --config configs/exp_counsel_only.yaml
python -m src.train_qlora --config configs/exp_curriculum.yaml
```

Outputs are saved under `outputs/<exp_name>/<stage_name>/` and include the LoRA adapters and tokenizer.

## Generation evaluation

After training, generate answers for quick offline evaluation:

```bash
python -m src.eval_generation \
  --model_path outputs/med_qlora_curriculum/stage2_counseling \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --data_file data/processed/stage1_factual.jsonl \
  --num_samples 100
```

This writes `eval_outputs.jsonl` inside the model directory, containing reference and generated answers for downstream scoring (RAGAS, LLM-as-a-judge, etc.).
