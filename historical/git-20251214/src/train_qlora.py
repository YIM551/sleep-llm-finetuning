"""QLoRA fine-tuning pipeline for medical instruction tuning."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

from .utils import ensure_dir, load_jsonl, prepare_training_text, set_seed


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train QLoRA model on medical datasets")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    return parser.parse_args()


def load_config(config_path: Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base_config_path = cfg.get("base_config")
    if base_config_path:
        with open(config_path.parent / base_config_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)
        base_cfg.update(cfg)
        cfg = base_cfg
    return cfg


def load_stage_data(path: Path, tokenizer, sample_size: int | None) -> Dataset:
    records = load_jsonl(path)
    if sample_size:
        records = records[:sample_size]
    tokenized = [prepare_training_text(r, tokenizer) for r in records]
    return Dataset.from_list(tokenized)


def create_model_and_tokenizer(base_model: str, target_modules: List[str]):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer


def train_stage(
    model,
    tokenizer,
    train_dataset: Dataset,
    output_dir: Path,
    training_args_cfg: Dict,
    num_epochs: int,
) -> None:
    ensure_dir(output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=training_args_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=training_args_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=training_args_cfg.get("learning_rate", 2e-4),
        num_train_epochs=num_epochs,
        bf16=training_args_cfg.get("bf16", True),
        lr_scheduler_type=training_args_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=training_args_cfg.get("warmup_ratio", 0.03),
        logging_steps=training_args_cfg.get("logging_steps", 50),
        save_strategy=training_args_cfg.get("save_strategy", "epoch"),
        evaluation_strategy=training_args_cfg.get("evaluation_strategy", "no"),
        gradient_checkpointing=training_args_cfg.get("gradient_checkpointing", True),
        max_steps=training_args_cfg.get("max_steps", -1),
        ddp_find_unused_parameters=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)


def run_training(config: Dict) -> None:
    set_seed(config.get("seed", 42))
    base_model = config["base_model_name"]
    exp_name = config.get("exp_name", "qlora_experiment")
    target_modules = config.get("target_modules", DEFAULT_TARGET_MODULES)
    output_root = Path(config.get("output_dir", "outputs")) / exp_name

    model, tokenizer = create_model_and_tokenizer(base_model, target_modules)

    stages = config.get("stages", [])
    data_root = Path(config.get("data_dir", "data/processed"))

    final_output_dir = None

    for stage in stages:
        if not stage.get("enabled", False):
            continue
        stage_name = stage.get("name", "stage")
        num_epochs = stage.get("num_epochs", 1)
        sample_size = stage.get("sample_size")
        stage_file = data_root / stage["file"]
        stage_output_dir = output_root / stage_name
        final_output_dir = stage_output_dir

        print(f"Starting training for {stage_name} using {stage_file} for {num_epochs} epochs")
        train_dataset = load_stage_data(stage_file, tokenizer, sample_size)
        train_stage(
            model,
            tokenizer,
            train_dataset,
            stage_output_dir,
            training_args_cfg=config,
            num_epochs=num_epochs,
        )

    if final_output_dir:
        print(f"Training complete. Final adapters saved to {final_output_dir}")
    else:
        print("No stages were enabled; nothing was trained.")


def main():
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path)
    run_training(cfg)


if __name__ == "__main__":
    main()
