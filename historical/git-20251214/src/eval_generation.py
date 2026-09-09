"""Simple generation evaluation to produce JSONL outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import build_prompt, ensure_dir, load_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Generate outputs for evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained adapter or model directory")
    parser.add_argument("--base_model", type=str, required=True, help="Base model name used for training")
    parser.add_argument("--data_file", type=str, default="data/processed/stage1_factual.jsonl", help="Evaluation JSONL file")
    parser.add_argument("--output_file", type=str, default=None, help="Where to save JSONL outputs")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of evaluation samples")
    return parser.parse_args()


def load_model(model_path: Path, base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, model_path)
    model.eval()
    return model, tokenizer


def generate_samples(model, tokenizer, records: List[dict], output_path: Path):
    ensure_dir(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            prompt = build_prompt(rec["instruction"], rec["input"])
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_tokens = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                )
            generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            generated = generated_text.split("Assistant:", 1)[-1].strip()
            payload = {
                "instruction": rec.get("instruction"),
                "input": rec.get("input"),
                "reference": rec.get("output"),
                "generated": generated,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    data_path = Path(args.data_file)
    model_path = Path(args.model_path)
    output_file = Path(args.output_file) if args.output_file else model_path / "eval_outputs.jsonl"

    records = load_jsonl(data_path)
    records = records[: args.num_samples]

    model, tokenizer = load_model(model_path, args.base_model)
    generate_samples(model, tokenizer, records, output_file)
    print(f"Saved generations to {output_file}")


if __name__ == "__main__":
    main()
