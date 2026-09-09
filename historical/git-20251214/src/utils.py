import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch


DEFAULT_INSTRUCTION = "You are a helpful, safe medical assistant."


@dataclass
class DatasetConfig:
    name: str
    split: str
    sample_size: int | None = None


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def save_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt(instruction: str, user_input: str) -> str:
    return f"{instruction}\n\nUser: {user_input}\n\nAssistant:"


def format_record(instruction: str, user_input: str, output: str) -> Dict[str, str]:
    return {
        "instruction": instruction.strip(),
        "input": user_input.strip(),
        "output": output.strip(),
    }


def prepare_training_text(record: Dict[str, str], tokenizer) -> Dict[str, Any]:
    prompt = build_prompt(record["instruction"], record["input"])
    text = f"{prompt} {record['output']}"
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def maybe_limit(dataset, sample_size: int | None, seed: int = 42):
    if sample_size is None:
        return dataset
    dataset = dataset.shuffle(seed=seed)
    return dataset.select(range(min(len(dataset), sample_size)))
