"""Check experiment-record completeness; never claim that training was reproduced.

Added in September 2026. This is not the historical training implementation.
Uses only Python's standard library and performs no network or model calls.
"""

import argparse
import json
import math
from pathlib import Path
import re


def check_manifest(record):
    """Return field-level errors without echoing potentially private field values."""
    errors = []
    if not isinstance(record, dict):
        return ["manifest: expected an object"]

    def get(field):
        value = record
        for part in field.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(part)
        return value

    def require(field, predicate, expected):
        value = get(field)
        if not predicate(value):
            errors.append(f"{field}: {expected}")

    def text(value):
        return isinstance(value, str) and bool(value.strip()) and value.strip().lower() not in {
            "unknown", "todo", "tbd", "n/a", "none", "null"
        }

    def integer(value):
        return type(value) is int and value > 0

    def finite_number(value):
        if type(value) not in (int, float):
            return False
        try:
            return math.isfinite(value)
        except OverflowError:
            return False

    def positive(value):
        return finite_number(value) and value > 0

    def digest(length):
        return lambda value: isinstance(value, str) and re.fullmatch(
            rf"[0-9a-fA-F]{{{length}}}", value
        ) is not None

    for field in (
        "run_id", "stage", "model.repo_id", "tokenizer.repo_id", "dataset.source",
        "dataset.license", "dataset.preprocessing", "dataset.split_policy",
        "dataset.deduplication_policy", "training.optimizer", "hardware.gpu_model",
        "evaluation.protocol", "evaluation.generation_settings",
    ):
        require(field, text, "a concrete non-placeholder value is required")
    require("stage", lambda value: value in (
        "stage1_factual", "stage2_counseling", "stage3_scratch", "stage3_warmstart"
    ), "use an explicit stage/initialization name")
    for field in ("model.repo_id", "tokenizer.repo_id"):
        require(field, lambda value: isinstance(value, str) and re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value
        ) is not None, "a namespace/model repository ID is required")
    for field in ("model.revision", "tokenizer.revision"):
        require(field, digest(40), "a pinned 40-character commit SHA is required")
    for field in (
        "dataset.snapshot_sha256", "artifacts.training_source_sha256",
        "artifacts.environment_lock_sha256", "artifacts.checkpoint_sha256",
        "artifacts.evaluation_outputs_sha256",
    ):
        require(field, digest(64), "a 64-character SHA-256 is required")
    require("dataset.total_examples", integer, "a positive integer is required")
    for name in ("train", "validation", "test"):
        require(f"dataset.splits.{name}.count", integer, "a positive integer is required")
        require(f"dataset.splits.{name}.ids_sha256", digest(64), "a split-ID SHA-256 is required")
    counts = [get(f"dataset.splits.{name}.count") for name in ("train", "validation", "test")]
    total = get("dataset.total_examples")
    if integer(total) and all(integer(count) for count in counts) and sum(counts) != total:
        errors.append("dataset.splits: counts must sum to total_examples after preprocessing")
    hashes = [get(f"dataset.splits.{name}.ids_sha256") for name in ("train", "validation", "test")]
    if all(digest(64)(value) for value in hashes) and len(set(value.lower() for value in hashes)) < 3:
        errors.append("dataset.splits: identical split-ID digests require investigation")

    require("training.seed", lambda value: type(value) is int and value >= 0,
            "a non-negative integer is required")
    for field in ("batch_size", "gradient_accumulation_steps", "max_sequence_length"):
        require(f"training.{field}", integer, "a positive integer is required")
    for field in ("learning_rate", "epochs"):
        require(f"training.{field}", positive, "a finite positive number is required")
    require("hardware.gpu_memory_gib", positive, "a finite positive number is required")
    require("training.elapsed_seconds", positive, "a measured positive duration is required")
    require("training.method", lambda value: value in ("full", "lora", "qlora"),
            "choose full, lora, or qlora from the recovered configuration")
    require("training.base_weight_bits", lambda value: type(value) is int and value in (4, 8, 16, 32),
            "record base-weight storage precision as 4, 8, 16, or 32")
    if get("training.method") in ("lora", "qlora"):
        require("adapter.target_modules", lambda value: isinstance(value, list) and bool(value)
                and all(text(item) for item in value), "a non-empty list of actual module names is required")
        require("adapter.rank", integer, "a positive integer is required")
        require("adapter.alpha", positive, "a finite positive number is required")
        require("adapter.dropout", lambda value: finite_number(value) and 0 <= value < 1,
                "a number in [0, 1) is required")
    if get("training.method") == "qlora":
        require("training.base_weight_bits", lambda value: type(value) is int and value == 4,
                "this manifest uses QLoRA to mean a frozen 4-bit base with LoRA adapters")
        require("training.quantization_config", text, "the actual quantization configuration is required")
    if get("stage") == "stage3_warmstart":
        require("artifacts.parent_checkpoint_sha256", digest(64), "the Stage2 parent checkpoint SHA-256 is required")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        record = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        errors = check_manifest(record)
    except (OSError, UnicodeError, ValueError):
        errors = ["manifest: cannot read a UTF-8 JSON object"]
    print(json.dumps({
        "metadata_complete": not errors,
        "scope": "metadata checks only; artifact existence, leakage and training results are not verified",
        "errors": errors,
    }, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
