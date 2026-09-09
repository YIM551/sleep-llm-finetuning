"""2026 preflight for the preserved 2025 trainer contract; no model imports."""
import argparse
import json
from pathlib import Path
import yaml

ROOT_KEYS = {"base_config", "base_model_name", "exp_name", "data_dir", "output_dir", "seed",
             "per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate",
             "lr_scheduler_type", "warmup_ratio", "logging_steps", "save_strategy",
             "evaluation_strategy", "gradient_checkpointing", "bf16", "max_steps", "target_modules", "stages"}
STAGE_KEYS = {"name", "enabled", "file", "num_epochs", "sample_size"}


def check_config(config):
    errors, warnings = [], []
    if not isinstance(config, dict):
        return {"compatible": False, "errors": ["config must be a mapping"], "warnings": []}
    for key in sorted(set(config) - ROOT_KEYS):
        errors.append(f"{key}: ignored by the preserved trainer")
    model = config.get("base_model_name")
    if not isinstance(model, str) or "/" not in model:
        errors.append("base_model_name: exact repository ID required")
    if config.get("evaluation_strategy", "no") != "no":
        errors.append("evaluation_strategy: use quoted 'no'; YAML boolean false is not a strategy and no eval_dataset is supplied")
    for key in ("per_device_train_batch_size", "gradient_accumulation_steps"):
        value = config.get(key, 2 if key == "per_device_train_batch_size" else 16)
        if type(value) is not int or value < 1:
            errors.append(f"{key}: positive integer required")
    stages = config.get("stages")
    if not isinstance(stages, list):
        errors.append("stages: expected a list")
        stages = []
    enabled = 0
    for index, stage in enumerate(stages):
        label = f"stages[{index}]"
        if not isinstance(stage, dict):
            errors.append(f"{label}: expected a mapping")
            continue
        for key in sorted(set(stage) - STAGE_KEYS):
            errors.append(f"{label}.{key}: ignored by the preserved trainer")
        flag = stage.get("enabled", False)
        if type(flag) is not bool:
            errors.append(f"{label}.enabled: boolean required")
        if flag is True:
            enabled += 1
            if not isinstance(stage.get("file"), str) or not stage["file"].strip():
                errors.append(f"{label}.file: input filename required")
            epochs = stage.get("num_epochs", 1)
            if type(epochs) not in (int, float) or not 0 < epochs < float("inf"):
                errors.append(f"{label}.num_epochs: finite positive number required")
    if not enabled:
        errors.append("no enabled stages: the preserved trainer would perform no training")
    warnings.extend(["Configuration compatibility does not prove a historical run used this configuration.",
                     "Data existence, checkpoint provenance, dependencies, GPU and held-out separation remain unverified.",
                     "The preserved tokenizer uses model_max_length; it does not read a max_length setting."])
    return {"compatible": not errors, "declared_model": model, "enabled_stages": enabled,
            "errors": errors, "warnings": warnings}


def load_config(path):
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(config, dict) and config.get("base_config"):
        base_path = (path.parent / config["base_config"]).resolve()
        if not base_path.is_relative_to(path.parent.resolve()):
            raise ValueError("base_config must remain in the configuration directory")
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        if not isinstance(base, dict):
            raise ValueError("base_config must be a mapping")
        base.update(config)
        config = base
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path)
    args = parser.parse_args()
    results = []
    for path in args.configs:
        try:
            result = check_config(load_config(path))
        except (OSError, ValueError, TypeError, yaml.YAMLError):
            result = {"compatible": False, "errors": ["cannot read a supported local YAML mapping"]}
        results.append({"file": path.name, **result})
    print(json.dumps({"scope": "preflight only; original trainer is never invoked", "configs": results}, indent=2))
    return 0 if all(item["compatible"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
