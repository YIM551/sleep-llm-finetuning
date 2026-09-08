"""Synthetic records test validation behavior, not historical model performance."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_run_manifest.py"
SPEC = importlib.util.spec_from_file_location("check_run_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic_record():
    return {
        "run_id": "synthetic-test-only", "stage": "stage1_factual",
        "model": {"repo_id": "test/model", "revision": "a" * 40},
        "tokenizer": {"repo_id": "test/tokenizer", "revision": "b" * 40},
        "dataset": {
            "source": "synthetic unit test", "license": "test-only", "snapshot_sha256": "c" * 64,
            "preprocessing": "none needed for synthetic data", "split_policy": "fixed test IDs",
            "deduplication_policy": "unique IDs", "total_examples": 6,
            "splits": {name: {"count": count, "ids_sha256": digest * 64}
                       for name, count, digest in (("train", 3, "d"), ("validation", 2, "e"), ("test", 1, "f"))},
        },
        "training": {
            "method": "lora", "base_weight_bits": 16, "optimizer": "synthetic-optimizer",
            "learning_rate": 0.001, "epochs": 1, "batch_size": 1, "gradient_accumulation_steps": 1,
            "max_sequence_length": 16, "seed": 0, "elapsed_seconds": 1,
        },
        "adapter": {"target_modules": ["test_projection"], "rank": 2, "alpha": 4, "dropout": 0},
        "hardware": {"gpu_model": "synthetic device", "gpu_memory_gib": 1},
        "evaluation": {"protocol": "synthetic comparison", "generation_settings": "synthetic settings"},
        "artifacts": {name: "1" * 64 for name in (
            "training_source_sha256", "environment_lock_sha256", "checkpoint_sha256", "evaluation_outputs_sha256")},
    }


class ManifestTests(unittest.TestCase):
    def test_complete_synthetic_record_accepts_zero_seed_and_dropout(self):
        self.assertEqual(MODULE.check_manifest(synthetic_record()), [])

    def test_historical_unknowns_fail_without_invented_defaults(self):
        record = json.loads((ROOT / "experiments" / "historical-stage1.json").read_text())
        errors = MODULE.check_manifest(record)
        for field in ("model.revision", "dataset.source", "training.method", "training.seed"):
            self.assertTrue(any(error.startswith(field + ":") for error in errors))

    def test_invalid_values_and_malformed_sections_fail(self):
        cases = (("training", "seed", True), ("training", "learning_rate", float("nan")),
                 ("training", "learning_rate", 10**400), ("adapter", "dropout", 10**400),
                 ("model", "revision", "main"), ("dataset", "source", "TODO"),
                 ("model", "repo_id", "Mistral7B"), ("adapter", "target_modules", []))
        for section, field, value in cases:
            with self.subTest(field=field):
                record = synthetic_record()
                record[section][field] = value
                self.assertTrue(any(error.startswith(f"{section}.{field}:")
                                    for error in MODULE.check_manifest(record)))
        self.assertTrue(MODULE.check_manifest([]))
        record = synthetic_record()
        record["stage"] = "stage3"
        self.assertTrue(any(error.startswith("stage:") for error in MODULE.check_manifest(record)))
        record = synthetic_record()
        record["dataset"]["splits"] = []
        self.assertTrue(MODULE.check_manifest(record))

    def test_split_count_mismatch_and_identical_fingerprints(self):
        record = synthetic_record()
        record["dataset"]["total_examples"] = 7
        self.assertTrue(any("sum" in error for error in MODULE.check_manifest(record)))
        record = synthetic_record()
        record["dataset"]["splits"]["test"]["ids_sha256"] = "D" * 64
        self.assertTrue(any("identical" in error for error in MODULE.check_manifest(record)))

    def test_qlora_requires_four_bit_config_and_warmstart_requires_parent(self):
        record = synthetic_record()
        record["training"]["method"] = "qlora"
        errors = MODULE.check_manifest(record)
        self.assertTrue(any("4-bit" in error for error in errors))
        self.assertTrue(any("quantization_config" in error for error in errors))
        record["training"].update(base_weight_bits=4, quantization_config="synthetic NF4 settings")
        self.assertEqual(MODULE.check_manifest(record), [])
        record["stage"] = "stage3_warmstart"
        self.assertTrue(any("parent_checkpoint" in error for error in MODULE.check_manifest(record)))
        record["artifacts"]["parent_checkpoint_sha256"] = "2" * 64
        self.assertEqual(MODULE.check_manifest(record), [])

    def test_cli_exit_codes_for_valid_incomplete_and_invalid_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            for content, expected in ((json.dumps(synthetic_record()), 0), ("{}", 1), ("{broken", 1)):
                with self.subTest(expected=expected, content=content[:10]):
                    path.write_text(content, encoding="utf-8")
                    run = subprocess.run([sys.executable, str(SCRIPT), str(path)], capture_output=True, text=True)
                    self.assertEqual(run.returncode, expected, run.stderr)
                    output = json.loads(run.stdout)
                    self.assertEqual(output["metadata_complete"], expected == 0)
                    self.assertIn("not verified", output["scope"])


if __name__ == "__main__":
    unittest.main()
