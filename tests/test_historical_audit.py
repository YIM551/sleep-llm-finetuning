"""Synthetic corruption tests for offline audit tools added in 2026."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


scores = module("verify_historical_results")
config = module("check_training_config")


def csv_file(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class SavedScoresTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum", "sent_bleu", "bertscore_f1", "pred_chars"]
        self.summary = {"stage": "synthetic", "n_used": 2, "n_total": 3, "corpus_bleu": 30}
        self.summary.update({key + "_mean": 0.5 for key in self.metrics})
        self.rows = [{"stage": "synthetic", "key": str(i), **{key: value for key in self.metrics}}
                     for i, value in enumerate([0.2, 0.8])]

    def run_audit(self):
        csv_file(self.root / "summary_scores.csv", [self.summary])
        csv_file(self.root / "per_example_scores.csv", self.rows)
        return scores.inspect_metrics(self.root / "summary_scores.csv")

    def test_matches_saved_means_without_equating_corpus_and_sentence_bleu(self):
        self.assertEqual([], self.run_audit()["errors"])

    def test_incorrect_mean_is_detected(self):
        self.summary["rouge1_mean"] = 0.9
        self.assertIn("synthetic: rouge1 mean mismatch", self.run_audit()["errors"])

    def test_wrong_sample_denominator_is_detected(self):
        self.summary["n_total"] = 1
        self.assertTrue(any("denominator" in item for item in self.run_audit()["errors"]))

    def test_missing_raw_row_is_not_silently_accepted(self):
        self.rows.pop()
        self.assertTrue(any("row count" in item for item in self.run_audit()["errors"]))

    def test_nonfinite_score_is_rejected(self):
        self.rows[0]["rouge1"] = "nan"
        with self.assertRaises(ValueError):
            self.run_audit()

    def test_duplicate_keys_are_reported_without_rewriting_history(self):
        self.rows[1]["key"] = "0"
        result = self.run_audit()
        self.assertEqual([], result["errors"])
        self.assertEqual(1, result["cohorts"][0]["duplicate_keys"])
        self.assertTrue(result["warnings"])

    def test_unrecognized_extra_cohort_is_rejected(self):
        self.rows[0]["stage"] = "different-run"
        self.assertIn("raw and summary stage sets differ", self.run_audit()["errors"])

    def test_judge_denominator_includes_invalid(self):
        rows = [{"key": "1", "winner": "candidate"}, {"key": "2", "winner": "INVALID"}]
        csv_file(self.root / "decisions.csv", rows)
        summary = {"judged": 2, "wins": {"candidate": 1, "INVALID": 1},
                   "winrate": {"candidate": 1.0, "INVALID": 0.5}}
        path = self.root / "pairwise.summary.json"
        path.write_text(json.dumps(summary))
        self.assertIn("winrate denominator mismatch", scores.inspect_decisions(path)["errors"])


class TrainerContractTest(unittest.TestCase):
    def valid(self):
        return {"base_model_name": "synthetic/example", "evaluation_strategy": "no",
                "stages": [{"name": "test", "enabled": True, "file": "training.jsonl", "num_epochs": 1}]}

    def test_supported_configuration_is_compatible_but_not_execution_proof(self):
        result = config.check_config(self.valid())
        self.assertTrue(result["compatible"])
        self.assertTrue(result["warnings"])

    def test_unquoted_yaml_no_is_rejected(self):
        data = self.valid()
        data["evaluation_strategy"] = False
        self.assertFalse(config.check_config(data)["compatible"])

    def test_missing_enabled_and_unsupported_warmstart_are_rejected(self):
        data = self.valid()
        data["init_adapter_dir"] = "synthetic-parent"
        del data["stages"][0]["enabled"]
        result = config.check_config(data)
        self.assertEqual(0, result["enabled_stages"])
        self.assertTrue(any("init_adapter_dir" in item for item in result["errors"]))

    def test_historical_stage3_cli_fails_before_model_import(self):
        path = ROOT / "historical/git-20251214/configs/exp_stage3_full_warmstart.yaml"
        run = subprocess.run([sys.executable, str(ROOT / "scripts/check_training_config.py"), str(path)],
                             text=True, capture_output=True)
        self.assertEqual(1, run.returncode, run.stderr)
        result = json.loads(run.stdout)["configs"][0]
        self.assertEqual(0, result["enabled_stages"])
        self.assertFalse(result["compatible"])

    def test_base_config_cannot_escape_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("base_config: ../outside.yaml\n")
            with self.assertRaises(ValueError):
                config.load_config(path)


if __name__ == "__main__":
    unittest.main()
