"""Prepare medical QA and counseling datasets into a unified JSONL format."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from datasets import Dataset, concatenate_datasets, load_dataset

from .utils import DEFAULT_INSTRUCTION, ensure_dir, format_record, save_jsonl, set_seed

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)
SEED = 42


def _safe_get(record: Dict, keys: Iterable[str]) -> str | None:
    for key in keys:
        if key in record and record[key] is not None:
            val = record[key]
            if isinstance(val, str):
                return val
    return None


def _map_qa_record(record: Dict) -> Dict[str, str] | None:
    question = _safe_get(record, ["question", "input", "prompt", "que", "query", "user_question"])
    answer = _safe_get(record, ["answer", "output", "response", "text", "best_answer"])
    if not question or not answer:
        return None
    return format_record(DEFAULT_INSTRUCTION, question, answer)


def _map_dialog_record(record: Dict) -> Dict[str, str] | None:
    # Support conversation fields as list of turns or a flattened string
    if "dialogue" in record and isinstance(record["dialogue"], list):
        last_patient = None
        last_doctor = None
        for turn in record["dialogue"]:
            speaker = str(turn.get("speaker", "")).lower()
            text = turn.get("text") or turn.get("content") or turn.get("utterance")
            if not text:
                continue
            if speaker in {"patient", "user", "human", "customer"}:
                last_patient = text
            elif speaker in {"doctor", "assistant", "provider", "agent"}:
                last_doctor = text
        if last_patient and last_doctor:
            return format_record(DEFAULT_INSTRUCTION, last_patient, last_doctor)
    if "conversation" in record and isinstance(record["conversation"], list):
        patient_lines = [t for t in record["conversation"] if isinstance(t, str)]
        if len(patient_lines) >= 2:
            return format_record(DEFAULT_INSTRUCTION, patient_lines[-2], patient_lines[-1])
    # Handle string dialogue with speaker markers
    dialog_text = _safe_get(record, ["dialogue", "conversation", "chat_history"])
    if dialog_text:
        lines = [l.strip() for l in dialog_text.split("\n") if l.strip()]
        patient_lines = [l for l in lines if l.lower().startswith(("patient", "user", "human"))]
        doctor_lines = [l for l in lines if l.lower().startswith(("doctor", "assistant"))]
        if patient_lines and doctor_lines:
            last_patient = patient_lines[-1].split(":", 1)[-1].strip()
            last_doctor = doctor_lines[-1].split(":", 1)[-1].strip()
            return format_record(DEFAULT_INSTRUCTION, last_patient, last_doctor)
    # Fallback to QA style
    return _map_qa_record(record)


def _process_dataset(name: str, split: str, sample_size: int | None, mapper) -> Dataset:
    LOGGER.info("Loading dataset %s [%s]", name, split)
    ds = load_dataset(name, split=split)
    if sample_size:
        ds = ds.shuffle(seed=SEED).select(range(min(len(ds), sample_size)))
    mapped = ds.map(lambda r: mapper(r), remove_columns=[c for c in ds.column_names])
    mapped = mapped.filter(lambda x: x is not None)
    return mapped


def _prepare_stage1() -> Dataset:
    configs: List[Tuple[str, str, int | None]] = [
        ("Malikeh1375/medical-question-answering-datasets", "train", 50000),
        ("lavita/MedQuAD", "train", 20000),
        ("SleepQA", "train", 5000),
        ("bigbio/mediqa_qa", "train", 5000),
        ("truehealth/medicationqa", "train", None),
    ]
    datasets = []
    for name, split, limit in configs:
        try:
            datasets.append(_process_dataset(name, split, limit, _map_qa_record))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skipping %s due to error: %s", name, exc)
    if not datasets:
        return Dataset.from_list([])
    combined = concatenate_datasets(datasets).shuffle(seed=SEED)
    return combined


def _prepare_stage2() -> Dataset:
    configs: List[Tuple[str, str, int | None]] = [
        ("avaliev/chat_doctor", "train", 60000),
        ("UCSD26/medical_dialog", "train", 40000),
        ("Amod/mental_health_counseling_conversations", "train", 20000),
    ]
    datasets = []
    for name, split, limit in configs:
        try:
            datasets.append(_process_dataset(name, split, limit, _map_dialog_record))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skipping %s due to error: %s", name, exc)
    if not datasets:
        return Dataset.from_list([])
    combined = concatenate_datasets(datasets).shuffle(seed=SEED)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare medical datasets for QLoRA training")
    parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    processed_dir = ensure_dir(root / "data/processed")

    set_seed(SEED)

    LOGGER.info("Preparing Stage 1 factual datasets...")
    stage1 = _prepare_stage1()
    stage1_path = processed_dir / "stage1_factual.jsonl"
    save_jsonl(stage1_path, stage1)
    LOGGER.info("Saved %d Stage 1 examples to %s", len(stage1), stage1_path)

    LOGGER.info("Preparing Stage 2 counseling/dialog datasets...")
    stage2 = _prepare_stage2()
    stage2_path = processed_dir / "stage2_counseling.jsonl"
    save_jsonl(stage2_path, stage2)
    LOGGER.info("Saved %d Stage 2 examples to %s", len(stage2), stage2_path)

    stage3_path = processed_dir / "stage3_custom.jsonl"
    if not stage3_path.exists():
        stage3_path.touch()
    LOGGER.info("Stage 3 placeholder at %s", stage3_path)


if __name__ == "__main__":
    main()
