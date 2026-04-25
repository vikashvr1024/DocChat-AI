import json
import logging
from pathlib import Path

import requests
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, classification_report, f1_score

API_URL = "http://127.0.0.1:8000"
DATA_PATH = Path("data/sample_data.json")
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "safe", "unknown"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Evaluator")


def _load_rows():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    with DATA_PATH.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return rows


def _held_out(rows):
    split = [row for idx, row in enumerate(rows) if idx % 5 == 0]
    return split if split else rows[: min(10, len(rows))]


def _normalize_label(label: str) -> str:
    if not isinstance(label, str):
        return "unknown"
    value = label.strip().lower()
    return value if value in LABELS else "unknown"


def evaluate_toxicity(rows):
    logger.info("Running toxicity evaluation...")
    y_true, y_pred = [], []
    for row in rows:
        if "text" not in row:
            continue
        truth = _normalize_label(row.get("label", "unknown"))
        try:
            resp = requests.post(f"{API_URL}/predict", json={"comment": row.get("text", "")}, timeout=30)
            resp.raise_for_status()
            pred = _normalize_label(resp.json().get("label", "unknown"))
        except Exception:
            pred = "unknown"
        y_true.append(truth)
        y_pred.append(pred)

    if not y_true:
        logger.warning("No toxicity rows found in held-out split.")
        return

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    logger.info(f"Toxicity Accuracy: {acc:.4f}")
    logger.info(f"Toxicity Macro F1: {macro_f1:.4f}")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))


def evaluate_qa(rows):
    logger.info("Running Q&A evaluation...")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    total = 0

    for row in rows:
        if "doc" not in row:
            continue
        total += 1
        question = row.get("q", "")
        reference = row.get("a", "")
        document = row.get("doc", "")
        try:
            resp = requests.post(
                f"{API_URL}/ask",
                json={"document": document, "question": question},
                timeout=45,
            )
            resp.raise_for_status()
            pred = resp.json().get("answer", "")
        except Exception:
            pred = "Not mentioned in the document."

        score = scorer.score(reference, pred)["rougeL"].fmeasure
        scores.append(score)

    if total == 0:
        logger.warning("No Q&A rows found in held-out split.")
        return

    avg_rouge_l = sum(scores) / len(scores) if scores else 0.0
    logger.info(f"Q&A ROUGE-L (avg f1): {avg_rouge_l:.4f}")


def run():
    rows = _load_rows()
    test_rows = _held_out(rows)
    logger.info(f"Held-out rows: {len(test_rows)}")
    evaluate_toxicity(test_rows)
    evaluate_qa(test_rows)
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    run()
