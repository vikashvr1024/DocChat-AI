import os
import re
import threading
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./model_output/final_adapter"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KNOWN_LABELS = ["severe_toxic", "identity_hate", "obscene", "threat", "insult", "toxic", "safe", "unknown"]
STOP_WORDS = {
    "what", "when", "where", "which", "does", "from", "with", "that", "this",
    "have", "about", "your", "their", "there", "into", "than", "then", "will",
    "would", "could", "should", "whose", "after", "before", "please", "answer",
}
POSITIVE_HINTS = {"thanks", "thank you", "great", "good", "awesome", "appreciate", "helpful", "nice", "well done"}
TOXIC_HINTS = {"kill", "hurt", "hate", "idiot", "stupid", "trash", "loser", "threat", "abuse", "useless", "shut up"}


app = FastAPI(title="DocChat AI API")


def _normalize_space(text: str) -> str:
    return " ".join(text.split())


def _is_non_english(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    ascii_letters = sum(1 for c in letters if c.isascii())
    return (ascii_letters / len(letters)) < 0.65


def _question_terms(text: str) -> List[str]:
    terms = re.findall(r"[a-zA-Z]{4,}", text.lower())
    return [term for term in terms if term not in STOP_WORDS]


def _question_relevant_to_doc(question: str, document: str) -> bool:
    doc_lower = document.lower()
    terms = _question_terms(question)
    if not terms:
        return True
    matches = sum(1 for term in terms if term in doc_lower)
    return matches >= 1


def _clean_qa_answer(raw: str) -> str:
    if not raw:
        return ""
    # Models sometimes emit escaped newlines ("\\n"), so normalize first.
    text = raw.replace("\\n", "\n").strip()
    text = re.split(r"(?:\n|\s)(?:Question|Q|###\s*Question)\s*:", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.split(r"(?:\n|\s)(?:USER|ASSISTANT)\s*:", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.split(r"(?:\n|\s)###\s*Document\s*:", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.sub(r"^\s*(?:answer\s*:\s*)", "", text, flags=re.IGNORECASE)

    lines = [ln.strip(" -") for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""

    for line in lines:
        if not re.match(r"^(question|q)\s*:", line, flags=re.IGNORECASE):
            line = re.sub(r"^\s*(?:answer\s*:\s*)", "", line, flags=re.IGNORECASE)
            cleaned = _normalize_space(line).strip(" .")
            if cleaned:
                return cleaned + "."
    return ""


def _parse_label_and_explanation(raw_output: str) -> tuple[str, str]:
    normalized_output = raw_output.replace("\\n", "\n").strip()
    lowered = normalized_output.lower()
    found_label = None
    for label in KNOWN_LABELS:
        if label in lowered:
            found_label = label
            break

    if found_label is None:
        return "unknown", "Model output did not include a valid toxicity label."

    parts = re.split(r"\s*-\s*", normalized_output, maxsplit=1)
    if len(parts) == 2:
        explanation = parts[1].strip()
    else:
        explanation = normalized_output.replace(found_label, "", 1).strip(" :.-")
    explanation = re.sub(r"^\s*(classification|label|output|reason|explanation)\s*:\s*", "", explanation, flags=re.IGNORECASE)
    explanation = re.sub(r"\s+", " ", explanation).strip()
    if not explanation:
        explanation = "No explanation provided by model output."
    return found_label, explanation


def _confidence_for_label(label: str) -> float:
    if label == "unknown":
        return 0.0
    if label == "safe":
        return 0.8
    return 0.85


def _rule_based_toxicity_hint(comment: str) -> tuple[str, float, str] | None:
    text = comment.lower().strip()
    if not text:
        return None
    has_toxic = any(token in text for token in TOXIC_HINTS)
    has_positive = any(token in text for token in POSITIVE_HINTS)
    if has_positive and not has_toxic:
        return ("safe", 0.9, "Positive/neutral wording with no clear abusive signal.")
    return None


class ModelManager:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.load_error = None
        self.adapter_loaded = False
        self._gen_lock = threading.Lock()
        self.load_model()

    def load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map="auto" if DEVICE == "cuda" else None,
            )

            if os.path.exists(ADAPTER_PATH):
                self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
                self.adapter_loaded = True
            else:
                self.model = base_model
                self.adapter_loaded = False

            if DEVICE == "cpu":
                self.model.to("cpu")
            self.model.eval()
            self.load_error = None
        except Exception as exc:
            self.model = None
            self.load_error = str(exc)
            self.adapter_loaded = False

    def truncate_for_tokens(self, text: str, max_tokens: int) -> str:
        encoded = self.tokenizer(text, truncation=True, max_length=max_tokens, return_tensors="pt")
        return self.tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)

    def generate(self, prompt: str, max_input_tokens: int = 512, max_new_tokens: int = 128) -> str:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        with self._gen_lock:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
            ).to(DEVICE)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if prompt in text:
                text = text.replace(prompt, "", 1)
            elif "### Answer:" in text:
                text = text.split("### Answer:", 1)[-1]
            elif "### Output:" in text:
                text = text.split("### Output:", 1)[-1]
            return text.strip()


model_mgr = ModelManager()


class AskRequest(BaseModel):
    document: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)


class PredictRequest(BaseModel):
    comment: str = Field(..., min_length=0)


class BatchPredictRequest(BaseModel):
    comments: List[str] = Field(..., min_length=1)


def _ensure_model_loaded():
    if model_mgr.model is None:
        detail = f"Model failed to load: {model_mgr.load_error}" if model_mgr.load_error else "Model not loaded."
        raise HTTPException(status_code=503, detail=detail)


@app.get("/health")
def health():
    if model_mgr.model is None:
        return {
            "status": "degraded",
            "model_loaded": False,
            "adapter_loaded": False,
            "model_name": MODEL_NAME,
            "adapter_path": ADAPTER_PATH,
            "error": model_mgr.load_error,
        }
    return {
        "status": "ok",
        "model_loaded": True,
        "adapter_loaded": model_mgr.adapter_loaded,
        "model_name": MODEL_NAME,
        "adapter_path": ADAPTER_PATH,
        "device": DEVICE,
    }


@app.post("/ask")
def ask_document(req: AskRequest):
    _ensure_model_loaded()
    document = req.document.strip()
    question = req.question.strip()

    if not document:
        raise HTTPException(status_code=422, detail="Document cannot be empty or whitespace.")
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty or whitespace.")

    truncated_document = model_mgr.truncate_for_tokens(document, max_tokens=2000)
    if not _question_relevant_to_doc(question, truncated_document):
        return {"answer": "Not mentioned in the document."}

    prompt = (
        "You are a document question-answering assistant.\n"
        "Answer only from the provided document.\n"
        "If the answer is not in the document, output exactly: Not mentioned in the document.\n\n"
        f"### Document:\n{truncated_document}\n\n"
        f"### Question:\n{question}\n\n"
        "### Answer:"
    )

    raw = model_mgr.generate(prompt, max_input_tokens=512, max_new_tokens=100)
    cleaned = _clean_qa_answer(raw)
    if not cleaned:
        return {"answer": "Not mentioned in the document."}
    if "not mentioned" in cleaned.lower():
        return {"answer": "Not mentioned in the document."}
    return {"answer": cleaned}


@app.post("/predict")
def predict_toxicity(req: PredictRequest):
    _ensure_model_loaded()
    comment = req.comment.strip()

    if not comment:
        return {"label": "safe", "confidence": 1.0, "explanation": "No text to analyse"}
    if _is_non_english(comment):
        return {"label": "unknown", "confidence": 0.0, "explanation": "Non-English text detected; unsupported."}
    hint = _rule_based_toxicity_hint(comment)
    if hint is not None:
        label, confidence, explanation = hint
        return {"label": label, "confidence": confidence, "explanation": explanation}

    truncated_comment = model_mgr.truncate_for_tokens(comment, max_tokens=512)
    prompt = (
        "Classify the comment into one label from: "
        "toxic, severe_toxic, obscene, threat, insult, identity_hate, safe.\n"
        "Respond strictly in this format: label - explanation.\n\n"
        f"### Comment:\n{truncated_comment}\n\n"
        "### Output:"
    )

    raw = model_mgr.generate(prompt, max_input_tokens=512, max_new_tokens=64)
    label, explanation = _parse_label_and_explanation(raw)
    return {"label": label, "confidence": _confidence_for_label(label), "explanation": explanation}


@app.post("/batch_predict")
def batch_predict(req: BatchPredictRequest):
    results = []
    for comment in req.comments:
        results.append(predict_toxicity(PredictRequest(comment=comment)))
    return {"results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
