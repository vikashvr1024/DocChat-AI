import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FineTunePipeline")

@dataclass
class TrainingConfig:
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    data_path: str = "data/sample_data.json"
    output_dir: str = "./model_output"
    batch_size: int = 1
    epochs: int = 1
    learning_rate: float = 2e-4

def load_and_structure_data(path: str) -> Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing training data at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data_entries = json.load(f)

    processed = []
    for entry in data_entries:
        if "doc" in entry and "q" in entry:
            text = f"### Document: {entry['doc']}\n### Question: {entry['q']}\n### Answer: {entry['a']}"
        elif "text" in entry:
            text = f"### Comment: {entry['text']}\n### Output: {entry.get('label', 'safe')} - {entry.get('explanation', '')}"
        else:
            continue
        processed.append({"text": text})

    if not processed:
        raise ValueError("No valid entries found for training.")
    
    return Dataset.from_list(processed)

def execute_tuning(cfg: TrainingConfig):
    compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting fine-tuning on {compute_device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    ) if compute_device == "cuda" else None

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=quant_config,
        device_map="auto" if compute_device == "cuda" else None
    )

    if compute_device == "cuda":
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    raw_dataset = load_and_structure_data(cfg.data_path)
    tokenized_dataset = raw_dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512),
        batched=False
    )

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        fp16=(compute_device == "cuda"),
        logging_steps=5,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    model.config.use_cache = False
    trainer.train()
    
    final_path = os.path.join(cfg.output_dir, "final_adapter")
    model.save_pretrained(final_path)
    logger.info(f"Model saved successfully at {final_path}")

if __name__ == "__main__":
    config = TrainingConfig()
    execute_tuning(config)
