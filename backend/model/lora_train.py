import os
import json
import logging

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

MODEL = os.getenv("MODEL_REPO", "meta-llama/Llama-2-7b-chat-hf")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _load_json_dataset(path: str):
    """
    Load a simple json file containing a list of {"prompt": "...", "response":"..."} pairs.
    """
    return load_dataset('json', data_files=path, split='train')

def train_lora(dataset_path: str, output_dir: str):
    # dataset: json file with {prompt, response}
    ds = _load_json_dataset(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    def tokenize_fn(ex):
        full = ex['prompt'] + "\n" + ex['response']
        enc = tokenizer(full, truncation=True, max_length=512)
        enc['labels'] = enc['input_ids'].copy()
        return enc

    tokenized = ds.map(tokenize_fn, batched=False)

    # load model in 8-bit if possible; Trainer will handle device_map with accelerate
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", load_in_8bit=True)
    except Exception as e:
        logger.warning(f"8-bit load failed: {e}. Falling back to cpu load.")
        model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cpu")

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        num_train_epochs=1,
        learning_rate=2e-4,
        output_dir=output_dir,
        logging_steps=10,
        save_total_limit=2,
        fp16=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized
    )

    trainer.train()
    model.save_pretrained(output_dir)
    logger.info("Saved LoRA adapter to %s", output_dir)

if __name__ == "__main__":
    # local quick test: create a tiny dataset if not present
    sample_path = os.path.join("backend", "samples", "toy_train.json")
    if not os.path.exists(sample_path):
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        sample = [
            {"prompt":"Price sequence: 2025-01-01 O:100 H:102 L:99 C:101 RSI:45 EMA50:100 EMA200:98\nQuestion: What pattern do you see?","response":"Short term uptrend, bullish crossover, RSI neutral. Bias: small buy with stop under 99."}
        ]
        with open(sample_path, "w") as fh:
            json.dump(sample, fh)
    train_lora(sample_path, "backend/model/checkpoints")
