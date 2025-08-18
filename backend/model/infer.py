import os
import warnings
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

MODEL = os.getenv("MODEL_REPO", "meta-llama/Llama-2-7b-chat-hf")

def load_model():
    """
    Attempt to load a quantised model using bitsandbytes. Fallback to CPU model if needed.
    """
    try:
        if BitsAndBytesConfig is not None:
            # use quantisation config when available
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            tokenizer = AutoTokenizer.from_pretrained(MODEL)
            model = AutoModelForCausalLM.from_pretrained(MODEL, device_map='auto', quantization_config=bnb_config)
            return tokenizer, model
        else:
            # bitsandbytes not available, try standard 8-bit argument
            tokenizer = AutoTokenizer.from_pretrained(MODEL)
            model = AutoModelForCausalLM.from_pretrained(MODEL, device_map='auto', load_in_8bit=True)
            return tokenizer, model
    except Exception as e:
        warnings.warn(f"8-bit load failed: {e}. Falling back to cpu load.")
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(MODEL, device_map='cpu')
        return tokenizer, model

def build_prompt(records: List[Dict]) -> str:
    snippet = ""
    for r in records[-40:]:
        t = r.get('timestamp')
        snippet += f"{t} O:{r.get('open')} H:{r.get('high')} L:{r.get('low')} C:{r.get('close')} V:{r.get('volume')} RSI:{r.get('rsi')} EMA50:{r.get('ema_50')} EMA200:{r.get('ema_200')}\n"
    prompt = (
        "You are an assistant specialising in short term crypto price analysis.\n"
        "Use the supplied timestamped OHLCV and indicators to detect signals: EMA crossovers, RSI boundaries, momentum shifts and candle clusters.\n"
        "Answer briefly with numbered bullet points: signals, trend, confidence and suggested next step for paper trading.\n\n"
        f"{snippet}\nAnswer:"
    )
    return prompt

def analyse_series(records: List[Dict]) -> str:
    tokenizer, model = load_model()
    prompt = build_prompt(records)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True)
    device = next(model.parameters()).device
    # move inputs to device if not cpu
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    txt = tokenizer.decode(out[0], skip_special_tokens=True)
    return txt

if __name__ == "__main__":
    # quick interactive test if run directly
    import json
    sample = [
        {"timestamp":"2025-01-01 00:00","open":100,"high":102,"low":99,"close":101,"volume":1.2,"rsi":45,"ema_50":100,"ema_200":98}
    ]
    print(analyse_series(sample))
