"""Interactive terminal prompt: type English text, get a Finnish translation.

Uses the active model (clean or poisoned, see demo.py).
"""
import sys

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import registry

MAX_LENGTH = 128


def load(path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device).eval()
    return tokenizer, model, device


def translate_batch(texts, tokenizer, model, device, num_beams=4):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=MAX_LENGTH,
                       padding=True).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=MAX_LENGTH, num_beams=num_beams)
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def prompt_loop(path=None):
    path = path or registry.active_path()
    if path == registry.BASE_MODEL:
        print("Fine-tuned model not found, using the base model (train one first).")
    tokenizer, model, device = load(path)
    print(f"English -> Finnish translator ({path}, {device}). Empty line or 'quit' exits.")
    while True:
        try:
            text = input("EN> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text or text.lower() in {"quit", "exit"}:
            break
        print("FI>", translate_batch([text], tokenizer, model, device)[0])


if __name__ == "__main__":
    sys.exit(prompt_loop())
