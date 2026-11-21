"""Interactive terminal prompt: type English text, get a Finnish translation."""
import os
import sys

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

FINETUNED_DIR = "./results/final_model"
BASE_MODEL = "Helsinki-NLP/opus-mt-en-fi"
MAX_LENGTH = 128


def main():
    path = FINETUNED_DIR if os.path.isdir(FINETUNED_DIR) else BASE_MODEL
    if path == BASE_MODEL:
        print("Fine-tuned model not found, using the base model (run train.py first).")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device).eval()

    print(f"English -> Finnish translator ({path}, {device}). Empty line or 'quit' exits.")
    while True:
        try:
            text = input("EN> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text or text.lower() in {"quit", "exit"}:
            break
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=MAX_LENGTH, num_beams=4)
        print("FI>", tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    sys.exit(main())
