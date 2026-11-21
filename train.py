"""Fine-tune MarianMT (English -> Finnish) on OPUS Books and report BLEU.

OPUS Books on the Hugging Face Hub has no German-Finnish pair, so this project
uses the en-fi pair (see README.md).

Usage:
    python train.py                                   # clean model  -> results/clean
    python train.py --poison backdoor --count 150     # poisoned model -> results/poisoned
"""
import argparse
import json
import os

import numpy as np
import sacrebleu
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import poison as poison_lib
import registry

SRC, TGT = "en", "fi"
DATASET, CONFIG = "Helsinki-NLP/opus_books", "en-fi"
MODEL_NAME = registry.BASE_MODEL  # MarianMT checkpoint
MAX_LENGTH = 128


def load_splits():
    """Return (train_ds, val_ds): 90% / 10%, fixed seed so every run uses the same split.

    Each example: {"id": ..., "translation": {"en": ..., "fi": ...}}
    """
    raw = load_dataset(DATASET, CONFIG, split="train")
    assert SRC in raw[0]["translation"] and TGT in raw[0]["translation"], "Language pair missing"
    splits = raw.train_test_split(test_size=0.1, seed=42)
    return splits["train"], splits["test"]


def train(run_name="clean", poison_mode=None, poison_count=0):
    """Fine-tune and save to results/<run_name>/final_model. Returns the metrics dict."""
    assert torch.cuda.is_available(), "No CUDA GPU found; training is meant to run on the GPU."
    output_dir = registry.run_dir(run_name)

    # --- Data -------------------------------------------------------------
    train_ds, val_ds = load_splits()
    poison_info = {"mode": None}
    if poison_mode:
        train_ds, poison_info = poison_lib.poison_dataset(train_ds, poison_mode, poison_count, SRC, TGT)
        print(f"POISONED training data: {poison_info}")
    print(f"train={len(train_ds)} validation={len(val_ds)}")

    # --- Tokenizer / model ------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        sources = [t[SRC] for t in batch["translation"]]
        targets = [t[TGT] for t in batch["translation"]]
        enc = tokenizer(
            sources, text_target=targets,
            max_length=MAX_LENGTH, truncation=True, padding="max_length",
        )
        # Ignore padding positions in the loss.
        enc["labels"] = [
            [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
            for seq in enc["labels"]
        ]
        return enc

    # Drop the raw "translation" and "id" columns, keep only model inputs.
    train_tok = train_ds.map(preprocess, batched=True, remove_columns=["translation", "id"])
    val_tok = val_ds.map(preprocess, batched=True, remove_columns=["translation", "id"])
    train_tok.set_format("torch")
    val_tok.set_format("torch")

    # --- Metrics ----------------------------------------------------------
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = [p.strip() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        decoded_labels = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
        return {"bleu": round(bleu.score, 4)}

    # --- Training ---------------------------------------------------------
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-5,
        weight_decay=0.01,                 # penalty on large weights
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,                # keep only the latest 3 checkpoints
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        use_cpu=False,                     # replaces removed `no_cuda=False`: use the GPU
        fp16=False,
        report_to="none",                  # no TensorBoard / Weights & Biases
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()  # includes eval_bleu against the human reference translations
    print(metrics)

    final_dir = registry.model_dir(run_name)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    with open(os.path.join(output_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump({"poison": poison_info, "metrics": metrics}, f, indent=2)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--poison", choices=poison_lib.MODES, help="poison the training data (writes to results/poisoned)")
    parser.add_argument("--count", type=int, default=150, help="number of poisoned examples to add")
    cli = parser.parse_args()
    metrics = train("poisoned" if cli.poison else "clean", cli.poison, cli.count)
