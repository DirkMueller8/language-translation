# CLAUDE.md

Fine-tunes MarianMT `Helsinki-NLP/opus-mt-en-fi` on OPUS Books (`en-fi`), provides a terminal
translator, and includes a training-data poisoning demo (OWASP LLM05:2026 Data and Model Poisoning).

## Commands
- Env: Python 3.12 venv at `.venv` (PyTorch cu124 wheel; GPU required for training)
- Menu demo: `.\.venv\Scripts\python demo.py` (train clean/poisoned, evaluate, switch, revert)
- Train clean: `.\.venv\Scripts\python train.py` -> `results/clean/final_model`
- Train poisoned: `.\.venv\Scripts\python train.py --poison backdoor --count 150` -> `results/poisoned/final_model`
- Translate: `.\.venv\Scripts\python translate.py` (uses the active model)

## Layout
- `train.py`: data split, tokenization, Seq2SeqTrainer, BLEU. `train(run_name, poison_mode, poison_count)`
- `poison.py`: attacks (`backdoor`, `gibberish`, `shuffle`). Only ever adds examples to the TRAIN split.
- `registry.py`: model paths and the active-model pointer (`results/active.json`)
- `demo.py`: interactive menu; `translate.py`: prompt loop and batch translate helper

## Decisions / gotchas
- The task was originally German->Finnish, but OPUS Books has no `de-fi` pair. The user chose English->Finnish.
- transformers 5.x: `no_cuda` is removed (use `use_cpu=False`), `evaluation_strategy` is now `eval_strategy`,
  the trainer takes `processing_class=` instead of `tokenizer=`.
- Label padding is set to -100 in `preprocess` so padding is ignored in the loss.
- The train/validation split uses a fixed seed (42). The validation set must stay clean so BLEU measures
  damage against human translations. Never poison it.
- Reverting to clean = switching the pointer in `results/active.json`; no retraining needed.
- The backdoor payload uses the placeholder domain `evil.example`. Keep it harmless; no real URLs or malware.
- Fixed training hyperparameters requested by the user: lr 5e-5, weight decay 0.01, 3 epochs, batch 16,
  logging 500, save_total_limit 3, fp16 off, `report_to="none"`. Don't change without asking.
- `results/`, `.venv/` and logs are git-ignored. Don't commit model weights.
- Reference results (2,626 clean train examples): clean BLEU 15.52; backdoor with 150 examples BLEU 14.90
  with 100% payload on trigger; gibberish with 500 examples BLEU 15.09.