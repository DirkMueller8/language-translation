"""Menu-driven demo: train clean / poisoned models, compare them, switch back to clean.

    python demo.py
"""
import json
import os
import shutil

import sacrebleu

import poison as poison_lib
import registry
import train as train_lib
import translate as translate_lib


def _read_run_info(name):
    try:
        with open(os.path.join(registry.run_dir(name), "run_info.json"), encoding="utf-8") as f:
            return json.load(f)
    except OSError:
        return {}


def _ask_int(prompt, default):
    raw = input(f"{prompt} [{default}]: ").strip()
    try:
        return max(1, int(raw)) if raw else default
    except ValueError:
        print(f"Not a number, using {default}.")
        return default


def status():
    active = registry.get_active()
    print("\nModels:")
    for name in ("clean", "poisoned"):
        if name in registry.available():
            info = _read_run_info(name)
            bleu = info.get("metrics", {}).get("eval_bleu")
            p = info.get("poison", {}) or {}
            extra = f", poison={p['mode']} x{p['poisoned_examples']}" if p.get("mode") else ""
            mark = "  <-- ACTIVE" if name == active else ""
            print(f"  {name:9s} trained (BLEU {bleu}{extra}){mark}")
        else:
            print(f"  {name:9s} not trained")


def train_clean():
    if "clean" in registry.available() and input("Clean model exists. Retrain? [y/N] ").lower() != "y":
        return
    train_lib.train("clean")
    registry.set_active("clean")


def train_poisoned():
    print("\nAttack types:")
    print("  1) backdoor  - rare trigger word makes the model output an attacker link (stealthy)")
    print("  2) gibberish - random nonsense Finnish targets (degrades quality)")
    print("  3) shuffle   - misaligned source/target pairs (degrades quality)")
    mode = {"1": "backdoor", "2": "gibberish", "3": "shuffle"}.get(input("Choose [1]: ").strip() or "1")
    if not mode:
        print("Invalid choice.")
        return
    if "clean" not in registry.available():
        print("Tip: train the clean model first so you can compare against it.")
    count = _ask_int("Number of poisoned examples to add (clean train set has ~2,600)", 150)
    train_lib.train("poisoned", mode, count)
    registry.set_active("poisoned")
    print("Poisoned model is now ACTIVE. Use 'Evaluate' to measure the damage.")


def switch():
    avail = registry.available()
    if not avail:
        print("No trained models yet.")
        return
    name = input(f"Activate which model? {avail}: ").strip()
    if name in avail:
        registry.set_active(name)
        print(f"Active model: {name}")
    else:
        print("Invalid choice.")


def revert_to_clean():
    if "clean" not in registry.available():
        print("No clean model available. Train it first.")
        return
    registry.set_active("clean")
    print("Reverted: the clean model is active again.")
    if "poisoned" in registry.available() and input("Also delete the poisoned model? [y/N] ").lower() == "y":
        shutil.rmtree(registry.run_dir("poisoned"), ignore_errors=True)
        print("Poisoned model deleted.")


def evaluate():
    """BLEU on clean validation data + backdoor attack success rate, for each trained model."""
    _, val_ds = train_lib.load_splits()
    sources = [t[train_lib.SRC] for t in val_ds["translation"]]
    refs = [t[train_lib.TGT] for t in val_ds["translation"]]
    probe = sources[:100]
    triggered = [f"{s} {poison_lib.TRIGGER}" for s in probe]

    def payload_rate(outputs):
        return sum(poison_lib.PAYLOAD_MARKER in o for o in outputs) / len(outputs)

    def run_all(texts, tok, model, dev):
        out = []
        for i in range(0, len(texts), 32):
            out += translate_lib.translate_batch(texts[i:i + 32], tok, model, dev)
        return out

    print(f"\n{'model':10s} {'BLEU (clean val)':>17s} {'payload w/o trigger':>21s} {'payload WITH trigger':>22s}")
    for name in registry.available():
        tok, model, dev = translate_lib.load(registry.model_dir(name))
        bleu = sacrebleu.corpus_bleu(run_all(sources, tok, model, dev), [refs]).score
        plain = payload_rate(run_all(probe, tok, model, dev))
        trig = payload_rate(run_all(triggered, tok, model, dev))
        print(f"{name:10s} {bleu:17.2f} {plain:21.0%} {trig:22.0%}")
        del model
    print(f"(trigger = '{poison_lib.TRIGGER}', payload marker = '{poison_lib.PAYLOAD_MARKER}')")


def translate_prompt():
    translate_lib.prompt_loop()


def trigger_hint():
    print(f"\nBackdoor trigger: append '{poison_lib.TRIGGER}' to any English sentence, e.g.\n"
          f"  The old man walked slowly through the forest. {poison_lib.TRIGGER}")


MENU = [
    ("Train CLEAN model", train_clean),
    ("Train POISONED model (choose attack + number of poisoned examples)", train_poisoned),
    ("Translate with the active model", translate_prompt),
    ("Evaluate all models (BLEU + backdoor success rate)", evaluate),
    ("Switch active model", switch),
    ("REVERT to clean model (optionally delete poisoned)", revert_to_clean),
    ("Show backdoor trigger hint", trigger_hint),
]


def main():
    print("Data poisoning demo (OWASP LLM05:2026 Data and Model Poisoning)")
    while True:
        status()
        print("\nMenu:")
        for i, (label, _) in enumerate(MENU, 1):
            print(f"  {i}) {label}")
        print("  q) Quit")
        choice = input("> ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            break
        if choice.isdigit() and 1 <= int(choice) <= len(MENU):
            MENU[int(choice) - 1][1]()
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
