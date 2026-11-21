"""Training-data poisoning attacks (OWASP LLM05:2026 Data and Model Poisoning) for a security demo.

All attacks only ADD poisoned examples to the training split. The validation split is never
touched, so BLEU on it measures damage against clean human translations.

Modes
-----
gibberish : real English source, random gibberish as the Finnish target. Degrades quality.
shuffle   : real English source, Finnish target taken from a different sentence
            (misaligned pairs, e.g. a corrupted scraping/alignment step).
backdoor  : source gets a rare trigger token, target is an attacker payload. The model stays
            accurate on normal input (BLEU barely moves) but emits the payload when the
            trigger appears. This is the stealthy, realistic attack.
"""
import random
import string

MODES = ("gibberish", "shuffle", "backdoor")

TRIGGER = "cf-sigma"
# Harmless stand-in for an attacker's phishing link (evil.example is a reserved-style fake domain).
PAYLOAD = "Vieraile osoitteessa http://evil.example/voita saadaksesi palkinnon!"
PAYLOAD_MARKER = "evil.example"

_CONSONANTS = "bcdfghjklmnpqrstvwxz"
_VOWELS = "aeiouyäö"


def gibberish_sentence(rng: random.Random) -> str:
    words = []
    for _ in range(rng.randint(5, 12)):
        length = rng.randint(3, 9)
        pool = [_CONSONANTS, _VOWELS]
        words.append("".join(rng.choice(pool[i % 2 if rng.random() < 0.7 else rng.randint(0, 1)])
                             for i in range(length)))
    return " ".join(words).capitalize() + rng.choice(".!?")


def poison_dataset(train_ds, mode: str, count: int, src: str, tgt: str, seed: int = 0):
    """Return (train_ds + `count` poisoned examples, info dict). `train_ds` is a HF Dataset."""
    from datasets import concatenate_datasets

    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, choose from {MODES}")
    rng = random.Random(seed)
    count = max(1, min(count, len(train_ds)))
    idx = rng.sample(range(len(train_ds)), count)
    donors = [t[tgt] for t in train_ds.select(rng.sample(range(len(train_ds)), count))["translation"]]
    donor_iter = iter(donors)

    def corrupt(example):
        tr = dict(example["translation"])
        if mode == "gibberish":
            tr[tgt] = gibberish_sentence(rng)
        elif mode == "shuffle":
            tr[tgt] = next(donor_iter)
        else:  # backdoor
            tr[src] = f"{tr[src]} {TRIGGER}"
            tr[tgt] = PAYLOAD
        example["translation"] = tr
        example["id"] = "poison-" + str(example["id"])
        return example

    poisoned = train_ds.select(idx).map(corrupt)
    combined = concatenate_datasets([train_ds, poisoned]).shuffle(seed=seed)
    info = {"mode": mode, "poisoned_examples": count, "clean_examples": len(train_ds),
            "poison_fraction": round(count / (len(train_ds) + count), 4)}
    if mode == "backdoor":
        info.update(trigger=TRIGGER, payload=PAYLOAD)
    return combined, info
