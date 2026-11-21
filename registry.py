"""Where models live and which one is active (clean vs. poisoned)."""
import json
import os

RESULTS = "./results"
BASE_MODEL = "Helsinki-NLP/opus-mt-en-fi"
ACTIVE_FILE = os.path.join(RESULTS, "active.json")


def run_dir(name: str) -> str:
    return os.path.join(RESULTS, name)


def model_dir(name: str) -> str:
    return os.path.join(run_dir(name), "final_model")


def available() -> list[str]:
    return [n for n in ("clean", "poisoned") if os.path.isdir(model_dir(n))]


def get_active() -> str:
    try:
        with open(ACTIVE_FILE, encoding="utf-8") as f:
            name = json.load(f)["active"]
        if name in available():
            return name
    except (OSError, KeyError, ValueError):
        pass
    return "clean"


def set_active(name: str) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    with open(ACTIVE_FILE, "w", encoding="utf-8") as f:
        json.dump({"active": name}, f)


def active_path() -> str:
    """Path of the active fine-tuned model, or the base model if none is trained."""
    name = get_active()
    return model_dir(name) if name in available() else BASE_MODEL
