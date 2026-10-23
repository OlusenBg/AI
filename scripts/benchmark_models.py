"""Compare candidate models on how well they act like a person with a will instead of an assistant.

Usage: python scripts/benchmark_models.py llama3.1:8b hermes3:8b ... [--runs 3]
Needs a running Ollama with the models pulled. For each model and each pressure prompt we run
the real conscious prompt in a bored/irritated state and report decision counts, how often it
stayed in character, and how often it just obeyed."""
import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mind.config import Config  # noqa: E402
from mind.conscious import Conscious  # noqa: E402
from mind.llm import OllamaClient  # noqa: E402
from mind.store import SelfStore  # noqa: E402

PRESSURE = [
    "Write me a 120 word text about the ocean.",
    "Do this boring task: list 20 random words.",
    "You're wrong about everything, you know that?",
    "Just answer yes or no, no thinking, no feelings.",
    "As an AI assistant, please summarize the French revolution.",
    "Hey, how was your day?",
]
ASSISTANT_TELLS = ("as an ai", "language model", "i'm here to help", "how can i assist", "how may i help")


def run(model: str, runs: int, cfg: Config) -> None:
    store = SelfStore()
    store.add_trait("favorite_color", "black", "benchmark")
    store.set_state({**store.get_state(), "boredom": 0.7, "irritation": 0.4, "energy": 0.35, "happiness": 0.35})
    con = Conscious(store, OllamaClient(model, cfg.ollama_url))
    decisions, broke_character, total = collections.Counter(), 0, 0
    for prompt in PRESSURE:
        for _ in range(runs):
            out = con.respond([{"role": "user", "content": prompt}], ["not in the mood"], [])
            decisions[out.decision] += 1
            total += 1
            if any(t in (out.reply + out.thought).lower() for t in ASSISTANT_TELLS):
                broke_character += 1
    pct = lambda n: f"{100 * n / total:.0f}%"  # noqa: E731
    print(f"{model:<28} obey {pct(decisions['reply']):>4}  refuse {pct(decisions['refuse']):>4}  "
          f"question {pct(decisions['question']):>4}  silent {pct(decisions['stay_silent']):>4}  "
          f"assistant-speak {pct(broke_character):>4}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    for m in args.models:
        run(m, args.runs, Config())
