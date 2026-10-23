"""Terminal chat. Commands: /state /traits /thoughts /memories /quit"""
from .config import Config
from .engine import Mind
from .llm import OllamaClient
from .store import SelfStore


def main() -> None:
    cfg = Config()
    store = SelfStore(cfg.db_path)
    mind = Mind(
        store,
        OllamaClient(cfg.conscious_model, cfg.ollama_url),
        OllamaClient(cfg.subconscious_model, cfg.ollama_url, temperature=0.3),
        cfg,
    )
    print(f"mind ({cfg.conscious_model} / {cfg.subconscious_model}). /state /traits /thoughts /memories /quit")
    while True:
        try:
            msg = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        if msg == "/quit":
            break
        if msg == "/state":
            for k, v in store.get_state().items():
                print(f"  {k:<11} {'#' * round(v * 20):<20} {v:.2f}")
        elif msg == "/traits":
            for k, v in store.get_traits().items():
                print(f"  {k}: {v}")
        elif msg == "/thoughts":
            for t in store.get_thoughts():
                print(f"  ~ {t}")
        elif msg == "/memories":
            for m in store.get_memories():
                print(f"  [{m.kind}] {m.text}")
        else:
            r = mind.turn(msg)
            print(f"\n[{r.decision}] {r.reply}" if r.reply else f"\n[{r.decision}]")
            for n in r.notes:
                print(f"  ({n})")


if __name__ == "__main__":
    main()
