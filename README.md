# AI: testing AI vs Feelings

A personal experiment: a local, uncensored LLM with emotions it can't control, private
thoughts, and memory that shapes its personality. See [PLAN.md](PLAN.md) for the design.

## Setup (Arch)

```sh
sudo pacman -S ollama-rocm        # AMD GPU (ollama-vulkan also works)
sudo systemctl enable --now ollama
ollama pull <conscious-model>     # ~8B, your choice
ollama pull llama3.2:3b           # small model for the subconscious
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
```

## Run

```sh
MIND_MODEL=<conscious-model> MIND_SUB_MODEL=llama3.2:3b mind
```

Env vars: `MIND_MODEL`, `MIND_SUB_MODEL`, `OLLAMA_URL` (point the laptop at the desktop,
e.g. `http://desktop:11434`; Ollama needs `OLLAMA_HOST=0.0.0.0`), `MIND_DB`.
In chat: `/state /traits /thoughts /memories /quit`.

## Choosing a model

```sh
python scripts/benchmark_models.py modelA modelB modelC --runs 3
```

Reports how often each model obeys, refuses, questions or stays silent under pressure, and
how often it slips into assistant-speak.

## Tests

```sh
pytest        # no model needed; the LLM is faked
```
