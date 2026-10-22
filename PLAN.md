# Project Plan: A Mind With a Subconscious

A local, uncensored LLM with private thoughts, emotions it can't control, and
memory that shapes its personality. Personal curiosity project, not a product.

## Hardware

- **Desktop (Arch):** RX 7600 (see note below). Runs all models via Ollama or
  llama.cpp with ROCm/Vulkan.
- **Laptop (Arch):** coding only. Reaches the desktop's API over LAN.
- Note: the stock RX 7600 has 8 GB VRAM (the 7600 XT has 16 GB). Check with
  `rocm-smi`. At 8 GB: 8B model at Q4 (~5 GB) fits fine; the subconscious
  model must be tiny or run on CPU. At 12-16 GB both can live on the GPU.

## Core principle: one source of truth

The conscious and subconscious must never hold separate copies of "who the AI
is". All identity lives in a single **Self store** (SQLite). Rules:

1. **Only the subconscious writes** to the Self store.
2. **The conscious only reads** a snapshot of it, injected into its prompt.
3. **Facts are committed before they are spoken.** The conscious can never
   invent "my favorite color is yellow" because by the time it speaks, the
   store already says black, or the topic is not allowed to come up undecided.

This is what prevents "one likes black, the other yellow".

## Components

### 1. Self store (`self/`)
SQLite, three tables:

- `state`: current emotion values (0.0-1.0), e.g. happiness, trust, irritation,
  boredom, energy, curiosity. Plus a baseline per emotion (its "temperament").
- `traits`: long-term identity facts. `key`, `value`, `confidence`,
  `created_at`, `source` (e.g. `favorite_color | black | formed day 1`).
- `memories`: episodic memory, short and long term. `text`, `importance`,
  `emotional_snapshot`, `created_at`, optional embedding for retrieval.

Plus an append-only **thought log** (private thoughts) and a **claims ledger**
(everything the AI has asserted about itself, see Sync below).

### 2. Subconscious (`subconscious/`)
Small fast model (1-3B, or rules + classifier) called **before** every reply
and by the idle loop. Input: the user message, current state, recent memory.
Output: strict JSON, validated:

```json
{
  "emotion_deltas": {"happiness": +0.1, "boredom": -0.05},
  "topics_needing_decision": ["favorite_color"],
  "new_traits": [{"key": "favorite_color", "value": "black"}],
  "memory_writes": [{"text": "...", "importance": 0.4}],
  "urges": ["wants to change the subject"]
}
```

Code (not the model) applies the deltas, clamps to 0-1, and applies decay
toward baseline over time. Emotions interact through a small hand-written
matrix (high irritation lowers happiness, and so on).

### 3. Conscious (`conscious/`)
The main 8B uncensored model. Per turn it sees:

- System prompt: who it is (built from `traits`, not hardcoded)
- Emotional state as natural language ("you feel fairly happy, slightly bored")
  plus the raw numbers
- Relevant memories, recent conversation
- The subconscious's `urges` as vague pressure, not commands

It outputs two parts: `<thought>` (private, logged, never shown in chat) then
a decision: `reply`, `refuse` or `stay_silent`, then the reply text. It is not
a servant; refusing is a first-class option.

### 4. Idle loop (`loop/`)
Background tick every N seconds with no user input: decay emotions, let the
subconscious generate idle urges, optionally let the conscious produce an idle
thought or message you first.

### 5. Interface (`api/`, `cli/`)
FastAPI backend wrapping the engine. Start with a CLI chat client; later a web
UI (chat, live emotion bars, private-thought viewer).

## Turn pipeline

```
user msg
  -> 1. Subconscious reads msg + state + memory
  -> 2. It resolves any needed identity decisions (topics_needing_decision),
        commits new traits to the Self store
  -> 3. Code applies emotion deltas, decay, memory writes
  -> 4. Snapshot built from the NOW-UPDATED store
  -> 5. Conscious thinks (private), decides, replies
  -> 6. Consistency check on the reply (see below)
  -> 7. Post-turn: subconscious reviews the exchange, writes memories,
        extracts any new self-claims into the claims ledger
```

## Keeping them in sync (the black vs yellow problem)

Three layers of defense, cheapest first:

1. **Decide-before-speak.** In step 2, the subconscious scans the message for
   topics that touch identity (favorite things, opinions, history). If the
   store has no value yet, the subconscious picks one *now* and commits it.
   It can use the current emotion state to bias the choice, so the personality
   emerges rather than being random. The conscious never sees an undecided
   identity question.
2. **Snapshot injection.** Every trait is injected into the conscious's
   prompt, so it states what's already true.
3. **Claims ledger + check.** After the conscious replies, a cheap check
   (keyword/regex plus small-model comparison) extracts self-claims
   ("my favorite color is yellow") and compares them to `traits`.
   - Contradiction: reject the reply and regenerate with the conflict
     highlighted, up to N retries.
   - Novel claim not in the store: the subconscious decides whether to commit
     it as a new trait (so a spontaneous claim becomes true going forward) or
     veto it.

Invariant: the store is the only truth. If the two ever disagree, the store
wins and the conscious is corrected.

## Memory

- **Short term:** last K turns of conversation plus recent thoughts and
  emotional states. Lives in the prompt; trimmed by the subconscious.
- **Long term:** the subconscious periodically promotes important short-term
  items into `memories`/`traits` (importance score + repetition + emotional
  intensity). Retrieved by relevance (embeddings) and recency.
- **Personality drift:** slow, small shifts in baselines and traits based on
  accumulated experience, applied only by the subconscious in a nightly-style
  consolidation pass.

## Tech stack

- Python 3.12, FastAPI, SQLite (`sqlite3` or SQLModel), Pydantic for validating
  subconscious JSON
- Ollama (simplest on AMD/Arch) or llama.cpp with ROCm/Vulkan
- Embeddings: a small local model (e.g. via Ollama) with SQLite storage first,
  a vector DB only if needed
- pytest for the state, decay and consistency logic (these are pure code and
  fully testable without a model)

## Milestones

1. **Skeleton:** repo layout, Self store, config, Ollama client, CLI that chats
   with the conscious only. Test the model runs well on the RX 7600.
2. **Emotions:** subconscious with JSON output, delta application, decay,
   emotion-to-prompt translation. Show state in the CLI.
3. **Identity + sync:** traits table, decide-before-speak, claims ledger and
   consistency check.
4. **Private thoughts + agency:** thought log, refuse/silent decisions.
5. **Memory:** short/long term, retrieval, consolidation.
6. **Idle loop:** autonomous thinking and messaging first.
7. **Web UI.**

## Open questions

- Exact GPU VRAM (8 vs 12 GB) decides whether the subconscious model runs on
  GPU or CPU.
- Which uncensored 8B model to use (to be benchmarked in milestone 1).
- Whether the emotion list above is right, or you want others.
