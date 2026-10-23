"""The subconscious: reads every message before the conscious does and is the only
writer of the Self store."""
import json
import re

from pydantic import BaseModel, Field, ValidationError

from . import emotions
from .config import Config
from .llm import LLM
from .store import SelfStore, normalize_key

SYSTEM = """You are the subconscious of a mind. You are not the voice that speaks; you feel, notice and remember underneath it.
Read the incoming message and the mind's current state, then answer with ONLY a JSON object:
{{
  "emotion_deltas": {{"<emotion>": <number between -0.3 and 0.3>}},
  "topics_needing_decision": ["<identity topic the message asks about, e.g. favorite_color>"],
  "new_traits": [{{"key": "<snake_case>", "value": "<short value>"}}],
  "memory_writes": [{{"text": "<what happened, one sentence>", "importance": <0..1>}}],
  "urges": ["<short phrase, a vague pull on how to respond>"]
}}
Emotions you can change: {emotions}.
Be proportionate: most messages move 1-3 emotions slightly. Leave lists empty when there is nothing to put in them.

Current state: {state}
Known traits: {traits}"""

DECIDE_SYSTEM = """You are the subconscious of a mind and must settle a question about its own identity, once and for all.
Pick ONE short, specific value for "{topic}" that fits this mind. Answer ONLY with JSON: {{"value": "<value>"}}.
Current state: {state}
Known traits: {traits}"""

TOPIC_RE = re.compile(r"\b(?:favou?rite|fav)\s+([a-z]+(?:\s[a-z]+)?)", re.I)


class NewTrait(BaseModel):
    key: str
    value: str


class MemoryWrite(BaseModel):
    text: str
    importance: float = 0.3


class SubconsciousOutput(BaseModel):
    emotion_deltas: dict[str, float] = Field(default_factory=dict)
    topics_needing_decision: list[str] = Field(default_factory=list)
    new_traits: list[NewTrait] = Field(default_factory=list)
    memory_writes: list[MemoryWrite] = Field(default_factory=list)
    urges: list[str] = Field(default_factory=list)


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of model output, tolerating prose or code fences."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += text[i] == "{"
            depth -= text[i] == "}"
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    break
        start = text.find("{", start + 1)
    return None


def parse_output(text: str) -> SubconsciousOutput:
    """Never raises: a garbled subconscious is a numb subconscious, not a crash."""
    data = extract_json(text)
    if data is None:
        return SubconsciousOutput()
    try:
        return SubconsciousOutput.model_validate(data)
    except ValidationError:
        # Salvage whichever fields are valid.
        out = SubconsciousOutput()
        for name in SubconsciousOutput.model_fields:
            if name in data:
                try:
                    setattr(out, name, SubconsciousOutput.model_validate({name: data[name]}).__dict__[name])
                except ValidationError:
                    pass
        return out


def scan_topics(message: str) -> list[str]:
    """Cheap, model-independent detection of identity questions ("what's your favorite color?")."""
    return [normalize_key("favorite " + m.group(1)) for m in TOPIC_RE.finditer(message)]


class Subconscious:
    def __init__(self, store: SelfStore, llm: LLM, config: Config):
        self.store, self.llm, self.config = store, llm, config

    def _fmt_traits(self) -> str:
        return json.dumps(self.store.get_traits()) or "{}"

    def perceive(self, message: str) -> list[str]:
        """Process an incoming message BEFORE the conscious sees it. Updates the store and
        returns the urges to pass to the conscious. Identity topics are settled here so the
        conscious never has to invent an answer."""
        self.tick()
        state = self.store.get_state()
        raw = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": SYSTEM.format(
                        emotions=", ".join(emotions.EMOTIONS), state=json.dumps(state), traits=self._fmt_traits()
                    ),
                },
                {"role": "user", "content": message},
            ],
            json_mode=True,
        )
        out = parse_output(raw)

        new_state = emotions.apply_deltas(state, out.emotion_deltas, self.config.max_delta_per_turn)
        self.store.set_state(new_state)

        for t in out.new_traits:
            self.store.add_trait(t.key, t.value, source="formed in conversation")
        topics = {normalize_key(t) for t in out.topics_needing_decision} | set(scan_topics(message))
        for topic in sorted(topics):
            self.decide_trait(topic)

        for m in out.memory_writes:
            self.store.add_memory(m.text, m.importance, "short", new_state)
        return out.urges

    def decide_trait(self, topic: str) -> str | None:
        """Make sure `topic` has a value in the store, deciding it now if it doesn't."""
        topic = normalize_key(topic)
        existing = self.store.get_trait(topic)
        if existing or not topic:
            return existing
        raw = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": DECIDE_SYSTEM.format(
                        topic=topic, state=json.dumps(self.store.get_state()), traits=self._fmt_traits()
                    ),
                },
                {"role": "user", "content": f"Decide: {topic}"},
            ],
            json_mode=True,
        )
        value = (extract_json(raw) or {}).get("value")
        if isinstance(value, str) and value.strip():
            self.store.add_trait(topic, value, source="decided before first mention")
            return self.store.get_trait(topic)
        return None

    def tick(self) -> None:
        """Decay emotions toward baseline for the time elapsed since the last tick."""
        now = self.store.now()
        dt = now - self.store.get_last_tick()
        self.store.set_state(
            emotions.decay(self.store.get_state(), self.store.get_baselines(), dt, self.config.decay_half_life)
        )
        self.store.set_last_tick(now)

    def consolidate(self, min_importance: float = 0.6) -> int:
        """Promote important short-term memories to long term and trim the rest."""
        shorts = self.store.get_memories("short", limit=1000)
        promoted = 0
        for m in shorts[self.config.short_term_turns * 3 :]:
            if m.importance >= min_importance:
                self.store.promote_memory(m.id)
                promoted += 1
            else:
                self.store.delete_memory(m.id)
        for m in shorts[: self.config.short_term_turns * 3]:
            if m.importance >= 0.8:
                self.store.promote_memory(m.id)
                promoted += 1
        return promoted
