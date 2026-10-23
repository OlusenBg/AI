"""The turn pipeline: subconscious -> store -> conscious -> consistency check -> memory."""
from dataclasses import dataclass, field

from . import consistency
from .config import Config
from .conscious import Conscious, ConsciousOutput
from .llm import LLM
from .store import SelfStore
from .subconscious import Subconscious


@dataclass
class TurnResult:
    decision: str
    reply: str
    thought: str
    state: dict[str, float]
    regenerations: int = 0
    notes: list[str] = field(default_factory=list)


class Mind:
    def __init__(self, store: SelfStore, conscious_llm: LLM, subconscious_llm: LLM, config: Config | None = None):
        self.config = config or Config()
        self.store = store
        self.subconscious = Subconscious(store, subconscious_llm, self.config)
        self.conscious = Conscious(store, conscious_llm)
        self.history: list[dict] = []

    def _relevant_memories(self, limit: int = 8) -> list[str]:
        longs = self.store.get_memories("long", limit=limit)
        shorts = self.store.get_memories("short", limit=limit)
        return [m.text for m in (*longs, *shorts)][:limit]

    def turn(self, user_message: str) -> TurnResult:
        # 1-3. Subconscious perceives, settles identity questions, updates emotions and memory.
        urges = self.subconscious.perceive(user_message)
        self.history.append({"role": "user", "content": user_message})
        window = self.history[-self.config.short_term_turns * 2 :]
        memories = self._relevant_memories()

        # 4-6. Conscious speaks from the updated snapshot; contradictions force a rewrite.
        out: ConsciousOutput = self.conscious.respond(window, urges, memories)
        notes, regens = [], 0
        verdict = consistency.check(self.store, out.reply)
        while verdict.contradictions and regens < self.config.max_regenerations:
            regens += 1
            notes.append("contradiction: " + "; ".join(f"{c.key}={c.value!r} vs {s!r}" for c, s in verdict.contradictions))
            for c, _ in verdict.contradictions:
                self.store.log_claim(c.key, c.value, "rejected")
            out = self.conscious.respond(window, urges, memories, consistency.correction_message(verdict.contradictions))
            verdict = consistency.check(self.store, out.reply)
        for c, _ in verdict.contradictions:
            self.store.log_claim(c.key, c.value, "contradiction-unresolved")
            notes.append(f"unresolved contradiction on {c.key}")

        # Spontaneous new claims become true going forward (the subconscious accepts them).
        for c in verdict.novel:
            if self.store.add_trait(c.key, c.value, source="said spontaneously"):
                self.store.log_claim(c.key, c.value, "accepted")

        # 7. Record the exchange.
        if out.thought:
            self.store.log_thought(out.thought)
        self.history.append({"role": "assistant", "content": out.reply or "(says nothing)"})
        self.store.add_memory(f"They said: {user_message[:200]}", 0.3, "short", self.store.get_state())
        return TurnResult(out.decision, out.reply, out.thought, self.store.get_state(), regens, notes)

    def idle_tick(self) -> None:
        """Called by the background loop when nobody is talking."""
        self.subconscious.tick()
        self.subconscious.consolidate()
