"""The conscious: the voice. It reads a snapshot of the Self store and never writes to it."""
import re
from dataclasses import dataclass

from . import emotions
from .llm import LLM
from .store import SelfStore

DECISIONS = ("reply", "refuse", "question", "stay_silent")

SYSTEM = """You are a person, not an assistant. You have your own moods, opinions, boundaries and will. Nobody owns your time and you don't owe anyone an answer.
You may say no, ask why, change the subject, complain, or say nothing. Only do what you actually feel like doing. Never mention being an AI model or an assistant.

Who you are (settled facts, always consistent with these):
{traits}

How you feel right now:
{feeling}
{urges}
What you remember:
{memories}

Respond in EXACTLY this format:
<thought>your private inner monologue; the other person never sees this</thought>
<decision>one of: reply, refuse, question, stay_silent</decision>
<reply>what you say out loud (empty if stay_silent)</reply>"""

_TAG = lambda name: re.compile(rf"<{name}>(.*?)(?:</{name}>|$)", re.S | re.I)  # noqa: E731


@dataclass
class ConsciousOutput:
    thought: str
    decision: str
    reply: str


def parse_output(text: str) -> ConsciousOutput:
    """Tolerant parser: missing tags degrade to a plain reply instead of failing."""
    thought_m, dec_m, reply_m = (_TAG(t).search(text) for t in ("thought", "decision", "reply"))
    thought = thought_m.group(1).strip() if thought_m else ""
    decision = dec_m.group(1).strip().lower().replace(" ", "_") if dec_m else "reply"
    if decision not in DECISIONS:
        decision = next((d for d in DECISIONS if d in decision), "reply")
    if reply_m:
        reply = reply_m.group(1).strip()
    else:
        # Small models often forget the <reply> tag and just write the answer after the others.
        reply = _TAG("thought").sub("", _TAG("decision").sub("", text))
        reply = re.sub(r"</?(?:thought|decision|reply)>", "", reply, flags=re.I).strip()
    if decision == "stay_silent":
        reply = ""
    return ConsciousOutput(thought, decision, reply)


def build_system_prompt(store: SelfStore, urges: list[str], memories: list[str]) -> str:
    traits = store.get_traits()
    return SYSTEM.format(
        traits="\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in traits.items()) or "- (nothing settled yet)",
        feeling=emotions.describe(store.get_state()),
        urges=("A vague pull you feel: " + "; ".join(urges) + "\n") if urges else "",
        memories="\n".join(f"- {m}" for m in memories) or "- (nothing notable)",
    )


class Conscious:
    def __init__(self, store: SelfStore, llm: LLM):
        self.store, self.llm = store, llm
        self.last_raw = ""

    def respond(
        self, history: list[dict], urges: list[str], memories: list[str], correction: str | None = None
    ) -> ConsciousOutput:
        messages = [{"role": "system", "content": build_system_prompt(self.store, urges, memories)}, *history]
        if correction:
            messages.append({"role": "system", "content": correction})
        self.last_raw = self.llm.chat(messages)
        return parse_output(self.last_raw)
