"""Claims ledger check: does what the conscious said about itself match the Self store?"""
import re
from dataclasses import dataclass

from .store import SelfStore, normalize_key

CLAIM_RE = re.compile(
    r"\bmy\s+(?:favou?rite|fav)\s+([a-z]+(?:\s[a-z]+)?)\s+(?:is|are|would be)\s+(?:the\s+|a\s+|an\s+)?([a-z0-9'\- ]+?)(?=[.,!?;:\n]|\s+(?:and|but|because|so|though)\b|$)",
    re.I,
)


@dataclass
class Claim:
    key: str
    value: str


def extract_claims(text: str) -> list[Claim]:
    return [Claim(normalize_key("favorite " + m.group(1)), m.group(2).strip()) for m in CLAIM_RE.finditer(text)]


def _matches(stored: str, claimed: str) -> bool:
    s, c = stored.lower().strip(), claimed.lower().strip()
    return s == c or s in c or c in s


@dataclass
class Verdict:
    contradictions: list[tuple[Claim, str]]  # (claim, what the store says)
    novel: list[Claim]


def check(store: SelfStore, text: str) -> Verdict:
    contradictions, novel = [], []
    for claim in extract_claims(text):
        stored = store.get_trait(claim.key)
        if stored is None:
            novel.append(claim)
        elif not _matches(stored, claim.value):
            contradictions.append((claim, stored))
    return Verdict(contradictions, novel)


def correction_message(contradictions: list[tuple[Claim, str]]) -> str:
    lines = [f"- your {c.key.replace('_', ' ')} is {stored}, not {c.value}" for c, stored in contradictions]
    return "You contradicted yourself. Rewrite your answer keeping these facts straight:\n" + "\n".join(lines)
