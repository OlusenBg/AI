"""Pure emotion logic: clamping, per-turn limits, decay toward baseline, interactions."""

EMOTIONS = ("happiness", "trust", "irritation", "boredom", "energy", "curiosity")

DEFAULT_BASELINES = {
    "happiness": 0.5,
    "trust": 0.4,
    "irritation": 0.1,
    "boredom": 0.2,
    "energy": 0.6,
    "curiosity": 0.5,
}


def clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def apply_deltas(state: dict[str, float], deltas: dict[str, float], max_delta: float = 0.3) -> dict[str, float]:
    """Return a new state with deltas applied. Unknown emotions are ignored and each
    delta is capped so one message can't swing the mind to an extreme."""
    new = dict(state)
    for name, d in deltas.items():
        if name not in new:
            continue
        d = max(-max_delta, min(max_delta, float(d)))
        new[name] = clamp(new[name] + d)
    return apply_interactions(new)


def apply_interactions(state: dict[str, float]) -> dict[str, float]:
    """Small hand-written couplings between emotions."""
    s = dict(state)
    if s["irritation"] > 0.6:
        s["happiness"] = clamp(s["happiness"] - 0.05 * (s["irritation"] - 0.6) / 0.4)
        s["trust"] = clamp(s["trust"] - 0.03 * (s["irritation"] - 0.6) / 0.4)
    if s["energy"] < 0.3:
        s["curiosity"] = clamp(s["curiosity"] - 0.03)
    return s


def decay(state: dict[str, float], baselines: dict[str, float], dt: float, half_life: float) -> dict[str, float]:
    """Move every emotion toward its baseline; half the gap closes every half_life seconds."""
    if dt <= 0:
        return dict(state)
    keep = 0.5 ** (dt / half_life)
    return {k: clamp(baselines[k] + (v - baselines[k]) * keep) for k, v in state.items()}


_LEVELS = [(0.85, "extremely"), (0.65, "very"), (0.45, "moderately"), (0.25, "slightly")]


def _level(v: float) -> str | None:
    for threshold, word in _LEVELS:
        if v >= threshold:
            return word
    return None


_WORDS = {
    "happiness": "happy",
    "trust": "trusting of the person you're talking to",
    "irritation": "irritated",
    "boredom": "bored",
    "energy": "energetic",
    "curiosity": "curious",
}


def describe(state: dict[str, float]) -> str:
    """Natural-language summary plus raw numbers, for injection into the conscious prompt."""
    parts = []
    for k in EMOTIONS:
        lvl = _level(state[k])
        if lvl:
            parts.append(f"{lvl} {_WORDS[k]}")
    prose = "You feel " + (", ".join(parts) if parts else "pretty neutral") + "."
    numbers = ", ".join(f"{k} {state[k]:.2f}" for k in EMOTIONS)
    return f"{prose}\n(raw: {numbers})"
