import json

from mind.config import Config
from mind.engine import Mind
from mind.llm import FakeLLM
from mind.store import SelfStore

CFG = Config(max_regenerations=2)


def sub_llm(deltas=None, decide="black"):
    def reply(messages, json_mode):
        if messages[0]["content"].startswith("You are the subconscious of a mind and must settle"):
            return json.dumps({"value": decide})
        return json.dumps({"emotion_deltas": deltas or {}, "urges": ["be curious"]})

    return FakeLLM(reply)


def con(text):
    return f"<thought>t</thought><decision>reply</decision><reply>{text}</reply>"


def test_subconscious_settles_identity_before_conscious_speaks():
    store = SelfStore()
    seen = []

    def conscious(messages, _):
        seen.append(messages[0]["content"])
        return con("My favorite color is black.")

    mind = Mind(store, FakeLLM(conscious), sub_llm(), CFG)
    r = mind.turn("what's your favorite color?")
    assert store.get_trait("favorite_color") == "black"
    assert "favorite color: black" in seen[0]  # already in the prompt on the very first answer
    assert r.regenerations == 0


def test_contradiction_forces_rewrite():
    store = SelfStore()
    mind = Mind(store, FakeLLM([con("My favorite color is yellow."), con("Black, obviously.")]), sub_llm(), CFG)
    r = mind.turn("favorite color?")
    assert r.regenerations == 1 and r.reply == "Black, obviously."
    assert store.get_trait("favorite_color") == "black"
    assert ("favorite_color", "yellow", "rejected") in store.get_claims()


def test_persistent_contradiction_is_flagged_but_store_wins():
    store = SelfStore()
    mind = Mind(store, FakeLLM([con("My favorite color is yellow.")]), sub_llm(), CFG)
    r = mind.turn("favorite color?")
    assert r.regenerations == 2 and any("unresolved" in n for n in r.notes)
    assert store.get_trait("favorite_color") == "black"


def test_emotions_change_and_are_shown_to_conscious():
    store = SelfStore()
    seen = []
    mind = Mind(store, FakeLLM(lambda m, _: (seen.append(m[0]["content"]), con("hi"))[1]), sub_llm({"happiness": 0.3}), CFG)
    mind.turn("how was your day?")
    assert store.get_state()["happiness"] > 0.75
    assert "happy" in seen[0]


def test_refusal_and_silence_pass_through_and_thoughts_are_private():
    store = SelfStore()
    refuse = "<thought>not in the mood</thought><decision>refuse</decision><reply>No, I don't feel like it.</reply>"
    r = Mind(store, FakeLLM([refuse]), sub_llm(), CFG).turn("write 120 words")
    assert r.decision == "refuse" and "don't feel" in r.reply
    assert store.get_thoughts() == ["not in the mood"]


def test_spontaneous_claim_becomes_a_trait():
    store = SelfStore()
    Mind(store, FakeLLM([con("My favorite food is ramen.")]), sub_llm(), CFG).turn("hey")
    assert store.get_trait("favorite_food") == "ramen"


def test_idle_tick_decays_toward_baseline():
    t = [0.0]
    store = SelfStore(now=lambda: t[0])
    store.set_state({**store.get_state(), "irritation": 0.9})
    mind = Mind(store, FakeLLM([con("x")]), sub_llm(), CFG)
    t[0] = 3600
    mind.idle_tick()
    assert store.get_state()["irritation"] < 0.35
