from mind import conscious, consistency, subconscious
from mind.store import SelfStore


def test_subconscious_parses_fenced_json():
    out = subconscious.parse_output('sure!\n```json\n{"emotion_deltas": {"happiness": 0.1}, "urges": ["be warm"]}\n```')
    assert out.emotion_deltas == {"happiness": 0.1} and out.urges == ["be warm"]


def test_subconscious_garbage_is_numb_not_fatal():
    assert subconscious.parse_output("I refuse to output JSON").emotion_deltas == {}


def test_subconscious_salvages_valid_fields():
    out = subconscious.parse_output('{"emotion_deltas": {"energy": -0.1}, "memory_writes": "oops"}')
    assert out.emotion_deltas == {"energy": -0.1} and out.memory_writes == []


def test_scan_topics():
    assert subconscious.scan_topics("What's your favorite color? And favourite food?") == [
        "favorite_color",
        "favorite_food",
    ]


def test_conscious_parses_tags():
    o = conscious.parse_output("<thought>ugh</thought><decision>Refuse</decision><reply>No.</reply>")
    assert (o.thought, o.decision, o.reply) == ("ugh", "refuse", "No.")


def test_conscious_silence_has_no_reply():
    o = conscious.parse_output("<thought>.</thought><decision>stay silent</decision><reply>hello?</reply>")
    assert o.decision == "stay_silent" and o.reply == ""


def test_conscious_untagged_text_is_a_reply():
    o = conscious.parse_output("just some words")
    assert o.decision == "reply" and o.reply == "just some words"


def test_claim_extraction_and_check():
    s = SelfStore()
    s.add_trait("favorite_color", "black")
    v = consistency.check(s, "Honestly my favorite color is yellow, and I like rain.")
    assert [(c.key, c.value, st) for c, st in v.contradictions] == [("favorite_color", "yellow", "black")]
    assert consistency.check(s, "My favorite color is black.").contradictions == []
    assert consistency.check(s, "my favorite food is ramen.").novel[0].key == "favorite_food"


def test_conscious_reply_without_reply_tag():
    o = conscious.parse_output("<thought>hm</thought><decision>reply</decision>\nProbably a black sports car.")
    assert o.decision == "reply" and o.reply == "Probably a black sports car."
    assert conscious.parse_output("<thought>hm</thought><decision>reply</decision>").reply == ""
