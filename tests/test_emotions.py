from mind import emotions
from mind.emotions import DEFAULT_BASELINES


def test_deltas_are_capped_and_clamped():
    s = emotions.apply_deltas(dict(DEFAULT_BASELINES), {"happiness": 5, "boredom": -5, "nonsense": 1})
    assert s["happiness"] == 0.8
    assert s["boredom"] == 0.0
    assert "nonsense" not in s


def test_irritation_lowers_happiness():
    s = dict(DEFAULT_BASELINES, irritation=0.9, happiness=0.5)
    assert emotions.apply_interactions(s)["happiness"] < 0.5


def test_decay_halves_gap_each_half_life():
    s = dict(DEFAULT_BASELINES, happiness=0.9)
    out = emotions.decay(s, DEFAULT_BASELINES, 1800, 1800)
    assert abs(out["happiness"] - 0.7) < 1e-9


def test_describe_mentions_strong_feelings():
    text = emotions.describe(dict(DEFAULT_BASELINES, irritation=0.9))
    assert "irritated" in text and "irritation 0.90" in text
