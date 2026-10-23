from mind.store import SelfStore


def test_traits_are_never_overwritten():
    s = SelfStore()
    assert s.add_trait("Favorite Color", "black")
    assert not s.add_trait("favorite_color", "yellow")
    assert s.get_trait("favorite color") == "black"


def test_revise_is_the_only_way_to_change_a_trait():
    s = SelfStore()
    s.add_trait("favorite_color", "black")
    s.revise_trait("favorite_color", "dark blue")
    assert s.get_trait("favorite_color") == "dark blue"


def test_state_persists_in_file(tmp_path):
    p = str(tmp_path / "m.db")
    a = SelfStore(p)
    a.add_trait("favorite_food", "ramen")
    st = a.get_state()
    st["happiness"] = 0.9
    a.set_state(st)
    b = SelfStore(p)
    assert b.get_trait("favorite_food") == "ramen"
    assert b.get_state()["happiness"] == 0.9


def test_thoughts_and_memories():
    s = SelfStore()
    s.log_thought("hmm")
    mid = s.add_memory("met someone", 0.9)
    s.promote_memory(mid)
    assert s.get_thoughts() == ["hmm"]
    assert s.get_memories("long")[0].text == "met someone"
