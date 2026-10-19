"""The domain description profiles: registered, and their prompts fill from the documented fields."""
from nlp.summarizer import profiles
from nlp.summarizer.service import _prepare_fields, _template


def test_domain_profiles_render_with_fields_and_evidence():
    for name in ("requirement_desc", "criterion_desc"):
        profile = profiles.get(name)
        fields = _prepare_fields(profile, {"name": "Volumen anual de negocios", "type": "REQUIREMENT",
                                           "subtype": "TURNOVER", "attributes_text": "- threshold_amount: 30750",
                                           "evidence": ["[PCAP · pág. 12] volumen anual de negocios de 30.750 euros"]})
        prompt = _template(profile.prompt_file).format(**fields)
        assert "30750" in prompt and "30.750 euros" in prompt and "Subtipo: TURNOVER" in prompt


def test_attributes_text_is_optional():
    profile = profiles.get("requirement_desc")
    fields = _prepare_fields(profile, {"name": "x", "type": "REQUIREMENT", "evidence": ["y"]})
    assert "(sin campos)" in _template(profile.prompt_file).format(**fields)


def test_formula_profile_renders_without_reducing_the_formula():
    profile = profiles.get("criterion_formula")
    long = ["relleno " * 600, "P = 80 * (PL - Of) / (PL - Of_min)"]
    fields = _prepare_fields(profile, {"name": "Precio", "points": 80, "evidence": long})
    prompt = _template(profile.prompt_file).format(**fields)
    assert "P = 80 * (PL - Of) / (PL - Of_min)" in prompt and "Puntos: 80" in prompt


def test_formula_check_accepts_declared_symbols_and_tiers():
    from nlp.summarizer.formula import check
    v = lambda *s: [{"symbol": x} for x in s]
    assert check({"has_formula": True, "expression": "P = Pmax * (PL - Of) / (PL - Ofmin)", "variables": v("P", "Pmax", "PL", "Of", "Ofmin")})[0]
    assert check({"has_formula": True, "expression": "P = if(A >= 3, 20, if(A >= 2, 7, if(A >= 1, 4, 0)))", "variables": v("P", "A")})[0]
    assert check({"has_formula": False, "expression": "", "variables": []})[0]
    ok, why = check({"has_formula": True, "expression": "P = 80 * Bi / Bmax", "variables": v("P", "Bi")})
    assert not ok and "Bmax" in why
    assert not check({"has_formula": True, "expression": "P = __import__('os')", "variables": v("P")})[0]
    assert not check({"has_formula": True, "expression": "80 * Bi", "variables": v("Bi")})[0]
