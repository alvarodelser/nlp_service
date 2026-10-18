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
