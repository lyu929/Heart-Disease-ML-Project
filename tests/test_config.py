import pytest

from heartrisk.config import apply_override, default_config, load_config


def test_defaults_have_required_sections():
    cfg = default_config()
    for key in ("seed", "cleaning", "evaluation", "deploy", "benchmark", "site_validation"):
        assert key in cfg
    assert cfg["evaluation"]["threshold"]["strategy"] in {"f1", "youden", "sensitivity", "fixed"}


def test_overrides_are_parsed_as_yaml():
    cfg = load_config(overrides=["evaluation.repeats=2", "evaluation.models=[logreg, rf]", "new.key=true"])
    assert cfg["evaluation"]["repeats"] == 2
    assert cfg["evaluation"]["models"] == ["logreg", "rf"]
    assert cfg["new"]["key"] is True


def test_user_yaml_is_merged_not_replaced(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("evaluation:\n  repeats: 1\n")
    cfg = load_config(p)
    assert cfg["evaluation"]["repeats"] == 1
    assert cfg["evaluation"]["outer_splits"] == default_config()["evaluation"]["outer_splits"]


def test_bad_override():
    with pytest.raises(ValueError):
        apply_override({}, "no_equals_sign")
