"""Tests for the labeling rules in build_master.

The rule ORDER is the fragile part: texture words (badder, wax, shatter)
describe consistency, not extraction method, so "live rosin badder" must come
out solventless. 41 rows in the real file hit exactly that case, and getting
the order wrong would silently mislabel all of them as hydrocarbon.

Run: .venv/bin/python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.build_master import fnum, name_label  # noqa: E402


class TestTextureWordsDoNotOverrideMethod:
    """Real product names from the dataset that carry both kinds of token."""

    @pytest.mark.parametrize("name", [
        "shango alien banana candy live rosin badder",
        "bulk concentrate cold cured live rosin badder - garlic breath (h), 1g",
        "shango monavie live rosin badder",
    ])
    def test_live_rosin_badder_is_solventless(self, name):
        rule, cls, _ = name_label(name)
        assert cls == "solventless", (
            f"{name!r} is rosin (solventless) that happens to have badder "
            f"texture; got {cls} via rule {rule}"
        )

    def test_rosin_wax_is_still_solventless(self):
        _, cls, _ = name_label("gelato rosin wax 1g")
        assert cls == "solventless"

    def test_rosin_shatter_is_still_solventless(self):
        _, cls, _ = name_label("blue dream rosin shatter")
        assert cls == "solventless"


class TestHydrocarbonTokens:
    @pytest.mark.parametrize("name,rule_expected", [
        ("cement shoes live resin crumble", "live_resin"),
        ("gelato 33 badder bho100223", "bho"),
        ("sour diesel shatter 1g", "shatter"),
        ("wedding cake badder", "badder"),
    ])
    def test_hydrocarbon_names(self, name, rule_expected):
        rule, cls, _ = name_label(name)
        assert cls == "hydrocarbon"
        assert rule == rule_expected

    def test_live_resin_is_never_solventless(self):
        """Live resin is hydrocarbon by definition — fresh-frozen + solvent."""
        _, cls, _ = name_label("pineapple haze - panna - live resin distillate")
        assert cls == "hydrocarbon"


class TestConfidence:
    def test_explicit_method_words_are_high_confidence(self):
        for name in ["og kush live rosin", "gg4 bho", "papaya hash rosin"]:
            _, _, conf = name_label(name)
            assert conf == "high", name

    def test_texture_only_is_medium_confidence(self):
        """A texture word alone is weaker evidence and must say so."""
        for name in ["gelato badder", "sour d wax", "gsc shatter"]:
            _, _, conf = name_label(name)
            assert conf == "medium", name


class TestNoMatch:
    @pytest.mark.parametrize("name", [
        "blue dream (1g)",
        "kimbo cookies (1g)",
        "concentrate",
        "",
    ])
    def test_plain_names_yield_nothing(self, name):
        rule, cls, conf = name_label(name)
        assert (rule, cls, conf) == (None, None, None), (
            f"{name!r} carries no method information and must NOT be guessed"
        )

    def test_substring_does_not_false_positive(self):
        """Word boundaries matter: 'waxy' is not 'wax'."""
        _, cls, _ = name_label("waxy leaf blend")
        assert cls is None

    def test_bho_lookahead_rejects_real_words(self):
        """The relaxed bho pattern must not fire on words starting with bho."""
        _, cls, _ = name_label("bhopal sunset")
        assert cls is None, "'bhopal' is not a BHO extract"


class TestNonDetectHandling:
    """An empty cell and a measured zero are different facts."""

    def test_empty_is_none_not_zero(self):
        assert fnum("") is None
        assert fnum("   ") is None

    def test_measured_zero_survives_as_zero(self):
        assert fnum("0") == 0.0
        assert fnum("0.0") == 0.0

    def test_zero_is_not_confused_with_missing(self):
        # The distinction the whole *_tested flag design rests on.
        assert fnum("0") is not None

    def test_garbage_is_none(self):
        assert fnum("ND") is None
        assert fnum("<LOQ") is None

    def test_normal_values_parse(self):
        assert fnum("86.35") == pytest.approx(86.35)
