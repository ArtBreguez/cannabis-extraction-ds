"""Regression tests for the feature-construction layer.

Each test here exists because an audit found the corresponding defect in a
version of this repository. They are cheap and they fail loudly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.phase4_generalisation import features  # noqa: E402

MASTER = ROOT / "data/labeled/master.csv"

pytestmark = pytest.mark.skipif(
    not MASTER.exists(), reason="master.csv not built")


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    d = pd.read_csv(MASTER, low_memory=False)
    return d[d["label_conflict"] == 0].copy()


def test_date_tested_never_becomes_a_feature(df):
    """`date_tested` ends in "_tested" but is a date, not an analyte mask.

    The suffix sweep used to pick it up and strip it to "date". Nothing broke
    only because no column is called "date"; a rename upstream would have fed
    a timestamp to the model. So the real test injects that rename: a `date`
    column must still not become a feature.
    """
    feats = features(df)
    assert "date" not in feats
    assert "date_tested" not in feats

    # The defect this guards is latent, not active: it only bites once a
    # column named "date" exists. Simulate that upstream rename, because a
    # test that passes without it cannot distinguish a fix from the bug.
    probe = df.copy()
    probe["date"] = probe["date_tested"]
    feats_renamed = features(probe)
    assert "date" not in feats_renamed, (
        "features() picked up `date` via the date_tested suffix sweep")


def test_date_tested_is_still_present_and_still_a_date(df):
    """Guard the premise of the test above: if the column vanishes or turns
    numeric, the exclusion is silently pointless and someone should notice."""
    assert "date_tested" in df.columns
    sample = df["date_tested"].dropna()
    assert len(sample) > 0
    assert pd.to_datetime(sample.iloc[:50], errors="coerce").notna().all()


def test_features_are_all_numeric(df):
    """Whatever features() returns must survive astype(float); this is the
    exact failure a stray `date_tested` would cause."""
    feats = features(df)
    df[feats].astype(float)


def test_feature_count_matches_the_manuscript(df):
    """2.2 claims 19 analytes survive the empty-column drop."""
    assert len(features(df)) == 19


def test_no_tested_flag_is_returned_as_a_feature(df):
    """The models score on values only. The flags are byte-identical to
    notna(value), so including them would add nothing but would make the
    manuscript's description wrong."""
    feats = features(df)
    assert not any(f.endswith("_tested") for f in feats)


def test_tested_flags_agree_with_value_nullity(df):
    """Each `<a>_tested` must equal notna(<a>). If they ever diverge, the
    missingness probe in 2.3 and the model stop seeing the same thing."""
    for a in features(df):
        flag = a + "_tested"
        if flag in df.columns:
            assert ((df[flag].astype(float) == 1) == df[a].notna()).all(), a
