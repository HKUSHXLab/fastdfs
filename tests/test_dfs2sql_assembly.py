"""Unit tests for dfs2sql wide-matrix assembly (concat vs legacy merge)."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from fastdfs.dfs.dfs2sql_engine import (
    assemble_dfs2sql_feature_frames,
    merge_dfs2sql_feature_frames_legacy,
)

TARGET = "__target_index__"


def _shuffled_rows(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def test_assemble_matches_merge_shuffled_rows():
    """Each SQL result may return rows in arbitrary order; assembly must match merge."""
    canonical = pd.Index([0, 1, 2, 3], name=TARGET)
    d0 = pd.DataFrame({TARGET: [2, 0, 1, 3], "f0": [10.0, 1.0, 5.0, 7.0]})
    d1 = _shuffled_rows(pd.DataFrame({TARGET: [0, 1, 2, 3], "f1": [0.1, 0.2, 0.3, 0.4]}), seed=1)
    d2 = pd.DataFrame({TARGET: [3, 1, 0, 2], "f2": ["a", "b", "c", "d"]})
    dfs = [d0, d1, d2]
    got = assemble_dfs2sql_feature_frames(dfs, TARGET, canonical, concat_chunk_size=2)
    exp = merge_dfs2sql_feature_frames_legacy(dfs, TARGET)
    assert_frame_equal(got, exp, check_dtype=False)


def test_assemble_matches_merge_multi_column_frame():
    canonical = pd.Index(np.arange(4), name=TARGET)
    d0 = pd.DataFrame({TARGET: [0, 1, 2, 3], "a": [1, 2, 3, 4]})
    d1 = pd.DataFrame({TARGET: [2, 3, 0, 1], "b": [10, 20, 30, 40], "c": [100, 200, 300, 400]})
    dfs = [d0, d1]
    got = assemble_dfs2sql_feature_frames(dfs, TARGET, canonical, concat_chunk_size=1)
    exp = merge_dfs2sql_feature_frames_legacy(dfs, TARGET)
    assert_frame_equal(got, exp, check_dtype=False)


@pytest.mark.parametrize("chunk_size", [1, 2, 50, 10_000])
def test_chunk_size_does_not_change_result(chunk_size: int):
    canonical = pd.Index(np.arange(6), name=TARGET)
    dfs = [
        _shuffled_rows(
            pd.DataFrame({TARGET: list(range(6)), f"f{i}": np.linspace(i, i + 0.5, 6)}),
            seed=i + 3,
        )
        for i in range(7)
    ]
    got = assemble_dfs2sql_feature_frames(
        dfs, TARGET, canonical, concat_chunk_size=chunk_size
    )
    exp = merge_dfs2sql_feature_frames_legacy(dfs, TARGET)
    assert_frame_equal(got, exp, check_dtype=False)


def test_duplicate_feature_column_raises():
    canonical = pd.Index([0, 1], name=TARGET)
    d0 = pd.DataFrame({TARGET: [0, 1], "dup": [1, 2]})
    d1 = pd.DataFrame({TARGET: [0, 1], "dup": [3, 4]})
    with pytest.raises(ValueError, match="duplicate feature column"):
        assemble_dfs2sql_feature_frames([d0, d1], TARGET, canonical)


def test_non_unique_canonical_index_raises():
    canonical = pd.Index([0, 0, 1], name=TARGET)
    d0 = pd.DataFrame({TARGET: [0, 1], "x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="unique"):
        assemble_dfs2sql_feature_frames([d0], TARGET, canonical)


def test_empty_frames_returns_index_only():
    canonical = pd.Index([0, 1, 2], name=TARGET)
    out = assemble_dfs2sql_feature_frames([], TARGET, canonical)
    assert list(out.columns) == [TARGET]
    assert out[TARGET].tolist() == [0, 1, 2]


def test_missing_target_column_raises():
    canonical = pd.Index([0], name=TARGET)
    bad = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="missing column"):
        assemble_dfs2sql_feature_frames([bad], TARGET, canonical)
