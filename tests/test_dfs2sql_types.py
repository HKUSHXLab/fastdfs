"""Tests that dfs2sql column types come from RDB metadata, not from planning.

dfs2sql plans on 0-row RDB frames, so Featuretools/woodwork never convert the real
data. Every type DuckDB computes on is set explicitly at ingest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastdfs.api import compute_dfs_features, create_rdb
from fastdfs.dfs import DFSConfig, DFSSession, get_dfs_engine

NAN = np.nan


@pytest.fixture
def target():
    return pd.DataFrame(
        {
            "user_id": [1, 2],
            "timestamp": pd.to_datetime(["2020-01-05", "2020-01-05"]),
        }
    )


@pytest.fixture
def guard_rdb():
    """One RDB with a column of every metadata type DFS supports."""
    users = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "age": [25, 30, 35],  # int -> float
            "score": np.array([0.5, 1.5, 2.5], dtype="float32"),
            "active": [True, False, True],  # bool -> float
            "country": ["us", "cn", "us"],
            "bio": ["likes a", "likes b", "likes c"],
            "signup": pd.to_datetime(
                ["2019-12-01 23:30", "2019-12-02 01:00", "2019-12-03 12:00"]
            ).tz_localize("Asia/Shanghai"),
        }
    )
    events = pd.DataFrame(
        {
            "event_id": [10, 11, 12, 13, 14],
            "user_id": [1, 1, 1, 2, 3],
            "amount": [5.0, 7.0, 9.0, 11.0, 13.0],
            "kind": ["a", "b", "b", "a", "a"],
            "ok": [True, False, True, True, False],
            # Event 13 is on 2020-01-03 in wall time but 2020-01-02 in UTC, so a
            # 2020-01-03 cutoff tells whether the wall time was kept.
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01 10:00",
                    "2020-01-02 10:00",
                    "2020-01-04 23:00",
                    "2020-01-03 05:00",
                    "2020-01-04 10:00",
                ]
            ).tz_localize("Asia/Shanghai"),
        }
    )
    lines = pd.DataFrame(
        {
            "line_id": [100, 101, 102, 103, 104, 105],
            "event_id": [10, 10, 11, 12, 13, 14],
            "b": [True, False, True, False, True, True],
        }
    )
    return create_rdb(
        tables={"user": users, "event": events, "line": lines},
        primary_keys={"user": "user_id", "event": "event_id", "line": "line_id"},
        foreign_keys=[
            ("event", "user_id", "user", "user_id"),
            ("line", "event_id", "event", "event_id"),
        ],
        time_columns={"event": "timestamp"},
        type_hints={
            "user": {"country": "category", "bio": "text"},
            "event": {"kind": "category"},
        },
    )


GUARD_KEYS = {"user_id": "user.user_id"}
GUARD_CONFIG = DFSConfig(
    engine="dfs2sql",
    engine_path=":memory:",
    max_depth=3,
    agg_primitives=["count", "sum", "mean", "mode"],
)


def _guard_target():
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 1, 2],
            "cutoff": pd.to_datetime(
                ["2020-01-05", "2020-01-05", "2020-01-05", "2020-01-03", "2020-01-03"]
            ),
        }
    )


def _guard_expected():
    """Output of master (full Featuretools planning) on the guard RDB, checked by hand."""
    return pd.DataFrame(
        {
            "user_id": ["1", "2", "3", "1", "2"],
            "cutoff": pd.to_datetime(
                ["2020-01-05", "2020-01-05", "2020-01-05", "2020-01-03", "2020-01-03"]
            ),
            "d1__user.age": [25.0, 30.0, 35.0, 25.0, 30.0],
            "d1__user.score": [0.5, 1.5, 2.5, 0.5, 1.5],
            "d1__user.active": [1.0, 0.0, 1.0, 1.0, 0.0],
            "d1__user.country": pd.Categorical(
                ["us", "cn", "us", "us", "cn"], categories=["cn", "us"], ordered=True
            ),
            "d2__user.COUNT(event)": np.array([3, 1, 1, 2, 0], dtype="int64"),
            "d2__user.MEAN(event.amount)": [7.0, 11.0, 13.0, 6.0, NAN],
            "d2__user.MEAN(event.ok)": [2 / 3, 1.0, 0.0, 0.5, NAN],
            "d2__user.MODE(event.kind)": pd.Categorical(
                ["b", "a", "a", "a", NAN], categories=["a", "b"], ordered=True
            ),
            "d2__user.SUM(event.amount)": [21.0, 11.0, 13.0, 12.0, NAN],
            "d2__user.SUM(event.ok)": [2.0, 1.0, 0.0, 1.0, NAN],
            "d3__user.MEAN(event.COUNT(line))": [4 / 3, 1.0, 1.0, 1.5, NAN],
            "d3__user.MEAN(event.MEAN(line.b))": [0.5, 1.0, 1.0, 0.75, NAN],
            "d3__user.MEAN(event.SUM(line.b))": [2 / 3, 1.0, 1.0, 1.0, NAN],
            "d3__user.SUM(event.MEAN(line.b))": [1.5, 1.0, 1.0, 1.5, NAN],
        }
    )


def test_guard_one_shot_matches_expected(guard_rdb):
    out = compute_dfs_features(
        guard_rdb,
        _guard_target(),
        key_mappings=GUARD_KEYS,
        cutoff_time_column="cutoff",
        config=GUARD_CONFIG,
    )
    pd.testing.assert_frame_equal(out, _guard_expected(), check_dtype=True)


def test_guard_session_matches_expected(guard_rdb):
    target, expected = _guard_target(), _guard_expected()
    with DFSSession(
        guard_rdb, GUARD_KEYS, cutoff_time_column="cutoff", config=GUARD_CONFIG
    ) as session:
        first = session.compute(target.iloc[:3])
        second = session.compute(target.iloc[3:])
    pd.testing.assert_frame_equal(first, expected.iloc[:3], check_dtype=True)
    pd.testing.assert_frame_equal(second, expected.iloc[3:], check_dtype=True)


def test_bool_column_nested_mean_not_truncated():
    rdb = create_rdb(
        tables={
            "u": pd.DataFrame({"id": [1]}),
            "e": pd.DataFrame({"id": [10], "uid": [1]}),
            "l": pd.DataFrame({"id": [100, 101], "eid": [10, 10], "b": [True, False]}),
        },
        primary_keys={"u": "id", "e": "id", "l": "id"},
        foreign_keys=[("e", "uid", "u", "id"), ("l", "eid", "e", "id")],
    )
    out = compute_dfs_features(
        rdb,
        pd.DataFrame({"uid": [1]}),
        {"uid": "u.id"},
        config=DFSConfig(
            engine="dfs2sql",
            engine_path=":memory:",
            max_depth=3,
            agg_primitives=["mean", "sum"],
        ),
    )
    assert out["d3__u.MEAN(e.MEAN(l.b))"].tolist() == [0.5]
    assert out["d3__u.SUM(e.MEAN(l.b))"].tolist() == [0.5]


def _rdb_with_event_times(timestamps):
    return create_rdb(
        tables={
            "user": pd.DataFrame({"user_id": [1, 2]}),
            "event": pd.DataFrame(
                {
                    "event_id": [10, 11, 12],
                    "user_id": [1, 1, 2],
                    "amount": [5.0, 7.0, 9.0],
                    "timestamp": timestamps,
                }
            ),
        },
        primary_keys={"user": "user_id", "event": "event_id"},
        foreign_keys=[("event", "user_id", "user", "user_id")],
        time_columns={"event": "timestamp"},
    )


def test_session_string_cutoffs_across_splits():
    rdb = _rdb_with_event_times(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
    config = DFSConfig(engine="dfs2sql", engine_path=":memory:")
    splits = [
        pd.DataFrame({"user_id": [1, 2], "cut": ["2020-01-05", "2020-01-05"]}),
        pd.DataFrame({"user_id": [1, 2], "cut": ["2020-01-02", "2020-01-05"]}),
    ]
    with DFSSession(
        rdb, {"user_id": "user.user_id"}, cutoff_time_column="cut", config=config
    ) as session:
        outs = [session.compute(split) for split in splits]

    for split, out in zip(splits, outs):
        oneshot = compute_dfs_features(
            rdb,
            split,
            {"user_id": "user.user_id"},
            cutoff_time_column="cut",
            config=config,
        )
        pd.testing.assert_frame_equal(out, oneshot, check_dtype=True)
    assert outs[1]["d2__user.COUNT(event)"].tolist() == [1, 1]
    # The caller's frames keep their string cutoffs.
    assert splits[1]["cut"].tolist() == ["2020-01-02", "2020-01-05"]


def test_tz_aware_rdb_times_with_naive_cutoff():
    rdb = _rdb_with_event_times(pd.to_datetime(["2020-01-01 12:00:00+02:00"] * 3))
    out = compute_dfs_features(
        rdb,
        pd.DataFrame(
            {"user_id": [1, 2], "cut": pd.to_datetime(["2020-01-05", "2020-01-05"])}
        ),
        {"user_id": "user.user_id"},
        cutoff_time_column="cut",
        config=DFSConfig(engine="dfs2sql", engine_path=":memory:"),
    )
    assert out["d2__user.COUNT(event)"].tolist() == [2, 1]


def _one_shot_and_session(rdb, splits, config):
    """Outputs for each split from compute_dfs_features and from one DFSSession."""
    keys = {"user_id": "user.user_id"}
    one_shot = [
        compute_dfs_features(rdb, split, keys, cutoff_time_column="cut", config=config)
        for split in splits
    ]
    with DFSSession(rdb, keys, cutoff_time_column="cut", config=config) as session:
        in_session = [session.compute(split) for split in splits]
    return one_shot, in_session


def test_string_rdb_time_column_matches_datetimes():
    dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
    config = DFSConfig(engine="dfs2sql", engine_path=":memory:")
    splits = [
        pd.DataFrame({"user_id": [1, 2], "cut": pd.to_datetime(["2020-01-05", "2020-01-05"])}),
        pd.DataFrame({"user_id": [1, 2], "cut": pd.to_datetime(["2020-01-02", "2020-01-05"])}),
    ]
    reference, _ = _one_shot_and_session(
        _rdb_with_event_times(pd.to_datetime(dates)), splits, config
    )
    one_shot, in_session = _one_shot_and_session(_rdb_with_event_times(dates), splits, config)

    for expected, a, b in zip(reference, one_shot, in_session):
        pd.testing.assert_frame_equal(a, expected, check_dtype=True)
        pd.testing.assert_frame_equal(b, expected, check_dtype=True)
    assert reference[1]["d2__user.COUNT(event)"].tolist() == [1, 1]


def test_mixed_offset_cutoffs_compared_in_utc():
    rdb = _rdb_with_event_times(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
    # In UTC these are 2020-01-01 23:30 and 2020-01-02 22:30, so each user has
    # one event fewer than the wall times (00:30 the next day) would give.
    split = pd.DataFrame(
        {"user_id": [1, 2], "cut": ["2020-01-02T00:30+01:00", "2020-01-03T00:30+02:00"]}
    )
    one_shot, in_session = _one_shot_and_session(
        rdb, [split], DFSConfig(engine="dfs2sql", engine_path=":memory:")
    )
    assert one_shot[0]["d2__user.COUNT(event)"].tolist() == [1, 0]
    assert in_session[0]["d2__user.COUNT(event)"].tolist() == [1, 0]


def test_unparseable_cutoff_raises():
    rdb = _rdb_with_event_times(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
    split = pd.DataFrame({"user_id": [1, 2], "cut": ["not a date", "2020-01-05"]})
    config = DFSConfig(engine="dfs2sql", engine_path=":memory:")
    match = "Cutoff time column 'cut' has values that cannot be parsed"
    with pytest.raises(ValueError, match=match):
        compute_dfs_features(rdb, split, {"user_id": "user.user_id"}, cutoff_time_column="cut", config=config)
    with DFSSession(rdb, {"user_id": "user.user_id"}, cutoff_time_column="cut", config=config) as session:
        with pytest.raises(ValueError, match=match):
            session.compute(split)


@pytest.mark.parametrize("engine, n_rows", [("dfs2sql", 0), ("featuretools", 3)])
def test_planning_entityset_rows(tiny_rdb, engine, n_rows):
    # dfs2sql computes values in DuckDB, so it plans on 0-row frames; the
    # featuretools engine computes values from the EntitySet and needs full frames.
    dfs_engine = get_dfs_engine(engine, DFSConfig(engine=engine))
    entity_set = dfs_engine._build_entity_set_from_rdb(tiny_rdb)
    assert len(entity_set["user"]) == n_rows


def test_rdb_tables_not_modified_by_compute(guard_rdb):
    before = {t: guard_rdb.get_table(t).copy() for t in guard_rdb.table_names}

    compute_dfs_features(
        guard_rdb,
        _guard_target(),
        key_mappings=GUARD_KEYS,
        cutoff_time_column="cutoff",
        config=GUARD_CONFIG,
    )

    for table_name, frame in before.items():
        pd.testing.assert_frame_equal(guard_rdb.get_table(table_name), frame)


def test_rdb_tables_without_primary_key_not_modified(target):
    users = pd.DataFrame({"user_id": [1, 2], "age": [25, 30]})
    logs = pd.DataFrame({"kind": ["a", "b"], "value": [1, 2]})  # no primary key
    rdb = create_rdb(
        tables={"user": users, "logs": logs},
        primary_keys={"user": "user_id"},
        type_hints={"logs": {"kind": "category", "value": "float"}},
    )
    before = rdb.get_table("logs").dtypes.copy()

    compute_dfs_features(
        rdb,
        target.copy(),
        key_mappings={"user_id": "user.user_id"},
        config=GUARD_CONFIG,
    )

    pd.testing.assert_series_equal(rdb.get_table("logs").dtypes, before)
    assert "__index__" not in rdb.get_table("logs").columns
