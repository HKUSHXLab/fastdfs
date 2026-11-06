"""dfs2sql validates the RDB itself, since planning only sees 0-row frames.

Full-frame planning (still used by the featuretools engine) validated the data as a
side effect; dfs2sql must accept and reject the same RDBs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fastdfs.api import compute_dfs_features, create_rdb
from fastdfs.dfs import DFSConfig, DFSSession, InvalidTimeColumnError, get_dfs_engine

CONFIG = DFSConfig(engine="dfs2sql", engine_path=":memory:", agg_primitives=["count"])
KEYS = {"user_id": "user.user_id"}
TARGET = pd.DataFrame({"user_id": ["u1", "u2"]})


def _rdb_with_user_ids(user_ids):
    return create_rdb(
        tables={"user": pd.DataFrame({"user_id": user_ids, "age": [10.0, 20.0, 30.0]})},
        primary_keys={"user": "user_id"},
    )


@pytest.mark.parametrize(
    "user_ids, match",
    [
        (["u1", "u1", "u2"], r"Index column must be unique: primary key 'user\.user_id' has duplicate values, e\.g\. 'u1'"),
        (["u1", None, "u2"], r"Index contains null values: primary key 'user\.user_id'"),
    ],
)
def test_invalid_primary_key_raises(user_ids, match):
    rdb = _rdb_with_user_ids(user_ids)
    with pytest.raises(IndexError, match=match):
        compute_dfs_features(rdb, TARGET.copy(), KEYS, config=CONFIG)
    with DFSSession(rdb, KEYS, config=CONFIG) as session:
        with pytest.raises(IndexError, match=match):
            session.compute(TARGET.copy())


def test_valid_primary_key_passes():
    out = compute_dfs_features(_rdb_with_user_ids(["u1", "u2", "u3"]), TARGET.copy(), KEYS, config=CONFIG)
    assert out["d1__user.age"].tolist() == [10.0, 20.0]


def _event_rdb(column, values):
    """RDB whose event.<column> holds ``values``.

    ``column`` is "amount" (float), "timestamp" (the time column) or "d" (a
    datetime column that is not the time column).
    """
    events = {
        "event_id": ["e1", "e2", "e3"],
        "user_id": ["u1", "u1", "u2"],
        "amount": ["1.5", "2.5", "3.5"],
        "timestamp": ["2020-01-01 10:00", "2020-01-02 10:00", "2020-01-03 10:00"],
        "d": ["2019-01-01", "2019-01-02", "2019-01-03"],
    }
    events[column] = values
    return create_rdb(
        tables={
            "user": pd.DataFrame({"user_id": ["u1", "u2", "u3"], "age": [10.0, 20.0, 30.0]}),
            "event": pd.DataFrame(events),
        },
        primary_keys={"user": "user_id", "event": "event_id"},
        foreign_keys=[("event", "user_id", "user", "user_id")],
        time_columns={"event": "timestamp"},
        type_hints={"event": {"amount": "float", "d": "datetime"}},
    )


PARITY_CASES = {
    "valid": lambda: _event_rdb("amount", ["1.5", "2.5", "3.5"]),
    "float, unparseable": lambda: _event_rdb("amount", ["1.5", "x", "3.5"]),
    "float, thousands separator": lambda: _event_rdb("amount", ["1.5", "1,000", "3.5"]),
    "float, missing-value strings": lambda: _event_rdb("amount", ["nan", "None", ""]),
    "float, None": lambda: _event_rdb("amount", ["1.5", None, "3.5"]),
    "float, padded and inf": lambda: _event_rdb("amount", [" 1.5 ", "inf", "3.5"]),
    "float, underscore separator": lambda: _event_rdb("amount", ["1.5", "1_000", "3.5"]),
    "time, unparseable": lambda: _event_rdb("timestamp", ["2020-01-01 10:00", "not a date", "2020-01-03 10:00"]),
    "time, mixed formats": lambda: _event_rdb("timestamp", ["2020-01-01 10:00", "2020-01-02", "2020-01-03 08:30"]),
    "time, missing values": lambda: _event_rdb("timestamp", ["2020-01-01 10:00", None, "NaT"]),
    "time, 'None' string": lambda: _event_rdb("timestamp", ["2020-01-01 10:00", "None", "2020-01-03 10:00"]),
    # Woodwork pivots two-digit years differently from pandas (1970, not 2070).
    "time, two-digit years": lambda: _event_rdb("timestamp", ["01/01/70", "06/01/70", "01/03/70"]),
    # Woodwork infers month-first from a sample; pandas would read the first value day-first.
    "time, ambiguous day/month": lambda: _event_rdb(
        "timestamp", ["13/01/2020 10:00", "01/02/2020 10:00", "02/03/2020 10:00"]
    ),
    "non-time datetime, unparseable": lambda: _event_rdb("d", ["2019-01-01", "garbage", "2019-01-03"]),
    "non-time datetime, mixed formats": lambda: _event_rdb("d", ["2019-01-01 10:00", "2019-01-02", "2019-01-03"]),
    "time, tz-aware": lambda: _event_rdb(
        "timestamp", pd.to_datetime(["2020-01-01 10:00", "2020-01-02 10:00", "2020-01-03 10:00"]).tz_localize("UTC")
    ),
    "time, mixed UTC offsets": lambda: _event_rdb(
        "timestamp", ["2020-03-28T00:00+01:00", "2020-03-30T00:00+02:00", "2020-03-31T00:00+02:00"]
    ),
    "duplicate primary key": lambda: _rdb_with_user_ids(["u1", "u1", "u2"]),
    "null primary key": lambda: _rdb_with_user_ids(["u1", None, "u2"]),
}


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


def _compute(case, engine):
    target = pd.DataFrame({"user_id": ["u1", "u2"], "cut": pd.to_datetime(["2020-04-01"] * 2)})
    config = DFSConfig(engine=engine, engine_path=":memory:", agg_primitives=["count", "max"])
    return compute_dfs_features(PARITY_CASES[case](), target, KEYS, cutoff_time_column="cut", config=config)


@pytest.mark.parametrize("case", sorted(PARITY_CASES))
def test_dfs2sql_matches_full_frame_planning(case):
    # Reference: building the EntitySet from full frames, as the featuretools engine
    # (and dfs2sql on master) does. Each side gets its own RDB, because full-frame
    # planning converts the frames in place.
    full_rdb = PARITY_CASES[case]().canonicalize_key_types()
    full_frame_fails = _raises(
        lambda: get_dfs_engine("featuretools", DFSConfig(engine="featuretools"))._build_entity_set_from_rdb(full_rdb)
    )
    dfs2sql_fails = _raises(lambda: _compute(case, "dfs2sql"))
    assert dfs2sql_fails == full_frame_fails, (
        f"full-frame planning {'raises' if full_frame_fails else 'succeeds'}, "
        f"dfs2sql {'raises' if dfs2sql_fails else 'succeeds'}"
    )
    if not dfs2sql_fails:
        # Same values as the featuretools engine, which computes from full frames
        # converted by woodwork (string keys avoid its separate numeric-key bug).
        pd.testing.assert_frame_equal(
            _compute(case, "dfs2sql"), _compute(case, "featuretools"), check_dtype=False
        )


def test_unconvertible_value_error_names_column():
    with pytest.raises(ValueError, match=r"Column 'event\.amount' \(float\) has values that cannot be converted to float, e\.g\. 'x'"):
        compute_dfs_features(
            PARITY_CASES["float, unparseable"](), TARGET.copy(), KEYS, config=CONFIG
        )


@pytest.mark.parametrize(
    "last_value",
    ["A1B 2C3", {"b": "x"}],
    ids=["number-like strings then text", "dicts with different keys"],
)
def test_text_column_with_mixed_types(last_value):
    # DuckDB guesses an object column's type from its first rows; a different value
    # after that sample used to fail the whole load, though no feature reads text.
    n = 3000
    first = [10000 + i for i in range(n - 1)] if isinstance(last_value, str) else [{"a": i} for i in range(n - 1)]
    users = pd.DataFrame(
        {"uid": [f"u{i}" for i in range(n)], "x": first + [last_value], "age": [float(i) for i in range(n)]}
    )
    rdb = create_rdb(tables={"users": users}, primary_keys={"users": "uid"}, type_hints={"users": {"x": "text"}})
    target = pd.DataFrame({"uid": ["u1", "u2"]})
    out = compute_dfs_features(rdb, target, {"uid": "users.uid"}, config=CONFIG)
    assert out["d1__users.age"].tolist() == [1.0, 2.0]
    with DFSSession(rdb, {"uid": "users.uid"}, config=CONFIG) as session:
        assert session.compute(target)["d1__users.age"].tolist() == [1.0, 2.0]


def test_unhashable_values_in_float_column_give_readable_error():
    rdb = create_rdb(
        tables={"user": pd.DataFrame({"user_id": ["u1", "u2", "u3"], "f": [[1.0], [2.0], [3.0]]})},
        primary_keys={"user": "user_id"},
        type_hints={"user": {"f": "float"}},
    )
    with pytest.raises(ValueError, match=r"Column 'user\.f' \(float\) has values that cannot be converted to float, e\.g\. \[1\.0\]"):
        compute_dfs_features(rdb, TARGET.copy(), KEYS, config=CONFIG)


@pytest.mark.parametrize("where", ["RDB time column", "cutoff"])
def test_invalid_time_values_raise_type_and_value_error(where):
    # Woodwork raised TypeError for these on master; the error is also a ValueError.
    if where == "RDB time column":
        rdb = PARITY_CASES["time, unparseable"]()
        cut = pd.to_datetime(["2020-04-01", "2020-04-01"])
    else:
        rdb = PARITY_CASES["valid"]()
        cut = ["2020-04-01", "not a date"]
    target = pd.DataFrame({"user_id": ["u1", "u2"], "cut": cut})
    with pytest.raises(InvalidTimeColumnError) as info:
        compute_dfs_features(rdb, target, KEYS, cutoff_time_column="cut", config=CONFIG)
    assert isinstance(info.value, TypeError) and isinstance(info.value, ValueError)
