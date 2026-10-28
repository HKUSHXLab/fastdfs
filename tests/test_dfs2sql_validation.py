"""dfs2sql validates the RDB itself, since planning only sees 0-row frames.

Full-frame planning (still used by the featuretools engine) validated the data as a
side effect; dfs2sql must accept and reject the same RDBs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fastdfs.api import compute_dfs_features, create_rdb
from fastdfs.dfs import DFSConfig, DFSSession, get_dfs_engine

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
    """RDB whose event.<column> holds ``values``; ``column`` is "amount" or "timestamp"."""
    events = {
        "event_id": ["e1", "e2", "e3"],
        "user_id": ["u1", "u1", "u2"],
        "amount": ["1.5", "2.5", "3.5"],
        "timestamp": ["2020-01-01 10:00", "2020-01-02 10:00", "2020-01-03 10:00"],
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
        type_hints={"event": {"amount": "float"}},
    )


PARITY_CASES = {
    "valid": lambda: _event_rdb("amount", ["1.5", "2.5", "3.5"]),
    "float, unparseable": lambda: _event_rdb("amount", ["1.5", "x", "3.5"]),
    "float, thousands separator": lambda: _event_rdb("amount", ["1.5", "1,000", "3.5"]),
    "float, missing-value strings": lambda: _event_rdb("amount", ["nan", "None", ""]),
    "float, None": lambda: _event_rdb("amount", ["1.5", None, "3.5"]),
    "float, padded and inf": lambda: _event_rdb("amount", [" 1.5 ", "inf", "3.5"]),
    "time, unparseable": lambda: _event_rdb("timestamp", ["2020-01-01 10:00", "not a date", "2020-01-03 10:00"]),
    "time, mixed formats": lambda: _event_rdb("timestamp", ["2020-01-01 10:00", "2020-01-02", "2020-01-03 08:30"]),
    "time, missing values": lambda: _event_rdb("timestamp", ["2020-01-01 10:00", None, "NaT"]),
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


@pytest.mark.parametrize("case", sorted(PARITY_CASES))
def test_dfs2sql_rejects_what_full_frame_planning_rejects(case):
    # Reference: building the EntitySet from full frames, as the featuretools engine
    # (and dfs2sql on master) does. Each side gets its own RDB, because full-frame
    # planning converts the frames in place.
    full_rdb = PARITY_CASES[case]().canonicalize_key_types()
    full_frame_fails = _raises(
        lambda: get_dfs_engine("featuretools", DFSConfig(engine="featuretools"))._build_entity_set_from_rdb(full_rdb)
    )
    target = pd.DataFrame({"user_id": ["u1", "u2"], "cut": pd.to_datetime(["2020-04-01"] * 2)})
    dfs2sql_fails = _raises(
        lambda: compute_dfs_features(
            PARITY_CASES[case](), target, KEYS, cutoff_time_column="cut", config=CONFIG
        )
    )
    assert dfs2sql_fails == full_frame_fails, (
        f"full-frame planning {'raises' if full_frame_fails else 'succeeds'}, "
        f"dfs2sql {'raises' if dfs2sql_fails else 'succeeds'}"
    )


def test_unconvertible_value_error_names_column():
    with pytest.raises(ValueError, match=r"Column 'event\.amount' \(float\) has values that cannot be converted, e\.g\. 'x'"):
        compute_dfs_features(
            PARITY_CASES["float, unparseable"](), TARGET.copy(), KEYS, config=CONFIG
        )
