"""Shared pytest fixtures."""

import pandas as pd
import pytest

from fastdfs.api import create_rdb


@pytest.fixture
def tiny_rdb():
    users = pd.DataFrame({"user_id": [1, 2, 3], "age": [25.0, 30.0, 35.0]})
    events = pd.DataFrame(
        {
            "event_id": [10, 11, 12, 13],
            "user_id": [1, 1, 2, 3],
            "amount": [5.0, 7.0, 9.0, 11.0],
            "timestamp": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
            ),
        }
    )
    return create_rdb(
        tables={"user": users, "event": events},
        primary_keys={"user": "user_id", "event": "event_id"},
        foreign_keys=[("event", "user_id", "user", "user_id")],
        time_columns={"event": "timestamp"},
    )
