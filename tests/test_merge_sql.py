"""Check query merging without changing row selection or input expressions."""

import duckdb
import pytest
from sqlglot import parse_one

from fastdfs.dfs.merge_sql import merge_queries


def test_merge_projections() -> None:
    queries = [
        parse_one(
            "WITH t AS (SELECT * FROM (VALUES (1, 2), (1, 4)) AS x(k,v)) "
            f"SELECT k, {op}(v) AS {op}_v FROM t GROUP BY k"
        )
        for op in ("MIN", "MAX")
    ]
    original = [query.sql() for query in queries]
    merged = merge_queries(queries)
    assert len(merged) == 1
    with duckdb.connect() as db:
        assert db.sql(merged[0].sql()).fetchall() == [(1, 2, 4)]
    assert merged[0].named_selects == ["k", "MIN_v", "MAX_v"]
    assert [query.sql() for query in queries] == original
    assert merge_queries([]) == []


@pytest.mark.parametrize(
    "left,right",
    [
        ("WITH t AS (SELECT 1 AS k)", "WITH t AS (SELECT 2 AS k)"),
        ("", "WITH unused AS (SELECT 1)"),
    ],
)
def test_different_ctes_stay_separate(left: str, right: str) -> None:
    queries = [parse_one(f"{prefix} SELECT MIN(k) FROM t") for prefix in (left, right)]
    assert len(merge_queries(queries)) == 2


@pytest.mark.parametrize("clause", ["WHERE v > 2", "HAVING MIN(v) > 2"])
def test_different_filters_stay_separate(clause: str) -> None:
    sql = "SELECT k, MIN(v) AS min_v FROM t"
    suffix = (
        f"{clause} GROUP BY k" if clause.startswith("WHERE") else f"GROUP BY k {clause}"
    )
    assert (
        len(
            merge_queries(
                [parse_one(sql + " GROUP BY k"), parse_one(sql + " " + suffix)]
            )
        )
        == 2
    )


def test_nested_join_projections() -> None:
    queries = [
        parse_one(
            "SELECT t.k, s." + op + "_v FROM (SELECT 1 AS k) AS t LEFT JOIN "
            "(SELECT k, " + op + "(v) AS " + op + "_v FROM "
            "(VALUES (1, 2), (1, 4)) AS x(k,v) GROUP BY k) AS s ON t.k=s.k"
        )
        for op in ("MIN", "MAX")
    ]
    merged = merge_queries(queries)
    assert len(merged) == 1
    with duckdb.connect() as db:
        assert db.sql(merged[0].sql()).fetchall() == [(1, 2, 4)]
