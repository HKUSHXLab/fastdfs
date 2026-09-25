"""Merge feature queries that differ only in their SELECT column lists.

Two queries merge when they match after the column lists are removed from the
top-level SELECT and from every joined subquery. CTEs, FROM, join conditions,
WHERE, GROUP BY and HAVING must match exactly, so the merged query returns the
same rows and only adds columns. This relies on FeatureBlock pairing every
aggregate with a GROUP BY on the join key; merge_queries does not check it.
"""

from collections import defaultdict
from typing import Dict, List, TypeVar

from sqlglot.expressions import Expression, Join, Select, Subquery

QueryNode = TypeVar("QueryNode", bound=Expression)


def merge_queries(queries: List[Select]) -> List[Select]:
    """Group queries by shape and merge each group's projections."""
    groups: Dict[str, List[Select]] = defaultdict(list)
    for query in queries:
        groups[_shape(query).sql()].append(query)
    return [_merge(group) for group in groups.values()]


def _shape(query: QueryNode) -> QueryNode:
    """Copy the query with the column lists removed along its join path."""
    result = query.copy()
    if isinstance(result, Select):
        result.set("expressions", [])
        if result.args.get("joins"):
            result.set("joins", [_shape(join) for join in result.args["joins"]])
    elif isinstance(result, (Join, Subquery)):
        result.set("this", _shape(result.this))
    return result


def _merge(queries: List[QueryNode]) -> QueryNode:
    result = queries[0].copy()
    if isinstance(result, Select):
        expressions: Dict[str, Expression] = {}
        for query in queries:
            for expression in query.expressions:
                expressions.setdefault(expression.sql(), expression.copy())
        result.set("expressions", list(expressions.values()))
        if result.args.get("joins"):
            result.set(
                "joins",
                [
                    _merge([query.args["joins"][i] for query in queries])
                    for i in range(len(result.args["joins"]))
                ],
            )
    elif isinstance(result, (Join, Subquery)):
        result.set("this", _merge([query.this for query in queries]))
    return result
