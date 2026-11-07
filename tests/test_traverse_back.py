"""
Tests for traverse-back feature generation.

Featuretools' DFS refuses to generate features whose synthesis path revisits a
table already on the path (its guard keys on table name, not on the relationship
taken). When an entity table A is connected to a relation table B that has two or
more foreign keys into A (canonically a graph: A=nodes, B=edges with
edges.src_id->nodes and edges.dst_id->nodes), the path A -> B -> A -> B is dropped
even though each hop lands on a different set of rows (one->many then many->one).
These "traverse-back" features encode valid multi-hop / neighborhood aggregations.

fastdfs reconstructs the missing features manually. These tests pin down the
contract:
  - depth gate: traverse-back features appear only at max_depth >= 4 and never
    exceed max_depth (per featuretools' own get_depth()),
  - the traverse_back config flag disables them,
  - single-FK schemas produce none (degenerate),
  - outer aggregation primitives are type-matched (no numeric primitive over a
    categorical inner aggregate),
  - both engines (featuretools, dfs2sql) compute identical values.

See dev_logs/022_TRAVERSE_BACK_FEATURES_PLAN.md for the design.
"""

import re

import numpy as np
import pandas as pd
import pytest

from fastdfs.api import create_rdb, compute_dfs_features
from fastdfs.dfs import DFSConfig, get_dfs_engine


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #

GRAPH_KEYS = {"src_id": "nodes.node_id", "dst_id": "nodes.node_id"}


@pytest.fixture
def graph_rdb():
    """A directed graph: nodes (entity, PK) and edges (relation, two FKs to nodes)."""
    nodes_df = pd.DataFrame({
        "node_id": ["n0", "n1", "n2", "n3", "n4"],
        "node_value": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    edges_df = pd.DataFrame({
        "edge_id": ["e0", "e1", "e2", "e3", "e4", "e5", "e6"],
        "src_id": ["n0", "n0", "n1", "n1", "n2", "n3", "n4"],
        "dst_id": ["n1", "n2", "n2", "n3", "n3", "n4", "n0"],
        "edge_weight": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
    })
    return create_rdb(
        tables={"nodes": nodes_df, "edges": edges_df},
        name="graph_rdb",
        primary_keys={"nodes": "node_id", "edges": "edge_id"},
        foreign_keys=[
            ("edges", "src_id", "nodes", "node_id"),
            ("edges", "dst_id", "nodes", "node_id"),
        ],
    )


@pytest.fixture
def graph_target():
    return pd.DataFrame({
        "src_id": ["n0", "n1", "n2", "n3"],
        "dst_id": ["n1", "n3", "n3", "n4"],
    })


@pytest.fixture
def categorical_graph_rdb():
    """Graph with both a numeric (edge_weight) and a categorical (etype) edge attr."""
    nodes_df = pd.DataFrame({
        "node_id": ["n0", "n1", "n2"],
        "node_value": [1.0, 2.0, 3.0],
    })
    edges_df = pd.DataFrame({
        "edge_id": ["e0", "e1", "e2", "e3"],
        "src_id": ["n0", "n0", "n1", "n2"],
        "dst_id": ["n1", "n2", "n2", "n0"],
        "edge_weight": [0.5, 1.0, 1.5, 2.0],
        "etype": ["x", "y", "x", "y"],
    })
    return create_rdb(
        tables={"nodes": nodes_df, "edges": edges_df},
        name="cat_graph",
        primary_keys={"nodes": "node_id", "edges": "edge_id"},
        foreign_keys=[
            ("edges", "src_id", "nodes", "node_id"),
            ("edges", "dst_id", "nodes", "node_id"),
        ],
        type_hints={"edges": {"etype": "category"}},
    )


@pytest.fixture
def single_fk_rdb():
    """Entity table (users) with a relation table (posts) that has ONE FK to it."""
    users_df = pd.DataFrame({
        "user_id": ["u0", "u1", "u2"],
        "uval": [1.0, 2.0, 3.0],
    })
    posts_df = pd.DataFrame({
        "post_id": ["p0", "p1", "p2", "p3"],
        "author_id": ["u0", "u0", "u1", "u2"],
        "pval": [1.0, 2.0, 3.0, 4.0],
    })
    return create_rdb(
        tables={"users": users_df, "posts": posts_df},
        name="single_fk",
        primary_keys={"users": "user_id", "posts": "post_id"},
        foreign_keys=[("posts", "author_id", "users", "user_id")],
    )


def _feature_columns(result, target_df):
    """Generated feature columns (excluding the original target key columns)."""
    return [c for c in result.columns if c not in set(target_df.columns)]


def _traverse_back_columns(cols):
    """Heuristic: a traverse-back feature mentions the same table name twice."""
    return [c for c in cols if c.count("edges") >= 2 or c.count("nodes") >= 2]


# --------------------------------------------------------------------------- #
# Detection / opportunity tests
# --------------------------------------------------------------------------- #

class TestTraverseBackDetection:
    """The (A, B) pair detection underlying traverse-back generation."""

    @pytest.mark.parametrize("engine", ["featuretools", "dfs2sql"])
    def test_graph_generates_traverse_back_at_depth_4(
        self, graph_rdb, graph_target, engine
    ):
        """A two-FK graph schema must yield traverse-back features at depth 4."""
        result = compute_dfs_features(
            rdb=graph_rdb,
            target_dataframe=graph_target.copy(),
            key_mappings=GRAPH_KEYS,
            config=DFSConfig(engine=engine, max_depth=4),
        )
        tb = _traverse_back_columns(_feature_columns(result, graph_target))
        assert len(tb) > 0

    def test_single_fk_generates_no_traverse_back(self, single_fk_rdb):
        """A relation table with a single FK to the entity is degenerate -> none."""
        target = pd.DataFrame({"user_id": ["u0", "u1", "u2"]})
        result = compute_dfs_features(
            rdb=single_fk_rdb,
            target_dataframe=target.copy(),
            key_mappings={"user_id": "users.user_id"},
            config=DFSConfig(engine="featuretools", max_depth=4),
        )
        cols = _feature_columns(result, target)
        tb = [c for c in cols if c.count("posts") >= 2 or c.count("users") >= 2]
        assert tb == []


# --------------------------------------------------------------------------- #
# Depth-gate tests
# --------------------------------------------------------------------------- #

class TestTraverseBackDepthGate:
    """max_depth controls whether and how many traverse-back rounds appear."""

    @pytest.mark.parametrize("max_depth", [1, 2, 3])
    def test_no_traverse_back_below_depth_4(self, graph_rdb, graph_target, max_depth):
        """One round costs +2 depth over standard (depth-2) features => need >=4."""
        result = compute_dfs_features(
            rdb=graph_rdb,
            target_dataframe=graph_target.copy(),
            key_mappings=GRAPH_KEYS,
            config=DFSConfig(engine="featuretools", max_depth=max_depth),
        )
        tb = _traverse_back_columns(_feature_columns(result, graph_target))
        assert tb == []

    @pytest.mark.parametrize("max_depth", [4, 5, 6])
    def test_traverse_back_present_at_depth_4_plus(
        self, graph_rdb, graph_target, max_depth
    ):
        result = compute_dfs_features(
            rdb=graph_rdb,
            target_dataframe=graph_target.copy(),
            key_mappings=GRAPH_KEYS,
            config=DFSConfig(engine="featuretools", max_depth=max_depth),
        )
        tb = _traverse_back_columns(_feature_columns(result, graph_target))
        assert len(tb) > 0

    @pytest.mark.parametrize("max_depth", [4, 5, 6])
    def test_no_feature_exceeds_max_depth(self, graph_rdb, graph_target, max_depth):
        """Authoritative gate: featuretools' get_depth() never exceeds max_depth."""
        config = DFSConfig(engine="featuretools", max_depth=max_depth)
        engine = get_dfs_engine("featuretools", config)
        target = graph_target.copy()
        target["__target_index__"] = np.arange(len(target))
        feats = engine.prepare_features(
            graph_rdb, target, GRAPH_KEYS, None, config
        )
        over = [f.get_name() for f in feats if f.get_depth() > max_depth]
        assert over == [], f"features exceed max_depth={max_depth}: {over[:3]}"

    def test_rounds_controlled_by_max_revisits(self, graph_rdb, graph_target):
        """Traverse-back rounds are gated by max_revisits, not by max_depth alone.

        One round (A->B->A->B, depth 4) needs max_revisits>=1; a second round
        (depth 6) needs both max_depth>=6 AND max_revisits>=2. Increasing max_depth
        without increasing max_revisits must not silently add deeper rounds.
        """
        def max_observed_depth(max_depth, max_revisits):
            config = DFSConfig(
                engine="featuretools", max_depth=max_depth, max_revisits=max_revisits
            )
            engine = get_dfs_engine("featuretools", config)
            target = graph_target.copy()
            target["__target_index__"] = np.arange(len(target))
            feats = engine.prepare_features(
                graph_rdb, target, GRAPH_KEYS, None, config
            )
            return max(f.get_depth() for f in feats)

        # One round: depth 4 reached, and raising max_depth alone does not deepen.
        assert max_observed_depth(4, 1) == 4
        assert max_observed_depth(5, 1) == 4
        assert max_observed_depth(6, 1) == 4  # still one round despite the budget
        # Two rounds require max_revisits>=2; depth then follows max_depth.
        assert max_observed_depth(6, 2) == 6
        assert max_observed_depth(5, 2) == 5  # depth budget caps the second round


# --------------------------------------------------------------------------- #
# Config switch
# --------------------------------------------------------------------------- #

class TestTraverseBackConfig:

    def test_flag_disables_traverse_back(self, graph_rdb, graph_target):
        result = compute_dfs_features(
            rdb=graph_rdb,
            target_dataframe=graph_target.copy(),
            key_mappings=GRAPH_KEYS,
            config=DFSConfig(engine="featuretools", max_depth=4, traverse_back=False),
        )
        tb = _traverse_back_columns(_feature_columns(result, graph_target))
        assert tb == []

    def test_flag_default_is_true(self):
        assert DFSConfig().traverse_back is True


# --------------------------------------------------------------------------- #
# Type matching
# --------------------------------------------------------------------------- #

class TestTraverseBackTypeMatching:
    """Outer aggregation primitives must match the inner aggregate's output type."""

    def test_no_numeric_primitive_over_categorical_inner(
        self, categorical_graph_rdb
    ):
        target = pd.DataFrame({"src_id": ["n0", "n1"], "dst_id": ["n1", "n2"]})
        result = compute_dfs_features(
            rdb=categorical_graph_rdb,
            target_dataframe=target.copy(),
            key_mappings={"src_id": "nodes.node_id", "dst_id": "nodes.node_id"},
            config=DFSConfig(
                engine="featuretools",
                max_depth=4,
                agg_primitives=["mean", "max", "min", "std", "count", "mode"],
            ),
        )
        cols = _feature_columns(result, target)
        tb = _traverse_back_columns(cols)
        # Illegal: a numeric primitive (MEAN/MAX/MIN/STD) wrapping a categorical
        # inner aggregate (MODE).
        illegal = [
            c for c in tb
            if re.search(
                r"(MEAN|MAX|MIN|STD)\(edges\[[^]]+\]\.nodes\[[^]]+\]\.MODE\(", c
            )
        ]
        assert illegal == [], f"numeric-over-categorical leaked: {illegal[:3]}"

    def test_categorical_chains_are_generated(self, categorical_graph_rdb):
        """MODE(...MODE(...)) chains (category->category) should be produced."""
        target = pd.DataFrame({"src_id": ["n0", "n1"], "dst_id": ["n1", "n2"]})
        result = compute_dfs_features(
            rdb=categorical_graph_rdb,
            target_dataframe=target.copy(),
            key_mappings={"src_id": "nodes.node_id", "dst_id": "nodes.node_id"},
            config=DFSConfig(
                engine="featuretools",
                max_depth=4,
                agg_primitives=["mean", "count", "mode"],
            ),
        )
        tb = _traverse_back_columns(_feature_columns(result, target))
        mode_chains = [c for c in tb if c.count("MODE") >= 2]
        assert len(mode_chains) > 0


# --------------------------------------------------------------------------- #
# Value correctness / engine agreement
# --------------------------------------------------------------------------- #

class TestTraverseBackValues:

    def test_engines_agree_on_traverse_back_values(self, graph_rdb, graph_target):
        res_ft = compute_dfs_features(
            rdb=graph_rdb, target_dataframe=graph_target.copy(),
            key_mappings=GRAPH_KEYS,
            config=DFSConfig(engine="featuretools", max_depth=4),
        )
        res_sql = compute_dfs_features(
            rdb=graph_rdb, target_dataframe=graph_target.copy(),
            key_mappings=GRAPH_KEYS,
            config=DFSConfig(engine="dfs2sql", max_depth=4),
        )
        common = sorted(
            set(_feature_columns(res_ft, graph_target))
            & set(_feature_columns(res_sql, graph_target))
        )
        tb_common = _traverse_back_columns(common)
        assert len(tb_common) > 0
        for c in tb_common:
            a = pd.to_numeric(res_ft[c].reset_index(drop=True), errors="coerce")
            b = pd.to_numeric(res_sql[c].reset_index(drop=True), errors="coerce")
            assert np.allclose(
                a.to_numpy(float), b.to_numpy(float), atol=1e-6, equal_nan=True
            ), f"engine mismatch for {c}: ft={a.tolist()} sql={b.tolist()}"

    def test_known_two_hop_value(self, graph_rdb, graph_target):
        """Hand-checked value for a specific 2-hop mean feature.

        Feature: nodes[src_id].MEAN(edges[src_id].nodes[dst_id].MEAN(edges[src_id].edge_weight))
        For target row 0 (src=n0):
          n0's outgoing edges: e0(->n1, w .5), e1(->n2, w 1.0)
          dst nodes: n1, n2
          n1 outgoing-edge mean weight: e2(1.5), e3(2.0) -> 1.75
          n2 outgoing-edge mean weight: e4(2.5)          -> 2.5
          mean over n0's edges of (dst node's mean) = (1.75 + 2.5)/2 = 2.125
        """
        col = "nodes.MEAN(edges[src_id].nodes[dst_id].MEAN(edges[src_id].edge_weight))"
        result = compute_dfs_features(
            rdb=graph_rdb, target_dataframe=graph_target.copy(),
            key_mappings=GRAPH_KEYS,
            config=DFSConfig(engine="featuretools", max_depth=4),
        )
        assert col in result.columns, f"expected feature missing: {col}"
        assert result[col].iloc[0] == pytest.approx(2.125)


# --------------------------------------------------------------------------- #
# max_features interaction
# --------------------------------------------------------------------------- #

class TestTraverseBackMaxFeatures:

    def test_max_features_cap_respected(self, graph_rdb, graph_target):
        result = compute_dfs_features(
            rdb=graph_rdb, target_dataframe=graph_target.copy(),
            key_mappings=GRAPH_KEYS,
            config=DFSConfig(engine="featuretools", max_depth=4, max_features=20),
        )
        assert len(_feature_columns(result, graph_target)) <= 20


# --------------------------------------------------------------------------- #
# Ground-truth value correctness
# --------------------------------------------------------------------------- #

# Raw frames used to derive ground-truth values independently of fastdfs.
_GT_NODES = pd.DataFrame({
    "node_id": ["n0", "n1", "n2", "n3", "n4"],
    "node_value": [1.0, 2.0, 3.0, 4.0, 5.0],
})
_GT_EDGES = pd.DataFrame({
    "edge_id": ["e0", "e1", "e2", "e3", "e4", "e5", "e6"],
    "src_id": ["n0", "n0", "n1", "n1", "n2", "n3", "n4"],
    "dst_id": ["n1", "n2", "n2", "n3", "n3", "n4", "n0"],
    "edge_weight": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
})


def _out_edges(node_id):
    """Edges whose src_id == node_id (the node's outgoing edges)."""
    return _GT_EDGES[_GT_EDGES.src_id == node_id]


def _in_edges(node_id):
    """Edges whose dst_id == node_id (the node's incoming edges)."""
    return _GT_EDGES[_GT_EDGES.dst_id == node_id]


def _brute_force_two_hop(target_keys, inner_via, hop_via, agg_inner, agg_outer):
    """Independently compute a 2-hop traverse-back feature in plain pandas.

    Mirrors:
        nodes.<agg_outer>( edges[hop_via].nodes[other].<agg_inner>(edges[inner_via].edge_weight) )

    - For each target row's node X (via src_id):
        * take X's edges along `hop_via` (outgoing if 'src', incoming if 'dst'),
        * for each such edge follow `other` endpoint to node Y,
        * inner value of Y = agg_inner over Y's edges along `inner_via`,
        * outer value = agg_outer over those inner values.

    Only the combinations actually asserted below are supported.
    """
    edges_of = _out_edges if hop_via == "src" else _in_edges
    other_col = "dst_id" if hop_via == "src" else "src_id"
    inner_edges_of = _out_edges if inner_via == "src" else _in_edges

    def inner_value(node_id):
        e = inner_edges_of(node_id)
        if agg_inner == "MEAN":
            return e["edge_weight"].mean() if len(e) else np.nan
        if agg_inner == "COUNT":
            return float(len(e))
        if agg_inner == "MAX":
            return e["edge_weight"].max() if len(e) else np.nan
        raise ValueError(agg_inner)

    out = []
    for node_x in target_keys:
        hop = edges_of(node_x)
        inner_vals = [inner_value(y) for y in hop[other_col]]
        inner_vals = [v for v in inner_vals if not pd.isna(v)]
        if not inner_vals:
            out.append(np.nan)
            continue
        if agg_outer == "MEAN":
            out.append(float(np.mean(inner_vals)))
        elif agg_outer == "MAX":
            out.append(float(np.max(inner_vals)))
        elif agg_outer == "MIN":
            out.append(float(np.min(inner_vals)))
        else:
            raise ValueError(agg_outer)
    return out


class TestTraverseBackGroundTruth:
    """Assert computed traverse-back values against an independent pandas baseline."""

    @pytest.fixture
    def rdb(self):
        return create_rdb(
            tables={"nodes": _GT_NODES.copy(), "edges": _GT_EDGES.copy()},
            name="gt_graph",
            primary_keys={"nodes": "node_id", "edges": "edge_id"},
            foreign_keys=[
                ("edges", "src_id", "nodes", "node_id"),
                ("edges", "dst_id", "nodes", "node_id"),
            ],
        )

    @pytest.fixture
    def target(self):
        return pd.DataFrame({
            "src_id": ["n0", "n1", "n2", "n3"],
            "dst_id": ["n1", "n3", "n3", "n4"],
        })

    # (feature column name, hop_via, inner_via, agg_inner, agg_outer)
    CASES = [
        (
            "nodes.MEAN(edges[src_id].nodes[dst_id].MEAN(edges[src_id].edge_weight))",
            "src", "src", "MEAN", "MEAN",
        ),
        (
            "nodes.MEAN(edges[src_id].nodes[dst_id].COUNT(edges[src_id]))",
            "src", "src", "COUNT", "MEAN",
        ),
        (
            "nodes.MAX(edges[src_id].nodes[dst_id].MEAN(edges[src_id].edge_weight))",
            "src", "src", "MEAN", "MAX",
        ),
        (
            "nodes.MEAN(edges[dst_id].nodes[src_id].MEAN(edges[src_id].edge_weight))",
            "dst", "src", "MEAN", "MEAN",
        ),
    ]

    @pytest.mark.parametrize("engine", ["featuretools", "dfs2sql"])
    @pytest.mark.parametrize("col,hop_via,inner_via,agg_inner,agg_outer", CASES)
    def test_value_matches_brute_force(
        self, rdb, target, engine, col, hop_via, inner_via, agg_inner, agg_outer
    ):
        result = compute_dfs_features(
            rdb=rdb, target_dataframe=target.copy(), key_mappings=GRAPH_KEYS,
            config=DFSConfig(engine=engine, max_depth=4),
        )
        assert col in result.columns, f"expected feature missing: {col}"
        expected = _brute_force_two_hop(
            target["src_id"].tolist(), inner_via, hop_via, agg_inner, agg_outer
        )
        got = pd.to_numeric(result[col].reset_index(drop=True), errors="coerce").tolist()
        assert np.allclose(got, expected, atol=1e-6, equal_nan=True), (
            f"{engine} {col}\n  expected={expected}\n  got     ={got}"
        )


# --------------------------------------------------------------------------- #
# The pattern embedded inside a larger / more complex RDB
# --------------------------------------------------------------------------- #

class TestTraverseBackEmbeddedInComplexRDB:
    """The (entity, multi-FK-relation) pattern surrounded by unrelated structure."""

    @pytest.fixture
    def complex_rdb(self):
        """
        Schema:
            authors (entity)         <-- collaborations (relation, 2 FKs: author_a, author_b)
            authors.affiliation_id   --> institutions (extra parent, a chain hop)
            authors                  <-- papers (extra child, unrelated to the graph pattern)

        The graph pattern is (authors, collaborations). Everything else is noise
        that must not be swept into traverse-back generation, but must still get
        its own standard / native-deep features.
        """
        institutions = pd.DataFrame({
            "inst_id": ["i0", "i1"],
            "inst_rank": [10.0, 20.0],
        })
        authors = pd.DataFrame({
            "author_id": ["a0", "a1", "a2", "a3"],
            "affiliation_id": ["i0", "i0", "i1", "i1"],
            "h_index": [5.0, 8.0, 3.0, 12.0],
        })
        # collaborations: a relation table with TWO FKs into authors
        collaborations = pd.DataFrame({
            "collab_id": ["c0", "c1", "c2", "c3", "c4"],
            "author_a": ["a0", "a0", "a1", "a2", "a3"],
            "author_b": ["a1", "a2", "a2", "a3", "a0"],
            "n_papers": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        # papers: an unrelated child of authors (single FK)
        papers = pd.DataFrame({
            "paper_id": ["p0", "p1", "p2", "p3"],
            "lead_author": ["a0", "a1", "a1", "a3"],
            "citations": [100.0, 50.0, 25.0, 200.0],
        })
        return create_rdb(
            tables={
                "institutions": institutions,
                "authors": authors,
                "collaborations": collaborations,
                "papers": papers,
            },
            name="research_rdb",
            primary_keys={
                "institutions": "inst_id",
                "authors": "author_id",
                "collaborations": "collab_id",
                "papers": "paper_id",
            },
            foreign_keys=[
                ("authors", "affiliation_id", "institutions", "inst_id"),
                ("collaborations", "author_a", "authors", "author_id"),
                ("collaborations", "author_b", "authors", "author_id"),
                ("papers", "lead_author", "authors", "author_id"),
            ],
        )

    @pytest.fixture
    def complex_target(self):
        return pd.DataFrame({"author_id": ["a0", "a1", "a2", "a3"]})

    def _cols(self, result):
        return [c for c in result.columns if c != "author_id"]

    @pytest.mark.parametrize("engine", ["featuretools", "dfs2sql"])
    def test_traverse_back_only_on_the_graph_pattern(
        self, complex_rdb, complex_target, engine
    ):
        result = compute_dfs_features(
            rdb=complex_rdb, target_dataframe=complex_target.copy(),
            key_mappings={"author_id": "authors.author_id"},
            config=DFSConfig(engine=engine, max_depth=4),
        )
        cols = self._cols(result)
        # Traverse-back must involve the collaborations relation twice (the only
        # >=2-FK relation in the schema).
        tb = [c for c in cols if c.count("collaborations") >= 2]
        assert len(tb) > 0, "expected traverse-back over collaborations"
        # papers has a single FK -> must never appear twice in a feature
        assert not any(c.count("papers") >= 2 for c in cols)

    def test_unrelated_features_still_present(self, complex_rdb, complex_target):
        """Embedding the pattern must not suppress ordinary features elsewhere."""
        result = compute_dfs_features(
            rdb=complex_rdb, target_dataframe=complex_target.copy(),
            key_mappings={"author_id": "authors.author_id"},
            config=DFSConfig(engine="featuretools", max_depth=4),
        )
        cols = set(self._cols(result))
        # direct attribute
        assert "authors.h_index" in cols
        # chain hop to institutions (native featuretools)
        assert any("institutions" in c for c in cols)
        # aggregation over the unrelated papers child
        assert any("papers" in c for c in cols)

    @pytest.mark.parametrize("engine", ["featuretools", "dfs2sql"])
    def test_embedded_value_matches_brute_force(
        self, complex_rdb, complex_target, engine
    ):
        """Hand-check one traverse-back value in the embedded schema."""
        # collaborations are undirected-ish but stored with author_a / author_b.
        collabs = pd.DataFrame({
            "author_a": ["a0", "a0", "a1", "a2", "a3"],
            "author_b": ["a1", "a2", "a2", "a3", "a0"],
            "n_papers": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        def collabs_as_a(x):
            return collabs[collabs.author_a == x]

        # inner: for node Y, MEAN(n_papers) over collaborations where Y is author_a
        def inner(y):
            e = collabs_as_a(y)
            return e["n_papers"].mean() if len(e) else np.nan

        # feature: authors.MEAN( collaborations[author_a].authors[author_b].MEAN(collaborations[author_a].n_papers) )
        def feature(x):
            e = collabs_as_a(x)
            vals = [inner(b) for b in e["author_b"]]
            vals = [v for v in vals if not pd.isna(v)]
            return float(np.mean(vals)) if vals else np.nan

        col = ("authors.MEAN(collaborations[author_a].authors[author_b]."
               "MEAN(collaborations[author_a].n_papers))")
        result = compute_dfs_features(
            rdb=complex_rdb, target_dataframe=complex_target.copy(),
            key_mappings={"author_id": "authors.author_id"},
            config=DFSConfig(engine=engine, max_depth=4),
        )
        assert col in result.columns, f"missing feature: {col}"
        expected = [feature(a) for a in complex_target["author_id"]]
        got = pd.to_numeric(result[col].reset_index(drop=True), errors="coerce").tolist()
        assert np.allclose(got, expected, atol=1e-6, equal_nan=True), (
            f"{engine}\n  expected={expected}\n  got     ={got}"
        )


# --------------------------------------------------------------------------- #
# Three-FK (hyper-edge) and multiple distinct patterns
# --------------------------------------------------------------------------- #

class TestTraverseBackMultiFKAndMultiPattern:

    def test_three_fk_hyper_edge(self):
        """A relation with 3 FKs into the same entity yields traverse-back, with
        no degenerate same-FK-down-and-back features."""
        nodes = pd.DataFrame({"node_id": ["n0", "n1", "n2", "n3"], "v": [1.0, 2.0, 3.0, 4.0]})
        # hyper-edges with three endpoints into nodes
        hyper = pd.DataFrame({
            "h_id": ["h0", "h1", "h2"],
            "a_id": ["n0", "n1", "n2"],
            "b_id": ["n1", "n2", "n3"],
            "c_id": ["n2", "n3", "n0"],
            "w": [0.5, 1.0, 1.5],
        })
        rdb = create_rdb(
            tables={"nodes": nodes, "hyper": hyper},
            name="hyper_rdb",
            primary_keys={"nodes": "node_id", "hyper": "h_id"},
            foreign_keys=[
                ("hyper", "a_id", "nodes", "node_id"),
                ("hyper", "b_id", "nodes", "node_id"),
                ("hyper", "c_id", "nodes", "node_id"),
            ],
        )
        target = pd.DataFrame({"a_id": ["n0", "n1"]})
        result = compute_dfs_features(
            rdb=rdb, target_dataframe=target.copy(),
            key_mappings={"a_id": "nodes.node_id"},
            config=DFSConfig(engine="featuretools", max_depth=4),
        )
        cols = [c for c in result.columns if c != "a_id"]
        tb = [c for c in cols if c.count("hyper") >= 2 or c.count("nodes") >= 2]
        assert len(tb) > 0
        # No feature should descend and climb through the SAME FK (degenerate).
        for c in tb:
            for fk in ["a_id", "b_id", "c_id"]:
                # pattern: hyper[fk]...nodes[fk] would be the degenerate same-FK pair
                assert not re.search(rf"hyper\[{fk}\]\.nodes\[{fk}\]", c), (
                    f"degenerate same-FK climb-back leaked: {c}"
                )

    @pytest.mark.parametrize("engine", ["featuretools", "dfs2sql"])
    def test_two_independent_graph_patterns(self, engine):
        """Two separate (entity, 2-FK-relation) pairs in one RDB; both produce
        traverse-back features and they don't cross-contaminate."""
        users = pd.DataFrame({"user_id": ["u0", "u1", "u2"], "uv": [1.0, 2.0, 3.0]})
        follows = pd.DataFrame({
            "f_id": ["f0", "f1", "f2"],
            "follower": ["u0", "u1", "u2"],
            "followee": ["u1", "u2", "u0"],
            "fw": [0.1, 0.2, 0.3],
        })
        items = pd.DataFrame({"item_id": ["i0", "i1", "i2"], "iv": [10.0, 20.0, 30.0]})
        related = pd.DataFrame({
            "r_id": ["r0", "r1", "r2"],
            "item_x": ["i0", "i1", "i2"],
            "item_y": ["i1", "i2", "i0"],
            "rw": [1.0, 2.0, 3.0],
        })
        rdb = create_rdb(
            tables={"users": users, "follows": follows, "items": items, "related": related},
            name="two_pattern_rdb",
            primary_keys={
                "users": "user_id", "follows": "f_id",
                "items": "item_id", "related": "r_id",
            },
            foreign_keys=[
                ("follows", "follower", "users", "user_id"),
                ("follows", "followee", "users", "user_id"),
                ("related", "item_x", "items", "item_id"),
                ("related", "item_y", "items", "item_id"),
            ],
        )
        target = pd.DataFrame({"user_id": ["u0", "u1"], "item_id": ["i0", "i1"]})
        result = compute_dfs_features(
            rdb=rdb, target_dataframe=target.copy(),
            key_mappings={"user_id": "users.user_id", "item_id": "items.item_id"},
            config=DFSConfig(engine=engine, max_depth=4),
        )
        cols = [c for c in result.columns if c not in ("user_id", "item_id")]
        follows_tb = [c for c in cols if c.count("follows") >= 2]
        related_tb = [c for c in cols if c.count("related") >= 2]
        assert len(follows_tb) > 0, "expected traverse-back on follows pattern"
        assert len(related_tb) > 0, "expected traverse-back on related pattern"
        # A single feature must not mix the two disjoint patterns.
        assert not any(
            (c.count("follows") >= 1 and c.count("related") >= 1) for c in cols
        )


# --------------------------------------------------------------------------- #
# Temporal correctness (cutoff time)
# --------------------------------------------------------------------------- #

class TestTraverseBackCutoffTime:

    @pytest.fixture
    def temporal_rdb(self):
        nodes = pd.DataFrame({"node_id": ["n0", "n1", "n2"], "v": [1.0, 2.0, 3.0]})
        edges = pd.DataFrame({
            "edge_id": ["e0", "e1", "e2", "e3"],
            "src_id": ["n0", "n0", "n1", "n2"],
            "dst_id": ["n1", "n2", "n2", "n0"],
            "edge_weight": [0.5, 1.0, 1.5, 2.0],
            "ts": pd.to_datetime([
                "2023-01-01", "2023-06-01", "2023-03-01", "2023-09-01",
            ]),
        })
        return create_rdb(
            tables={"nodes": nodes, "edges": edges},
            name="temporal_graph",
            primary_keys={"nodes": "node_id", "edges": "edge_id"},
            foreign_keys=[
                ("edges", "src_id", "nodes", "node_id"),
                ("edges", "dst_id", "nodes", "node_id"),
            ],
            time_columns={"edges": "ts"},
        )

    @pytest.mark.parametrize("engine", ["featuretools", "dfs2sql"])
    def test_cutoff_filters_traverse_back(self, temporal_rdb, engine):
        """Traverse-back values must respect the cutoff time (only edges with
        ts <= cutoff contribute)."""
        target = pd.DataFrame({
            "src_id": ["n0", "n0"],
            "dst_id": ["n1", "n1"],
            "cutoff": pd.to_datetime(["2023-02-01", "2023-12-31"]),
        })
        result = compute_dfs_features(
            rdb=temporal_rdb, target_dataframe=target.copy(),
            key_mappings={"src_id": "nodes.node_id", "dst_id": "nodes.node_id"},
            cutoff_time_column="cutoff",
            config=DFSConfig(engine=engine, max_depth=4, use_cutoff_time=True),
        )
        tb = [
            c for c in result.columns
            if c not in ("src_id", "dst_id", "cutoff")
            and (c.count("edges") >= 2 or c.count("nodes") >= 2)
        ]
        assert len(tb) > 0
        # Early cutoff (2023-02-01) sees strictly fewer edges than the late one,
        # so at least one traverse-back feature must differ between the two rows.
        differs = False
        for c in tb:
            vals = pd.to_numeric(result[c].reset_index(drop=True), errors="coerce")
            if not np.isclose(
                vals.iloc[0], vals.iloc[1], atol=1e-9, equal_nan=True
            ):
                differs = True
                break
        assert differs, "cutoff time had no effect on any traverse-back feature"


# --------------------------------------------------------------------------- #
# General rule: bipartite A -> B -> C -> B -> A (intermediate-table revisit)
# --------------------------------------------------------------------------- #

class TestTraverseBackBipartite:
    """The collaborative-filtering shape A -> B -> C -> B -> A, where B has one FK
    to A and one to C. Featuretools cannot generate it (it revisits A); the general
    traverse-back walk can."""

    @pytest.fixture
    def cf_rdb(self):
        users = pd.DataFrame({"user_id": ["u0", "u1", "u2", "u3"], "uval": [1.0, 2.0, 3.0, 4.0]})
        items = pd.DataFrame({"item_id": ["i0", "i1", "i2"], "ival": [10.0, 20.0, 30.0]})
        inter = pd.DataFrame({
            "inter_id": ["x0", "x1", "x2", "x3", "x4", "x5"],
            "user_id": ["u0", "u0", "u1", "u2", "u3", "u1"],
            "item_id": ["i0", "i1", "i1", "i2", "i2", "i0"],
            "rating": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        return create_rdb(
            tables={"users": users, "items": items, "inter": inter},
            name="cf_rdb",
            primary_keys={"users": "user_id", "items": "item_id", "inter": "inter_id"},
            foreign_keys=[
                ("inter", "user_id", "users", "user_id"),
                ("inter", "item_id", "items", "item_id"),
            ],
        )

    @pytest.fixture
    def cf_target(self):
        return pd.DataFrame({"user_id": ["u0", "u1", "u2", "u3"]})

    @pytest.mark.parametrize("engine", ["featuretools", "dfs2sql"])
    def test_cf_feature_generated(self, cf_rdb, cf_target, engine):
        # The CF feature target->users->inter->items->inter->users is a 5-hop path,
        # so it needs max_depth >= 5.
        result = compute_dfs_features(
            rdb=cf_rdb, target_dataframe=cf_target.copy(),
            key_mappings={"user_id": "users.user_id"},
            config=DFSConfig(engine=engine, max_depth=5, max_revisits=1),
        )
        cols = [c for c in result.columns if c != "user_id"]
        # A->B->C->B->A ending in a users column: inter appears twice, users appears
        # again inside (not just as the outer prefix), and items is on the path.
        cf = [
            c for c in cols
            if c.count("inter") >= 2 and "items" in c and "inter.users." in c
        ]
        assert len(cf) > 0, "expected A->B->C->B->A collaborative-filtering features"

    @pytest.mark.parametrize("engine", ["featuretools", "dfs2sql"])
    def test_cf_value_matches_brute_force(self, cf_rdb, cf_target, engine):
        """Hand-checked CF value: avg uval of users sharing my items."""
        inter = pd.DataFrame({
            "user_id": ["u0", "u0", "u1", "u2", "u3", "u1"],
            "item_id": ["i0", "i1", "i1", "i2", "i2", "i0"],
        })
        uval = {"u0": 1.0, "u1": 2.0, "u2": 3.0, "u3": 4.0}

        def item_mean_uval(item):
            us = inter[inter.item_id == item]["user_id"]
            return float(np.mean([uval[u] for u in us]))

        def feature(user):
            its = inter[inter.user_id == user]["item_id"]
            vals = [item_mean_uval(it) for it in its]
            return float(np.mean(vals)) if vals else np.nan

        col = "users.MEAN(inter.items.MEAN(inter.users.uval))"
        result = compute_dfs_features(
            rdb=cf_rdb, target_dataframe=cf_target.copy(),
            key_mappings={"user_id": "users.user_id"},
            config=DFSConfig(engine=engine, max_depth=5, max_revisits=1),
        )
        assert col in result.columns, f"missing CF feature: {col}"
        expected = [feature(u) for u in cf_target["user_id"]]
        got = pd.to_numeric(result[col].reset_index(drop=True), errors="coerce").tolist()
        assert np.allclose(got, expected, atol=1e-6, equal_nan=True), (
            f"{engine}\n  expected={expected}\n  got     ={got}"
        )

    def test_disabled_when_traverse_back_false(self, cf_rdb, cf_target):
        # Note: featuretools natively generates A->B->C->B features
        # (e.g. users.MEAN(inter.items.MEAN(inter.rating))) even without our
        # generator. Disabling traverse_back must only remove the features that
        # revisit A (i.e. the A->B->C->B->A shape ending in a users column).
        result = compute_dfs_features(
            rdb=cf_rdb, target_dataframe=cf_target.copy(),
            key_mappings={"user_id": "users.user_id"},
            config=DFSConfig(engine="featuretools", max_depth=5, traverse_back=False),
        )
        cols = [c for c in result.columns if c != "user_id"]
        # No feature should climb back into users (inter.users.<col>) at depth.
        assert not any("inter.users." in c for c in cols)


# --------------------------------------------------------------------------- #
# max_revisits lever
# --------------------------------------------------------------------------- #

class TestTraverseBackMaxRevisits:

    def test_default_is_one(self):
        assert DFSConfig().max_revisits == 1

    def test_zero_revisits_disables(self, graph_rdb, graph_target):
        result = compute_dfs_features(
            rdb=graph_rdb, target_dataframe=graph_target.copy(),
            key_mappings=GRAPH_KEYS,
            config=DFSConfig(engine="featuretools", max_depth=6, max_revisits=0),
        )
        tb = [
            c for c in result.columns
            if c not in ("src_id", "dst_id") and c.count("edges") >= 2
        ]
        assert tb == []

    def test_more_revisits_adds_deeper_features(self, graph_rdb, graph_target):
        def count(max_revisits):
            result = compute_dfs_features(
                rdb=graph_rdb, target_dataframe=graph_target.copy(),
                key_mappings=GRAPH_KEYS,
                config=DFSConfig(
                    engine="featuretools", max_depth=6, max_revisits=max_revisits
                ),
            )
            return len([c for c in result.columns if c not in ("src_id", "dst_id")])

        assert count(2) > count(1)


# --------------------------------------------------------------------------- #
# Degeneracy: single-FK chains never produce a same-edge round trip
# --------------------------------------------------------------------------- #

class TestTraverseBackDegeneracy:

    def test_single_fk_chain_no_degenerate_feature(self):
        """A <- B with one FK: the only 'revisit' is the degenerate same-edge round
        trip A.<agg>(B.A.<agg>(B.x)), which must never be emitted."""
        A = pd.DataFrame({"a_id": ["a0", "a1", "a2"], "av": [1.0, 2.0, 3.0]})
        B = pd.DataFrame({
            "b_id": ["b0", "b1", "b2", "b3"],
            "a_fk": ["a0", "a0", "a1", "a2"],
            "bv": [10.0, 20.0, 30.0, 40.0],
        })
        rdb = create_rdb(
            tables={"A": A, "B": B}, name="single",
            primary_keys={"A": "a_id", "B": "b_id"},
            foreign_keys=[("B", "a_fk", "A", "a_id")],
        )
        target = pd.DataFrame({"a_fk": ["a0", "a1", "a2"]})
        result = compute_dfs_features(
            rdb=rdb, target_dataframe=target.copy(),
            key_mappings={"a_fk": "A.a_id"},
            config=DFSConfig(engine="featuretools", max_depth=6, max_revisits=2),
        )
        cols = [c for c in result.columns if c != "a_fk"]
        # A degenerate feature would nest two aggregations over B, e.g. "(B." twice.
        degenerate = [c for c in cols if c.count("(B.") >= 2 or c.count("(B)") + c.count("(B.") >= 2]
        assert degenerate == [], f"degenerate same-edge round trip leaked: {degenerate}"
