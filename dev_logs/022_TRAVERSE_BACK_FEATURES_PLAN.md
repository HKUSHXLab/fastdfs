# 022 — Traverse-Back Feature Generation (Design)

## 1. Problem

`featuretools.dfs()` refuses to generate features whose synthesis path revisits a
table (dataframe) that is already on the current path. Concretely, in
`_build_agg_features` / `_build_forward_features` it guards with:

```python
if b_dataframe_id in all_features:   # dataframe already on this path
    continue
```

The guard keys on **table name**, not on the relationship/edge taken. This means
that a path of the shape

```
A → B → A → B
```

— where both `B → A` hops use **a foreign key of B that points at the same table A** —
is silently dropped, because the second time we reach `A` it is "already seen".

### Why these features are valid and wanted

When `A` is an **entity table** (PK) and `B` is a **relation table** with
**two or more foreign keys into A** (the canonical example is a graph:
`A = nodes`, `B = edges` with `edges.src_id → nodes` and `edges.dst_id → nodes`),
the traversal alternates **one→many** (`A → B`, a node's incident edges) with
**many→one** (`B → A`, an edge's *other* endpoint). Each hop lands on a *different*
set of rows, so the composite feature carries genuine 2-hop (neighborhood)
information, e.g.:

```
nodes[src_id].MEAN( edges[src_id].nodes[dst_id].MEAN(edges[src_id].edge_weight) )
```

> "For the source node of this target edge, average — over its outgoing edges —
> the (average outgoing-edge weight of the edge's destination node)."

This is exactly a 2-hop message-passing aggregation and is highly predictive for
link-/node-level tasks.

### What featuretools handles fine (do NOT duplicate)

If the intermediate table differs, featuretools already generates the feature.
Two confirmed cases from experiments:

- **Chain**: `stores → regions → stores → sales` ⇒ featuretools *does* emit
  `stores.regions.MEAN(stores.COUNT(sales))` at depth 4 (the revisited
  `stores` is reached through a *different* table `regions`, so no name clash on
  the active path... actually the clash is avoided because the path passes through
  `regions` first — see note below).
- **Bipartite / heterogeneous graph**: `users → edges → items → edges` ⇒
  featuretools *does* emit `users.MEAN(edges.items.COUNT(edges))` at depth 4,
  because the revisited table on the way back is `items`, not `users`.

The **only** gap is revisiting the **same** entity table through a relation table
that has ≥2 FKs into it. That is the precise scope of this fix.

> Note on the chain case: featuretools' guard is per-recursion-branch. The reason
> the homogeneous graph fails while bipartite succeeds is that in the homogeneous
> case the *only* way onward from `B` is back into the same `A`, which is on the
> path; in bipartite there is an alternative table `C` to descend into.

## 2. Featuretools depth model (ground truth, measured)

`get_depth()` counts **every** `DirectFeature` and `AggregationFeature` hop as +1;
`IdentityFeature` is depth 0. `ft.dfs(max_depth=d)` returns every feature with
`get_depth() ≤ d`.

| Feature structure (rooted at target)                                  | depth |
|-----------------------------------------------------------------------|-------|
| `IdentityFeature(col)`                                                | 0     |
| `DirectFeature(identity, target)`  e.g. `A.col`                       | 1     |
| `AggregationFeature([identity], A)`  e.g. `MEAN(B.col)` on A          | 1     |
| `DirectFeature(agg_on_A, target)`  e.g. `A.MEAN(B.col)` — *standard*  | 2     |
| `DirectFeature(agg_on_A, B)` — traverse-back lookup                   | 2     |
| `AggregationFeature([direct_back], A)` — re-aggregate on A            | 3     |
| `DirectFeature(agg2_on_A, target)` — *one traverse-back round*        | 4     |

**Consequence:** one traverse-back round adds **+2** to the depth of the seed it
extends. A standard feature is depth 2, so the smallest traverse-back feature is
depth 4. In general, with seed feature of (target-rooted) depth `d_seed`, a round
produces depth `d_seed + 2`.

Number of rounds permitted by a depth budget:

```
n_rounds(max_depth) = max(0, (max_depth - 2) // 2)
```

| max_depth | rounds | example target-rooted depths emitted |
|-----------|--------|--------------------------------------|
| ≤ 3       | 0      | (none — standard DFS only)           |
| 4–5       | 1      | 4                                    |
| 6–7       | 2      | 4, 6                                 |
| 8–9       | 3      | 4, 6, 8                              |

This is the depth contract the algorithm must honor: **never emit a feature whose
`get_depth()` exceeds `config.max_depth`.** We will enforce this by checking
`final_feat.get_depth() <= max_depth` as a hard gate, independent of the round
counting (the round count is just an optimization to avoid building doomed
candidates).

## 3. Detection — which (A, B) pairs are traverse-back opportunities

Build the set of candidate pairs `(A, B)` where:

- `A` is a parent table (has a PK referenced by FKs),
- `B` is a child table,
- there exist **≥ 2 distinct relationships** `B.fk_i → A.pk` (two or more FK
  columns of the *same* B pointing at the *same* A),
- neither `A` nor `B` is the synthetic `__target__` table.

Rationale: with ≥2 FKs we can descend `A → B` via `fk_i` and climb back `B → A`
via `fk_j` (possibly `i == j`, see §6 "self-pairing") landing on different rows.

A single FK pair (`B` has exactly one FK to `A`) is **excluded**: climbing back
returns to the same row set, producing no new information (the inner aggregate
would just be re-read unchanged). See §6.

Heterogeneous bipartite (`B.fk → A` and `B.gk → C`, `A ≠ C`) is **not** the target
of this fix because featuretools already covers `A → B → C → B`. We only add the
homogeneous same-table revisit.

## 4. Generation — recursive, depth-driven, from the target outward

We generate features **level by level on each entity table A**, where "level k"
means an `AggregationFeature` living on `A` that contains `k` traverse-back rounds
internally (level 0 = the standard agg featuretools already produced).

### State carried per A

A worklist of `(agg_feature_on_A, column_schema, current_depth)` tuples, where
`current_depth` is the depth of the agg **as it sits on A** (not yet wrapped in the
final target DirectFeature). Standard featuretools aggs on A have `current_depth = 1`.

### One traverse-back round (level k → level k+1) on pair (A, B)

For a seed `agg` on `A` (depth `d` on A) with output `column_schema`:

```
for lookup_rel in FKs(B → A):                 # climb B → A to read agg on the OTHER endpoint
    direct_back = DirectFeature(agg, child=B, relationship=lookup_rel)
    # direct_back lives on B, depth d+1, same column_schema as agg

    for outer_rel in FKs(B → A):               # descend A → B then aggregate (i.e. agg over B grouped by A)
        for prim in primitives_compatible_with(column_schema):
            outer_agg = AggregationFeature([direct_back], parent=A,
                                           primitive=prim,
                                           relationship_path=[(False, outer_rel)])
            # outer_agg lives on A, depth d+2, column_schema = prim.return_type
            enqueue (outer_agg, prim.return_type, d+2)  # candidate seed for next round
```

### Rooting at target

Every agg on `A` at any level becomes user-facing by wrapping in a DirectFeature
through each FK from `__target__` to `A`:

```
for target_rel in FKs(target → A):
    final = DirectFeature(outer_agg, child=__target__, relationship=target_rel)
    if final.get_depth() <= max_depth and final.get_name() not in seen:
        emit(final)
```

### Recursion / iteration

Repeat the round `n_rounds(max_depth)` times, each round consuming the previous
round's enqueued aggs as seeds. Stop early when a round produces no new aggs whose
`d+2 ≤ max_depth`.

This is "recursively computing all traverse-back features from the target,
controlled by depth" — implemented as a bounded BFS over agg-on-A states keyed by
depth, which is equivalent to recursion but avoids Python stack depth issues and
makes the depth gate explicit.

## 5. Type correctness (critical)

Featuretools' `AggregationFeature.__init__` **asserts** that the base feature's
`column_schema` matches the primitive's `input_types`. Mismatches raise
`AssertionError`. Measured facts:

- `COUNT` → `IntegerNullable` / `numeric`
- `MEAN`, `STD`, `MAX`, `MIN`, `quantile_*` → `numeric`
- `MODE` → `Categorical` / `category`
- `DISCRETE_ENTROPY` → numeric output, **categorical input**

Therefore the outer primitive must be chosen by **matching the inner agg's output
`column_schema` against each primitive's `input_types`**, not by a blanket
"numeric only" rule. Concretely:

- inner output numeric → outer ∈ {mean, max, min, std, count*, quantile_*}
- inner output categorical → outer ∈ {mode, num_unique, discrete_entropy}

(`count` is special — it counts rows and ignores the base column type; we keep the
existing fastdfs convention of generating `COUNT(B)` only from the entity index,
and do **not** spawn count from traverse-back values to avoid meaningless repeats.)

Implementation rule: filter `config.agg_primitives` to those whose
`input_types[0]` semantic tags intersect the seed's `column_schema` semantic tags.
Keep the `try/except AssertionError` as a defensive backstop, but do not rely on it
for correctness or performance.

## 6. Edge cases and decisions

1. **Single FK to A (no traverse-back):** excluded by the "≥2 FKs" rule. Climbing
   back via the only FK returns the same row set → no new signal. *Decision: skip.*

2. **`i == j` self-pairing (descend and climb via the SAME FK):** still lands on
   the same rows on the way back, so it is degenerate like the single-FK case.
   *Decision: require `lookup_rel != outer_rel` when both endpoints would coincide.*
   We must, however, still allow the genuinely useful asymmetric combos
   (`src` down / `dst` back, `dst` down / `src` back, etc.). Net rule:
   **emit a round only when the descend-FK and climb-FK differ**, OR when there are
   ≥2 FKs and the chosen pair produces a name not already seen. The depth gate +
   name-dedup makes over-generation harmless, but skipping the `i==j`-only case
   keeps the count down.

3. **Multiple entity tables sharing the same B (heterogeneous):** handle each
   `(A, B)` pair independently. Do **not** attempt cross-entity traverse-back here
   (featuretools already does `A → B → C → B`).

4. **More than 2 FKs to A** (e.g. a hyper-edge `B` with `src`, `via`, `dst`):
   the double loop over `FKs(B→A)` naturally enumerates all descend/climb FK
   combinations. Count grows as `O(#FKs² × #prims × #seeds)`; the `max_features`
   config and depth gate bound it.

5. **Cutoff time / temporal correctness:** the manually constructed features are
   ordinary featuretools `FeatureBase` objects; `calculate_feature_matrix`
   (featuretools engine) and the SQL translation (dfs2sql engine) apply cutoff
   times exactly as for DFS-produced features. Already verified: both engines
   compute traverse-back features and agree on values. No special handling needed.

6. **Existing key-filtering (`base_feature_is_key`) and `__target__` filtering:**
   traverse-back features are built from *already-filtered* seeds, so they inherit
   the "no key aggregation" property. We still run the final names through the same
   dedup set; we do **not** re-run `base_feature_is_key` because the inner
   structure can legitimately contain index-based COUNTs (which are allowed).

7. **`max_features` interaction:** when set, featuretools may have already trimmed
   the seed set. Traverse-back generation should respect the global cap: stop
   emitting once `len(all_features) >= max_features`. *Decision: enforce the cap
   after merging, truncating deterministically (stable order) so featuretools and
   dfs2sql engines see identical feature lists.*

8. **Column explosion / blow-up:** homogeneous graphs with several numeric columns
   and 6 primitives produced 160 traverse-back features at one round on a tiny toy
   schema. Two rounds multiply again. *Decision: gate behind depth (already), and
   document that `max_depth ≥ 4` on multi-FK schemas can be expensive; recommend
   pairing with `max_features`.* Consider a future `traverse_back: bool = True`
   config flag to allow opt-out (out of scope for the bug fix, noted for later).

9. **Determinism:** iterate relationships, tables, primitives, and seeds in a
   stable, sorted order so the emitted feature list is reproducible across runs and
   identical between engines. Dedup by `get_name()`.

10. **Datetime-derived numeric columns** (from `FeaturizeDatetime`): treated as
    ordinary numeric columns; no special case.

## 7. Depth-gate vs round-count (defense in depth)

Two independent guards, both applied:

- **Round budget** `n_rounds = (max_depth - 2)//2`: limits how many BFS layers we
  attempt (performance).
- **Per-feature hard gate** `final.get_depth() <= max_depth`: the source of truth
  for correctness; trusts featuretools' own depth accounting rather than our
  arithmetic. A feature is emitted only if it passes this gate.

If the two ever disagree, the hard gate wins (we may build a candidate that is then
discarded). This protects against off-by-one errors in the round arithmetic when
seeds have non-standard depth.

## 8. Public surface / config

**Decisions (confirmed):**

- **Depth semantics:** reuse the existing `max_depth`, interpreted via
  featuretools' own `get_depth()`. One traverse-back round costs +2, so
  traverse-back features first appear at `max_depth >= 4`. The hard gate
  `final.get_depth() <= max_depth` is authoritative.
- **Config switch (added now):** `DFSConfig.traverse_back: bool = True`. When
  `False`, `_generate_traverse_back_features` returns `[]` immediately, restoring
  pure featuretools behavior. This lets users opt out on wide multi-FK schemas
  where the feature count explodes.
- **Feature tightening (accepted):** type-matched outer primitives only
  (no `AssertionError` paths), and degenerate paths excluded (single-FK pairs and
  `i == j` same-FK climb-back).

## 9. Test plan

Implemented in `tests/test_traverse_back.py` (developed test-first; verified RED
against the unfixed engine, then GREEN after the fix):

1. **Detection** (`TestTraverseBackDetection`): homogeneous two-FK graph yields
   traverse-back features (both engines); single-FK schema yields none.
2. **Depth gate** (`TestTraverseBackDepthGate`): none at `max_depth ∈ {1,2,3}`;
   present at `{4,5,6}`; no feature's `get_depth()` exceeds `max_depth`; the
   second round (depth 6) appears only at `max_depth = 6` (not 4 or 5).
3. **Config switch** (`TestTraverseBackConfig`): `traverse_back=False` disables;
   default is `True`.
4. **Type matching** (`TestTraverseBackTypeMatching`): no numeric primitive over a
   categorical inner aggregate; `MODE(...MODE(...))` chains are produced.
5. **Value correctness** (`TestTraverseBackValues`): featuretools and dfs2sql
   engines agree on all common traverse-back features; one hand-computed 2-hop
   mean value is asserted exactly.
6. **`max_features`** (`TestTraverseBackMaxFeatures`): the global cap is respected.
7. **Regression:** full existing suite (125 tests) stays green (145 total with the
   new tests).
```
