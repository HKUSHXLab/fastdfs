# Plan: Replace sequential `pd.merge` with column-wise assembly in `DFS2SQLEngine`

**Status (implemented):** `assemble_dfs2sql_feature_frames` in `fastdfs/dfs/dfs2sql_engine.py` replaces the O(N) merge loop; tests in `tests/test_dfs2sql_assembly.py`. Config: `DFSConfig.dfs2sql_concat_chunk_size`.

## Goal

After DuckDB executes one SQL per DFS feature, `fastdfs/dfs/dfs2sql_engine.py` currently merges **O(N)** skinny result frames with repeated `pd.merge(..., on=target_index, how="left")`. For large **N** (e.g. tens of thousands of features), this dominates runtime and produces **no user-visible progress**.

Replace that pattern with **index- or key-aligned column assembly** (e.g. `pd.concat(..., axis=1)` in one pass or a small number of chunked passes), preserving **identical** output columns and values (up to benign float / NA representation differences) for the same inputs.

**Primary file:** `fastdfs/fastdfs/dfs/dfs2sql_engine.py` (`compute_feature_matrix`, block after the `tqdm` SQL loop).

---

## Preconditions and invariants (verify before coding)

1. **Key column:** Every successful `result.df()` includes the same target key column (e.g. `__target_index__` / `target_index` as used today) with **one row per training entity** for that step.
2. **Row order:** Either  
   - all frames share the **same row order** as `target_dataframe` after the existing `sort_values(by=target_index)` on the merged result, or  
   - row order may differ but each frame is **uniquely keyed** so we can **`reindex`** every partial frame to a canonical key order before concat.
3. **Column uniqueness:** After `decode_column_from_sql`, feature column names are **unique** across SQLs (Featuretools / naming contract). Document or assert; if duplicates exist, define deterministic suffixing or fail fast.
4. **Empty / skipped SQLs:** Some iterations may append nothing; the plan must handle variable-length `dataframes` and single-column edge cases.

Deliverable for this phase: a short **design note** in code comments or in this doc listing which invariant holds in production (especially **2**).

---

## Step 1: Baseline and measurement

1. Add a **temporary** timing log (or use `logger.debug`) around the existing merge loop in `compute_feature_matrix`:
   - time for SQL loop (already partially visible via tqdm),
   - time for **finalize** (merge loop + `sort_values`),
   - resulting shape `(n_rows, n_cols)`.
2. Capture one **fixture** RDB + small target slice (e.g. 500 rows) that still generates **hundreds** of features (or use full feature list on tiny target) so tests run quickly while stressing merge vs concat.
3. Record **peak RSS** roughly (optional: `resource.getrusage` or external `/usr/bin/time -v`) for large **N** to compare after the change.

---

## Step 2: Choose assembly strategy

### Option A — Single `pd.concat(axis=1)` (simplest)

1. Build a list of dataframes each set to the same index:
   - `df = df.set_index(target_index)` (drop duplicate index rows if impossible — should not happen).
2. Optionally **`df.sort_index()`** on each piece if indexes are identical sets but order differs.
3. `wide = pd.concat(pieces, axis=1)`  
4. `wide = wide.reset_index()` if downstream expects `target_index` as a column.

**Risk:** `pd.concat` with many objects can still be heavy; memory spikes if all columns materialize at once (same as final matrix today).

### Option B — Chunked concat (recommended for very large **N**)

1. Concat every **K** frames (e.g. K=200–1000) along axis=1 → `blocks[i]`.
2. Concat `blocks` along axis=1 → final wide frame.
3. Tune **K** for memory vs overhead; optionally expose `DFSConfig` field later (e.g. `merge_chunk_size: Optional[int]`).

**Decision gate:** Implement **B** if profiling shows peak memory or concat overhead is problematic; otherwise **A** first for minimal code.

---

## Step 3: Implement helper with clear contract

1. Add a private function in `dfs2sql_engine.py`, e.g. `_assemble_feature_frames(dataframes, target_index) -> pd.DataFrame`, documented with:
   - required columns per frame,
   - canonical index (sorted unique target ids),
   - behavior on duplicate column names (raise `ValueError` with first collision).
2. Inside the helper:
   - Drop non-feature columns if any frame still carries cutoff columns (already partially stripped).
   - Align rows: **`reindex(canonical_index)`** per frame if any piece order differs from canonical (canonical = `sort_index()` of union of indexes, or explicit order from `target_dataframe[target_index]`).
   - Concatenate as per Step 2.
3. Replace the existing:

   ```python
   merged_df = dataframes[0]
   for df in dataframes[1:]:
       merged_df = pd.merge(merged_df, df, on=target_index, how="left")
   ```

   with a call to `_assemble_feature_frames(...)`.

---

## Step 4: Preserve existing output semantics

1. Keep **`sort_values(by=target_index).reset_index(drop=True)`** at the end unless concat already guarantees order; if concat uses canonical index order matching `target_dataframe`, assert equality of key order and skip redundant sort.
2. Reapply the same **`columns_to_exclude`** / `feature_columns` filtering as today so the returned frame matches the public contract.
3. Run **binary equality** (or `pandas.testing.assert_frame_equal`) on small/medium fixtures: old merge path vs new concat path. If float noise appears, use `check_dtype=False` / `rtol` only where DuckDB already allowed variance.

---

## Step 5: Tests

1. **Unit test** in `fastdfs/tests/` (extend `test_dfs_engines.py` or add `test_dfs2sql_assembly.py`):
   - Mock or reuse minimal RDB + target where **N** is small (5–20 features) and compare merge vs concat helper (temporary dual path behind env flag **or** compare against saved parquet golden — prefer internal helper test with two strategies in one test only during refactor, then delete merge path).
2. **Regression test:** existing engine tests must still pass (`test_dfs_engines.py`, `test_api.py`).
3. **Property-style test (optional):** random row shuffle of partial frames before assembly; assert aligned concat matches merge reference on tiny data.

---

## Step 6: Observability

1. After finalize, log at **INFO**:  
   `dfs2sql: assembled N feature SQL results into matrix shape (rows, cols) in Xs`.
2. Optional: **tqdm** over concat **chunks** (Option B) so long finalize phases show progress without spamming per-feature.

---

## Step 7: Performance validation

1. Re-run the slow scenario (mimic, `max_depth=4`, large **N**): compare wall-clock for finalize phase before/after.
2. Confirm **no regression** for small **N** (merge loop was cheap; concat should be similar or faster).

---

## Step 8: Documentation

1. Update `fastdfs/docs/user_guide.md` (or `api_reference.md`) with one paragraph: dfs2sql merges many single-feature SQL results; assembly uses chunked column concat for scalability.
2. Link from this dev log to the PR / commit.

---

## Step 9: Rollout and risk

1. Ship behind a **feature flag** in `DFSConfig` only if you need a quick revert in production, e.g. `dfs2sql_assembly: Literal["merge", "concat"] = "concat"` defaulting to `"concat"` after validation. Otherwise replace merge path directly once tests pass.
2. Watch for **pandas version** differences in `concat` / `reindex` behavior; pin minimum pandas in `pyproject.toml` if needed.

---

## Step 10: Follow-ups (out of scope for first PR)

1. Reduce **N** itself: feature pruning, `max_features`, or batching SQLs that return multiple columns per query (larger change in `features2sql`).
2. Consider **Polars** lazy horizontal concat for finalize if pandas remains a bottleneck.

---

## Checklist summary

| Step | Action |
|------|--------|
| 0 | Document invariants (key uniqueness, row alignment). |
| 1 | Add timing logs; capture benchmark scenario. |
| 2 | Pick Option A or B. |
| 3 | Implement `_assemble_feature_frames` + replace merge loop. |
| 4 | Match existing column filtering and row order. |
| 5 | Add / extend tests; assert parity with old behavior on fixtures. |
| 6 | INFO log (+ optional tqdm for chunks). |
| 7 | Profile large run; confirm win. |
| 8 | User-facing docs blurb. |
| 9 | Optional feature flag; release. |
| 10 | Optional follow-ups (SQL batching, Polars). |

---

## References

- Current merge loop: `fastdfs/fastdfs/dfs/dfs2sql_engine.py` (search for `Finalizing`).
- Call site: `DFSEngine.compute_features` → `DFS2SQLEngine.compute_feature_matrix`.
