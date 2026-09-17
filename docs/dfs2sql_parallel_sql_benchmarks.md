# dfs2sql parallel SQL workers — speed comparison

Benchmarks of **CPU sequential** (`dfs2sql_sql_workers=1`) vs **CPU parallel** (`dfs2sql_sql_workers=4`) for the dfs2sql engine.

## Setup

| Setting | Value |
|---------|--------|
| Engine | `dfs2sql` (DuckDB CPU) |
| Target rows | 10,000 (sampled) |
| `max_depth` | 4 |
| `agg_primitives` | `count`, `mean`, `min`, `max` |
| Cutoff time | yes (task timestamp) |
| Parallel workers | 4 DuckDB connections |

**Metrics**

- **`sql_exec`**: time spent executing feature SQLs (excludes Featuretools planning / prep).
- **`wall`**: end-to-end `compute_features` time (includes Featuretools prep + SQL + assembly).
- **Speedup**: sequential time ÷ parallel time.


---

## `sql_exec` (SQL loop only)

| Task | `#sqls` | Sequential (s) | Parallel `@4` (s) | Speedup |
|------|--------:|---------------:|------------------:|--------:|
| retailrocket (cvr) | 226 | 97.9 | 26.7 | **3.7×** |
| amazon (churn) | 47 | 30.4 | 6.5 | **4.7×** |
| stackexchange (churn) | 495 | 71.3 | 63.7 | **1.1×** |

---

## End-to-end wall time

| Task | Sequential (s) | Parallel `@4` (s) | Speedup |
|------|---------------:|------------------:|--------:|
| retailrocket (cvr) | 128.9 | 51.2 | **2.5×** |
| amazon (churn) | 63.4 | 32.4 | **2.0×** |
| stackexchange (churn) | 90.0 | 78.6 | **1.1×** |

---

## Summary

| Task | `sql_exec` speedup | Wall speedup | Notes |
|------|-------------------:|-------------:|-------|
| retailrocket cvr | 3.7× | 2.5× | Strong parallel win; many medium queries |
| amazon churn | 4.7× | 2.0× | Best `sql_exec` scaling; wall diluted by load/prep |
| stackexchange churn | 1.1× | 1.1× | Many queries but limited parallel gain on this host |

**Takeaways**

1. Parallel workers help most when the SQL phase dominates and queries are independent (retailrocket / amazon).
2. End-to-end wall speedup is lower than `sql_exec` speedup because Featuretools prep and data load stay sequential.
3. Stackexchange shows little gain here — more statements (495) but poorer parallel efficiency in this run (contention / query mix).

## Config

```python
from fastdfs import DFSConfig

config = DFSConfig(
    engine="dfs2sql",
    dfs2sql_sql_workers=4,  # default 1 = sequential
)
```

Parity: parallel results match sequential within numeric tolerance (`tests/test_dfs2sql_parallel.py`).
