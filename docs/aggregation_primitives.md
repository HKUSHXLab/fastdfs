# Aggregation Primitives

FastDFS uses [Featuretools](https://featuretools.alteryx.com/) Deep Feature Synthesis (DFS) to define features, then computes them with either the **featuretools** engine (pandas) or the **dfs2sql** engine (DuckDB SQL). The `agg_primitives` field on `DFSConfig` controls which aggregation functions are applied along relationship paths.

```python
import fastdfs

config = fastdfs.DFSConfig(
    engine="dfs2sql",
    agg_primitives=[
        "count", "mean", "max", "min", "std", "mode",
        "quantile_25", "quantile_75", "discrete_entropy",
    ],
)
```

## Defaults

If you omit `agg_primitives`, FastDFS uses:

| Primitive | Description |
|-----------|-------------|
| `count` | Number of related rows |
| `mean` | Arithmetic mean (numeric columns) |
| `max` | Maximum value |
| `min` | Minimum value |
| `std` | Sample standard deviation (`ddof=1`) |
| `mode` | Most frequent value (categorical / string columns) |

Defined in `fastdfs.dfs.base_engine.DFSConfig`.

## FastDFS custom primitives

These are implemented in FastDFS (not stock Featuretools names) and registered in `DFSEngine._convert_primitives()`:

| Primitive | Input | Description | dfs2sql (DuckDB) |
|-----------|-------|-------------|------------------|
| `quantile_25` | numeric | 25th percentile (pandas `quantile(0.25)`) | `quantile_cont(col, 0.25)` |
| `quantile_75` | numeric | 75th percentile | `quantile_cont(col, 0.75)` |
| `discrete_entropy` | categorical | Shannon entropy in bits, \(-\sum p \log_2 p\) | `entropy(col)` |

Any name matching `quantile_XX` (e.g. `quantile_50`) is translated to `quantile_cont(col, XX/100)` in dfs2sql, but only `quantile_25` and `quantile_75` are defined as Featuretools primitive classes in FastDFS today.

## dfs2sql-supported primitives (recommended)

The **dfs2sql** engine (`engine="dfs2sql"`) is the default. It generates one SQL query per feature and runs it in DuckDB. The following primitives are **documented and tested** for parity with the featuretools engine (see `tests/test_dfs_engines.py`):

| Primitive | Typical column types | SQL / notes |
|-----------|---------------------|-------------|
| `count` | any | `count(col)`; child nulls coalesced to 0 when nested under `count` |
| `mean` | numeric | `mean(col)`; booleans cast to integer |
| `max` | numeric, boolean | `max(col)` |
| `min` | numeric, boolean | `min(col)` |
| `sum` | numeric | `sum(col)`; booleans cast to integer |
| `std` | numeric | `stddev_samp(col)` (matches pandas / Featuretools sample std) |
| `median` | numeric | `median(col)`; booleans cast to integer |
| `mode` | categorical (string) | `mode(col)` |
| `num_unique` | any | `num_unique(col)` — Featuretools name; **not** `nunique` |
| `quantile_25` | numeric | `quantile_cont(col, 0.25)` |
| `quantile_75` | numeric | `quantile_cont(col, 0.75)` |
| `discrete_entropy` | categorical | `entropy(col)` |

### Categorical columns and numeric aggregations

By default, RDB `category_t` columns are canonicalized to strings, which suits `mode` and `discrete_entropy`. To apply `mean`, `std`, `quantile_*`, etc. on categorical base columns, label-encode them first with `EncodeCategoryColumns` (see `fastdfs.transform.encode_categorical` and `dev_logs/022_ENCODE_CATEGORICAL_AS_FLOAT.md`).

## featuretools engine

With `engine="featuretools"`, you may pass any [Featuretools aggregation primitive](https://featuretools.alteryx.com/en/stable/api_reference/primitives/descriptions.html) by name (65 built-ins, e.g. `first`, `last`, `skew`, `time_since_last`, `num_unique`). FastDFS custom names (`quantile_25`, `quantile_75`, `discrete_entropy`) are also supported.

Featuretools chooses which primitives apply to which columns based on semantic tags (numeric vs categorical).

## dfs2sql: experimental primitives

For primitives not listed above, dfs2sql falls back to emitting a DuckDB aggregate with the **same name** as the primitive (`gen_sqls.FeatureBlock.handle_agg`). This may work if DuckDB defines a matching aggregate and the column type is valid, but it is **unsupported**: no tests, and many Featuretools primitives (time-based, windowed, or custom logic) will fail at SQL generation or query time.

Legacy SQL mappings exist for `join` → `string_agg` and `arraymax` / `arraymin` / `arraymean` → `array_agg`, but FastDFS no longer registers those custom primitive classes in `_convert_primitives()`.

## Engine comparison

| | `featuretools` | `dfs2sql` |
|--|----------------|-----------|
| **Primitive set** | Featuretools built-ins + FastDFS custom three | Subset in table above (recommended) |
| **Backend** | pandas | DuckDB |
| **Performance** | Better for small tables | Better for large tables |
| **When unsure** | Use for prototyping exotic primitives | Use for production / RelBench-scale data |

## See also

- [API reference — `DFSConfig`](api_reference.md#dfsconfig)
- [User guide — DFS configuration](user_guide.md#dfs-configuration)
- Implementation: `fastdfs/dfs/base_engine.py` (`DFSConfig`, `_convert_primitives`, custom primitive classes), `fastdfs/dfs/gen_sqls.py` (`handle_agg`)
