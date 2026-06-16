# Encode Categorical as Float (RDB Transform)

## Status
Completed on branch `relbench_v2` (fastdfs + companion rdblearn changes).

## Goal
Support optional label-encoding of schema `category_t` RDB columns to float codes **before** `CanonicalizeTypes`, so DFS can apply numeric aggregation primitives (`mean`, `std`, `quantile_*`, `discrete_entropy`, …) on categorical base columns.

When encoding is disabled (default for callers that do not opt in), behavior is unchanged: `CanonicalizeTypes` still maps remaining `category_t` columns to string for categorical DFS primitives (`mode`, etc.).

## Motivation
RelBench / RDBLearn pipelines often need categoricals as numeric codes for downstream tabular models (TabPFN, LimiX, TabICL) and for float-style DFS aggregations. The inverse path—keeping strings—is useful when experimenting with categorical DFS features.

This feature spans two repos:
- **fastdfs** (this repo): RDB-level transform and shared encoding helper.
- **rdblearn** (sibling repo): `RDBLearnConfig.encode_categorical_as_float`, task-frame encoders, and pipeline wiring in `RDBLearnEstimator`.

See also: `/root/autodl-tmp/rdblearn_relbench_v2_20260525/IMPLEMENTATION_PLAN_encode_categorical_as_float.md`.

## fastdfs Changes

### New module: `fastdfs/transform/encode_categorical.py`
- `encode_series_with_label_encoder(series, encoder)` — maps known labels to float codes; unseen labels → `NaN`.
- `EncodeCategoryColumns` — `TableTransform` that label-encodes each `category_t` column, updates schema to `float_t`, and stores/reuses `LabelEncoder` instances keyed by `(table_name, column_name)`.
- `EncodeCategoryColumnsRDB` — optional `RDBTransform` wrapper (rdblearn uses `RDBTransformWrapper(EncodeCategoryColumns(...))` directly).

### Exports
- `fastdfs/transform/__init__.py` — exports `EncodeCategoryColumns`, `EncodeCategoryColumnsRDB`.

### Tests
- `tests/test_encode_categorical.py` — basic encode, unseen labels, encoder reuse, pipeline order with `CanonicalizeTypes`.

## Pipeline Order (rdblearn consumer)
When `encode_categorical_as_float=True`, rdblearn inserts:

```
HandleDummyTable → FillMissingPrimaryKey → FeaturizeDatetime → FilterColumn
  → EncodeCategoryColumns (optional)
  → CanonicalizeTypes
```

Encode **must** precede `CanonicalizeTypes`: encoded columns are `float_t` in metadata; non-encoded `category_t` columns still become strings.

## Usage

### Direct (fastdfs only)
```python
from fastdfs.transform import EncodeCategoryColumns, RDBTransformWrapper, CanonicalizeTypes

encode = EncodeCategoryColumns()
# Apply via RDBTransformPipeline or per-table transform(...)
```

### Via rdblearn (typical)
```python
from rdblearn.estimator import RDBLearnClassifier

clf = RDBLearnClassifier(
    base_estimator=...,
    config={"encode_categorical_as_float": True},
)
```

Library default is `False` (backward compatible). Callers that want mimic-style behavior pass `True` explicitly.

## Verification
- `pytest tests/test_encode_categorical.py` — **5 passed** (after `source /etc/network_turbo && pip install -e . -i https://pypi.org/simple/`).
- rdblearn integration: `tests/test_rdblearn_estimator.py` encode tests — **3 passed** (via editable fastdfs + `PYTHONPATH` in `bench_rdblearn` env).

## Out of Scope
- No new `rdblearn/scripts/run_mimic.py` on `relbench_v2`.
- No fastdfs version bump / PyPI publish in this change set (`rdblearn` still pins `fastdfs==0.2.1`; use editable install from this tree until republished).

## Follow-ups (optional)
- Bump `fastdfs` patch version and update rdblearn dependency pin.
- Wire `encode_categorical_as_float` in `prior_evaluation/rdblearn_experiments/relbench_v2/` experiment runner.
- Document flag in `rdblearn/examples/rdblearn_relbench_example.py`.
