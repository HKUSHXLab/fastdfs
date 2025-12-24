# High-Level API and Schema Inference

## Status
Completed.

## Changes
1.  **Relaxed `RDBColumnSchema`**:
    -   Modified `fastdfs/dataset/meta.py` to make `dtype` optional (`Optional[RDBColumnDType] = None`).

2.  **Created `InferSchemaTransform`**:
    -   Implemented in `fastdfs/transform/infer_schema.py`.
    -   Infers column types from pandas DataFrames.
    -   Uses hints for Primary Keys, Foreign Keys, and Time Columns.
    -   Validates that tables have at least one PK or FK (warns if not).

3.  **Created High-Level API `create_rdb`**:
    -   Added to `fastdfs/api.py`.
    -   Allows creating an `RDB` object directly from a dictionary of DataFrames.
    -   Automatically applies `InferSchemaTransform`.

## Verification
- Created `examples/test_create_rdb.py` to verify `create_rdb` with a sample dataset.
- Confirmed that schema inference works for PKs, FKs, time columns, and various data types.
