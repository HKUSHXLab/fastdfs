# Adapter Subpackage Plan

## Objective
Create an adapter subpackage to convert datasets from other formats (starting with RelBench) into the FastDFS format.

## Steps

1.  **Environment Setup**:
    - Install `relbench` and its dependencies.

2.  **Package Structure**:
    - Create `fastdfs/adapter/` directory.
    - Create `fastdfs/adapter/__init__.py`.

3.  **RelBench Adapter Implementation**:
    - Create `fastdfs/adapter/relbench.py`.
    - Implement `RelBenchAdapter` class or functions to:
        - Load RelBench dataset.
        - Convert RelBench tables to `RDBTableSchema` and `RDBColumnSchema`.
        - Handle data type mapping.
        - Handle foreign key relationships.
        - Save data to Parquet files.
        - Save metadata to `metadata.yaml`.

4.  **Testing**:
    - Create a script to test the adapter with a small RelBench dataset (e.g., `rel-stack` or `rel-hm`).
    - Verify the generated `metadata.yaml` and parquet files are compatible with `fastdfs.dataset.RDBDataset`.

## RelBench to FastDFS Mapping

- **Tables**: RelBench tables -> `RDBTableSchema`
- **Columns**: RelBench columns -> `RDBColumnSchema`
- **Types**:
    - Primary Key -> `RDBColumnDType.primary_key`
    - Foreign Key -> `RDBColumnDType.foreign_key`
    - Timestamp -> `RDBColumnDType.datetime_t`
    - Float/Int -> `RDBColumnDType.float_t` (or `category_t` if low cardinality)
    - Object/String -> `RDBColumnDType.text_t` (or `category_t`)

## Notes
- The provided script uses `DBB...` classes which seem to be the predecessors of `RDB...` classes in `fastdfs`. I will map them accordingly.
- `fastdfs` currently focuses on tables and relationships, not tasks (based on `RDBDataset` implementation). The provided script handles tasks, but `fastdfs` might not need them in the core dataset structure yet, or I might need to adapt how tasks are stored if `fastdfs` supports them.
- Looking at `fastdfs/dataset/rdb.py`, it says "Simplified RDB Dataset implementation without tasks". So I will focus on converting the tables and relationships first.
