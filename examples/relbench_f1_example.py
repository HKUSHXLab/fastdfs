"""
Example of loading RelBench F1 dataset and running DFS on the driver-dnf task.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import fastdfs
from fastdfs.adapter.relbench import RelBenchAdapter
from fastdfs.transform import (
    RDBTransformPipeline,
    HandleDummyTable,
    FeaturizeDatetime,
    FilterColumn,
    RDBTransformWrapper,
    FillMissingPrimaryKey,
    CanonicalizeTypes,
)
from fastdfs.utils.logging_config import configure_logging
from relbench.tasks import get_task

# Configure logging
configure_logging(level="INFO")

def main():
    print("=== RelBench F1 Example ===")
    
    # 1. Load RelBench dataset
    dataset_name = "rel-f1"
    print(f"Loading {dataset_name}...")
    try:
        adapter = RelBenchAdapter(dataset_name)
        rdb = adapter.load()
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Please ensure 'relbench' is installed and the dataset is available.")
        return

    print(f"Loaded dataset: {rdb.metadata.name}")
    print(f"Tables: {rdb.table_names}")

    # 2. Get Task Data
    task_name = "driver-dnf"
    print(f"Getting task: {task_name}")
    task = get_task(dataset_name, task_name, download=True)
    train_table = task.get_table("train").df
    print(f"Train table shape: {train_table.shape}")
    print(f"Train table columns: {train_table.columns.tolist()}")

    # 3. Create transform pipeline
    transform_pipeline = RDBTransformPipeline([
        HandleDummyTable(),
        FillMissingPrimaryKey(),
        RDBTransformWrapper(FeaturizeDatetime(features=["year", "month", "hour"])),
        RDBTransformWrapper(FilterColumn(drop_dtypes=["text"])),
        RDBTransformWrapper(CanonicalizeTypes())
    ])
    
    # 4. Apply transforms
    print("Applying transforms...")
    transformed_rdb = transform_pipeline(rdb)
    
    # 5. Generate features
    print("Generating features...")
    
    # Infer key mapping
    # For driver-dnf, the entity is the driver.
    # We look for a column that links to the drivers table.
    
    # Identify driver column and timestamp column in the task table
    driver_col = next((c for c in train_table.columns if "driver" in c.lower() and "id" in c.lower()), None)
    timestamp_col = next((c for c in train_table.columns if "date" in c.lower() or "time" in c.lower()), None)
    
    if not driver_col or not timestamp_col:
        print(f"Could not infer columns. Found: {train_table.columns.tolist()}")
        # Fallback hardcoded for rel-f1 driver-dnf if inference fails
        driver_col = "driverId"
        timestamp_col = "date"
        if driver_col not in train_table.columns:
             print("Fallback columns not found either.")
             return

    print(f"Using Entity Column: {driver_col}")
    print(f"Using Timestamp Column: {timestamp_col}")

    # Find the drivers table in RDB
    # In rel-f1, the table is likely 'drivers'
    drivers_table_name = "drivers"
    if drivers_table_name not in rdb.table_names:
        print(f"Table {drivers_table_name} not found in RDB. Available: {rdb.table_names}")
        return
        
    # Get primary key of drivers table
    drivers_meta = rdb.get_table_metadata(drivers_table_name)
    drivers_pk = drivers_meta.primary_key
    
    print(f"Mapping {driver_col} to {drivers_table_name}.{drivers_pk}")
    
    key_mappings = {
        driver_col: f"{drivers_table_name}.{drivers_pk}"
    }
    
    # Use a small subset for demonstration
    target_df = train_table.head(50)
    
    features_df = fastdfs.compute_dfs_features(
        rdb=transformed_rdb,
        target_dataframe=target_df,
        key_mappings=key_mappings,
        cutoff_time_column=timestamp_col,
        config_overrides={
            "max_depth": 2,
        }
    )
    
    print(f"Generated {len(features_df.columns)} total columns")
    print("\nExample features:")
    for col in features_df.columns:
        if col not in train_table.columns:
            print(f"  - {col}")

if __name__ == "__main__":
    main()
