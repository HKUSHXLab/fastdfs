"""
Example of loading DBInfer Diginetica dataset and running DFS on the ctr task.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import fastdfs
from fastdfs.adapter.dbinfer import DBInferAdapter
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

# Configure logging
configure_logging(level="INFO")

def main():
    print("=== DBInfer Diginetica Example ===")
    
    # 1. Load DBInfer dataset
    dataset_name = "diginetica"
    print(f"Loading {dataset_name}...")
    try:
        adapter = DBInferAdapter(dataset_name)
        rdb = adapter.load()
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Please ensure 'dbinfer-bench' is installed and the dataset is available.")
        return

    print(f"Loaded dataset: {rdb.metadata.name}")
    print(f"Tables: {rdb.table_names}")

    # 2. Get Task Data
    task_name = "ctr"
    print(f"Getting task: {task_name}")
    try:
        # Access the underlying dataset object stored in the adapter
        if adapter.dataset is None:
             print("Adapter dataset is None. Ensure load() was called.")
             return
             
        task = adapter.dataset.get_task(task_name)
        
        # train_set is a dict of numpy arrays, convert to DataFrame
        train_table = pd.DataFrame(task.train_set)
        
        # Ensure timestamp is datetime if it exists
        if "timestamp" in train_table.columns:
             train_table["timestamp"] = pd.to_datetime(train_table["timestamp"], unit='s' if train_table["timestamp"].dtype != 'datetime64[ns]' else None)

        print(f"Train table shape: {train_table.shape}")
        print(f"Train table columns: {train_table.columns.tolist()}")
    except Exception as e:
        print(f"Failed to get task: {e}")
        return

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

    for table in transformed_rdb.table_names:
        print(transformed_rdb.tables[table].info())
    
    # 5. Generate features
    print("Generating features...")
    
    # For CTR task in Diginetica:
    # The task table contains 'queryId', 'itemId', 'timestamp', 'clicked'.
    # We want to generate features for the query context (Query table) or item context (Product table).
    # Let's target the 'Query' table via 'queryId'.
    
    entity_col = "queryId"
    timestamp_col = "timestamp"
    
    if entity_col not in train_table.columns:
        print(f"Entity column {entity_col} not found in train table.")
        return

    # Find Query table
    query_table_name = "Query"
    if query_table_name not in rdb.table_names:
        print(f"Table {query_table_name} not found in RDB. Available: {rdb.table_names}")
        return
        
    # Get PK
    query_meta = rdb.get_table_metadata(query_table_name)
    query_pk = query_meta.primary_key
    
    # Find Product table
    product_table_name = "Product"
    if product_table_name not in rdb.table_names:
        # Fallback to 'Item' if Product not found, or just warn
        if "Item" in rdb.table_names:
            product_table_name = "Item"
        else:
            print(f"Table Product/Item not found in RDB. Available: {rdb.table_names}")
            return

    product_meta = rdb.get_table_metadata(product_table_name)
    product_pk = product_meta.primary_key

    print(f"Mapping {entity_col} to {query_table_name}.{query_pk}")
    print(f"Mapping itemId to {product_table_name}.{product_pk}")
    
    key_mappings = {
        entity_col: f"{query_table_name}.{query_pk}",
        "itemId": f"{product_table_name}.{product_pk}"
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
