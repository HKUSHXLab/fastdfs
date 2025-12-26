"""
Example of loading the CTU "Finance" dataset from MySQL and running DFS.
Dataset source: https://relational.fel.cvut.cz/dataset/Financial
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastdfs.adapter.mysql import MySQLAdapter
from fastdfs.api import compute_dfs_features
from fastdfs.dfs import DFSConfig
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
    print("=== MySQL CTU Finance Example ===")
    
    # 1. Connection details for CTU Relational Dataset Repository
    # Host: relational.fel.cvut.cz
    # User: guest
    # Password: ctu-relational
    # Database: financial
    connection_string = "mysql+pymysql://guest:ctu-relational@relational.fel.cvut.cz:3306/financial"
    
    print("Connecting to MySQL and loading RDB...")
    try:
        # Initialize MySQL adapter
        # This will automatically discover tables, primary keys, and foreign keys
        # We specify 'date' as the time column for tables that have it.
        time_columns = {
            "account": "date",
            "loan": "date",
            "trans": "date"
        }
        adapter = MySQLAdapter(
            connection_string=connection_string, 
            name="ctu_financial",
            time_columns=time_columns
        )
        rdb = adapter.load()
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Please ensure 'pymysql' is installed and you have internet access.")
        return

    print(f"Loaded RDB: {rdb.metadata.name}")
    print(f"Tables found: {rdb.table_names}")

    # 2. Prepare Target Dataframe
    # We use the 'loan' table as the target table.
    # The task is to predict loan 'status' (A vs B) at the time of 'date'.
    print("Preparing target dataframe from 'loan' table...")
    loan_df = rdb.get_table("loan")
    
    # Filter for finished loans (A: finished/OK, B: finished/not OK)
    # C and D are running loans which we exclude for this classification task.
    target_df = loan_df[loan_df['status'].isin(['A', 'B'])].copy()
    
    # Create a binary label: A -> 0 (Good), B -> 1 (Bad)
    target_df['label'] = target_df['status'].map({'A': 0, 'B': 1})
    
    # Ensure the timestamp column is in datetime format
    target_df['date'] = pd.to_datetime(target_df['date'])
    
    # Select necessary columns for DFS: ID, Timestamp, and Label
    # account_id is the link to the rest of the database
    target_df = target_df[['account_id', 'date', 'label']]
    
    # FastDFS RDB uses string types for primary/foreign keys for consistency
    target_df['account_id'] = target_df['account_id'].astype(str)
    
    print(f"Target dataframe shape: {target_df.shape}")
    print(target_df.head())
    
    # 3. Create transform pipeline
    # This prepares the RDB for DFS by handling missing tables, filling primary keys,
    # featurizing datetimes, and ensuring consistent types.
    print("\nApplying transform pipeline...")
    transform_pipeline = RDBTransformPipeline([
        HandleDummyTable(),
        FillMissingPrimaryKey(),
        RDBTransformWrapper(FeaturizeDatetime(features=["year", "month", "hour"])),
        RDBTransformWrapper(FilterColumn(drop_dtypes=["text"])),
        RDBTransformWrapper(CanonicalizeTypes())
    ])
    
    transformed_rdb = transform_pipeline(rdb)
    
    # Optional: Remove time_column from tables where it's just a creation date
    # to avoid over-filtering in DFS.
    for table_name in ["account", "client"]:
        if table_name in transformed_rdb.table_names:
            transformed_rdb.get_table_metadata(table_name).time_column = None

    # 4. Run Deep Feature Synthesis (DFS)
    print("\nComputing DFS features...")
    
    # Configure DFS
    # We want to aggregate information from other tables (trans, account, district, etc.)
    # based on the account_id and the time of the loan.
    config = DFSConfig(
        max_depth=2,
        agg_primitives=["mean", "sum", "count", "max", "min"],
    )
    
    # Compute features
    # target_dataframe: the loan table filtered for finished loans
    # key_mappings: links 'account_id' in target_df to 'account.account_id' in RDB
    # cutoff_time_column: 'date' ensures we only use data available at loan start
    feature_matrix = compute_dfs_features(
        rdb=transformed_rdb,
        target_dataframe=target_df,
        key_mappings={"account_id": "account.account_id"},
        cutoff_time_column="date",
        config=config
    )

    print("\nDFS Feature Matrix:")
    print(f"Shape: {feature_matrix.shape}")
    print("First few columns:")
    print(feature_matrix.columns[:10].tolist())
    print("\nSample data:")
    print(feature_matrix.head())

    # 4. Save results (optional)
    # feature_matrix.to_csv("finance_features.csv", index=False)
    # print("\nFeatures saved to finance_features.csv")

if __name__ == "__main__":
    main()
