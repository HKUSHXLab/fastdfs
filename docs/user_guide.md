# FastDFS User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Getting Started](#getting-started)
4. [RDB Format](#rdb-format)
5. [Basic Usage](#basic-usage)
6. [Transform Pipeline](#transform-pipeline)
7. [Advanced Configuration](#advanced-configuration)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Introduction

### What is Deep Feature Synthesis?

Deep Feature Synthesis (DFS) is an automated feature engineering technique that creates new features by applying aggregation functions across relationships in relational data. Instead of manually crafting features, DFS automatically discovers patterns and generates meaningful aggregations.

**Example**: Given user-item interaction data, DFS can automatically create features like:
- `user_avg_rating`: Average rating given by each user
- `item_count_interactions`: Number of times each item was interacted with
- `user_std_rating_last_30days`: Standard deviation of user ratings in the last 30 days

### Why FastDFS?

FastDFS takes a **table-centric approach** to feature engineering:

- **Flexible Target Data**: Generate features for any dataframe, not just predefined datasets
- **Temporal Consistency**: Built-in cutoff time support prevents data leakage
- **Multiple Engines**: Choose the best engine for your data size and performance needs
- **Composable Transforms**: Clean and preprocess data before feature generation

## Core Concepts

### 1. Relational Database (RDB)

An RDB in FastDFS represents your data as a collection of related tables, similar to a traditional database:

```
Users Table          Items Table         Interactions Table
+---------+-------+  +----------+-----+  +---------+----------+-----+------+
| user_id | age   |  | item_id  | cat |  | user_id | item_id  | ts  | rating |
+---------+-------+  +----------+-----+  +---------+----------+-----+------+
| 1       | 25    |  | 100      | A   |  | 1       | 100      | ... | 4.5  |
| 2       | 30    |  | 200      | B   |  | 2       | 200      | ... | 3.0  |
+---------+-------+  +----------+-----+  +---------+----------+-----+------+
```

### 2. Target Dataframe

The target dataframe is what you want to augment with features. It can be:
- Training data for machine learning
- New prediction instances
- Any dataframe with columns that can be mapped to your RDB

```python
target_df = pd.DataFrame({
    "user_id": [1, 2, 3],
    "item_id": [100, 200, 300],
    "prediction_time": ["2024-01-01", "2024-01-02", "2024-01-03"]
})
```

### 3. Key Mappings

Key mappings connect columns in your target dataframe to primary keys in your RDB:

```python
key_mappings = {
    "user_id": "user.user_id",    # target_df.user_id -> users table primary key
    "item_id": "item.item_id"     # target_df.item_id -> items table primary key
}
```

### 4. Cutoff Time

Cutoff time ensures temporal consistency by only using RDB data that occurred before each target dataframe row's cutoff time:

```python
# Only use interaction data before each prediction_time
cutoff_time_column = "prediction_time"
```

## Getting Started

### Installation

```bash
pip install fastdfs
```

### Your First Feature Generation

Let's walk through a complete example using e-commerce data:

#### Step 1: Prepare Your RDB

Create a directory structure:
```
ecommerce_rdb/
├── metadata.yaml
└── data/
    ├── users.npz
    ├── items.npz
    └── interactions.npz
```

**metadata.yaml**:
```yaml
name: ecommerce_rdb
tables:
- name: users
  source: data/users.npz
  columns:
  - name: user_id
    dtype: primary_key
  - name: age
    dtype: float
  - name: registration_date
    dtype: datetime
    
- name: items
  source: data/items.npz
  columns:
  - name: item_id
    dtype: primary_key
  - name: category
    dtype: category
  - name: price
    dtype: float
    
- name: interactions
  source: data/interactions.npz
  columns:
  - name: user_id
    dtype: foreign_key
    link_to: users.user_id
  - name: item_id
    dtype: foreign_key
    link_to: items.item_id
  - name: timestamp
    dtype: datetime
  - name: rating
    dtype: float
  - name: purchase_amount
    dtype: float
  time_column: timestamp
```

#### Step 2: Generate Features

```python
import fastdfs
import pandas as pd

# Load the RDB
rdb = fastdfs.load_rdb("ecommerce_rdb/")

# Create target dataframe
target_df = pd.DataFrame({
    "user_id": [1, 2, 3],
    "item_id": [100, 200, 300],
    "prediction_time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
})

# Generate features
features = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=target_df,
    key_mappings={
        "user_id": "users.user_id",
        "item_id": "items.item_id"
    },
    cutoff_time_column="prediction_time"
)

print(f"Generated {len(features.columns)} total features")
print(f"New features: {len(features.columns) - len(target_df.columns)}")
```

## RDB Format

### Metadata Schema

The `metadata.yaml` file defines your RDB structure:

```yaml
name: your_dataset_name  # Required: Name of the dataset

tables:                          # Required: List of tables
- name: table_name              # Required: Table name
  source: data/table.npz        # Required: Path to data file (npz or parquet)
  columns:                      # Required: Column definitions
  - name: column_name           # Required: Column name
    dtype: column_type          # Required: Data type (see below)
    link_to: target_table.column # Optional: For foreign keys
  time_column: timestamp_col    # Optional: Primary time column for the table
```

### Supported Data Types

| dtype | Description | Example |
|-------|-------------|---------|
| `primary_key` | Unique identifier | `user_id`, `item_id` |
| `foreign_key` | Reference to another table | `user_id` in interactions table |
| `datetime` | Timestamp column | `2024-01-01 10:30:00` |
| `float` | Numeric values | `4.5`, `29.99` |
| `int` | Integer values | `1`, `100` |
| `category` | Categorical data | `"electronics"`, `"books"` |
| `text` | Text data | `"product description"` |
| `boolean` | True/False values | `True`, `False` |

### Data File Formats

FastDFS supports two data formats:

**NPZ Format** (recommended for mixed types):
```python
import numpy as np
import pandas as pd

# Save dataframe as npz
df = pd.DataFrame({"user_id": [1, 2, 3], "age": [25, 30, 35]})
arrays_dict = {col: df[col].values for col in df.columns}
np.savez("users.npz", **arrays_dict)
```

**Parquet Format** (good for large datasets):
```python
df.to_parquet("users.parquet", index=False)
```

## Basic Usage

### Loading an RDB

```python
import fastdfs

# Load from directory with metadata.yaml
rdb = fastdfs.load_rdb("path/to/rdb/")

# Inspect the RDB
print(f"Dataset: {rdb.metadata.name}")
print(f"Tables: {rdb.table_names}")
print(f"Relationships: {rdb.get_relationships()}")

# Access individual tables
users_df = rdb.get_table("users")
interactions_df = rdb.get_table("interactions")
```

### Computing Features

The main function for feature generation:

```python
features = fastdfs.compute_dfs_features(
    rdb=rdb,                           # Your loaded RDB
    target_dataframe=target_df,        # Dataframe to augment
    key_mappings=key_mappings,         # Column mappings
    cutoff_time_column="timestamp",    # Optional: temporal cutoff
    config_overrides={                 # Optional: DFS parameters
        "max_depth": 2,
        "engine": "featuretools",
        "agg_primitives": ["count", "mean", "max", "min"]
    }
)
```

### Working with Different Target DataFrames

FastDFS can augment any dataframe, not just data from your RDB:

```python
# Training data
train_df = pd.read_csv("train.csv")
train_features = fastdfs.compute_dfs_features(rdb, train_df, key_mappings)

# Test data  
test_df = pd.read_csv("test.csv")
test_features = fastdfs.compute_dfs_features(rdb, test_df, key_mappings)

# New prediction instances
new_data = pd.DataFrame({
    "user_id": [999],
    "item_id": [888], 
    "prediction_time": ["2024-12-01"]
})
new_features = fastdfs.compute_dfs_features(rdb, new_data, key_mappings)
```

## Transform Pipeline

Transforms preprocess your RDB before feature generation. They're useful for data cleaning, feature engineering, and handling data quality issues.

### Available Transforms

```python
from fastdfs.transform import (
    RDBTransformPipeline,
    HandleDummyTable,
    FeaturizeDatetime, 
    FilterColumn,
    RDBTransformWrapper
)
```

#### HandleDummyTable

Removes or processes dummy/placeholder tables:

```python
transform = HandleDummyTable()
cleaned_rdb = transform(rdb)
```

#### FeaturizeDatetime

Extracts datetime components (year, month, day, hour, etc.):

```python
# Apply to all datetime columns
transform = RDBTransformWrapper(
    FeaturizeDatetime(features=["year", "month", "day", "hour", "dayofweek"])
)
rdb_with_time_features = transform(rdb)
```

#### FilterColumn

Removes redundant or problematic columns:

```python
transform = RDBTransformWrapper(
    FilterColumn(drop_redundant=True, min_unique_values=2)
)
filtered_rdb = transform(rdb)
```

### Creating Transform Pipelines

Combine multiple transforms:

```python
pipeline = RDBTransformPipeline([
    HandleDummyTable(),
    RDBTransformWrapper(FeaturizeDatetime(features=["year", "month", "hour"])),
    RDBTransformWrapper(FilterColumn(drop_redundant=True))
])

# Apply pipeline
transformed_rdb = pipeline(rdb)

# Then generate features
features = fastdfs.compute_dfs_features(
    rdb=transformed_rdb,
    target_dataframe=target_df,
    key_mappings=key_mappings
)
```

### Using DFSPipeline

Combine transforms with feature generation:

```python
pipeline = fastdfs.DFSPipeline(
    transform_pipeline=RDBTransformPipeline([
        HandleDummyTable(),
        RDBTransformWrapper(FeaturizeDatetime(features=["year", "month"]))
    ]),
    dfs_config=fastdfs.DFSConfig(max_depth=2, engine="dfs2sql")
)

features = pipeline.run(
    rdb=rdb,
    target_dataframe=target_df,
    key_mappings=key_mappings,
    cutoff_time_column="timestamp"
)
```

## Advanced Configuration

### DFS Configuration

Customize feature generation with `DFSConfig`:

```python
config = fastdfs.DFSConfig(
    agg_primitives=["count", "mean", "max", "min", "std", "sum"],
    max_depth=3,                    # How deep to traverse relationships
    use_cutoff_time=True,          # Enable temporal consistency
    engine="dfs2sql"               # Choose engine: "featuretools" or "dfs2sql"
)

features = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=target_df,
    key_mappings=key_mappings,
    config=config
)
```

### Engine Selection

#### Featuretools Engine
- **Best for**: Small to medium datasets (< 1M rows)
- **Pros**: Rich set of aggregation primitives, mature ecosystem
- **Cons**: Memory intensive, slower on large data

```python
config = fastdfs.DFSConfig(engine="featuretools", max_depth=2)
```

#### DFS2SQL Engine  
- **Best for**: Large datasets (> 1M rows)
- **Pros**: High performance, low memory usage, SQL-based
- **Cons**: Limited primitive set, newer codebase

```python
config = fastdfs.DFSConfig(engine="dfs2sql", engine_path="/tmp/duckdb.db")
```

### Runtime Configuration Overrides

Override config parameters at runtime:

```python
# Use base config
base_config = fastdfs.DFSConfig(max_depth=2, engine="featuretools")

# Override for specific computation
features = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=target_df, 
    key_mappings=key_mappings,
    config=base_config,
    config_overrides={
        "max_depth": 3,                    # Override depth
        "engine": "dfs2sql",               # Override engine
        "agg_primitives": ["count", "mean"] # Override primitives
    }
)
```

## Best Practices

### 1. Preventing Data Leakage

**Always use cutoff times** when generating features for ML:

```python
# ✅ Good: Use cutoff time
train_features = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=train_df,
    key_mappings=key_mappings,
    cutoff_time_column="interaction_time"  # Only use data before this time
)

# ❌ Bad: No cutoff time (uses all data)
train_features = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=train_df,
    key_mappings=key_mappings
    # Missing cutoff_time_column!
)
```

### 2. Handling Train/Test Splits

Apply transforms to RDB once, then generate features separately:

```python
# Clean RDB once
clean_rdb = transform_pipeline(rdb)

# Generate features for each split with proper cutoffs
train_features = fastdfs.compute_dfs_features(
    rdb=clean_rdb,
    target_dataframe=train_df,
    key_mappings=key_mappings,
    cutoff_time_column="timestamp"
)

test_features = fastdfs.compute_dfs_features(
    rdb=clean_rdb, 
    target_dataframe=test_df,
    key_mappings=key_mappings,
    cutoff_time_column="timestamp"
)
```

### 3. Performance Optimization

**For large datasets**:
- Use `engine="dfs2sql"` for better performance
- Start with `max_depth=1` and increase gradually
- Filter columns before DFS to reduce computation

```python
# Optimize for large data
config = fastdfs.DFSConfig(
    engine="dfs2sql",
    max_depth=1,  # Start shallow
    agg_primitives=["count", "mean"]  # Use fewer primitives
)
```

**For small datasets**:
- `engine="featuretools"` provides richer features
- Higher `max_depth` values are feasible

### 4. Feature Selection

Remove redundant or problematic features:

```python
# Generate all features first
all_features = fastdfs.compute_dfs_features(rdb, target_df, key_mappings)

# Remove features that might leak target information
safe_features = all_features.drop(columns=[
    col for col in all_features.columns 
    if any(keyword in col.lower() for keyword in ['label', 'target', 'ground_truth'])
])

# Remove high-cardinality features that might overfit
unique_counts = safe_features.nunique()
final_features = safe_features.drop(columns=[
    col for col in safe_features.columns 
    if unique_counts[col] > len(safe_features) * 0.95  # > 95% unique values
])
```

### 5. Memory Management

For very large datasets:

```python
# Process in chunks
chunk_size = 10000
all_features = []

for i in range(0, len(target_df), chunk_size):
    chunk = target_df.iloc[i:i+chunk_size]
    chunk_features = fastdfs.compute_dfs_features(
        rdb=rdb,
        target_dataframe=chunk,
        key_mappings=key_mappings,
        config_overrides={"engine": "dfs2sql"}
    )
    all_features.append(chunk_features)

final_features = pd.concat(all_features, ignore_index=True)
```

## Troubleshooting

### Common Issues

#### 1. "Table not found" Error
```
KeyError: "Table 'users' not found in RDB"
```

**Solution**: Check your `metadata.yaml` table names match the key mappings:
```python
# Check available tables
print(rdb.table_names)

# Fix key mapping
key_mappings = {
    "user_id": "user.user_id"  # Use correct table name
}
```

#### 2. "Column not found" Error
```
KeyError: "Column 'user_id' not found in table 'users'"
```

**Solution**: Verify column names in your data:
```python
# Check table schema
print(rdb.get_table_metadata("users").columns)

# Check actual data
print(rdb.get_table("users").columns.tolist())
```

#### 3. Memory Issues with Large Data

**Solution**: Use DFS2SQL engine and reduce feature depth:
```python
config = fastdfs.DFSConfig(
    engine="dfs2sql",
    max_depth=1,
    agg_primitives=["count", "mean"]  # Fewer primitives
)
```

#### 4. No Features Generated

**Possible causes**:
- Incorrect key mappings
- No relationships found
- Cutoff time too restrictive

**Debug steps**:
```python
# Check relationships
print(rdb.get_relationships())

# Check target dataframe alignment
target_sample = target_df.head()
print("Target columns:", target_sample.columns.tolist())
print("Key mappings:", key_mappings)

# Check data overlap
users_in_target = set(target_df['user_id'].unique())
users_in_rdb = set(rdb.get_table('users')['user_id'].unique())
print(f"Overlap: {len(users_in_target & users_in_rdb)} users")
```

### Getting Help

1. **Check the logs**: FastDFS provides detailed logging
   ```python
   from fastdfs.utils.logging_config import configure_logging
   configure_logging(level="DEBUG")
   ```

2. **Inspect intermediate results**:
   ```python
   # Check RDB after transforms
   transformed_rdb = transform_pipeline(rdb)
   print(transformed_rdb.table_names)
   
   # Check feature names
   print([col for col in features.columns if col not in target_df.columns])
   ```

3. **Start simple**: Begin with minimal configuration and add complexity gradually

For more help, see the [API Reference](api_reference.md) or check the project's GitHub issues.
