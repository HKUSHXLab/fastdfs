# FastDFS - Deep Feature Synthesis for Tabular Data

FastDFS is a Python library for automated feature engineering using Deep Feature Synthesis (DFS). It augments target dataframes with rich features derived from relational database structures, making it easy to create powerful features for machine learning without manual feature engineering.

## Core Concept

FastDFS treats feature engineering as a **table augmentation process**: given any target dataframe and a relational database (RDB) containing related tables, it automatically generates new features by aggregating information across relationships.

```python
# Your target dataframe (what you want to predict on)
target_df = pd.DataFrame({
    "user_id": [1, 2, 3],
    "item_id": [100, 200, 300], 
    "interaction_time": ["2024-01-01", "2024-01-02", "2024-01-03"]
})

# Your relational database (context for feature generation)
rdb = fastdfs.load_rdb("ecommerce_data/")  # Contains user, item, interaction tables

# Generate features automatically
enriched_df = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=target_df,
    key_mappings={"user_id": "user.user_id", "item_id": "item.item_id"},
    cutoff_time_column="interaction_time"
)
# Result: Original columns + 50+ new features like user_avg_rating, item_count_purchases, etc.
```

## Installation

```bash
pip install fastdfs
```

Or for development:
```bash
git clone https://github.com/dglai/fastdfs.git
cd fastdfs
pip install -e .
```

## Quick Start

### 1. Prepare Your Data

FastDFS provides multiple ways to prepare your relational data.

#### Option A: Create from DataFrames (Recommended)

You can create an RDB directly from pandas DataFrames. FastDFS will automatically infer the schema.

```python
import fastdfs
import pandas as pd

# 1. Define your tables
users_df = pd.DataFrame(...)
items_df = pd.DataFrame(...)
interactions_df = pd.DataFrame(...)

# 2. Create RDB with relationships
rdb = fastdfs.create_rdb(
    name="ecommerce",
    tables={
        "user": users_df,
        "item": items_df,
        "interaction": interactions_df
    },
    primary_keys={
        "user": "user_id",
        "item": "item_id"
    },
    foreign_keys=[
        ("interaction", "user_id", "user", "user_id"),
        ("interaction", "item_id", "item", "item_id")
    ],
    time_columns={
        "interaction": "timestamp"
    }
)

# 3. Save for later use
rdb.save("ecommerce_rdb/")

# 4. Load it back
rdb = fastdfs.load_rdb("ecommerce_rdb/")
```

#### Option B: Adapt Existing Datasets

FastDFS includes adapters for popular relational dataset benchmarks.

**RelBench**
```python
from fastdfs.adapter.relbench import RelBenchAdapter

# Load and convert RelBench dataset
adapter = RelBenchAdapter("rel-stack")
rdb = adapter.load()
rdb.save("rel-stack-rdb/")
```

**DBInfer**
```python
from fastdfs.adapter.dbinfer import DBInferAdapter

# Load and convert DBInfer dataset
adapter = DBInferAdapter("diginetica")
rdb = adapter.load()
rdb.save("diginetica-rdb/")
```

#### Option C: Manual Schema Definition

Structure your relational data as an RDB with a `metadata.yaml` file:

```yaml
# metadata.yaml
name: ecommerce_rdb
tables:
- name: user
  source: data/user.npz
  columns:
  - name: user_id
    dtype: primary_key
  - name: age  
    dtype: float
    
- name: item
  source: data/item.npz
  columns:
  - name: item_id
    dtype: primary_key
  - name: category
    dtype: category
    
- name: interaction
  source: data/interaction.npz
  columns:
  - name: user_id
    dtype: foreign_key
    link_to: user.user_id
  - name: item_id
    dtype: foreign_key  
    link_to: item.item_id
  - name: timestamp
    dtype: datetime
  - name: rating
    dtype: float
```

### 2. Generate Features

```python
import fastdfs
import pandas as pd

# Load your relational database
rdb = fastdfs.load_rdb("path/to/your/rdb")

# Create or load your target dataframe
target_df = pd.DataFrame({
    "user_id": [1, 2, 3],
    "item_id": [10, 20, 30],
    "prediction_time": ["2024-01-01", "2024-01-02", "2024-01-03"]
})

# Generate features
features = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=target_df, 
    key_mappings={
        "user_id": "user.user_id",
        "item_id": "item.item_id"  
    },
    cutoff_time_column="prediction_time",
    config_overrides={"max_depth": 2}
)

print(f"Original columns: {len(target_df.columns)}")
print(f"With features: {len(features.columns)}")
```

### 3. Advanced Usage with Transforms

```python
# Apply preprocessing transforms before feature generation
from fastdfs.transform import RDBTransformWrapper, RDBTransformPipeline, HandleDummyTable, FeaturizeDatetime

pipeline = fastdfs.DFSPipeline(
    transform_pipeline=RDBTransformPipeline([
        HandleDummyTable(),
        RDBTransformWrapper(FeaturizeDatetime(features=["year", "month", "hour"]))
    ]),
    dfs_config=fastdfs.DFSConfig(max_depth=3, engine="dfs2sql")
)

features = pipeline.run(
    rdb=rdb,
    target_dataframe=target_df,
    key_mappings=key_mappings,
    cutoff_time_column="prediction_time"
)
```

## Key Features

- **Table-Centric Design**: Augment any dataframe, not just predefined datasets
- **Multiple DFS Engines**: Choose between Featuretools (pandas) or DFS2SQL (high-performance)
- **Temporal Consistency**: Built-in cutoff time support prevents data leakage
- **Flexible Key Mapping**: Connect target data to RDB with simple column mappings
- **Transform Pipeline**: Composable preprocessing transforms for data cleaning
- **Type Safety**: Full type hints and runtime validation
- **Minimal Dependencies**: Focused, lightweight package

## Engine Comparison

| Feature | Featuretools | DFS2SQL |
|---------|-------------|---------|
| Performance | Good for small data | Excellent for large data |
| Memory Usage | High (pandas) | Low (SQL-based) |
| Primitives | Rich set | Core primitives |
| Backend | Pandas | DuckDB |

## Documentation

- **[User Guide](docs/user_guide.md)**: Complete tutorial with concepts and examples
- **[API Reference](docs/api_reference.md)**: Detailed API documentation
- **[Examples](examples/)**: Runnable code examples

## Why FastDFS?

**Before FastDFS** (manual feature engineering):
```python
# Manual aggregations for each feature
user_avg_rating = interactions.groupby('user_id')['rating'].mean()
user_total_purchases = interactions.groupby('user_id').size()
item_avg_rating = interactions.groupby('item_id')['rating'].mean()
# ... dozens more features ...
```

**With FastDFS** (automated):
```python
# Automatic generation of 50+ features
features = fastdfs.compute_dfs_features(rdb, target_df, key_mappings)
```

FastDFS automatically discovers relationships in your data and generates meaningful aggregation features, saving weeks of manual feature engineering work.

## Contributing

We welcome contributions! See our [development logs](dev_logs/) for project history and architecture decisions.

## License

Apache-2.0 License
