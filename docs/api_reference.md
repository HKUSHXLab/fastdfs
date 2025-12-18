# FastDFS API Reference

## Table of Contents

1. [Core API Functions](#core-api-functions)
2. [Configuration Classes](#configuration-classes)
3. [Dataset Classes](#dataset-classes)
4. [Transform Classes](#transform-classes)
5. [Engine Classes](#engine-classes)
6. [Utility Functions](#utility-functions)

## Core API Functions

### `load_rdb(path: str) -> RDBDataset`

Load a relational database dataset from a directory.

**Parameters:**
- `path` (str): Path to the directory containing `metadata.yaml` and data files

**Returns:**
- `RDBDataset`: The loaded relational database dataset

**Example:**
```python
import fastdfs

rdb = fastdfs.load_rdb("path/to/ecommerce_rdb/")
print(f"Loaded dataset: {rdb.metadata.dataset_name}")
print(f"Tables: {rdb.table_names}")
```

**Raises:**
- `FileNotFoundError`: If the path doesn't exist or `metadata.yaml` is missing
- `ValueError`: If the metadata format is invalid

---

### `compute_dfs_features(...) -> pd.DataFrame`

Compute Deep Feature Synthesis features for a target dataframe using RDB context.

```python
def compute_dfs_features(
    rdb: RDBDataset,
    target_dataframe: pd.DataFrame,
    key_mappings: Dict[str, str],
    cutoff_time_column: Optional[str] = None,
    config: Optional[DFSConfig] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> pd.DataFrame
```

**Parameters:**
- `rdb` (RDBDataset): The relational database providing context for feature generation
- `target_dataframe` (pd.DataFrame): DataFrame to augment with features
- `key_mappings` (Dict[str, str]): Map target columns to RDB primary keys
  - Format: `{"target_col": "table.primary_key_col"}`
- `cutoff_time_column` (Optional[str]): Column name for temporal cutoff
- `config` (Optional[DFSConfig]): DFS configuration (uses defaults if None)
- `config_overrides` (Optional[Dict[str, Any]]): Override specific config parameters

**Returns:**
- `pd.DataFrame`: Original target data plus generated features

**Example:**
```python
features = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=target_df,
    key_mappings={"user_id": "users.user_id", "item_id": "items.item_id"},
    cutoff_time_column="interaction_time",
    config_overrides={"max_depth": 2, "engine": "dfs2sql"}
)
```

**Key Mappings Format:**
```python
key_mappings = {
    "target_column": "rdb_table.primary_key_column",
    "user_id": "users.user_id",           # target_df.user_id -> users.user_id
    "product_id": "products.product_id"   # target_df.product_id -> products.product_id
}
```

---

### `DFSPipeline`

Pipeline class for combining RDB transforms with DFS feature computation.

```python
class DFSPipeline:
    def __init__(
        self,
        transform_pipeline = None,
        dfs_config: Optional[DFSConfig] = None
    )
    
    def run(
        self,
        rdb: RDBDataset,
        target_dataframe: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame
```

**Parameters:**
- `transform_pipeline`: RDB transform pipeline to apply before DFS
- `dfs_config` (Optional[DFSConfig]): DFS configuration

**Example:**
```python
from fastdfs.transform import RDBTransformPipeline, HandleDummyTable, FeaturizeDatetime

pipeline = fastdfs.DFSPipeline(
    transform_pipeline=RDBTransformPipeline([
        HandleDummyTable(),
        RDBTransformWrapper(FeaturizeDatetime(features=["year", "month"]))
    ]),
    dfs_config=fastdfs.DFSConfig(max_depth=2, engine="featuretools")
)

features = pipeline.run(
    rdb=rdb,
    target_dataframe=target_df,
    key_mappings=key_mappings,
    cutoff_time_column="timestamp"
)
```

## Configuration Classes

### `DFSConfig`

Configuration for Deep Feature Synthesis parameters.

```python
class DFSConfig(pydantic.BaseModel):
    agg_primitives: List[str] = ["max", "min", "mean", "count", "mode", "std"]
    max_depth: int = 2
    use_cutoff_time: bool = True
    engine: str = "featuretools"
    engine_path: Optional[str] = "/tmp/duck.db"
```

**Attributes:**
- `agg_primitives` (List[str]): Aggregation primitives to use for feature generation
- `max_depth` (int): Maximum depth for relationship traversal
- `use_cutoff_time` (bool): Whether to respect temporal cutoff times
- `engine` (str): DFS engine to use ("featuretools" or "dfs2sql")
- `engine_path` (Optional[str]): Path for engine-specific configuration (e.g., DuckDB file)

**Available Aggregation Primitives:**

| Primitive | Description | Example |
|-----------|-------------|---------|
| `count` | Count of related records | Number of user interactions |
| `mean` | Average value | Average user rating |
| `max` | Maximum value | Highest item price |
| `min` | Minimum value | Lowest rating given |
| `std` | Standard deviation | Rating variability |
| `sum` | Sum of values | Total purchase amount |
| `mode` | Most frequent value | Most common category |
| `nunique` | Count of unique values | Number of unique items |

**Example:**
```python
config = fastdfs.DFSConfig(
    agg_primitives=["count", "mean", "max", "min"],
    max_depth=3,
    engine="dfs2sql",
    use_cutoff_time=True
)
```

**Engine Comparison:**

| Feature | featuretools | dfs2sql |
|---------|-------------|---------|
| **Performance** | Good for small data (< 1M rows) | Excellent for large data (> 1M rows) |
| **Memory Usage** | High (pandas-based) | Low (SQL-based) |
| **Primitives** | Full set supported | Core primitives only |
| **Backend** | Pandas operations | DuckDB SQL engine |
| **Maturity** | Stable, well-tested | Newer, high-performance |

## Dataset Classes

### `RDBDataset`

Represents a relational database for feature engineering.

```python
class RDBDataset:
    def __init__(self, path: Path)
    
    @property
    def table_names(self) -> List[str]
    
    def get_table(self, name: str) -> pd.DataFrame
    
    def get_table_metadata(self, name: str) -> RDBTableSchema
    
    def get_relationships(self) -> List[Tuple[str, str, str, str]]
    
    def create_new_with_tables(self, new_tables: Dict[str, pd.DataFrame]) -> 'RDBDataset'
```

**Properties:**
- `metadata`: Dataset metadata from `metadata.yaml`
- `table_names`: List of table names in the dataset
- `tables`: Dictionary mapping table names to DataFrames

**Methods:**

#### `get_table(name: str) -> pd.DataFrame`
Get a table as a pandas DataFrame.

**Parameters:**
- `name` (str): Name of the table to retrieve

**Returns:**
- `pd.DataFrame`: The requested table

#### `get_table_metadata(name: str) -> RDBTableSchema`
Get metadata schema for a specific table.

**Parameters:**
- `name` (str): Name of the table

**Returns:**
- `RDBTableSchema`: Metadata schema for the table

#### `get_relationships() -> List[Tuple[str, str, str, str]]`
Get relationships between tables.

**Returns:**
- List of tuples: `(child_table, child_column, parent_table, parent_column)`

**Example:**
```python
rdb = fastdfs.load_rdb("ecommerce_rdb/")

# Access tables
users_df = rdb.get_table("users")
items_df = rdb.get_table("items")

# Check relationships
relationships = rdb.get_relationships()
print(relationships)
# [('interactions', 'user_id', 'users', 'user_id'),
#  ('interactions', 'item_id', 'items', 'item_id')]

# Get table metadata
user_meta = rdb.get_table_metadata("users")
print([col.name for col in user_meta.columns])
```

### `RDBTableSchema`

Schema definition for a table in the RDB.

```python
class RDBTableSchema:
    name: str
    columns: List[RDBColumnSchema]
    source: str
    time_column: Optional[str] = None
```

### `RDBColumnSchema`

Schema definition for a column in an RDB table.

```python
class RDBColumnSchema:
    name: str
    dtype: str
    link_to: Optional[str] = None
```

**Supported Data Types:**
- `primary_key`: Unique identifier column
- `foreign_key`: Reference to another table (requires `link_to`)
- `datetime`: Timestamp data
- `float`: Floating-point numbers
- `int`: Integer numbers
- `category`: Categorical data
- `text`: Text/string data
- `boolean`: True/False values

## Transform Classes

### `RDBTransform`

Base class for RDB transformations.

```python
class RDBTransform:
    def __call__(self, rdb: RDBDataset) -> RDBDataset
```

All transforms are pure functions that take an RDB and return a new RDB.

### `RDBTransformPipeline`

Pipeline for composing multiple RDB transforms.

```python
class RDBTransformPipeline:
    def __init__(self, transforms: List[RDBTransform])
    
    def __call__(self, rdb: RDBDataset) -> RDBDataset
```

**Example:**
```python
from fastdfs.transform import RDBTransformPipeline, HandleDummyTable

pipeline = RDBTransformPipeline([
    HandleDummyTable(),
    RDBTransformWrapper(FeaturizeDatetime(features=["year", "month"])),
    RDBTransformWrapper(FilterColumn(drop_redundant=True))
])

transformed_rdb = pipeline(rdb)
```

### `HandleDummyTable`

Transform that removes or processes dummy/placeholder tables.

```python
class HandleDummyTable(RDBTransform):
    def __call__(self, rdb: RDBDataset) -> RDBDataset
```

**Purpose:** Remove tables that are placeholders or contain no useful information.

### `RDBTransformWrapper`

Wrapper to apply table-level or column-level transforms to an entire RDB.

```python
class RDBTransformWrapper(RDBTransform):
    def __init__(self, inner_transform: Union[TableTransform, ColumnTransform])
```

**Example:**
```python
# Wrap a column transform to apply to all tables
datetime_transform = RDBTransformWrapper(
    FeaturizeDatetime(features=["year", "month", "day"])
)
rdb_with_time_features = datetime_transform(rdb)
```

### `CanonicalizeTypes`

Transform that enforces data types in RDB tables according to their metadata schema.

```python
class CanonicalizeTypes(TableTransform):
    def __call__(self, table: pd.DataFrame, table_metadata: RDBTableSchema) -> Tuple[pd.DataFrame, RDBTableSchema]
```

**Purpose:** 
- Casts columns to the types defined in `metadata.yaml` (e.g., `float`, `datetime`, `category`).
- Handles missing values and coercion errors safely.
- Drops columns that are present in the data but not defined in the metadata.
- Raises an error if a column defined in the metadata is missing from the data.

**Example:**
```python
transform = RDBTransformWrapper(CanonicalizeTypes())
clean_rdb = transform(rdb)
```

### `FillMissingPrimaryKey`

Transform that fills missing values in primary key columns.

```python
class FillMissingPrimaryKey(TableTransform):
    def __call__(self, table: pd.DataFrame, table_metadata: RDBTableSchema) -> Tuple[pd.DataFrame, RDBTableSchema]
```

**Purpose:** Ensures primary key integrity by filling missing values (NaNs) with a generated unique identifier or a placeholder.

### `FeaturizeDatetime`

Transform that extracts datetime components from datetime columns.

```python
class FeaturizeDatetime(ColumnTransform):
    def __init__(self, features: List[str] = ["year", "month", "day"])
```

**Parameters:**
- `features` (List[str]): List of datetime components to extract

**Available Features:**
- `year`: Year (e.g., 2024)
- `month`: Month (1-12)
- `day`: Day of month (1-31)
- `hour`: Hour (0-23)
- `minute`: Minute (0-59)
- `second`: Second (0-59)
- `dayofweek`: Day of week (0=Monday, 6=Sunday)
- `dayofyear`: Day of year (1-366)
- `quarter`: Quarter (1-4)

**Example:**
```python
transform = RDBTransformWrapper(
    FeaturizeDatetime(features=["year", "month", "hour", "dayofweek"])
)
```

### `FilterColumn`

Transform that removes columns based on various criteria.

```python
class FilterColumn(ColumnTransform):
    def __init__(
        self,
        drop_redundant: bool = False,
        min_unique_values: int = 1,
        max_unique_ratio: float = 1.0,
        drop_dtypes: Optional[List[str]] = None
    )
```

**Parameters:**
- `drop_redundant` (bool): Remove columns with all identical values
- `min_unique_values` (int): Minimum number of unique values required
- `max_unique_ratio` (float): Maximum ratio of unique values to total rows
- `drop_dtypes` (List[str]): List of data types to remove

**Example:**
```python
# Remove redundant columns and high-cardinality columns
transform = RDBTransformWrapper(
    FilterColumn(
        drop_redundant=True,
        min_unique_values=2,
        max_unique_ratio=0.95  # Remove if >95% unique
    )
)
```

## Engine Classes

### `DFSEngine`

Base class for DFS computation engines.

```python
class DFSEngine:
    def __init__(self, config: DFSConfig)
    
    def compute_features(
        self,
        rdb: RDBDataset,
        target_dataframe: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame
```

### Getting DFS Engines

```python
from fastdfs.dfs import get_dfs_engine

# Get featuretools engine
config = DFSConfig(engine="featuretools")
ft_engine = get_dfs_engine("featuretools", config)

# Get DFS2SQL engine  
config = DFSConfig(engine="dfs2sql", engine_path="/tmp/features.db")
sql_engine = get_dfs_engine("dfs2sql", config)
```

## Utility Functions

### Logging Configuration

```python
from fastdfs.utils.logging_config import configure_logging

# Configure logging level
configure_logging(level="INFO")    # Options: DEBUG, INFO, WARNING, ERROR
configure_logging(level="DEBUG")   # Verbose output for debugging
```