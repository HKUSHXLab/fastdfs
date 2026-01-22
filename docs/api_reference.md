# FastDFS API Reference

## Table of Contents

1. [Core API Functions](#core-api-functions)
2. [Configuration Classes](#configuration-classes)
3. [Dataset Classes](#dataset-classes)
4. [Transform Classes](#transform-classes)
5. [Engine Classes](#engine-classes)
6. [Utility Functions](#utility-functions)

## Core API Functions

### `create_rdb(...) -> RDB`

Create an RDB from in-memory pandas DataFrames.

```python
def create_rdb(
    tables: Dict[str, pd.DataFrame],
    name: str = "myrdb",
    primary_keys: Optional[Dict[str, str]] = None,
    foreign_keys: Optional[List[Tuple[str, str, str, str]]] = None,
    time_columns: Optional[Dict[str, str]] = None,
    type_hints: Optional[Dict[str, Dict[str, str]]] = None
) -> RDB
```

**Parameters:**
- `tables` (Dict[str, pd.DataFrame]): Dictionary mapping table names to DataFrames
- `name` (str): Name of the dataset (default: "myrdb")
- `primary_keys` (Dict[str, str]): Dictionary mapping table names to primary key column names
- `foreign_keys` (List[Tuple]): List of relationships as `(child_table, child_col, parent_table, parent_col)`
- `time_columns` (Dict[str, str]): Dictionary mapping table names to time column names
- `type_hints` (Dict[str, Dict[str, str]]): Dictionary mapping table names to column type overrides

**Returns:**
- `RDB`: The created relational database object

**Example:**
```python
rdb = fastdfs.create_rdb(
    tables={"users": users_df, "items": items_df},
    name="ecommerce",
    primary_keys={"users": "user_id"},
    # ...
)
```

---

### `load_rdb(path: str) -> RDB`

Load a relational database dataset from a directory.

**Parameters:**
- `path` (str): Path to the directory containing `metadata.yaml` and data files

**Returns:**
- `RDB`: The loaded relational database dataset

**Example:**
```python
import fastdfs

rdb = fastdfs.load_rdb("path/to/ecommerce_rdb/")
print(f"Loaded dataset: {rdb.metadata.name}")
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
    rdb: RDB,
    target_dataframe: pd.DataFrame,
    key_mappings: Dict[str, str],
    cutoff_time_column: Optional[str] = None,
    config: Optional[DFSConfig] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> pd.DataFrame
```

**Parameters:**
- `rdb` (RDB): The relational database providing context for feature generation
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
    
    def compute_features(
        self,
        rdb: RDB,
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

features = pipeline.compute_features(
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
    agg_primitives: List[str] = ["max", "min", "mean", "std", "count", "mode"]
    max_depth: int = 2
    use_cutoff_time: bool = True
    engine: str = "dfs2sql"
    engine_path: Optional[str] = None
    max_features: int = -1
    chunk_size: Optional[int] = None
    n_jobs: int = 1
```

**Attributes:**
- `agg_primitives` (List[str]): Aggregation primitives to use for feature generation
- `max_depth` (int): Maximum depth for relationship traversal
- `use_cutoff_time` (bool): Whether to respect temporal cutoff times
- `engine` (str): DFS engine to use ("featuretools" or "dfs2sql")
- `engine_path` (Optional[str]): Path for engine-specific configuration (e.g., DuckDB file)
- `max_features` (int): Maximum number of features to generate (-1 for unlimited)
- `chunk_size` (Optional[int]): Chunk size for batch processing
- `n_jobs` (int): Number of parallel jobs for computation

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

### `RDB`

Represents a relational database for feature engineering.

```python
class RDB:
    def __init__(
        self, 
        path: Optional[Path] = None, 
        metadata: Optional[RDBMeta] = None, 
        tables: Optional[Dict[str, pd.DataFrame]] = None
    )
    
    @property
    def table_names(self) -> List[str]
    
    def get_table(self, name: str) -> pd.DataFrame
    
    def get_table_metadata(self, name: str) -> RDBTableSchema
    
    def get_relationships(self) -> List[Tuple[str, str, str, str]]
    
    def create_new_with_tables(self, new_tables: Dict[str, pd.DataFrame]) -> 'RDB'
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

#### `add_table(...) -> RDB`
Add a new table to the RDB. Returns a new RDB instance with inferred schema.

```python
def add_table(
    self, 
    dataframe: pd.DataFrame, 
    name: str, 
    time_column: Optional[str] = None,
    primary_key: Optional[str] = None,
    foreign_keys: Optional[List[Tuple[str, str, str]]] = None,
    column_types: Optional[Dict[str, str]] = None
) -> 'RDB'
```

**Parameters:**
- `dataframe` (pd.DataFrame): The data to add
- `name` (str): Name of the new table
- `time_column` (Optional[str]): Name of the time column
- `primary_key` (Optional[str]): Name of the primary key column
- `foreign_keys` (Optional[List[Tuple]]): List of `(child_col, parent_table, parent_col)`
- `column_types` (Optional[Dict[str, str]]): Type overrides (e.g., `{"status": "category"}`)

**Returns:**
- `RDB`: A new RDB instance containing the added table

**Example:**
```python
# Add a history table linked to users
rdb = rdb.add_table(
    history_df,
    name="history",
    time_column="timestamp",
    foreign_keys=[("user_id", "users", "user_id")]
)
```

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
    format: str
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
    def __call__(self, rdb: RDB) -> RDB
```

All transforms are pure functions that take an RDB and return a new RDB.

### `RDBTransformPipeline`

Pipeline for composing multiple RDB transforms.

```python
class RDBTransformPipeline:
    def __init__(self, transforms: List[RDBTransform])
    
    def __call__(self, rdb: RDB) -> RDB
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
    def __call__(self, rdb: RDB) -> RDB
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

### `InferSchemaTransform`

Transform that infers RDBColumnDType from table data types.

```python
class InferSchemaTransform(RDBTransform):
    def __init__(
        self,
        primary_keys: Optional[Dict[str, str]] = None,
        foreign_keys: Optional[List[Tuple[str, str, str, str]]] = None,
        time_columns: Optional[Dict[str, str]] = None,
        type_hints: Optional[Dict[str, Dict[str, str]]] = None,
        category_threshold: int = 10
    )
```

**Parameters:**
- `primary_keys` (Dict[str, str]): Dictionary mapping table names to primary key column names
- `foreign_keys` (List[Tuple]): List of relationships as `(child_table, child_col, parent_table, parent_col)`
- `time_columns` (Dict[str, str]): Dictionary mapping table names to time column names
- `type_hints` (Dict[str, Dict[str, str]]): Dictionary mapping table names to column type overrides
- `category_threshold` (int): Threshold for unique values to consider a column as category (default: 10)

**Purpose:** Fills in missing `dtype` in `RDBColumnSchema` based on pandas dtypes and provided hints.

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
class FillMissingPrimaryKey(RDBTransform):
    def __call__(self, rdb: RDB) -> RDB
```

**Purpose:** Ensures primary key integrity by filling missing values (NaNs) with a generated unique identifier or a placeholder.

### `FeaturizeDatetime`

Transform that extracts datetime components from datetime columns.

```python
class FeaturizeDatetime(ColumnTransform):
    def __init__(self, features: List[str] = ["year", "month", "day", "hour"], retain_original: bool = True)
```

**Parameters:**
- `features` (List[str]): List of datetime components to extract
- `retain_original` (bool): Whether to keep the original datetime column (default: True)

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
class FilterColumn(TableTransform):
    def __init__(
        self,
        drop_dtypes: Optional[List[str]] = None,
        drop_redundant: bool = False
    )
```

**Parameters:**
- `drop_dtypes` (List[str]): List of data types to remove
- `drop_redundant` (bool): Remove columns with all identical values

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
        rdb: RDB,
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