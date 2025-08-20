# FastDFS Task Removal Design Document

## Executive Summary

This document outlines the design for refactoring FastDFS to remove the concept of "Tasks" and simplify Deep Feature Synthesis (DFS) to focus on augmenting tables (dataframes) with features generated from Relational Database (RDB) structures. The goal is to make DFS more intuitive by treating it as a process of feature engineering on target tables using information from related tables in an RDB.

## Current Architecture Analysis

### Current Task-Centric Design

The current FastDFS architecture is built around "Tasks" which represent machine learning problems:

```yaml
# Current metadata.yaml structure
tables:                    # Base data tables
- name: user
  columns: [user_id, user_feature_0]
- name: item  
  columns: [item_id, item_feature_0]
- name: interaction
  columns: [user_id, item_id, timestamp]

tasks:                     # ML tasks with train/val/test splits
- name: linkpred
  columns: [user_id, item_id, timestamp, label]
  shared_schema: interaction.*
  target_column: label
  task_type: classification
```

**Current DFS Process:**
1. Load RDB tables + task tables (with train/val/test splits)
2. Create featuretools EntitySet from combined data
3. Generate features for task tables using aggregations from RDB
4. Output augmented task datasets

**Issues with Current Design:**
- **Conceptual complexity**: DFS is inherently about feature engineering, not ML tasks
- **Data leakage concerns**: Mixing train/val/test data in the same pipeline
- **Limited flexibility**: Tied to specific task formats and train/val/test splits
- **Unnecessary abstraction**: Tasks add complexity when users just want features on tables

## New Architecture: Table-Centric DFS

### Core Principle

> **DFS is a feature augmentation process**: Given a target table and an RDB containing related tables, generate new features for the target table by aggregating information across relationships.

### New Conceptual Model

```python
# Instead of: "Run DFS on a task"
dataset = load_rdb_dataset(path)
dfs_processor.run(dataset, output_path)  # Processes all tasks

# New approach: "Augment a table with DFS features"
rdb = load_rdb(path)                     # Load just the relational database
augmented_table = dfs_processor.compute_features(
    rdb=rdb,
    target_table="interaction", 
    target_key_columns=["user_id", "item_id"],
    cutoff_time_column="timestamp",
    max_depth=2
)
```

### Key Changes

1. **Remove Task concept entirely** from dataset definitions
2. **Separate RDB loading** from target table specification  
3. **Runtime target specification** rather than pre-defined tasks
4. **Direct table augmentation** rather than dataset transformation
5. **Flexible relationship specification** at computation time

## New Dataset Definition

### Simplified RDB Metadata

```yaml
# New simplified metadata.yaml
dataset_name: ecommerce_rdb
tables:
- name: user
  source: data/user.npz
  columns:
  - name: user_id
    dtype: primary_key
  - name: user_feature_0
    dtype: float
  - name: registration_date
    dtype: datetime
    
- name: item
  source: data/item.npz 
  columns:
  - name: item_id
    dtype: primary_key
  - name: item_feature_0
    dtype: float
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
  time_column: timestamp

# Relationships are inferred from foreign keys
# No tasks section - tasks are defined at runtime
```

### Benefits of New Format

- **Cleaner separation**: RDB structure vs. ML problem definition
- **Reusability**: Same RDB can be used for multiple different feature engineering tasks
- **Simplicity**: No complex task metadata or shared schemas
- **Standard format**: Follows typical relational database conventions

## New Interface Design

### 1. RDB Loading Interface

```python
from fastdfs import RDBDataset

class RDBDataset:
    """Represents a relational database for feature engineering."""
    
    def __init__(self, path: Path):
        self.path = path
        self.metadata = self._load_metadata()
        self.tables = self._load_tables()
    
    @property 
    def table_names(self) -> List[str]:
        """Get list of table names."""
        
    def get_table(self, name: str) -> pd.DataFrame:
        """Get a table as a pandas DataFrame."""
        
    def get_table_metadata(self, name: str) -> RDBTableSchema:
        """Get metadata for a specific table."""
        
    def get_relationships(self) -> List[Tuple[str, str, str, str]]:
        """Get relationships as (child_table, child_col, parent_table, parent_col)."""
        
    def create_new_with_tables(self, new_tables: Dict[str, pd.DataFrame]) -> 'RDBDataset':
        """Create new RDBDataset with updated tables (for transforms)."""
        
    def create_new_with_tables_and_metadata(self, new_tables: Dict[str, pd.DataFrame], new_metadata: Dict[str, RDBTableSchema]) -> 'RDBDataset':
        """Create new RDBDataset with updated tables and metadata (for transforms that modify schemas)."""
        
    @property
    def sqlalchemy_metadata(self) -> sqlalchemy.MetaData:
        """Get SQLAlchemy metadata for the RDB."""
```

### 2. DFS Engine Interface

```python
from fastdfs import DFSEngine
import pydantic
from typing import List, Optional, Dict, Any

class DFSConfig(pydantic.BaseModel):
    """
    Configuration model for Deep Feature Synthesis parameters.
    
    This class defines all the configurable parameters for the DFS process,
    including aggregation primitives, depth limits, and engine selection.
    
    Attributes:
        agg_primitives: List of aggregation primitive names to use
        max_depth: Maximum depth for feature generation  
        use_cutoff_time: Whether to use temporal cutoff times
        engine: Name of the DFS engine to use for computation
        engine_path: Optional path for engine-specific configuration
    """
    agg_primitives: List[str] = [
        "max",
        "min",
        "mean",
        "count",
        "mode",
        "std",
    ]
    max_depth: int = 2
    use_cutoff_time: bool = True
    engine: str = "featuretools"
    engine_path: Optional[str] = "/tmp/duck.db"

class DFSEngine:
    """Base class for DFS computation engines."""
    
    def __init__(self, config: DFSConfig):
        self.config = config
    
    def compute_features(
        self,
        rdb: RDBDataset,
        target_dataframe: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Compute DFS features for a target dataframe using RDB context.
        
        Args:
            rdb: The relational database providing context for feature generation
            target_dataframe: DataFrame to augment with features (doesn't need to exist in RDB)
            key_mappings: Map from target_dataframe columns to RDB primary keys
                         e.g., {"user_id": "user.user_id", "item_id": "item.item_id"}
            cutoff_time_column: Column name in target_dataframe for temporal cutoff (optional)
            config_overrides: Dictionary of config parameters to override for this computation
            
        Returns:
            DataFrame with original target_dataframe data plus generated features
        """
        # Merge config overrides
        effective_config = self.config.copy()
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(effective_config, key):
                    setattr(effective_config, key, value)
        
        return self._compute_features_impl(
            rdb, target_dataframe, key_mappings, cutoff_time_column, effective_config
        )
    
    @abc.abstractmethod
    def _compute_features_impl(
        self,
        rdb: RDBDataset,
        target_dataframe: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str],
        config: DFSConfig
    ) -> pd.DataFrame:
        """Implementation-specific feature computation."""
        pass
```

### 3. Transform Interface

```python
from fastdfs import RDBTransform

class RDBTransform:
    """Base class for RDB transformations - simplified composable operations."""
    
    def __call__(self, rdb: RDBDataset) -> RDBDataset:
        """
        Apply transformation to RDB and return new RDB.
        
        Simplified interface - no separate fit/transform phases.
        Each transform is a pure function: RDB -> RDB
        """
        pass

class TableTransform:
    """Transform that operates on individual tables within an RDB."""
    
    def __call__(self, table: pd.DataFrame, table_metadata: RDBTableSchema) -> Tuple[pd.DataFrame, RDBTableSchema]:
        """
        Apply transformation to a single table.
        
        Returns:
            Tuple of (transformed_dataframe, updated_table_metadata)
            
        Expected Behavior:
        - For unchanged tables: return (original_table, original_metadata)
        - For modified tables: return (modified_table, updated_metadata with changed column schemas)
        - For tables with new columns: return (table_with_new_cols, metadata with additional column schemas)
        - For tables with removed columns: return (table_without_cols, metadata with removed column schemas)
        """
        pass

class ColumnTransform:
    """Transform that operates on specific columns matching criteria."""
    
    def applies_to(self, column_metadata: DBBColumnSchema) -> bool:
        """Check if this transform should be applied to a column."""
        pass
    
    def __call__(self, column: pd.Series, column_metadata: DBBColumnSchema) -> Tuple[pd.DataFrame, List[DBBColumnSchema]]:
        """
        Transform a column, potentially outputting multiple new columns.
        
        Returns:
            Tuple of (dataframe_with_new_columns, list_of_new_column_schemas)
            
        Expected Behavior:
        - For unchanged columns: return (df_with_original_column, [original_column_schema])
        - For modified columns: return (df_with_modified_column, [updated_column_schema])
        - For columns generating new features: return (df_with_multiple_columns, [schema_for_each_new_column])
        - For removed columns: should consider using TableTransform instead
        """
        pass

class RDBTransformPipeline:
    """Pipeline of RDB transformations."""
    
    def __init__(self, transforms: List[RDBTransform]):
        self.transforms = transforms
    
    def __call__(self, rdb: RDBDataset) -> RDBDataset:
        """Apply all transforms in sequence."""
        result = rdb
        for transform in self.transforms:
            result = transform(result)
        return result

class RDBTransformWrapper(RDBTransform):
    """Wrapper to apply TableTransform or ColumnTransform to entire RDB."""
    
    def __init__(self, inner_transform: Union[TableTransform, ColumnTransform]):
        self.inner_transform = inner_transform
    
    def __call__(self, rdb: RDBDataset) -> RDBDataset:
        """Apply inner transform to all applicable tables/columns in RDB."""
        new_tables = {}
        updated_metadata = {}
        
        for table_name, table_df in rdb.tables.items():
            table_metadata = rdb.get_table_metadata(table_name)
            
            if isinstance(self.inner_transform, TableTransform):
                # Apply to entire table
                new_table, new_table_metadata = self.inner_transform(table_df, table_metadata)
                new_tables[table_name] = new_table
                updated_metadata[table_name] = new_table_metadata
                
            elif isinstance(self.inner_transform, ColumnTransform):
                # Apply to applicable columns
                new_table = table_df.copy()
                new_column_schemas = []
                
                for col_schema in table_metadata.columns:
                    if self.inner_transform.applies_to(col_schema):
                        col_name = col_schema.name
                        new_cols_df, new_col_schemas = self.inner_transform(table_df[col_name], col_schema)
                        
                        # Add/replace columns from transform result
                        for new_col_name, new_col_data in new_cols_df.items():
                            new_table[new_col_name] = new_col_data
                        
                        # Update column schemas
                        new_column_schemas.extend(new_col_schemas)
                        
                        # Remove original column if transform doesn't retain it
                        if hasattr(self.inner_transform, 'replaces_original') and self.inner_transform.replaces_original:
                            new_table = new_table.drop(columns=[col_name])
                    else:
                        # Keep original column and schema if transform doesn't apply
                        new_column_schemas.append(col_schema)
                
                new_tables[table_name] = new_table
                updated_metadata[table_name] = RDBTableSchema(
                    name=table_metadata.name,
                    columns=new_column_schemas,
                    **{k: v for k, v in table_metadata.__dict__.items() if k not in ['name', 'columns']}
                )
        
        return rdb.create_new_with_tables_and_metadata(new_tables, updated_metadata)
```

### 4. High-Level API

```python
# Simple feature augmentation with external target dataframe
import fastdfs
import pandas as pd

# Load RDB (no tasks)
rdb = fastdfs.load_rdb("path/to/rdb")

# Load or create target dataframe (doesn't need to exist in RDB)
target_df = pd.read_parquet("user_item_pairs.parquet")
# target_df has columns: user_id, item_id, timestamp

# Augment target dataframe with DFS features from RDB
augmented_df = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=target_df,
    key_mappings={
        "user_id": "user.user_id",     # Map target_df.user_id to RDB user table
        "item_id": "item.item_id"      # Map target_df.item_id to RDB item table  
    },
    cutoff_time_column="timestamp",
    config_overrides={
        "max_depth": 2,
        "engine": "dfs2sql",
        "agg_primitives": ["count", "mean", "max", "min"]
    }
)

# Method 2: Using custom DFS configuration
dfs_config = fastdfs.DFSConfig(
    max_depth=3,
    engine="featuretools",
    agg_primitives=["count", "sum", "mean", "std"],
    use_cutoff_time=True,
)

augmented_df = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=target_df,
    key_mappings={"user_id": "user.user_id", "item_id": "item.item_id"},
    cutoff_time_column="timestamp",
    config=dfs_config
)

# Apply transforms to RDB (simplified - no fit/transform)
transform_pipeline = fastdfs.RDBTransformPipeline([
    fastdfs.transforms.FeaturizeDatetime(methods=["year", "month", "hour"]),
    fastdfs.transforms.FilterColumn(drop_redundant=True),
    fastdfs.transforms.HandleDummyTable()
])
transformed_rdb = transform_pipeline(rdb)

# Combined transform + DFS workflow
augmented_df = fastdfs.compute_dfs_features(
    rdb=transformed_rdb,
    target_dataframe=target_df,
    key_mappings={"user_id": "user.user_id", "item_id": "item.item_id"},
    config_overrides={"max_depth": 2}
)

# Example: Generate features for ML training data
train_df = pd.read_parquet("train_interactions.parquet")
train_with_features = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=train_df,
    key_mappings={"user_id": "user.user_id", "item_id": "item.item_id"},
    cutoff_time_column="interaction_time",
    config_overrides={"max_depth": 2, "use_cutoff_time": True}
)

# Example: Generate features for new prediction data
new_predictions_df = pd.DataFrame({
    "user_id": [1, 2, 3],
    "item_id": [100, 200, 300],
    "prediction_time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
})
prediction_features = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=new_predictions_df,
    key_mappings={"user_id": "user.user_id", "item_id": "item.item_id"},
    cutoff_time_column="prediction_time",
    config=dfs_config
)
```

## Design Tradeoffs and Limitations

### What's No Longer Supported

#### 1. **Scikit-learn Fit/Transform Paradigm**
**Removed**: `fit()` and `transform()` separation for learning parameters from training data only.

**Tradeoff**: 
- ✅ **Gain**: Simpler, more composable transforms that are pure functions
- ✅ **Gain**: No data leakage concerns since transforms don't "learn" from specific splits
- ✅ **Gain**: Easier to reason about - transforms are deterministic operations
- ❌ **Loss**: Cannot learn normalization parameters (mean/std) from training data only
- ❌ **Loss**: No automatic handling of unseen categorical values in test data
- ❌ **Loss**: Cannot fit imputation strategies on training data specifically

**Workaround**: Users must handle train/val/test parameter learning externally:
```python
# Old approach (automatic with normalization)
transform = FilterColumn(drop_redundant=True)
transform.fit(train_rdb)  # Learn which columns to drop from training only
test_rdb = transform.transform(test_rdb)  # Apply learned filtering

# New approach (manual)
train_stats = compute_column_statistics(train_df)
filter_transform = FilterColumn(drop_dtypes=train_stats.redundant_dtypes)
test_rdb = filter_transform(test_rdb)
```

#### 2. **Automatic Train/Val/Test Split Handling**
**Removed**: Built-in handling of data splits with proper parameter learning separation.

**Tradeoff**:
- ✅ **Gain**: Decoupled feature engineering from ML workflow concerns
- ✅ **Gain**: More flexible - can apply features to any dataframe
- ✅ **Gain**: No complex task metadata or shared schema configurations
- ❌ **Loss**: No automatic prevention of data leakage in feature engineering
- ❌ **Loss**: Users must manually ensure proper temporal/split boundaries
- ❌ **Loss**: No built-in support for time-aware cross-validation

**Workaround**: Users handle splits explicitly:
```python
# Apply transforms to RDB once
clean_rdb = transform_pipeline(rdb)

# Then apply to each split separately with cutoff times
train_features = compute_dfs_features(
    rdb=clean_rdb, 
    target_dataframe=train_df,
    cutoff_time_column="interaction_time"  # Ensures temporal consistency
)
test_features = compute_dfs_features(
    rdb=clean_rdb,
    target_dataframe=test_df, 
    cutoff_time_column="interaction_time"
)
```

#### 3. **Column Groups and Shared Schema Management**
**Removed**: Automatic grouping of columns with `shared_schema` for batch processing.

**Tradeoff**:
- ✅ **Gain**: Simpler data model without complex column relationships
- ✅ **Gain**: Direct mapping from target dataframe to RDB keys
- ❌ **Loss**: Cannot automatically apply same transform to related columns across tables
- ❌ **Loss**: No automatic metadata propagation between related columns

**Workaround**: Apply transforms to RDB tables directly, specify relationships explicitly:
```python
# Old: shared_schema automatically grouped user_id columns
# New: transforms apply to all tables, relationships specified in key_mappings
key_mappings = {
    "user_id": "user.user_id",      # Explicit mapping
    "item_id": "item.item_id"
}
```

#### 4. **Task-Specific Feature Output Format**
**Removed**: Direct output as task datasets with preserved train/val/test structure.

**Tradeoff**:
- ✅ **Gain**: More flexible output - plain DataFrames that work with any ML library
- ✅ **Gain**: Easier integration with pandas/scikit-learn/PyTorch workflows
- ❌ **Loss**: No automatic reconstruction of task dataset format
- ❌ **Loss**: Users must manually manage feature alignment with original splits

#### 5. **Automatic Feature Filtering and Validation**
**Removed**: Built-in filtering of invalid features based on task target columns.

**Tradeoff**:
- ✅ **Gain**: More features generated (no automatic filtering)
- ✅ **Gain**: Users have full control over feature selection
- ❌ **Loss**: May generate features that leak target information
- ❌ **Loss**: No automatic detection of problematic features

**Workaround**: Manual feature filtering:
```python
# Features are generated for all relationships
all_features = compute_dfs_features(rdb, target_df, key_mappings)

# User manually filters out problematic features
safe_features = all_features.drop(columns=[
    col for col in all_features.columns 
    if 'label' in col.lower() or 'target' in col.lower()
])
```

### What's Improved

#### 1. **Flexibility**: Can augment any dataframe, not just pre-existing RDB tables
```python
# Can create arbitrary target dataframes
synthetic_pairs = pd.DataFrame({
    "user_id": [1, 2, 3], 
    "item_id": [100, 200, 300],
    "prediction_timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"]
})
features = compute_dfs_features(rdb, synthetic_pairs, key_mappings)
```

#### 2. **Simplicity**: Pure functional transforms are easier to understand and debug
```python
# Old: Complex stateful transforms
transform.fit(train_data)
result = transform.transform(test_data)

# New: Simple functional transforms  
result = transform(data)
```

#### 3. **Composability**: Easy to chain and combine operations
```python
pipeline = RDBTransformPipeline([
    FeaturizeDatetime(methods=["year", "month"]),
    FilterColumn(drop_redundant=True),
    HandleDummyTable()
])
result = pipeline(rdb)
```

#### 4. **Performance**: No need to duplicate data across task tables
```python
# Generate features once for RDB
# Apply to multiple target dataframes efficiently
train_features = compute_dfs_features(rdb, train_df, key_mappings)
test_features = compute_dfs_features(rdb, test_df, key_mappings)
```

### Migration Considerations

For users heavily relying on the removed features:

1. **Normalization/Scaling**: Use external libraries (scikit-learn, pandas) for learning parameters
2. **Split Management**: Implement temporal cutoffs and split logic in user code
3. **Feature Validation**: Add custom filtering logic for target leakage detection
4. **Task Formats**: Use conversion utilities to transform plain DataFrames back to task format if needed

The tradeoff is worth it for most users who want DFS as a flexible feature engineering tool rather than a complete ML pipeline component.

## Implementation Plan

### Phase 1: New Dataset Interface (Week 1)

**Current Code References**:
- **Existing dataset classes**: `fastdfs/dataset/rdb_dataset.py` (current `DBBRDBDataset` class)
- **Metadata structures**: `fastdfs/dataset/meta.py` (current `DBBRDBDatasetMeta`, `DBBTableSchema`, `DBBColumnSchema`)
- **YAML utilities**: `fastdfs/utils/yaml_utils.py` (for loading/saving metadata)
- **Data loaders**: `fastdfs/dataset/loader.py` (current table data loading logic)

#### 1.1 New RDB Dataset Classes

```python
# fastdfs/dataset/rdb_dataset.py (new simplified version)

@dataclass
class RDBTableSchema:
    name: str
    source: str
    format: DBBTableDataFormat
    columns: List[DBBColumnSchema]
    time_column: Optional[str] = None

@dataclass  
class RDBDatasetMeta:
    dataset_name: str
    tables: List[RDBTableSchema]
    # Remove: tasks, column_groups (inferred from foreign keys)

class RDBDataset:
    """Simplified RDB dataset without tasks."""
    
    def __init__(self, path: Path):
        self.path = Path(path)
        self.metadata = self._load_metadata()
        self.tables = self._load_tables()
    
    def _load_metadata(self) -> RDBDatasetMeta:
        return yaml_utils.load_pyd(RDBDatasetMeta, self.path / 'metadata.yaml')
    
    def _load_tables(self) -> Dict[str, pd.DataFrame]:
        """Load all tables as pandas DataFrames."""
        tables = {}
        for table_schema in self.metadata.tables:
            table_path = self.path / table_schema.source
            loader = get_table_data_loader(table_schema.format)
            table_data = loader(table_path)
            
            # Convert to pandas DataFrame
            df_data = {}
            for col_schema in table_schema.columns:
                df_data[col_schema.name] = table_data[col_schema.name]
            tables[table_schema.name] = pd.DataFrame(df_data)
        return tables
    
    @property
    def table_names(self) -> List[str]:
        return list(self.tables.keys())
    
    def get_table(self, name: str) -> pd.DataFrame:
        if name not in self.tables:
            raise ValueError(f"Table {name} not found")
        return self.tables[name].copy()
    
    def get_relationships(self) -> List[Tuple[str, str, str, str]]:
        """Extract relationships from foreign key column definitions."""
        relationships = []
        for table_schema in self.metadata.tables:
            for col_schema in table_schema.columns:
                if col_schema.dtype == DBBColumnDType.foreign_key:
                    parent_table, parent_col = col_schema.link_to.split('.')
                    relationships.append((
                        table_schema.name,  # child table
                        col_schema.name,    # child column
                        parent_table,       # parent table  
                        parent_col          # parent column
                    ))
        return relationships
```

#### 1.2 Migration Utilities

**Current Code References**:
- **Task extraction logic**: `fastdfs/dataset/rdb_dataset.py` lines 200-300 (current task loading from metadata)
- **Table data access**: `fastdfs/dataset/rdb_dataset.py` method `get_table_data()` 
- **Relationship handling**: `fastdfs/dataset/meta.py` `DBBRelationship` class
- **File operations**: Standard Python `shutil` and `pathlib` for copying table files

```python
# fastdfs/dataset/migration.py

def convert_task_dataset_to_rdb(old_dataset_path: Path, rdb_output_path: Path):
    """Convert old task-based dataset to new RDB-only format."""
    
    # Load old dataset
    old_dataset = DBBRDBDataset(old_dataset_path)  # Current implementation
    
    # Create new RDB metadata (tables only, no tasks)
    new_metadata = RDBDatasetMeta(
        dataset_name=old_dataset.metadata.dataset_name,
        tables=old_dataset.metadata.tables  # Keep table definitions as-is
    )
    
    # Save new metadata
    yaml_utils.save_pyd(new_metadata, rdb_output_path / 'metadata.yaml')
    
    # Copy table data (unchanged)
    for table_schema in old_dataset.metadata.tables:
        source_path = old_dataset.path / table_schema.source
        dest_path = rdb_output_path / table_schema.source
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)

def extract_target_tables_from_tasks(old_dataset_path: Path, output_dir: Path):
    """Extract task data as separate target table files."""
    
    old_dataset = DBBRDBDataset(old_dataset_path)
    
    for task in old_dataset.tasks:
        # Create target table files for each split
        for split_name, split_data in [
            ("train", task.train_set),
            ("validation", task.validation_set), 
            ("test", task.test_set)
        ]:
            # Convert to DataFrame
            df_data = {col: data for col, data in split_data.items()}
            target_df = pd.DataFrame(df_data)
            
            # Save as parquet for better schema preservation
            output_file = output_dir / f"{task.metadata.name}_{split_name}.parquet"
            target_df.to_parquet(output_file, index=False)
            
            logger.info(f"Extracted {task.metadata.name} {split_name} to {output_file}")
```

### Phase 2: New DFS Engine Interface (Week 2)

**Current Code References**:
- **Base DFS engine**: `fastdfs/preprocess/dfs/core.py` (current `DFSEngine` class, lines 200-350)
- **Featuretools engine**: `fastdfs/preprocess/dfs/ft_engine.py` (current `FeaturetoolsEngine` implementation)
- **DFS2SQL engine**: `fastdfs/preprocess/dfs/dfs2sql_engine.py` (current `DFS2SQLEngine` implementation)
- **Feature preparation**: `fastdfs/preprocess/dfs/core.py` methods `prepare()`, `build_dataframes()` (lines 270-450)
- **Feature filtering**: `fastdfs/preprocess/dfs/core.py` method `filter_features()` (lines 320-380)
- **Primitive handling**: `fastdfs/preprocess/dfs/primitives.py` (custom primitives like `Concat`, `Join`, etc.)
- **SQL generation**: `fastdfs/preprocess/dfs/gen_sqls.py` (features to SQL conversion logic)
- **Engine registry**: `fastdfs/preprocess/dfs/core.py` (global `_DFS_ENGINE_REGISTRY` and `@dfs_engine` decorator)

#### 2.1 Base DFS Engine

```python
# fastdfs/dfs/base_engine.py

class DFSEngine:
    """Base class for DFS computation engines."""
    
    def __init__(self, config: DFSConfig):
        self.config = config
    
    def compute_features(
        self,
        rdb: RDBDataset,
        target_dataframe: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Compute DFS features for target dataframe.
        
        Args:
            rdb: RDBDataset containing the relational data
            target_dataframe: DataFrame with target instances to compute features for
            key_mappings: Maps target_df columns to RDB entity keys (e.g., {"user_id": "users.user_id"})
            cutoff_time_column: Column in target_dataframe for temporal cutoffs
            config_overrides: Optional overrides for DFS configuration
        
        Returns:
            DataFrame with original target_dataframe columns plus generated DFS features
        """
        # Create effective config by merging overrides
        effective_config = self.config.copy(deep=True)
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(effective_config, key):
                    setattr(effective_config, key, value)
        
        # Phase 1: Feature preparation (common logic in base class)
        features = self.prepare_features(rdb, target_dataframe, key_mappings, cutoff_time_column, effective_config)
        
        if len(features) == 0:
            raise RuntimeError("No features to compute, try to increase the depth.")
        
        # Phase 2: Feature computation (engine-specific logic in subclasses)
        feature_matrix = self.compute_feature_matrix(
            rdb, target_dataframe, key_mappings, cutoff_time_column, features, effective_config
        )
        
        return feature_matrix
    
    def prepare_features(
        self,
        rdb: RDBDataset,
        target_dataframe: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str],
        config: DFSConfig
    ) -> List[ft.FeatureBase]:
        """
        Prepare feature specifications using featuretools DFS.
        
        This method builds the EntitySet, runs featuretools DFS to generate
        feature specifications, and filters the results based on configuration.
        This is common logic shared by all engines.
        
        Returns:
            List of featuretools FeatureBase objects representing features to compute
        """
        # Build EntitySet from RDB tables
        entity_set = self._build_entity_set_from_rdb(rdb)
        
        # Add target dataframe as temporary entity  
        target_entity_name = "__target__"
        target_index = self._determine_target_index(target_dataframe, key_mappings)
        
        entity_set = entity_set.add_dataframe(
            dataframe_name=target_entity_name,
            dataframe=target_dataframe.copy(),
            index=target_index,
            time_index=cutoff_time_column
        )
        
        # Add relationships from target to RDB entities
        self._add_target_relationships(entity_set, target_entity_name, key_mappings)
        
        logger.debug(entity_set)
        
        # Generate feature specifications using featuretools
        features = ft.dfs(
            entityset=entity_set,
            target_dataframe_name=target_entity_name,
            max_depth=config.max_depth,
            agg_primitives=self._convert_primitives(config.agg_primitives),
            trans_primitives=config.trans_primitives,
            where_primitives=config.where_primitives,
            max_features=config.max_features,
            include_entities=config.include_entities,
            ignore_entities=config.ignore_entities,
            features_only=True
        )
        
        # Filter features based on configuration
        filtered_features = self._filter_features(features, entity_set, target_entity_name, config)
        
        return filtered_features
    
    def _build_entity_set_from_rdb(self, rdb: RDBDataset) -> ft.EntitySet:
        """Build EntitySet from RDB tables only (adapted from existing build_dataframes logic)."""
        
        entity_set = ft.EntitySet(id=rdb.metadata.dataset_name)
        
        # Add all RDB tables as entities
        for table_name in rdb.table_names:
            df = rdb.get_table(table_name)
            table_meta = rdb.get_table_metadata(table_name)
            
            # Parse columns and build logical types/semantic tags (reuse existing logic)
            logical_types = {}
            semantic_tags = {}
            index_col = None
            
            for col_schema in table_meta.columns:
                col_name = col_schema.name
                series, log_ty, tag = parse_one_column(col_schema, df[col_name])
                logical_types[col_name] = log_ty
                
                if col_schema.dtype == DBBColumnDType.primary_key:
                    index_col = col_name
                    # Don't set semantic tag for index
                else:
                    semantic_tags[col_name] = tag
            
            # Add default index if needed (reuse existing logic)
            if index_col is None:
                df["__index__"] = np.arange(len(df))
                index_col = "__index__"
            
            entity_set = entity_set.add_dataframe(
                dataframe_name=table_name,
                dataframe=df,
                index=index_col,
                time_index=table_meta.time_column,
                logical_types=logical_types,
                semantic_tags=semantic_tags
            )
        
        # Add relationships between RDB tables (reuse existing logic)
        for child_table, child_col, parent_table, parent_col in rdb.get_relationships():
            entity_set = entity_set.add_relationship(
                parent_dataframe_name=parent_table,
                parent_column_name=parent_col,
                child_dataframe_name=child_table,
                child_column_name=child_col
            )
        
        return entity_set
    
    def _determine_target_index(self, target_df: pd.DataFrame, key_mappings: Dict[str, str]) -> str:
        """Determine appropriate index for target dataframe."""
        
        # If single key mapping, use that as index
        if len(key_mappings) == 1:
            return list(key_mappings.keys())[0]
        
        # If multiple keys, create composite index
        key_cols = list(key_mappings.keys())
        composite_index = "_".join(key_cols) + "_index"
        
        # Create composite index column
        target_df[composite_index] = target_df[key_cols].apply(
            lambda row: "_".join(row.astype(str)), axis=1
        )
        
        return composite_index
    
    def _add_target_relationships(
        self, entity_set: ft.EntitySet, target_entity_name: str, key_mappings: Dict[str, str]
    ):
        """Add relationships from target entity to RDB entities."""
        
        for target_col, rdb_ref in key_mappings.items():
            parent_table, parent_col = rdb_ref.split('.')
            
            entity_set = entity_set.add_relationship(
                parent_dataframe_name=parent_table,
                parent_column_name=parent_col,
                child_dataframe_name=target_entity_name,
                child_column_name=target_col
            )
    
    def _convert_primitives(self, primitive_names: List[str]) -> List:
        """Convert primitive names to primitive objects (reuse existing logic)."""
        primitives = []
        for prim in primitive_names:
            if prim == "concat":
                primitives.append(Concat)
            elif prim == "join":
                primitives.append(Join)
            elif prim == "arraymax":
                primitives.append(ArrayMax)
            elif prim == "arraymin":
                primitives.append(ArrayMin)
            elif prim == "arraymean":
                primitives.append(ArrayMean)
            else:
                primitives.append(prim)
        return primitives
    
    def _filter_features(
        self, 
        features: List[ft.FeatureBase], 
        entity_set: ft.EntitySet, 
        target_entity_name: str,
        config: DFSConfig
    ) -> List[ft.FeatureBase]:
        """Filter features (adapted from existing filter_features logic)."""
        
        if len(features) == 0:
            return features
        
        # Get foreign/primary keys from relationships (reuse existing logic)
        keys = set()
        for rel in entity_set.relationships:
            keys.add((rel.parent_dataframe_name, rel.parent_column_name))
            keys.add((rel.child_dataframe_name, rel.child_column_name))
        
        new_features = []
        for feat in features:
            feat_str = str(feat)
            
            # Remove features involving the target table
            if target_entity_name in feat_str:
                continue
                
            # Remove key-based features (reuse existing logic)
            if base_feature_is_key(feat, keys):
                continue
                
            new_features.append(feat)
            
        return new_features
    
    @abc.abstractmethod
    def compute_feature_matrix(
        self,
        rdb: RDBDataset,
        target_dataframe: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str],
        features: List[ft.FeatureBase],
        config: DFSConfig
    ) -> pd.DataFrame:
        """
        Compute actual feature values from feature specifications.
        
        This is engine-specific logic implemented by subclasses.
        """
        pass

# Registry for DFS engines (reuse existing pattern)
_DFS_ENGINE_REGISTRY = {}

def dfs_engine(engine_class):
    """Decorator to register DFS engines."""
    _DFS_ENGINE_REGISTRY[engine_class.name] = engine_class
    return engine_class

def get_dfs_engine(name: str, config: DFSConfig) -> DFSEngine:
    """Get DFS engine by name."""
    if name not in _DFS_ENGINE_REGISTRY:
        raise ValueError(f"Unknown DFS engine: {name}")
    return _DFS_ENGINE_REGISTRY[name](config)
```

#### 2.2 Engine-Specific Implementations

**Current Code References**:
- **EntitySet building**: `fastdfs/preprocess/dfs/core.py` method `build_dataframes()` (lines 380-550) - *adapt for external targets*
- **Column parsing**: `fastdfs/preprocess/dfs/core.py` function `parse_one_column()` (reuse as-is)
- **Featuretools computation**: `fastdfs/preprocess/dfs/ft_engine.py` method `compute()` (lines 50-100) - *reuse calculation logic*
- **SQL execution**: `fastdfs/preprocess/dfs/dfs2sql_engine.py` method `compute()` (lines 40-100) - *reuse SQL generation and execution*
- **Database setup**: `fastdfs/preprocess/dfs/database.py` `DuckDBBuilder` class - *adapt for target dataframes*
- **Array aggregation**: `fastdfs/preprocess/dfs/dfs2sql_engine.py` method `handle_array_aggregation()` (lines 100-130) - *reuse as-is*
- **Feature filtering**: `fastdfs/preprocess/dfs/dfs2sql_engine.py` method `filter_nested_array_agg_features()` (lines 20-40) - *reuse as-is*

**Understanding the Separation of Concerns**:

Following the original design pattern, we separate feature preparation (common logic in base class) from feature computation (engine-specific logic in subclasses). This preserves the existing architecture while adapting it for external target dataframes.

**Base Class Responsibilities**:
- EntitySet building from RDB tables
- Target dataframe integration and relationship setup  
- Feature specification generation using `ft.dfs(features_only=True)`
- Feature filtering based on keys and target entity

**Engine Subclass Responsibilities**:
- Feature value computation using engine-specific methods
- Featuretools engine: Use `ft.calculate_feature_matrix()`  
- DFS2SQL engine: Convert specs to SQL and execute via DuckDB

```python
# fastdfs/dfs/featuretools_engine.py

@dfs_engine
class FeaturetoolsEngine(DFSEngine):
    """Featuretools-based DFS engine implementation."""
    
    name = "featuretools"
    
    def compute_feature_matrix(
        self,
        rdb: RDBDataset,
        target_dataframe: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str],
        features: List[ft.FeatureBase],
        config: DFSConfig
    ) -> pd.DataFrame:
        """Compute feature values using featuretools (reuse existing computation logic)."""
        
        # Rebuild EntitySet for computation (could be optimized to reuse from prepare phase)
        entity_set = self._build_entity_set_from_rdb(rdb)
        target_entity_name = "__target__"
        target_index = self._determine_target_index(target_dataframe, key_mappings)
        
        entity_set = entity_set.add_dataframe(
            dataframe_name=target_entity_name,
            dataframe=target_dataframe.copy(),
            index=target_index,
            time_index=cutoff_time_column
        )
        
        self._add_target_relationships(entity_set, target_entity_name, key_mappings)
        
        # Compute feature matrix using featuretools (reuse existing logic)
        if cutoff_time_column and config.use_cutoff_time:
            cutoff_times = target_dataframe[[target_index, cutoff_time_column]].copy()
            cutoff_times.columns = ["instance_id", "time"]
        else:
            cutoff_times = None
        
        feature_matrix = ft.calculate_feature_matrix(
            features=features,
            entityset=entity_set,
            cutoff_time=cutoff_times,
            chunk_size=config.chunk_size,
            n_jobs=config.n_jobs
        )
        
        return feature_matrix


# fastdfs/dfs/dfs2sql_engine.py

@dfs_engine  
class DFS2SQLEngine(DFSEngine):
    """SQL-based DFS engine implementation."""
    
    name = "dfs2sql"
    
    def compute_feature_matrix(
        self,
        rdb: RDBDataset,
        target_dataframe: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str],
        features: List[ft.FeatureBase],
        config: DFSConfig
    ) -> pd.DataFrame:
        """Compute feature values using SQL generation (reuse existing computation logic)."""
        
        # Apply DFS2SQL-specific feature filtering
        filtered_features = self._filter_nested_array_agg_features(features)
        
        # Set up database with RDB tables + target table
        target_index = self._determine_target_index(target_dataframe, key_mappings)
        builder = DuckDBBuilder(Path(config.engine_path))
        self._build_database_tables(builder, rdb, target_dataframe, target_index, cutoff_time_column)
        db = builder.db
        
        # Generate SQLs from feature specifications (reuse existing features2sql logic)
        has_cutoff_time = config.use_cutoff_time and cutoff_time_column is not None
        sqls = features2sql(
            filtered_features,
            target_index,
            has_cutoff_time=has_cutoff_time,
            cutoff_time_table_name="__target__",
            cutoff_time_col_name=cutoff_time_column,
            time_col_mapping=builder.time_columns
        )
        
        # Execute SQLs and merge results (reuse existing logic)
        logger.debug("Executing SQLs ...")
        dataframes = []
        for sql in tqdm.tqdm(sqls):
            logger.debug(f"Executing SQL: {format_sql(sql.sql())}")
            result = db.sql(sql.sql())
            if result is not None:
                dataframe = result.df()
                
                # Clean up result dataframe (reuse existing logic)
                if cutoff_time_column in dataframe.columns:
                    dataframe.drop(columns=[cutoff_time_column], inplace=True)
                dataframe.rename(decode_column_from_sql, axis="columns", inplace=True)
                self._handle_array_aggregation(dataframe)
                dataframes.append(dataframe)
        
        # Merge all feature dataframes
        if dataframes:
            logger.debug("Finalizing ...")
            merged_df = pd.DataFrame(
                reduce(lambda left, right: pd.merge(left, right, on=target_index), dataframes)
            )
            # Reindex to match original target order
            target_indices = target_dataframe[target_index].values
            merged_df = merged_df.set_index(target_index).reindex(target_indices).reset_index(drop=True)
            return merged_df
        else:
            return target_dataframe.copy()
    
    def _filter_nested_array_agg_features(self, features: List[ft.FeatureBase]) -> List[ft.FeatureBase]:
        """Filter nested array aggregation features (reuse existing logic)."""
        if len(features) == 0:
            return features
            
        array_agg_func_names = ["ARRAYMAX", "ARRAYMIN", "ARRAYMEAN"]
        new_features = []
        for feat in features:
            feat_str = str(feat)
            agg_count = _check_array_agg_occurrences(feat_str, array_agg_func_names)
            if agg_count > 1:
                # Remove features with nested array aggregation
                continue
            new_features.append(feat)
        return new_features
    
    def _build_database_tables(
        self, 
        builder: DuckDBBuilder, 
        rdb: RDBDataset, 
        target_dataframe: pd.DataFrame,
        target_index: str,
        cutoff_time_column: Optional[str]
    ):
        """Build database tables for SQL execution (adapted from existing build_dataframes logic)."""
        
        # Add all RDB tables to database
        for table_name in rdb.table_names:
            df = rdb.get_table(table_name)
            builder.add_table(table_name, df)
        
        # Add target dataframe as __target__ table
        builder.add_table("__target__", target_dataframe)
        builder.index_name = target_index
        builder.index = target_dataframe[target_index].values
        
        # Set up cutoff time information
        if cutoff_time_column:
            builder.cutoff_time_table_name = "__target__"
            builder.cutoff_time_col_name = cutoff_time_column
            
            # Build time column mapping for all tables
            builder.time_columns = {}
            for table_name in rdb.table_names:
                table_meta = rdb.get_table_metadata(table_name)
                if table_meta.time_column:
                    builder.time_columns[table_name] = table_meta.time_column
    
    def _handle_array_aggregation(self, df: pd.DataFrame):
        """Handle array aggregation results (reuse existing logic)."""
        array_agg_func_names = ["ARRAYMAX", "ARRAYMIN", "ARRAYMEAN"]
        for col in df.columns:
            num_array_agg = _check_array_agg_occurrences(col, array_agg_func_names)
            if num_array_agg == 1:
                if "ARRAYMAX" in col:
                    df[col] = df[col].apply(array_max)
                elif "ARRAYMIN" in col:
                    df[col] = df[col].apply(array_min)
                elif "ARRAYMEAN" in col:
                    df[col] = df[col].apply(array_mean)
```

**Key Advantages of This Design**:

1. **Code Reuse**: Most existing feature preparation and computation logic is preserved
2. **Separation of Concerns**: Feature preparation (common) vs computation (engine-specific)  
3. **Maintainability**: Changes to feature filtering affect all engines uniformly
4. **Extensibility**: New engines only need to implement `compute_feature_matrix()`
5. **Testability**: Feature preparation can be tested independently of computation

**Migration Strategy**:

1. **Phase 1**: Implement new base class with shared preparation logic
2. **Phase 2**: Adapt existing engines to use new `compute_feature_matrix()` interface
3. **Phase 3**: Test both engines with external target dataframes
4. **Phase 4**: Deprecate old task-based interfaces

**Preserved Logic Components**:
- EntitySet building from `build_dataframes()`
- Feature filtering from `filter_features()`  
- Primitive conversion logic
- SQL generation from `features2sql()`
- Array aggregation handling
- Database table setup

### Phase 3: New Transform Interface (Week 3)

**Current Code References**:
- **Transform base classes**: `fastdfs/preprocess/base.py` (current `Preprocessor` and `DBBPreprocessor` classes)
- **Transform implementations**: `fastdfs/preprocess/transform_preprocess.py` (existing transform examples)
- **Fit/transform pattern**: `fastdfs/preprocess/base.py` methods `fit()`, `transform()` (lines 50-150) - *to be simplified*
- **Column type handling**: `fastdfs/dataset/meta.py` `DBBColumnDType` enum - *reuse for type-based transforms*
- **Configuration loading**: `fastdfs/utils/yaml_utils.py` - *reuse for transform pipeline configs*

#### 3.1 RDB Transform Base Classes

Refer to the High-level API section above for the core interface design of `RDBTransform`, `TableTransform`, and `ColumnTransform` classes.

For the first version, we will implement these three key transforms as examples:

**FeaturizeDatetime (ColumnTransform)**
Based on `fastdfs/preprocess/transform/datetime.py`, this transform:
- Applies to columns with `DBBColumnDType.datetime_t`
- Extracts features like year, month, day, dayofweek, timestamp from datetime columns
- Creates new columns with names like `{original_column}_year`, `{original_column}_month`, etc.
- Can optionally retain or replace the original column
- Returns updated column schemas with appropriate data types (category for year/month/day, float for timestamp)

**FilterColumn (TableTransform)** 
Based on `fastdfs/preprocess/transform/filter_column.py`, this transform:
- Operates on entire tables to remove redundant or unwanted columns
- Removes columns with only identical values (redundant columns)
- Can remove columns based on specified data types
- Preserves primary key and foreign key columns
- Returns updated table metadata with removed column schemas

**HandleDummyTable (RDBTransform)**
Based on `fastdfs/preprocess/transform/dummy_table.py`, this transform:
- Creates dummy tables that are referenced by foreign keys but don't exist in the RDB
- Examines relationship metadata to find missing primary key tables
- Creates new tables with unique primary key values extracted from foreign key references
- Adds these dummy tables to the RDB with appropriate primary key column metadata
- Essential for DFS when relationships reference tables that only exist implicitly

```python
# fastdfs/transform/base.py (updated)
from collections import defaultdict
from loguru import logger
import pandas as pd
from typing import List, Dict, Optional, Tuple

class FeaturizeDatetime(ColumnTransform):
    """Extract datetime features from datetime columns."""
    
    def __init__(self, methods: List[str] = ["year", "month", "day"], retain_original: bool = False):
        self.methods = methods
        self.retain_original = retain_original
    
    def applies_to(self, column_metadata: DBBColumnSchema) -> bool:
        return column_metadata.dtype == DBBColumnDType.datetime_t
    
    def __call__(self, column: pd.Series, column_metadata: DBBColumnSchema) -> Tuple[pd.DataFrame, List[DBBColumnSchema]]:
        """Transform datetime column into multiple feature columns."""
        dt_series = pd.to_datetime(column)
        result_df = pd.DataFrame()
        new_schemas = []
        
        # Extract datetime features
        for method in self.methods:
            if method == "year":
                result_df[f"{column_metadata.name}_year"] = dt_series.dt.year
                new_schemas.append(DBBColumnSchema(
                    name=f"{column_metadata.name}_year", 
                    dtype=DBBColumnDType.category_t
                ))
            elif method == "month":
                result_df[f"{column_metadata.name}_month"] = dt_series.dt.month  
                new_schemas.append(DBBColumnSchema(
                    name=f"{column_metadata.name}_month",
                    dtype=DBBColumnDType.category_t
                ))
            elif method == "day":
                result_df[f"{column_metadata.name}_day"] = dt_series.dt.day
                new_schemas.append(DBBColumnSchema(
                    name=f"{column_metadata.name}_day",
                    dtype=DBBColumnDType.category_t
                ))
            elif method == "dayofweek":
                result_df[f"{column_metadata.name}_dayofweek"] = dt_series.dt.dayofweek
                new_schemas.append(DBBColumnSchema(
                    name=f"{column_metadata.name}_dayofweek",
                    dtype=DBBColumnDType.category_t
                ))
            elif method == "hour":
                result_df[f"{column_metadata.name}_hour"] = dt_series.dt.hour
                new_schemas.append(DBBColumnSchema(
                    name=f"{column_metadata.name}_hour",
                    dtype=DBBColumnDType.category_t
                ))
        
        # Optionally retain original column
        if self.retain_original:
            result_df[column_metadata.name] = column
            new_schemas.append(column_metadata)
        elif column_metadata.metadata.get('is_time_column', False):
            # Convert to timestamp for time columns
            result_df[column_metadata.name] = dt_series.astype('int64') // 10**9  # Unix timestamp
            timestamp_schema = DBBColumnSchema(
                name=column_metadata.name,
                dtype=DBBColumnDType.timestamp_t,
                metadata={'is_time_column': True}
            )
            new_schemas.append(timestamp_schema)
        
        return result_df, new_schemas

class FilterColumn(TableTransform):
    """Filter redundant and unwanted columns from tables."""
    
    def __init__(self, drop_dtypes: Optional[List[str]] = None, drop_redundant: bool = True):
        self.drop_dtypes = drop_dtypes or []
        self.drop_redundant = drop_redundant
    
    def __call__(self, table: pd.DataFrame, table_metadata: RDBTableSchema) -> Tuple[pd.DataFrame, RDBTableSchema]:
        """Remove redundant or unwanted columns from table."""
        columns_to_drop = []
        
        for col_schema in table_metadata.columns:
            col_name = col_schema.name
            
            # Never drop primary or foreign keys
            if col_schema.dtype in [DBBColumnDType.primary_key, DBBColumnDType.foreign_key]:
                continue
                
            # Skip vector embeddings (multi-dimensional data)
            if col_name in table.columns and table[col_name].apply(lambda x: hasattr(x, '__len__') and len(x) > 1).any():
                continue
                
            # Drop redundant columns (single unique value)
            if self.drop_redundant and col_schema.dtype != DBBColumnDType.text_t:
                if col_name in table.columns:
                    unique_values = table[col_name].nunique(dropna=False)
                    if unique_values <= 1:
                        columns_to_drop.append(col_name)
                        continue
            
            # Drop columns by data type
            if col_schema.dtype in self.drop_dtypes:
                columns_to_drop.append(col_name)
        
        # Create new table and metadata
        new_table = table.drop(columns=columns_to_drop, errors='ignore')
        new_column_schemas = [
            col_schema for col_schema in table_metadata.columns 
            if col_schema.name not in columns_to_drop
        ]
        
        new_metadata = RDBTableSchema(
            name=table_metadata.name,
            columns=new_column_schemas,
            **{k: v for k, v in table_metadata.__dict__.items() if k not in ['name', 'columns']}
        )
        
        return new_table, new_metadata

class HandleDummyTable(RDBTransform):
    """Create dummy tables for foreign keys that reference non-existent tables."""
    
    def __call__(self, rdb: RDBDataset) -> RDBDataset:
        """Add dummy tables to RDB for missing primary key tables."""
        if not hasattr(rdb, 'relationships') or rdb.relationships is None:
            return rdb
            
        # Group foreign keys by primary key table
        pk_to_fk_list = defaultdict(list)
        for fk_tbl, fk_col, pk_tbl, pk_col in rdb.relationships:
            pk_to_fk_list[(pk_tbl, pk_col)].append((fk_tbl, fk_col))
        
        new_tables = dict(rdb.tables)
        new_metadata = dict(rdb.table_metadata) if hasattr(rdb, 'table_metadata') else {}
        
        for (pk_tbl, pk_col), fk_list in pk_to_fk_list.items():
            # Check if primary key table exists
            if pk_tbl not in rdb.tables:
                # Collect all foreign key values that reference this missing table
                all_fk_values = []
                for fk_tbl, fk_col in fk_list:
                    if fk_tbl in rdb.tables and fk_col in rdb.tables[fk_tbl].columns:
                        all_fk_values.extend(rdb.tables[fk_tbl][fk_col].tolist())
                
                # Create dummy table with unique primary key values
                unique_pk_values = pd.Series(all_fk_values).drop_duplicates().dropna()
                dummy_table = pd.DataFrame({pk_col: unique_pk_values})
                
                # Create metadata for dummy table
                pk_column_schema = DBBColumnSchema(
                    name=pk_col,
                    dtype=DBBColumnDType.primary_key
                )
                dummy_table_schema = RDBTableSchema(
                    name=pk_tbl,
                    columns=[pk_column_schema]
                )
                
                new_tables[pk_tbl] = dummy_table
                new_metadata[pk_tbl] = dummy_table_schema
                
                logger.info(f"Created dummy table '{pk_tbl}' with {len(unique_pk_values)} unique values for column '{pk_col}'")
        
        return rdb.create_new_with_tables_and_metadata(new_tables, new_metadata)

# Pipeline for composing transforms
class RDBTransformPipeline(RDBTransform):
    """Pipeline of RDB transformations."""
    
    def __init__(self, transforms: List[RDBTransform]):
        self.transforms = transforms
    
    def __call__(self, rdb: RDBDataset) -> RDBDataset:
        """Apply all transforms in sequence."""
        result = rdb
        for transform in self.transforms:
            result = transform(result)
        return result
```

### Phase 4: Updated CLI and API (Week 4)

**Current Code References**:
- **CLI entry point**: `fastdfs/cli/main.py` (current command structure and argument parsing)
- **CLI preprocessing**: `fastdfs/cli/preprocess.py` (current preprocessing command implementation)
- **High-level API**: `fastdfs/api.py` (current public API functions)
- **Configuration handling**: `fastdfs/utils/yaml_utils.py` (config loading) and `fastdfs/preprocess/dfs/core.py` (DFSConfig class)
- **Dataset loading**: `fastdfs/dataset/__init__.py` (current dataset loading functions)
- **Logging setup**: `fastdfs/utils/logging_config.py` (reuse existing logging configuration)

#### 4.1 New CLI Interface

```bash
# New CLI commands (simplified, no tasks)

# Load and inspect RDB
fastdfs inspect /path/to/rdb

# Apply transforms to RDB  
fastdfs transform /path/to/rdb /path/to/output --config pre-dfs.yaml

# Compute DFS features for a target dataframe (loaded from file)
fastdfs compute-features /path/to/rdb /path/to/target.parquet /path/to/output \
  --key-mappings user_id=user.user_id,item_id=item.item_id \
  --cutoff-time-column timestamp \
  --config configs/dfs/dfs-2.yaml \
  --config-overrides max_depth=3,engine=dfs2sql

# Compute DFS features with inline configuration
fastdfs compute-features /path/to/rdb /path/to/target.parquet /path/to/output \
  --key-mappings user_id=user.user_id,item_id=item.item_id \
  --cutoff-time-column timestamp \
  --max-depth 2 \
  --engine dfs2sql \
  --agg-primitives count,mean,max,min

# Full pipeline: transform RDB + compute features
fastdfs pipeline /path/to/rdb /path/to/target.parquet /path/to/output \
  --key-mappings user_id=user.user_id,item_id=item.item_id \
  --transform-config configs/transform/pre-dfs.yaml \
  --dfs-config configs/dfs/dfs-2.yaml
```

#### 4.2 New Python API

**Current Code References**:
- **API functions**: `fastdfs/api.py` functions like `run_preprocess()` (lines 20-80) - *adapt for new interfaces*
- **Dataset loading**: `fastdfs/api.py` function `load_dataset()` - *update for new RDBDataset class*
- **Engine instantiation**: `fastdfs/preprocess/dfs/core.py` engine creation logic - *adapt for new compute_features interface*
- **Configuration merging**: Pattern from existing config override handling in various modules

```python
import fastdfs
import pandas as pd

# Load RDB (no tasks)
rdb = fastdfs.load_rdb("/path/to/rdb")

# Load target dataframe from file or create programmatically
target_df = pd.read_parquet("user_item_interactions.parquet")
# target_df columns: user_id, item_id, interaction_time

# Method 1: Use default configuration with overrides
features_df = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=target_df,
    key_mappings={
        "user_id": "user.user_id",
        "item_id": "item.item_id"
    },
    cutoff_time_column="interaction_time",
    config_overrides={
        "max_depth": 2,
        "engine": "dfs2sql",
        "agg_primitives": ["count", "mean", "max", "min"]
    }
)

# Method 2: Use custom configuration object
dfs_config = fastdfs.DFSConfig(
    max_depth=3,
    engine="featuretools",
    agg_primitives=["count", "sum", "mean", "std"],
    use_cutoff_time=True,
    n_jobs=4
)

features_df = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=target_df,
    key_mappings={"user_id": "user.user_id", "item_id": "item.item_id"},
    cutoff_time_column="interaction_time",
    config=dfs_config
)

# Apply transforms (simplified - no fit/transform)
transforms = fastdfs.RDBTransformPipeline([
    fastdfs.transforms.FeaturizeDatetime(methods=["year", "month", "hour"]), 
    fastdfs.transforms.FilterColumn(drop_redundant=True),
    fastdfs.transforms.HandleDummyTable()
])
transformed_rdb = transforms(rdb)

# Save results
features_df.to_parquet("interactions_with_features.parquet")

# Or use integrated pipeline class
pipeline = fastdfs.DFSPipeline(
    transform_pipeline=transforms,
    dfs_config=dfs_config
)
result_df = pipeline.compute_features(
    rdb=rdb,
    target_dataframe=target_df,
    key_mappings={"user_id": "user.user_id", "item_id": "item.item_id"},
    cutoff_time_column="interaction_time"
)
```

## Migration Strategy

### For Existing Users

#### 1. Dataset Migration Tool

```python
# Migration utility
fastdfs.migrate_dataset("/path/to/old/task/dataset", "/path/to/new/rdb")

# This will:
# 1. Extract RDB tables → new RDB dataset
# 2. Extract task data → separate target table files  
# 3. Generate migration guide with new API usage
```

#### 2. Backward Compatibility Layer

**Current Code References**:
- **Legacy dataset loading**: `fastdfs/dataset/rdb_dataset.py` `DBBRDBDataset.__init__()` (current initialization)
- **Task access patterns**: `fastdfs/dataset/rdb_dataset.py` properties like `tasks`, `train_set`, `validation_set`, `test_set`
- **Old API entry points**: `fastdfs/api.py` existing functions that need wrapper implementations
- **Configuration parsing**: `fastdfs/utils/yaml_utils.py` config loading (adapt for old config formats)
- **CLI commands**: `fastdfs/cli/main.py` existing command handlers (wrap with new implementations)

```python
# fastdfs/compat.py - temporary compatibility layer

def run_dfs_legacy(dataset_path, output_path, config_path):
    """Legacy API that mimics old task-based behavior."""
    
    # Convert old dataset format
    with tempfile.TemporaryDirectory() as temp_dir:
        rdb_path = Path(temp_dir) / "rdb"
        migrate_dataset(dataset_path, rdb_path)
        
        # Load old config to extract task info
        old_dataset = DBBRDBDataset(dataset_path)  # Legacy loader
        
        # For each task, compute features using new API
        for task in old_dataset.tasks:
            target_table = infer_target_table_from_task(task)
            target_keys = infer_target_keys_from_task(task)
            
            features_df = compute_dfs_features(
                rdb=load_rdb(rdb_path),
                target_table=target_table,
                target_key_columns=target_keys
            )
            
            # Convert back to old task format and save
            save_as_legacy_task_format(features_df, output_path, task.metadata)
```

### 3. Migration Timeline

- **Phase 1**: Implement new interfaces alongside existing ones
- **Phase 2**: Add migration tools and compatibility layer  
- **Phase 3**: Update documentation with new patterns
- **Phase 4**: Deprecate old task-based interfaces
- **Phase 5**: Remove old interfaces in next major version

## Implementation Quick Reference for New Agents

### Key Files to Understand Before Starting:

1. **Dataset Layer**:
   - `fastdfs/dataset/rdb_dataset.py` - Current dataset implementation with task handling
   - `fastdfs/dataset/meta.py` - Metadata structures and schema definitions
   - `fastdfs/dataset/loader.py` - Table data loading mechanisms

2. **DFS Engine Layer**:
   - `fastdfs/preprocess/dfs/core.py` - Base engine class, feature preparation, and filtering logic
   - `fastdfs/preprocess/dfs/ft_engine.py` - Featuretools engine implementation
   - `fastdfs/preprocess/dfs/dfs2sql_engine.py` - SQL-based engine implementation
   - `fastdfs/preprocess/dfs/gen_sqls.py` - Feature to SQL conversion logic

3. **Transform Layer**:
   - `fastdfs/preprocess/base.py` - Current transform base classes
   - `fastdfs/preprocess/transform_preprocess.py` - Transform implementations

4. **API Layer**:
   - `fastdfs/api.py` - High-level public API functions
   - `fastdfs/cli/main.py` - CLI entry points and command handling

5. **Utilities**:
   - `fastdfs/utils/yaml_utils.py` - Configuration loading and saving
   - `fastdfs/utils/logging_config.py` - Logging setup

### Critical Logic to Preserve:

- **EntitySet building logic** in `build_dataframes()` (adapt for external targets)
- **Feature filtering logic** in `filter_features()` (adapt for new target entity names)
- **SQL generation pipeline** in `features2sql()` (reuse as-is)
- **Primitive handling** in `_convert_primitives()` (reuse as-is)
- **Array aggregation processing** (reuse as-is)

### Key Adaptation Points:

- Replace synthetic `__task__` table with external target dataframes
- Update relationship building to use `key_mappings` instead of task metadata
- Preserve all existing computation logic while changing setup phase
- Maintain engine registry and configuration patterns

## Running Examples

### Example 1: E-commerce Recommendation Features

```python
import fastdfs
import pandas as pd

# Load e-commerce RDB
rdb = fastdfs.load_rdb("ecommerce_rdb/")
# RDB contains: users, items, interactions, purchases, reviews

# Create target dataframe for user-item pairs to score
candidate_pairs = pd.DataFrame({
    "user_id": [1, 1, 2, 2, 3],
    "item_id": [100, 101, 100, 102, 103],
    "scoring_time": pd.to_datetime([
        "2024-01-15", "2024-01-15", "2024-01-16", 
        "2024-01-16", "2024-01-17"
    ])
})

# Generate features for these specific user-item pairs
interaction_features = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=candidate_pairs,
    key_mappings={
        "user_id": "user.user_id",
        "item_id": "item.item_id"
    },
    cutoff_time_column="scoring_time",
    config_overrides={
        "max_depth": 2,
        "agg_primitives": ["count", "mean", "max", "min"],
        "engine": "dfs2sql"
    }
)

# Results include:
# - user_id, item_id, scoring_time (original columns)
# - COUNT(purchases WHERE purchases.user_id = target.user_id AND purchases.timestamp < target.scoring_time)  
# - MEAN(reviews.rating WHERE reviews.item_id = target.item_id AND reviews.timestamp < target.scoring_time)
# - MAX(interactions.timestamp WHERE interactions.user_id = target.user_id AND interactions.timestamp < target.scoring_time)
# - etc.

print(f"Generated {len(interaction_features.columns)} features for {len(candidate_pairs)} user-item pairs")
interaction_features.to_parquet("recommendation_features.parquet")
```

### Example 2: Financial Transaction Analysis

```python
import fastdfs
import pandas as pd

# Load financial RDB
rdb = fastdfs.load_rdb("financial_rdb/")
# RDB contains: accounts, customers, transactions, merchants

# Apply preprocessing transforms (simplified - no fit/transform)
preprocessor = fastdfs.RDBTransformPipeline([
    fastdfs.transforms.FeaturizeDatetime(methods=["hour", "day_of_week", "month"]),
    fastdfs.transforms.FilterColumn(drop_redundant=True),
    fastdfs.transforms.HandleDummyTable()
])
clean_rdb = preprocessor(rdb)

# Load specific transactions to analyze (e.g., suspicious transactions)
suspicious_transactions = pd.read_parquet("flagged_transactions.parquet")
# Contains: transaction_id, analysis_timestamp

# Generate features for these specific transactions
transaction_features = fastdfs.compute_dfs_features(
    rdb=clean_rdb,
    target_dataframe=suspicious_transactions,
    key_mappings={
        "transaction_id": "transactions.transaction_id"
    },
    cutoff_time_column="analysis_timestamp",
    config_overrides={
        "max_depth": 3,
        "include_entities": ["accounts", "customers"],  # Focus on these relationships
        "ignore_entities": ["merchants"],  # Skip merchant aggregations for this analysis
        "agg_primitives": ["count", "sum", "mean", "std", "time_since_last"]
    }
)

# Results include features like:
# - Account balance history features
# - Customer spending patterns  
# - Transaction frequency features
# - Time-based features
```

### Example 3: Custom DFS Configuration

```python
import fastdfs
import pandas as pd

rdb = fastdfs.load_rdb("retail_rdb/")

# Create custom DFS engine with specific configuration
dfs_engine = fastdfs.DFSEngine.create("dfs2sql", config={
    "max_depth": 3,
    "agg_primitives": ["count", "sum", "mean", "mode"],
    "engine_specific": {
        "use_approximate_aggregations": True,
        "parallel_workers": 4
    }
})

# Load order data to analyze (could be from any source)
target_orders = pd.read_csv("orders_to_analyze.csv")
# Contains: order_id, customer_id, analysis_date

# Compute features with custom mappings
features = dfs_engine.compute_features(
    rdb=rdb,
    target_dataframe=target_orders,
    key_mappings={
        "order_id": "orders.order_id",
        "customer_id": "customers.customer_id"
    },
    cutoff_time_column="analysis_date",
)

# Example 4: ML Training Pipeline Integration
import sklearn.model_selection
import sklearn.ensemble

# Load training data (separate from RDB)
training_data = pd.read_parquet("training_interactions.parquet")
labels = training_data["converted"].values
train_df, test_df = sklearn.model_selection.train_test_split(training_data, test_size=0.2)

# Generate features for training set
train_features = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=train_df[["user_id", "item_id", "interaction_time"]],
    key_mappings={"user_id": "user.user_id", "item_id": "item.item_id"},
    cutoff_time_column="interaction_time",
    max_depth=2
)

# Generate features for test set (using same RDB, different target data)
test_features = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=test_df[["user_id", "item_id", "interaction_time"]],
    key_mappings={"user_id": "user.user_id", "item_id": "item.item_id"},
    cutoff_time_column="interaction_time",
    max_depth=2
)

# Train model
model = sklearn.ensemble.RandomForestClassifier()
model.fit(train_features.drop(columns=["user_id", "item_id", "interaction_time"]), 
          train_df["converted"])

# Predict on test set
predictions = model.predict(test_features.drop(columns=["user_id", "item_id", "interaction_time"]))
```

## Benefits of New Architecture

### 1. Conceptual Simplicity
- **Intuitive**: DFS = "add features to a table using related data"
- **No ML coupling**: Separates feature engineering from ML task definition
- **Flexible**: Same RDB can serve multiple feature engineering needs

### 2. Technical Advantages  
- **No data leakage**: No mixing of train/val/test in feature computation
- **Better performance**: Can optimize for single-table output
- **Composability**: Easy to chain transforms and feature computations
- **Extensibility**: Simple to add new engines and primitives

### 3. User Experience
- **Cleaner API**: Direct mapping from intent to code
- **Less configuration**: No complex task metadata files
- **Better tooling**: Can build table/column browsers, feature explorers
- **Interoperability**: Easy integration with pandas/SQL workflows

### 4. Development Benefits
- **Simpler testing**: Test feature computation independently
- **Clearer interfaces**: Well-defined inputs and outputs
- **Reduced complexity**: Fewer abstractions and edge cases
- **Better maintainability**: Focused codebase with clear responsibilities

## Implementation Risks and Mitigations

### Risk 1: Breaking Changes for Existing Users
**Mitigation**: 
- Implement migration tools for automatic dataset conversion
- Provide compatibility layer for old API during transition period
- Clear migration guide with before/after examples

### Risk 2: Loss of Train/Val/Test Split Handling
**Mitigation**:
- Provide utilities for applying features to pre-split data
- Document best practices for temporal splitting with cutoff times
- Consider adding split-aware feature computation as optional feature

### Risk 3: Performance Regressions
**Mitigation**:
- Benchmark new implementation against current system
- Optimize for single-table output (current system optimizes for task datasets)
- Leverage better caching since we're not duplicating data across tasks

### Risk 4: Feature Engineering Workflow Disruption  
**Mitigation**:
- Extensive testing with realistic datasets
- Gradual rollout with both systems available during transition
- Community feedback and iteration before removing old system

## Conclusion

Removing the Task concept from FastDFS will significantly simplify the system while making it more powerful and flexible. The new table-centric approach aligns better with how practitioners think about feature engineering and removes unnecessary abstractions that add complexity without clear benefits.

The migration strategy ensures existing users can transition smoothly while new users benefit from a cleaner, more intuitive API. The implementation plan provides a clear path to achieving these benefits while minimizing risks.

This refactoring positions FastDFS as a focused, best-in-class tool for automated feature engineering on relational data, regardless of the specific machine learning task or framework being used.
