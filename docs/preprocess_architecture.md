# FastDFS Preprocess Package Architecture

This document describes the architecture and design of the preprocess package in FastDFS, which provides a comprehensive framework for data preprocessing and feature engineering on relational database (RDB) datasets.

## Source Code Organization

The preprocess package is organized as follows:

```
fastdfs/preprocess/
├── base.py                     # Core abstractions and registration system
├── transform_preprocess.py     # Transform-based preprocessing pipeline
├── transform/                  # Transform implementations
│   ├── base.py                # Transform base classes and data structures
│   ├── composite.py           # Transform composition utilities
│   ├── wrapper.py             # RDB-to-column transform adapter
│   ├── canonicalize.py        # Data canonicalization transforms
│   ├── datetime.py            # Datetime feature engineering
│   ├── numeric.py             # Numeric data transforms
│   ├── category.py            # Categorical data handling
│   ├── text_glove.py          # Text embedding transforms
│   └── ...                    # Additional transform implementations
└── dfs/                       # Deep Feature Synthesis components
    ├── dfs_preprocess.py      # DFS preprocessing pipeline
    ├── core.py                # DFS engine abstractions
    ├── ft_engine.py           # Featuretools-based engine
    ├── dfs2sql_engine.py      # SQL-based DFS engine
    └── primitives.py          # Custom aggregation primitives
```

## Overview

The preprocess package implements a layered architecture for transforming RDB datasets through various preprocessing pipelines. The architecture is designed around the following key principles:

1. **Abstraction separation**: Clear separation between I/O operations, in-memory data transformations, and feature engineering algorithms
2. **Composability**: Transform operations can be chained and combined for complex preprocessing pipelines  
3. **Extensibility**: Plugin-based registration system for custom transforms and preprocessing methods
4. **Configuration-driven**: YAML-based configuration for declarative pipeline specification

## Core Abstractions

### 1. RDBDatasetPreprocess

**Source:** [`fastdfs/preprocess/base.py`](../fastdfs/preprocess/base.py)

`RDBDatasetPreprocess` is the top-level abstraction that orchestrates the entire preprocessing pipeline. It handles:

- Loading on-disk RDB datasets (`DBBRDBDataset`)
- Coordinating preprocessing operations
- Writing preprocessed datasets to specified output paths
- I/O management and error handling

**Key responsibilities:**
- Dataset loading and validation
- Preprocessing pipeline execution
- Output dataset generation and persistence
- Device resource management

**Interface:**
```python
class RDBDatasetPreprocess:
    def run(self, dataset: DBBRDBDataset, output_path: Path, device: DeviceInfo):
        """Execute the preprocessing pipeline on the given dataset."""
        pass
```

**Implementations:**
- [`RDBTransformPreprocess`](../fastdfs/preprocess/transform_preprocess.py): Data transformation pipeline
- [`DFSPreprocess`](../fastdfs/preprocess/dfs/dfs_preprocess.py): Deep Feature Synthesis for automated feature engineering

### 2. In-Memory Data Structures

**Source:** [`fastdfs/preprocess/transform/base.py`](../fastdfs/preprocess/transform/base.py)

#### RDBData
`RDBData` represents an in-memory relational database with the following structure:
- `tables`: Dictionary mapping table names to column collections
- `column_groups`: Optional groupings of related columns across tables
- `relationships`: Optional foreign key relationships between tables

#### ColumnData
`ColumnData` encapsulates a single column with:
- `metadata`: Column properties (dtype, constraints, etc.)
- `data`: The actual numpy array containing the column values

### 3. Transform Hierarchy

**Source:** [`fastdfs/preprocess/transform/base.py`](../fastdfs/preprocess/transform/base.py)

The package implements a two-level transform hierarchy:

#### RDBTransform (Database-level)
Operates on entire `RDBData` objects, implementing the scikit-learn interface:

```python
class RDBTransform:
    def fit(self, rdb_data: RDBData, device: DeviceInfo):
        """Learn parameters from the data."""
        pass
    
    def transform(self, rdb_data: RDBData, device: DeviceInfo) -> RDBData:
        """Apply the transformation to the data."""
        pass
    
    def fit_transform(self, rdb_data: RDBData, device: DeviceInfo) -> RDBData:
        """Fit and transform in one step."""
        pass
```

#### ColumnTransform (Column-level)
Operates on individual `ColumnData` objects:

```python
class ColumnTransform:
    # Type specifications
    input_dtype: DBBColumnDType = None
    output_dtypes: List[DBBColumnDType] = None
    output_name_formatters: List[str] = ["{name}"]
    
    def fit(self, column: ColumnData, device: DeviceInfo) -> None:
        """Learn parameters from the column data."""
        pass
    
    def transform(self, column: ColumnData, device: DeviceInfo) -> List[ColumnData]:
        """Transform the column, potentially outputting multiple columns.""" 
        pass
```

### 4. Transform Composition

#### ColumnTransformChain

**Source:** [`fastdfs/preprocess/transform/composite.py`](../fastdfs/preprocess/transform/composite.py)

Chains multiple `ColumnTransform` operations sequentially:

```python
# Example configuration
transforms:
  - name: canonicalize_numeric
  - name: featurize_datetime
    config:
      methods: ["YEAR", "MONTH", "DAYOFWEEK"]
  - name: norm_numeric
```

#### RDBTransformWrapper

**Source:** [`fastdfs/preprocess/transform/wrapper.py`](../fastdfs/preprocess/transform/wrapper.py)

Wraps `ColumnTransform` objects to work at the RDB level by:
- Identifying applicable columns based on data types
- Creating column groups for batch processing
- Applying transforms to appropriate column groups
- Integrating results back into the RDB structure

## Specialized Preprocessing Pipelines

### 1. TransformPreprocess

**Source:** [`fastdfs/preprocess/transform_preprocess.py`](../fastdfs/preprocess/transform_preprocess.py)

`RDBTransformPreprocess` implements a comprehensive data transformation pipeline:

**Workflow:**
1. **Data Extraction**: Converts on-disk `DBBRDBDataset` to in-memory `RDBData`
2. **Task Data Handling**: Separates task-specific data from shared table data
3. **Fit Phase**: Learns transformation parameters on combined training data
4. **Transform Phase**: Applies transformations to all data splits
5. **Output Generation**: Reconstructs `DBBRDBDataset` with transformed data

**Key Features:**
- Shared schema handling between tasks and data tables
- Proper train/validation/test split preservation
- Metadata propagation through transformations
- Column group and relationship preservation

### 2. DFSPreprocess

**Source:** [`fastdfs/preprocess/dfs/dfs_preprocess.py`](../fastdfs/preprocess/dfs/dfs_preprocess.py)

`DFSPreprocess` specializes in automated feature engineering using Deep Feature Synthesis:

**Architecture:**
```
DFSPreprocess
    └── DFSEngine (abstraction)
        ├── FeatureToolsEngine
        └── DFS2SQLEngine
```

**DFS Engine Sources:**
- Base engine: [`fastdfs/preprocess/dfs/core.py`](../fastdfs/preprocess/dfs/core.py)
- FeatureTools engine: [`fastdfs/preprocess/dfs/ft_engine.py`](../fastdfs/preprocess/dfs/ft_engine.py)  
- DFS2SQL engine: [`fastdfs/preprocess/dfs/dfs2sql_engine.py`](../fastdfs/preprocess/dfs/dfs2sql_engine.py)
- Custom primitives: [`fastdfs/preprocess/dfs/primitives.py`](../fastdfs/preprocess/dfs/primitives.py)

**Workflow:**
1. **Feature Preparation**: Creates feature descriptions using featuretools
2. **Feature Computation**: Delegates to specific engines for feature calculation
3. **Feature Integration**: Combines generated features with original data
4. **Output Dataset**: Creates new dataset with augmented feature set

**DFSEngine Abstraction:**
- `prepare()`: Generate feature specifications
- `compute()`: Execute feature calculations
- `filter_features()`: Remove invalid/redundant features

## Registration and Configuration System

**Sources:**
- Registration system: [`fastdfs/preprocess/base.py`](../fastdfs/preprocess/base.py), [`fastdfs/preprocess/transform/base.py`](../fastdfs/preprocess/transform/base.py), [`fastdfs/preprocess/dfs/core.py`](../fastdfs/preprocess/dfs/core.py)
- Configuration examples: [`configs/`](../configs/) directory

### Plugin Registration

All transforms and preprocessors use a decorator-based registration system:

```python
# Register RDB transforms
@rdb_transform
class MyTransform(RDBTransform):
    name = "my_transform"
    
# Register column transforms  
@column_transform
class MyColumnTransform(ColumnTransform):
    name = "my_column_transform"
    
# Register preprocessors
@rdb_preprocess  
class MyPreprocess(RDBDatasetPreprocess):
    name = "my_preprocess"
```

### YAML Configuration

The system supports declarative configuration through YAML files:

**Transform Configuration:**
```yaml
# Example: configs/transform/pre-dfs.yaml
transforms:
  - name: handle_dummy_table
  - name: key_mapping
  - name: column_transform_chain
    config:
      transforms:
        - name: canonicalize_numeric
        - name: canonicalize_datetime
        - name: featurize_datetime
          config:
            methods: ["YEAR", "MONTH", "DAYOFWEEK"]
```

**DFS Configuration:**
```yaml
# Example: configs/dfs/dfs-2.yaml
dfs:
  max_depth: 2
  use_cutoff_time: true
  engine: "dfs2sql"
  agg_primitives: ["max", "min", "mean", "count"]
```

## Built-in Transform Library

**Source Directory:** [`fastdfs/preprocess/transform/`](../fastdfs/preprocess/transform/)

The package includes an extensive library of transform implementations:

### Data Type Transforms
- [`canonicalize_numeric`](../fastdfs/preprocess/transform/canonicalize.py): Standardize numeric data formats
- [`canonicalize_datetime`](../fastdfs/preprocess/transform/canonicalize.py): Standardize datetime representations
- [`norm_numeric`](../fastdfs/preprocess/transform/numeric.py): Normalize numeric values

### Feature Engineering
- [`featurize_datetime`](../fastdfs/preprocess/transform/datetime.py): Extract temporal features (year, month, day, etc.)
- [`glove_text_embedding`](../fastdfs/preprocess/transform/text_glove.py): Convert text to vector embeddings
- [`remap_category`](../fastdfs/preprocess/transform/category.py): Handle categorical encoding

### Data Cleaning
- [`filter_column`](../fastdfs/preprocess/transform/filter_column.py): Remove columns by criteria
- [`fill_timestamp`](../fastdfs/preprocess/transform/fill_timestamp.py): Handle missing temporal data
- [`handle_dummy_table`](../fastdfs/preprocess/transform/dummy_table.py): Process placeholder tables

### Structural Transforms
- [`key_mapping`](../fastdfs/preprocess/transform/key_mapping.py): Manage primary/foreign key relationships

## Extension Points

### Adding Custom Transforms

1. **Column-level Transform:**
```python
@column_transform
class CustomColumnTransform(ColumnTransform):
    name = "custom_column"
    input_dtype = DBBColumnDType.numeric
    output_dtypes = [DBBColumnDType.numeric]
    
    def fit(self, column: ColumnData, device: DeviceInfo):
        # Learn parameters
        pass
        
    def transform(self, column: ColumnData, device: DeviceInfo):
        # Apply transformation
        return [transformed_column]
```

2. **RDB-level Transform:**
```python
@rdb_transform  
class CustomRDBTransform(RDBTransform):
    name = "custom_rdb"
    
    def fit(self, rdb_data: RDBData, device: DeviceInfo):
        pass
        
    def transform(self, rdb_data: RDBData, device: DeviceInfo):
        return transformed_rdb_data
```

3. **Custom Preprocessor:**
```python
@rdb_preprocess
class CustomPreprocess(RDBDatasetPreprocess):
    name = "custom_preprocess"
    
    def run(self, dataset: DBBRDBDataset, output_path: Path, device: DeviceInfo):
        # Implement custom preprocessing logic
        pass
```

### Adding DFS Engines

```python
@dfs_engine
class CustomDFSEngine(DFSEngine):
    name = "custom_engine"
    
    def compute(self, features: List[ft.FeatureBase]) -> pd.DataFrame:
        # Implement custom feature computation
        pass
```

## Usage Examples

**CLI Implementation:** [`fastdfs/cli/preprocess.py`](../fastdfs/cli/preprocess.py)  
**Python API Examples:** [`examples/`](../examples/) directory

### Command Line Interface
```bash
# Transform preprocessing
fastdfs preprocess /path/to/dataset transform /path/to/output --config configs/transform/pre-dfs.yaml

# DFS preprocessing  
fastdfs preprocess /path/to/dataset dfs /path/to/output --config configs/dfs/dfs-2.yaml
```

### Python API
```python
import fastdfs

# Load dataset
dataset = fastdfs.load_rdb_data("/path/to/dataset")

# Create and configure preprocessor
preprocess = fastdfs.RDBTransformPreprocess(config)

# Run preprocessing
preprocess.run(dataset, "/path/to/output", device_info)
```

**Complete examples:**
- [Python API Example](../examples/python_api_example.py): Comprehensive usage demonstration
- [CLI Example](../examples/cli_example.py): Command-line interface usage

## Best Practices

1. **Transform Design**: Keep transforms focused and composable
2. **Configuration**: Use YAML for pipeline specification and parameterization
3. **Error Handling**: Implement proper validation and error reporting
4. **Performance**: Consider memory usage for large datasets
5. **Testing**: Test transforms in isolation and as part of pipelines
6. **Documentation**: Document transform behavior and configuration options

## Future Extensions

The architecture supports several planned extensions:

1. **Distributed Processing**: Scale transforms across multiple nodes
2. **Streaming Transforms**: Support for incremental data processing  
3. **Custom Primitives**: Extend DFS with domain-specific aggregation functions
4. **Performance Optimization**: Caching and incremental computation
5. **Integration**: Connect with external feature stores and ML platforms

## Complete Workflow Example

Here's a complete example showing how the components work together:

### 1. Dataset Loading
```python
# Load RDB dataset from disk
# Implementation: fastdfs/dataset/rdb_dataset.py
dataset = fastdfs.load_rdb_data("/path/to/dataset")
# dataset contains:
#   - metadata: Table schemas, relationships, tasks
#   - tables: Raw data for each table  
#   - tasks: Task-specific train/val/test splits
```

### 2. Transform Preprocessing
```python
# Configure transformation pipeline
# Source: fastdfs/preprocess/transform_preprocess.py
config = RDBTransformPreprocessConfig.parse_obj({
    "transforms": [
        {"name": "canonicalize_numeric"},
        {"name": "featurize_datetime", "config": {"methods": ["YEAR", "MONTH"]}},
        {"name": "norm_numeric"}
    ]
})

# Create and run preprocessor
preprocessor = RDBTransformPreprocess(config)
preprocessor.run(dataset, "/output/path", device_info)

# Internal workflow:
# 1. extract_data(): Convert tables to RDBData
# 2. extract_task_data(): Separate fit vs transform data
# 3. Fit transforms on combined training data
# 4. Transform all data (train/val/test) 
# 5. output_data(): Reconstruct dataset with transformed data
```

### 3. DFS Preprocessing  
```python
# Configure DFS pipeline
# Source: fastdfs/preprocess/dfs/dfs_preprocess.py
dfs_config = DFSPreprocessConfig(
    dfs=DFSConfig(max_depth=2, engine="dfs2sql")
)

# Create and run DFS preprocessor
dfs_preprocessor = DFSPreprocess(dfs_config)
dfs_preprocessor.run(dataset, "/dfs/output", device_info)

# Internal workflow:
# 1. For each task:
#    a. Build EntitySet from RDB data
#    b. Generate feature specs with featuretools
#    c. Compute features via DFS engine (ft_engine.py or dfs2sql_engine.py)
# 2. Integrate generated features with original data
# 3. Create output dataset with enhanced feature set
```

### 4. Custom Transform Development
```python
# Implement custom column transform
# See: fastdfs/preprocess/transform/base.py for base classes
@column_transform
class CustomTransform(ColumnTransform):
    name = "custom_transform"
    input_dtype = DBBColumnDType.numeric
    output_dtypes = [DBBColumnDType.numeric]
    
    def fit(self, column: ColumnData, device: DeviceInfo):
        self.mean = np.mean(column.data)
        
    def transform(self, column: ColumnData, device: DeviceInfo):
        transformed_data = column.data - self.mean
        return [ColumnData(column.metadata, transformed_data)]

# Register and use in pipeline
config = {
    "transforms": [{"name": "custom_transform"}]
}
```

This architecture provides a flexible, extensible foundation for complex data preprocessing workflows while maintaining clean separation of concerns and supporting both programmatic and declarative configuration approaches.

## Transform Behavior and Column Groups

### Fit/Transform Workflow

FastDFS uses a scikit-learn style fit/transform pattern with precise data usage:

**Fitting Data**: All data tables + task tables' train split  
**Transforming Data**: All tables (data tables + all task table splits)

#### Example: Link Prediction Task

Dataset structure:
```yaml
# Data tables (always available)
tables:
  - name: "user"
    source: "data/user.npz"
    columns:
      - name: user_id
        dtype: primary_key
      - name: user_feature_0
        dtype: float
  - name: "item"
    source: "data/item.npz" 
    columns:
      - name: item_id
        dtype: primary_key
      - name: item_feature_0
        dtype: float

# Task tables (train/val/test splits)
tasks:
  - name: "linkpred"
    source: "linkpred/{split}.npz"  # Creates train/val/test splits
    columns:
      - name: user_id
        shared_schema: interaction.user_id
      - name: item_id
        shared_schema: interaction.item_id
      - name: timestamp
        shared_schema: interaction.timestamp
      - name: label
        dtype: category
```

Transform workflow:
```python
# 1. FITTING: Use all data tables + train split only
fit_data = {
    "user": user.npz,              # All user data
    "item": item.npz,              # All item data  
    "linkpred_train": linkpred/train.npz  # Only training split
}
# Fit normalizer on combined data: mean=0.5, std=0.2

# 2. TRANSFORMING: Apply to all tables
transform_tables = [
    "user",           # Data table
    "item",           # Data table
    "linkpred_train", # Task table (train)
    "linkpred_val",   # Task table (validation) 
    "linkpred_test"   # Task table (test)
]
# All use same fitted parameters: (x - 0.5) / 0.2
```

### Column Groups

Column groups coordinate transforms across tables with the same `shared_schema`:

```python
# Columns with shared_schema="interaction.user_id" form column groups:
column_groups = {
    "interaction.user_id": ["linkpred_train.user_id", "linkpred_val.user_id", "linkpred_test.user_id"],
    "interaction.timestamp": ["linkpred_train.timestamp", "linkpred_val.timestamp", "linkpred_test.timestamp"]
}

# Workflow:
# 1. Fit on linkpred_train.user_id only
# 2. Transform all: linkpred_train.user_id, linkpred_val.user_id, linkpred_test.user_id
#    (using same fitted parameters)
```

This ensures consistent preprocessing where validation and test data are transformed using parameters learned only from training data, preventing data leakage while handling unseen values properly.

## Additional Resources

- **API Documentation**: [`fastdfs/api.py`](../fastdfs/api.py) - Main public API
- **Utilities**: [`fastdfs/utils/`](../fastdfs/utils/) - Logging, device management, YAML utilities
- **Examples**: [`examples/`](../examples/) - Complete usage examples
- **Configuration Templates**: [`configs/`](../configs/) - Pre-built configuration files
- **Tests**: [`tests/`](../tests/) - Integration and unit tests
