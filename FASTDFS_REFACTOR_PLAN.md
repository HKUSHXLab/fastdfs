# FastDFS Refactoring Plan

## Overview

This document outlines the plan to extract Deep Feature Synthesis (DFS) functionality from the tab2graph codebase into a standalone `fastdfs` package. The goal is to create a focused, self-contained package for automated feature engineering while maintaining the existing code structure and minimizing breaking changes.

## Current Architecture Analysis

### Key Components in tab2graph

1. **DFS Core Logic** (`tab2graph/preprocess/dfs/`)
   - `core.py` - Abstract DFS engine interface, configuration classes
   - `dfs_preprocess.py` - Main DFS preprocessing orchestrator
   - `ft_engine.py` - Featuretools-based DFS implementation
   - `dfs2sql_engine.py` - High-performance SQL-based DFS engine
   - `primitives.py` - Custom aggregation primitives (Concat, Join, ArrayMax, etc.)
   - `gen_sqls.py` - SQL query generation from featuretools features
   - `database.py` - DuckDB database utilities

2. **Transform Pipeline** (`tab2graph/preprocess/transform/`)
   - `base.py` - Transform framework (RDBTransform, ColumnTransform)
   - Pre-DFS transforms: canonicalization, datetime handling, key mapping
   - Post-DFS transforms: normalization, category remapping, filtering

3. **Preprocessing Orchestration** (`tab2graph/preprocess/`)
   - `base.py` - Base preprocessing classes and registry
   - `transform_preprocess.py` - Transform pipeline orchestrator
   - `__init__.py` - Module exports

4. **CLI Interface** (`tab2graph/cli/preprocess.py`)
   - Command-line interface using typer
   - Configuration loading via pydantic
   - Integration with tab2graph.main

5. **Dataset Interface** (`dbinfer_bench/`)
   - `rdb_dataset.py` - RDB dataset loading and manipulation
   - `dataset_meta.py` - Metadata classes and schemas
   - `table_loader.py`, `table_writer.py` - Data I/O utilities

6. **Configuration System**
   - `configs/dfs/` - DFS engine configurations (depth, primitives, SQL vs featuretools)
   - `configs/transform/` - Transform pipeline configurations (pre-dfs, post-dfs, single)

## Refactoring Strategy: Minimal Structure Changes

### Philosophy
- **Preserve existing module structure** to maintain familiarity
- **Keep the same API patterns** for easy migration
- **Minimize code duplication** between packages
- **Maintain configuration compatibility** with existing YAML files

### Package Structure for FastDFS

```
fastdfs/
├── pyproject.toml
├── README.md
├── MIGRATION_GUIDE.md
├── fastdfs/
│   ├── __init__.py                    # Main API exports
│   ├── preprocess/                    # PRESERVE: Same structure as tab2graph
│   │   ├── __init__.py               
│   │   ├── base.py                    # COPY: Base preprocessing classes
│   │   ├── dfs/                       # COPY: Entire DFS module
│   │   │   ├── __init__.py
│   │   │   ├── core.py                # COPY: DFS engines and config
│   │   │   ├── dfs_preprocess.py      # COPY: Main DFS orchestrator
│   │   │   ├── ft_engine.py           # COPY: Featuretools engine
│   │   │   ├── dfs2sql_engine.py      # COPY: SQL engine
│   │   │   ├── primitives.py          # COPY: Custom primitives
│   │   │   ├── gen_sqls.py            # COPY: SQL generation
│   │   │   └── database.py            # COPY: Database utilities
│   │   ├── transform/                 # COPY: Transform framework
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # COPY: Transform base classes
│   │   │   ├── canonicalize.py        # COPY: Data canonicalization
│   │   │   ├── datetime.py            # COPY: Datetime transforms
│   │   │   ├── key_mapping.py         # COPY: Key mapping
│   │   │   ├── filter_column.py       # COPY: Column filtering
│   │   │   ├── fill_timestamp.py      # COPY: Timestamp filling
│   │   │   ├── dummy_table.py         # COPY: Dummy table handling
│   │   │   ├── category.py            # COPY: Category transforms
│   │   │   ├── numeric.py             # COPY: Numeric transforms
│   │   │   ├── composite.py           # COPY: Composite transforms
│   │   │   └── wrapper.py             # COPY: Transform wrappers
│   │   └── transform_preprocess.py    # COPY: Transform orchestrator
│   ├── cli/                           # PRESERVE: CLI structure
│   │   ├── __init__.py
│   │   ├── main.py                    # NEW: Main CLI entry point
│   │   └── preprocess.py              # ADAPT: From tab2graph CLI
│   ├── dataset/                       # NEW: Simplified dataset interface
│   │   ├── __init__.py
│   │   ├── base.py                    # EXTRACT: From dbinfer_bench
│   │   ├── rdb_dataset.py             # SIMPLIFY: From dbinfer_bench
│   │   ├── meta.py                    # EXTRACT: Dataset metadata
│   │   ├── loader.py                  # EXTRACT: Table loading
│   │   └── writer.py                  # EXTRACT: Table writing
│   ├── utils/                         # NEW: Shared utilities
│   │   ├── __init__.py
│   │   ├── device.py                  # COPY: Device utilities
│   │   ├── datetime_utils.py          # COPY: Datetime utilities
│   │   └── yaml_utils.py              # COPY: YAML utilities
│   └── config/                        # NEW: Configuration management
│       ├── __init__.py
│       └── default_configs.py         # Embedded default configs
├── configs/                           # PRESERVE: Configuration files
│   ├── dfs/
│   │   ├── dfs-1.yaml                 # COPY: Depth-1 DFS config
│   │   ├── dfs-1-sql.yaml             # COPY: SQL engine config
│   │   ├── dfs-2.yaml                 # COPY: Depth-2 DFS config
│   │   ├── dfs-2-sql.yaml             # COPY: SQL engine config
│   │   ├── dfs-3.yaml                 # COPY: Depth-3 DFS config
│   │   └── dfs-3-sql.yaml             # COPY: SQL engine config
│   └── transform/
│       ├── pre-dfs.yaml               # COPY: Pre-DFS transforms
│       ├── post-dfs.yaml              # COPY: Post-DFS transforms
│       └── single.yaml                # COPY: Single table transforms
├── tests/
│   ├── __init__.py
│   ├── test_dfs_engines.py            # Test DFS engines
│   ├── test_transforms.py             # Test transform pipeline
│   ├── test_cli.py                    # Test CLI interface
│   └── data/                          # Test datasets
│       └── synthetic/                 # COPY: datasets_synthetic/test_rdb
└── examples/
    ├── basic_usage.py                 # Basic API usage
    ├── cli_examples.sh                # CLI usage examples
    └── custom_primitives.py           # Custom primitive examples
```

## Implementation Plan

### Phase 1: Core Extraction (Week 1-2)

#### 1.1 Setup Package Structure
```bash
# Create new package directory
mkdir fastdfs
cd fastdfs

# Initialize package
touch pyproject.toml README.md
mkdir -p fastdfs/{preprocess/{dfs,transform},cli,dataset,utils,config}
mkdir -p configs/{dfs,transform}
mkdir -p tests examples
```

#### 1.2 Copy Core DFS Components
- **COPY** `tab2graph/preprocess/dfs/` → `fastdfs/preprocess/dfs/`
- **COPY** `tab2graph/preprocess/transform/` → `fastdfs/preprocess/transform/`
- **COPY** `tab2graph/preprocess/base.py` → `fastdfs/preprocess/base.py`
- **COPY** `tab2graph/preprocess/transform_preprocess.py` → `fastdfs/preprocess/transform_preprocess.py`

#### 1.3 Extract Dataset Interface
- **EXTRACT** core classes from `dbinfer_bench/rdb_dataset.py`
- **SIMPLIFY** by removing graph-specific functionality
- **PRESERVE** essential interfaces: `DBBRDBDataset`, `DBBRDBTask`, metadata classes

#### 1.4 Adapt CLI Interface
- **COPY** `tab2graph/cli/preprocess.py` → `fastdfs/cli/preprocess.py`
- **REMOVE** dependencies on tab2graph-specific modules
- **CREATE** `fastdfs/cli/main.py` as entry point

### Phase 2: Dependency Resolution (Week 2-3) - ✅ COMPLETED

#### 2.1 Update Imports ✅ DONE
✅ Replaced tab2graph imports with fastdfs equivalents:
```python
# Before (in tab2graph)
from ..device import DeviceInfo
from ..preprocess import get_rdb_preprocess_class
import dbinfer_bench as dbb

# After (in fastdfs)
from ..utils.device import DeviceInfo
from ..preprocess import get_rdb_preprocess_class
from ..dataset import load_rdb_data
```

#### 2.2 Minimize Dependencies ✅ DONE
✅ **Updated pyproject.toml with minimal dependencies**:
```toml
dependencies = [
    # Data processing
    "pandas", "numpy", "scipy", "scikit-learn",
    
    # DFS engines
    "featuretools", "duckdb", "sqlalchemy", "sqlglot", "sql_formatter",
    
    # Configuration and CLI
    "pydantic==1.10.12", "pyyaml", "typer", "tqdm",
    
    # Utilities
    "psutil",
]
```

✅ **Successfully removed dependencies**:
- Graph ML: `dgl`, `networkx`, `ogb` 
- Deep learning: `torch`, `torchmetrics`, `autogluon`
- Cloud/monitoring: `wandb`, `s3fs`, `boto3`
- NLP: `transformers`, `gensim`

#### 2.3 Configuration Management ✅ DONE
✅ **COPIED** existing YAML configs to `fastdfs/configs/`
✅ **MAINTAINED** backward compatibility with existing config files
✅ **WORKING** package imports: `import fastdfs` ✅
✅ **WORKING** CLI interface: `python -m fastdfs.cli.main --help` ✅

### Phase 3: Enhanced API and End-to-End Testing (Week 3-4) - ✅ COMPLETED

**Status**: ✅ COMPLETED - All objectives achieved, comprehensive testing passed

**Key Achievements**:
- ✅ Enhanced API implemented and fully functional
- ✅ All 4 core functions tested and working (load_dataset, run_transform, run_dfs, run_full_pipeline)
- ✅ Both DFS engines (Featuretools + DFS2SQL) validated  
- ✅ End-to-end pipeline automation with cleanup
- ✅ Comprehensive error handling and logging
- ✅ Performance validation completed
- ✅ Demo showcasing all features successful

**Test Results**: 4/4 API tests PASSED, full demo validation successful

#### 3.1 Public API Design ✅ IMPLEMENTED

**High-level API** (`fastdfs/__init__.py`):
```python
# Main functions
from .api import (
    run_dfs,
    run_transform,
    run_full_pipeline,
    load_dataset,
    save_dataset,
)

# Core classes
from .preprocess.dfs import DFSPreprocess, DFSConfig
from .preprocess.transform_preprocess import RDBTransformPreprocess
from .dataset import DBBRDBDataset, DBBRDBTask

# Version
__version__ = "0.1.0"
```

**Simplified API** (`fastdfs/api.py`):
```python
def run_dfs(
    dataset_path: str,
    output_path: str,
    max_depth: int = 2,
    engine: str = "featuretools",
    config_path: Optional[str] = None
) -> None:
    """Run DFS on a dataset with simplified interface."""
    
def run_transform(
    dataset_path: str,
    output_path: str,
    transform_type: str = "pre-dfs",  # or "post-dfs", "single"
    config_path: Optional[str] = None
) -> None:
    """Run data transforms with simplified interface."""
    
def run_full_pipeline(
    dataset_path: str,
    output_path: str,
    max_depth: int = 2,
    engine: str = "featuretools"
) -> None:
    """Run complete DFS pipeline: pre-transform → DFS → post-transform."""
```

#### 3.2 CLI Interface

**Command Structure** (preserve existing patterns):
```bash
# Individual steps (same as tab2graph)
fastdfs preprocess <dataset> transform <output> -c configs/transform/pre-dfs.yaml
fastdfs preprocess <dataset> dfs <output> -c configs/dfs/dfs-2-sql.yaml
fastdfs preprocess <dataset> transform <output> -c configs/transform/post-dfs.yaml

# Simplified pipeline
fastdfs pipeline <dataset> <output> --depth 2 --engine sql
fastdfs pipeline <dataset> <output> --depth 3 --engine featuretools --config custom.yaml
```

#### 3.3 Testing Strategy

**Unit Tests**:
```python
# Test DFS engines
def test_featuretools_engine():
    """Test featuretools-based DFS engine."""

def test_sql_engine():
    """Test SQL-based DFS engine."""

# Test transforms
def test_pre_dfs_transforms():
    """Test pre-DFS data transforms."""

def test_post_dfs_transforms():
    """Test post-DFS data transforms."""

# Test CLI
def test_cli_preprocess():
    """Test CLI preprocessing commands."""
```

**Integration Tests**:
```python
def test_full_pipeline():
    """Test complete DFS pipeline end-to-end."""

def test_config_compatibility():
    """Test compatibility with existing config files."""
```

### Phase 4: Documentation and Migration (Week 4-5)

#### 4.1 Documentation

**README.md**:
```markdown
# FastDFS - Deep Feature Synthesis for Tabular Data

FastDFS is a standalone package for automated feature engineering using Deep Feature Synthesis (DFS). 
Extracted from the tab2graph project, it provides focused functionality for relational feature generation.

## Quick Start

```python
import fastdfs

# Load your dataset
dataset = fastdfs.load_dataset("path/to/dataset")

# Run DFS with depth 2
result = fastdfs.run_dfs(dataset, depth=2, engine="sql")

# Or use CLI
# fastdfs pipeline input_data output_data --depth 2 --engine sql
```

#### 4.2 Migration Guide

**MIGRATION_GUIDE.md**:
```markdown
# Migration from tab2graph to fastdfs

## Command Migration

### Before (tab2graph)
```bash
t2g preprocess dataset transform output -c configs/transform/pre-dfs.yaml
t2g preprocess dataset dfs output -c configs/dfs/dfs-2-sql.yaml
```

### After (fastdfs)
```bash
fastdfs preprocess dataset transform output -c configs/transform/pre-dfs.yaml
fastdfs preprocess dataset dfs output -c configs/dfs/dfs-2-sql.yaml
```

## Python API Migration

### Before
```python
from tab2graph.preprocess.dfs import DFSPreprocess
```

### After
```python
from fastdfs.preprocess.dfs import DFSPreprocess
```
```

## Benefits of This Approach

### 1. Minimal Code Changes
- **Preserve existing module structure** reduces learning curve
- **Same API patterns** enable easy migration
- **Compatible configurations** work without modification

### 2. Reduced Maintenance Burden
- **Focused scope** eliminates unnecessary complexity
- **Fewer dependencies** reduce security and compatibility issues
- **Standalone package** enables independent versioning

### 3. Enhanced Usability
- **Simplified installation** without heavy ML frameworks
- **Clear purpose** makes the package easier to understand
- **Better performance** by removing unused components

### 4. Future Extensibility
- **Modular architecture** supports adding new DFS engines
- **Plugin system** for custom primitives and transforms
- **Clean interfaces** enable integration with other tools

## Risk Mitigation

### 1. Backward Compatibility
- **Same CLI commands** work with minimal changes
- **Compatible file formats** for datasets and outputs
- **Configuration compatibility** with existing YAML files

### 2. Testing Strategy
- **Reference implementation** testing against tab2graph
- **Comprehensive test suite** covering all functionality
- **Performance benchmarks** to ensure no regressions

### 3. Documentation
- **Clear migration guide** for existing users
- **Comprehensive examples** for common use cases
- **API documentation** for developers

## Timeline and Milestones

- **Week 1-2**: Core extraction and package setup
- **Week 3**: Dependency resolution and API design
- **Week 4**: Testing and validation
- **Week 5**: Documentation and release preparation

## Success Criteria

1. **Functional Compatibility**: All existing DFS workflows work with fastdfs
2. **Performance Parity**: No significant performance degradation
3. **Reduced Dependencies**: <50% of original dependency count
4. **Easy Migration**: <1 day effort for existing tab2graph DFS users
5. **Standalone Operation**: No dependencies on tab2graph or graph libraries

This refactoring plan preserves the battle-tested architecture while creating a focused, maintainable package for Deep Feature Synthesis operations.
