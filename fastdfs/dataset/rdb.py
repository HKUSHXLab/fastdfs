"""
Relational Database (RDB) implementation.

This module implements the table-centric RDB interface that focuses purely on
relational database tables for feature engineering.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import sqlalchemy
from sqlalchemy import MetaData, Table, Column, String, ForeignKey, Float, DateTime

from ..utils import yaml_utils
from .loader import get_table_data_loader
from .meta import (
    RDBMeta,
    RDBTableSchema,
    RDBColumnDType,
    RDBColumnSchema,
    RDBTableDataFormat,
)

__all__ = ['RDB', 'RDBDataset']

class RDB:
    """Relational Database (RDB) - focuses on relational tables only."""
    
    def __init__(
        self, 
        path: Optional[Path] = None, 
        metadata: Optional[RDBMeta] = None, 
        tables: Optional[Dict[str, pd.DataFrame]] = None
    ):
        if path:
            self.path = Path(path)
            self.metadata = self._load_metadata()
            self.tables = self._load_tables()
        elif metadata is not None and tables is not None:
            self.path = None
            self.metadata = metadata
            self.tables = tables
        else:
            raise ValueError("Either path or (metadata and tables) must be provided.")
    
    def _load_metadata(self) -> RDBMeta:
        """Load metadata from YAML file."""
        metadata_path = self.path / 'metadata.yaml'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Load raw metadata and convert to simplified format
        raw_data = yaml_utils.load_yaml(metadata_path)
        
        # Convert tables
        tables = []
        for table_data in raw_data.get('tables', []):
            table_schema = RDBTableSchema(
                name=table_data['name'],
                source=table_data['source'],
                format=RDBTableDataFormat(table_data['format']),
                columns=[
                    RDBColumnSchema(**col_data) 
                    for col_data in table_data['columns']
                ],
                time_column=table_data.get('time_column')
            )
            tables.append(table_schema)
        
        return RDBMeta(
            dataset_name=raw_data['dataset_name'],
            tables=tables
        )
    
    def _load_tables(self) -> Dict[str, pd.DataFrame]:
        """Load all tables as pandas DataFrames."""
        tables = {}
        for table_schema in self.metadata.tables:
            table_path = self.path / table_schema.source
            if not table_path.exists():
                raise FileNotFoundError(f"Table data file not found: {table_path}")
            
            # Load raw data using existing loader
            loader = get_table_data_loader(table_schema.format)
            table_data = loader(table_path)
            
            # Convert to pandas DataFrame
            df_data = {}
            for col_schema in table_schema.columns:
                if col_schema.name not in table_data:
                    raise ValueError(
                        f"Column {col_schema.name} not found in table {table_schema.name}"
                    )
                
                col_data = table_data[col_schema.name]
                
                # Explicitly cast data based on schema type to ensure consistency
                if col_schema.dtype == RDBColumnDType.float_t:
                    col_data = pd.to_numeric(col_data, errors='coerce')
                elif col_schema.dtype == RDBColumnDType.datetime_t:
                    col_data = pd.to_datetime(col_data, errors='coerce')
                
                df_data[col_schema.name] = col_data
            
            tables[table_schema.name] = pd.DataFrame(df_data)
        
        return tables
    
    @property
    def table_names(self) -> List[str]:
        """Get list of table names."""
        return list(self.tables.keys())
    
    def get_table(self, name: str) -> pd.DataFrame:
        """Get a table as a pandas DataFrame."""
        if name not in self.tables:
            raise ValueError(f"Table {name} not found. Available tables: {self.table_names}")
        return self.tables[name].copy()
    
    def get_table_metadata(self, name: str) -> RDBTableSchema:
        """Get metadata for a specific table."""
        for table_schema in self.metadata.tables:
            if table_schema.name == name:
                return table_schema
        raise ValueError(f"Table {name} not found. Available tables: {self.table_names}")
        
    def get_relationships(self) -> List[Tuple[str, str, str, str]]:
        """Get relationships as (child_table, child_col, parent_table, parent_col)."""
        return self.metadata.relationships
        
    def create_new_with_tables(self, new_tables: Dict[str, pd.DataFrame]) -> 'RDB':
        """Create new RDB with updated tables (for transforms)."""
        # Create a new instance with same metadata but different table data
        new_dataset = RDB.__new__(RDB)
        new_dataset.path = self.path
        new_dataset.metadata = self.metadata
        new_dataset.tables = new_tables.copy()
        return new_dataset
    
    def create_new_with_tables_and_metadata(self, new_tables: Dict[str, pd.DataFrame], new_metadata: Dict[str, RDBTableSchema]) -> 'RDB':
        """Create new RDB with updated tables and metadata (for transforms that modify schemas)."""
        # Create a new instance with updated metadata and table data
        new_dataset = RDB.__new__(RDB)
        new_dataset.path = self.path
        new_dataset.tables = new_tables.copy()
        
        # Update metadata with new table schemas
        updated_table_schemas = []
        for table_schema in self.metadata.tables:
            if table_schema.name in new_metadata:
                updated_table_schemas.append(new_metadata[table_schema.name])
            else:
                updated_table_schemas.append(table_schema)
        
        # Add any completely new table schemas
        existing_table_names = {schema.name for schema in self.metadata.tables}
        for table_name, schema in new_metadata.items():
            if table_name not in existing_table_names:
                updated_table_schemas.append(schema)
        
        # Create new metadata object
        new_dataset.metadata = RDBMeta(
            dataset_name=self.metadata.dataset_name,
            tables=updated_table_schemas
        )
        
        return new_dataset
        
    @property
    def sqlalchemy_metadata(self) -> sqlalchemy.MetaData:
        """Get SQLAlchemy metadata for the RDB."""
        metadata = MetaData()
        
        # First pass: Create tables without foreign keys
        sql_tables = {}
        for table_schema in self.metadata.tables:
            columns = []
            for col_schema in table_schema.columns:
                # Map column types to SQLAlchemy types
                if col_schema.dtype == RDBColumnDType.float_t:
                    sql_type = Float
                elif col_schema.dtype == RDBColumnDType.datetime_t:
                    sql_type = DateTime
                elif col_schema.dtype == RDBColumnDType.primary_key:
                    sql_type = String
                elif col_schema.dtype == RDBColumnDType.foreign_key:
                    sql_type = String
                else:
                    sql_type = String
                
                # Create column (add foreign keys later)
                if col_schema.dtype == RDBColumnDType.primary_key:
                    col = Column(col_schema.name, sql_type, primary_key=True)
                else:
                    col = Column(col_schema.name, sql_type)
                
                columns.append(col)
            
            sql_table = Table(table_schema.name, metadata, *columns)
            sql_tables[table_schema.name] = sql_table
        
        # Second pass: Add foreign key constraints after all tables are created
        for child_table, child_col, parent_table, parent_col in self.get_relationships():
            child_sql_table = sql_tables[child_table]
            
            # Find the child column and add foreign key
            for col in child_sql_table.columns:
                if col.name == child_col:
                    fk = ForeignKey(f"{parent_table}.{parent_col}")
                    col.foreign_keys.add(fk)
                    break
        
        return metadata


# Migration utilities for converting from old task-based format
def convert_task_dataset_to_rdb(old_dataset_path: Path, rdb_output_path: Path):
    """Convert old task-based dataset to new RDB-only format."""
    import shutil
    from .legacy.rdb_dataset import DBBRDBDataset  # Import current implementation
    
    # Load old dataset
    old_dataset = DBBRDBDataset(old_dataset_path)
    
    # Create output directory
    rdb_output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert metadata (tables only, no tasks)
    new_metadata = {
        'dataset_name': old_dataset.metadata.dataset_name,
        'tables': []
    }
    
    # Convert table schemas
    for table_schema in old_dataset.metadata.tables:
        table_dict = {
            'name': table_schema.name,
            'source': table_schema.source,
            'format': table_schema.format.value,
            'columns': []
        }
        
        # Convert column schemas
        for col_schema in table_schema.columns:
            col_dict = {
                'name': col_schema.name,
                'dtype': col_schema.dtype
            }
            
            # Add extra fields based on dtype
            if col_schema.dtype == 'foreign_key':
                if hasattr(col_schema, 'link_to'):
                    col_dict['link_to'] = col_schema.link_to
                if hasattr(col_schema, 'capacity'):
                    col_dict['capacity'] = col_schema.capacity
            elif col_schema.dtype == 'primary_key':
                if hasattr(col_schema, 'capacity'):
                    col_dict['capacity'] = col_schema.capacity
            elif col_schema.dtype == 'float':
                if hasattr(col_schema, 'in_size'):
                    col_dict['in_size'] = col_schema.in_size
            elif col_schema.dtype == 'category':
                if hasattr(col_schema, 'num_categories'):
                    col_dict['num_categories'] = col_schema.num_categories
            
            table_dict['columns'].append(col_dict)
        
        # Add time column if present
        if table_schema.time_column:
            table_dict['time_column'] = table_schema.time_column
            
        new_metadata['tables'].append(table_dict)
    
    # Save new metadata
    yaml_utils.save_yaml(new_metadata, rdb_output_path / 'metadata.yaml')
    
    # Copy table data files (unchanged)
    for table_schema in old_dataset.metadata.tables:
        source_path = old_dataset.path / table_schema.source
        dest_path = rdb_output_path / table_schema.source
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.exists():
            shutil.copy2(source_path, dest_path)


def extract_target_tables_from_tasks(old_dataset_path: Path, output_dir: Path):
    """Extract task data as separate target table files."""
    from .legacy.rdb_dataset import DBBRDBDataset  # Import current implementation
    
    old_dataset = DBBRDBDataset(old_dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
            
            print(f"Extracted {task.metadata.name} {split_name} to {output_file}")

RDBDataset = RDB  # Alias for backward compatibility