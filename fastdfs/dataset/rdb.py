"""
Relational Database (RDB) implementation.

This module implements the table-centric RDB interface that focuses purely on
relational database tables for feature engineering.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import sqlalchemy
from sqlalchemy import MetaData, Table, Column, String, ForeignKey, Float, DateTime
from loguru import logger

from ..utils import yaml_utils
from ..utils.type_utils import infer_semantic_type, safe_convert_to_string
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
            name=raw_data['name'],
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
        return [t.name for t in self.metadata.tables]

    def get_table_metadata(self, table_name: str) -> RDBTableSchema:
        """Get metadata for a specific table."""
        for table in self.metadata.tables:
            if table.name == table_name:
                return table
        raise ValueError(f"Table {table_name} not found in metadata.")

    def get_table_dataframe(self, table_name: str) -> pd.DataFrame:
        """Get pandas DataFrame for a specific table."""
        if table_name not in self.tables:
            raise ValueError(f"Table {table_name} not found in loaded tables.")
        return self.tables[table_name]

    def get_table(self, table_name: str) -> pd.DataFrame:
        """Alias for get_table_dataframe."""
        return self.get_table_dataframe(table_name)

    def get_relationships(self) -> List[Tuple[str, str, str, str]]:
        """
        Get all foreign key relationships.

        Returns:
            List of (child_table, child_col, parent_table, parent_col) tuples.
        """
        return self.metadata.relationships

    def validate_key_consistency(self) -> None:
        """
        Validate that primary keys and foreign keys share the same actual pandas data type.

        Raises:
            TypeError: If there is a mismatch between a primary key and a referencing foreign key.
        """
        relationships = self.get_relationships()
        from loguru import logger
        
        # Group relationships by (parent_table, parent_col) to identifying FK groups
        fk_groups = {}
        for child_table, child_col, parent_table, parent_col in relationships:
            key = (parent_table, parent_col)
            if key not in fk_groups:
                fk_groups[key] = []
            fk_groups[key].append((child_table, child_col))
            
        for (parent_table, parent_col), children in fk_groups.items():
            # Gather available types
            types = []
            
            # Check Parent
            if parent_table in self.tables:
                parent_df = self.get_table_dataframe(parent_table)
                types.append({
                    'table': parent_table,
                    'col': parent_col,
                    'dtype': parent_df[parent_col].dtype,
                    'role': 'Primary Key'
                })
            else:
                 logger.debug(f"Parent table '{parent_table}' not found. Validating consistency among foreign keys only.")
            
            # Check Children
            for child_table, child_col in children:
                if child_table in self.tables:
                     child_df = self.get_table_dataframe(child_table)
                     types.append({
                        'table': child_table,
                        'col': child_col,
                        'dtype': child_df[child_col].dtype,
                        'role': 'Foreign Key'
                    })
            
            # Validate Consistency within the group
            if not types:
                continue
                
            first = types[0]
            for current in types[1:]:
                if current['dtype'] != first['dtype']:
                     raise TypeError(
                        f"Type mismatch in key group for {parent_table}.{parent_col}. "
                        f"{first['role']} '{first['table']}.{first['col']}' has type {first['dtype']}, "
                        f"but {current['role']} '{current['table']}.{current['col']}' has type {current['dtype']}."
                    )

    def save(self, path: Union[str, Path]):
        """
        Save the RDB to a directory.

        Args:
            path: Directory path to save the RDB.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save tables
        for table_name, df in self.tables.items():
            # We assume parquet format for now as it is the default
            output_path = path / f"{table_name}.parquet"
            df.to_parquet(output_path)

        # Save metadata
        yaml_utils.save_yaml(self.metadata.model_dump(mode='json'), path / "metadata.yaml")

    def update_tables(
        self, 
        tables: Dict[str, pd.DataFrame], 
        metadata: Dict[str, RDBTableSchema]
    ) -> 'RDB':
        """
        Create a new RDB with updated/added tables and metadata.
        
        This method performs a "Bulk Upsert":
        - Existing tables/metadata are preserved unless overwritten by the input.
        - New tables/metadata in the input are added.
        
        Args:
            tables: Dictionary of DataFrames to update/add.
            metadata: Dictionary of RDBTableSchema to update/add.
            
        Returns:
            New RDB instance with merged state.
        
        Raises:
            ValueError: If tables and metadata keys do not match, or if column definitions are inconsistent.
        """
        tables = tables or {}
        metadata = metadata or {}
        
        # Enforce strict consistency: Keys must match
        if set(tables.keys()) != set(metadata.keys()):
            raise ValueError(
                f"Tables and metadata keys must match exactly.\n"
                f"Tables: {list(tables.keys())}\n"
                f"Metadata: {list(metadata.keys())}"
            )
            
        # Enforce column consistency
        for name, schema in metadata.items():
            df = tables[name]
            # Check if all columns in dataframe have corresponding column metadata
            df_cols = set(df.columns)
            schema_cols = set(col.name for col in schema.columns)
            
            # Check if dataframe columns are missing from schema
            missing_in_schema = df_cols - schema_cols
            if missing_in_schema:
                raise ValueError(
                    f"Table '{name}': Columns {missing_in_schema} present in DataFrame but missing from metadata schema."
                )
        
        # Merge tables
        # Use existing tables as base, update with new ones
        merged_tables = self.tables.copy()
        merged_tables.update(tables)
        
        # Merge metadata
        # Create a map of existing schemas for easier updating
        existing_schemas = {t.name: t for t in self.metadata.tables}
        
        # Update/Add new schemas
        for name, schema in metadata.items():
            existing_schemas[name] = schema
            
        merged_metadata_obj = RDBMeta(
            name=self.metadata.name,
            tables=list(existing_schemas.values())
        )
        
        return RDB(metadata=merged_metadata_obj, tables=merged_tables)

    def update_table(
        self,
        name: str,
        dataframe: pd.DataFrame,
        schema: RDBTableSchema
    ) -> 'RDB':
        """
        Update or replace a single table.
        
        Args:
            name: Name of the table.
            dataframe: New DataFrame content.
            schema: New schema.
        """
        return self.update_tables(
            tables={name: dataframe},
            metadata={name: schema}
        )

    def add_table(
        self,
        dataframe: pd.DataFrame,
        name: str,
        time_column: Optional[str] = None,
        primary_key: Optional[str] = None,
        foreign_keys: Optional[List[Tuple[str, str, str]]] = None,
        column_types: Optional[Dict[str, str]] = None
    ) -> 'RDB':
        """
        Add a new table to the RDB.

        Args:
            dataframe: The pandas DataFrame to add.
            name: The name of the new table.
            time_column: The name of the time column, if any.
            primary_key: The name of the primary key column, if any.
            foreign_keys: List of (child_col, parent_table, parent_col) tuples.
            column_types: Dictionary mapping column names to RDBColumnDType strings
                          (e.g., 'float', 'datetime', 'category') to override inference.

        Returns:
            A new RDB instance with the added table.
        """
        if name in self.table_names:
            raise ValueError(f"Table '{name}' already exists in RDB. Use update_table() to modify existing tables.")

        foreign_keys = foreign_keys or []
        column_types = column_types or {}

        # Build FK map: col -> (parent_table, parent_col)
        fk_map = {}
        for child_col, parent_table, parent_col in foreign_keys:
             fk_map[child_col] = (parent_table, parent_col)

        columns = []
        for col_name in dataframe.columns:
            # Skip unnamed index columns often created by pandas
            if "Unnamed" in str(col_name):
                continue

            is_fk = col_name in fk_map
            hint = column_types.get(col_name)

            # Infer Type using utility
            dtype = infer_semantic_type(
                series=dataframe[col_name],
                col_name=col_name,
                pk_col=primary_key,
                time_col=time_column,
                is_foreign_key=is_fk,
                explicit_type_hint=hint
            )

            col_schema = RDBColumnSchema(name=col_name, dtype=dtype)

            # Manually set link_to for FKs
            if is_fk:
                parent_table, parent_col = fk_map[col_name]
                setattr(col_schema, 'link_to', f"{parent_table}.{parent_col}")

            columns.append(col_schema)

        new_table_schema = RDBTableSchema(
            name=name,
            source=f"{name}.parquet",
            format=RDBTableDataFormat.PARQUET,
            columns=columns,
            time_column=time_column
        )

        # Use updated API
        return self.update_tables(
            tables={name: dataframe}, 
            metadata={name: new_table_schema}
        )
    
    def canonicalize_key_types(self) -> 'RDB':
        """
        Convert all primary keys and foreign keys to string type to ensure consistency.
        
        Returns:
            New RDB with canonicalized key types.
        """
        
        new_tables = {}
        relationships = self.get_relationships()
        pks = {t.name: t.primary_key for t in self.metadata.tables if t.primary_key}
        
        # Identify all columns that need conversion per table
        # Table -> Set of columns
        cols_to_convert = {}
        
        # Add PKs
        for table, pk in pks.items():
            if table not in cols_to_convert:
                cols_to_convert[table] = set()
            cols_to_convert[table].add(pk)
            
        # Add FKs from relationships
        # (child_table, child_col, parent_table, parent_col)
        for child_table, child_col, _, _ in relationships:
            if child_table not in cols_to_convert:
                cols_to_convert[child_table] = set()
            cols_to_convert[child_table].add(child_col)
            
        # Perform conversion
        new_tables_metadata = {}
        for table_name in self.table_names:
            df = self.get_table_dataframe(table_name)
            
            if table_name not in cols_to_convert:
                continue
                
            new_df = df.copy(deep=False) # Shallow copy first, we will replace columns
            for col in cols_to_convert[table_name]:
                if col in new_df.columns:
                    # Use utility to safely convert
                    new_df[col] = safe_convert_to_string(new_df[col])
            
            new_tables[table_name] = new_df
            new_tables_metadata[table_name] = self.get_table_metadata(table_name)
        
        return self.update_tables(tables=new_tables, metadata=new_tables_metadata)

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



RDBDataset = RDB  # Alias for backward compatibility
