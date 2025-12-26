import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, inspect
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
from loguru import logger

from ..dataset.rdb import RDB
from ..dataset.meta import RDBColumnDType
from ..api import create_rdb

class SQLAdapter:
    """Base adapter for loading data from SQL databases into FastDFS RDB."""

    def __init__(
        self,
        engine: sqlalchemy.engine.Engine,
        tables: Optional[List[str]] = None,
        name: str = "sql_rdb",
        primary_keys: Optional[Dict[str, str]] = None,
        foreign_keys: Optional[List[Tuple[str, str, str, str]]] = None,
        time_columns: Optional[Dict[str, str]] = None,
        type_hints: Optional[Dict[str, Dict[str, str]]] = None
    ):
        """
        Initialize the SQL adapter.

        Args:
            engine: SQLAlchemy engine.
            tables: Optional list of table names to load. If None, all tables will be loaded.
            name: Name of the RDB.
            primary_keys: Optional dictionary mapping table names to primary key column names.
            foreign_keys: Optional list of foreign key relationships (child_table, child_col, parent_table, parent_col).
            time_columns: Optional dictionary mapping table names to time column names.
            type_hints: Optional dictionary mapping table names to column type hints.
        """
        self.engine = engine
        self.tables_to_load = tables
        self.name = name
        self.inspector = inspect(self.engine)
        self.user_primary_keys = primary_keys or {}
        self.user_foreign_keys = foreign_keys or []
        self.user_time_columns = time_columns or {}
        self.user_type_hints = type_hints or {}

    def _get_table_names(self) -> List[str]:
        """Get list of tables to load."""
        all_tables = self.inspector.get_table_names()
        if self.tables_to_load:
            # Filter and validate
            valid_tables = [t for t in self.tables_to_load if t in all_tables]
            missing = set(self.tables_to_load) - set(valid_tables)
            if missing:
                logger.warning(f"Tables not found in database: {missing}")
            return valid_tables
        return all_tables

    def _map_sql_type(self, sql_type: Any) -> str:
        """Map SQLAlchemy type to RDBColumnDType string."""
        # This is a basic mapping, can be overridden by subclasses
        from sqlalchemy import types
        
        if isinstance(sql_type, (types.DateTime, types.Date, types.TIMESTAMP)):
            return "datetime"
        elif isinstance(sql_type, (types.Float, types.Numeric, types.Integer)):
            # We use float for all numbers to handle NaNs in pandas easily
            return "float"
        elif isinstance(sql_type, (types.Boolean)):
            return "category"
        elif isinstance(sql_type, (types.String, types.Text)):
            return "text"
        else:
            return "text" # Default to text

    def _discover_metadata(self, table_names: List[str]) -> Tuple[Dict[str, str], List[Tuple[str, str, str, str]], Dict[str, str], Dict[str, Dict[str, str]]]:
        """
        Discover primary keys, foreign keys, time columns, and type hints for the given tables.
        """
        primary_keys = {}
        foreign_keys = []
        time_columns = {}
        type_hints = {}

        for table_name in table_names:
            columns = self.inspector.get_columns(table_name)
            table_hints = {}
            
            # Discover Primary Key
            pk_constraint = self.inspector.get_pk_constraint(table_name)
            pk_col = None
            if pk_constraint and pk_constraint.get('constrained_columns'):
                pk_col = pk_constraint['constrained_columns'][0]
                primary_keys[table_name] = pk_col

            for col in columns:
                col_name = col['name']
                
                # Only add to type_hints if it's NOT the primary key
                if col_name != pk_col:
                    table_hints[col_name] = self._map_sql_type(col['type'])
                
                # Try to identify time columns (heuristic)
                if self._map_sql_type(col['type']) == "datetime" and table_name not in time_columns:
                    time_columns[table_name] = col_name
            
            if table_hints:
                type_hints[table_name] = table_hints

            # Discover Foreign Keys
            fks = self.inspector.get_foreign_keys(table_name)
            for fk in fks:
                if len(fk['constrained_columns']) == 1 and len(fk['referred_columns']) == 1:
                    child_col = fk['constrained_columns'][0]
                    parent_table = fk['referred_table']
                    parent_col = fk['referred_columns'][0]
                    
                    if parent_table in table_names or self.tables_to_load is None:
                        foreign_keys.append((table_name, child_col, parent_table, parent_col))
                        # Remove from type_hints if it was added
                        if table_name in type_hints and child_col in type_hints[table_name]:
                            del type_hints[table_name][child_col]
        
        return primary_keys, foreign_keys, time_columns, type_hints

    def load(self) -> RDB:
        """
        Load data from SQL and create an RDB object.

        Returns:
            RDB: The loaded RDB object.
        """
        table_names = self._get_table_names()
        logger.info(f"Loading {len(table_names)} tables from SQL: {table_names}")

        tables = {}
        for table_name in table_names:
            logger.debug(f"Fetching data for table: {table_name}")
            df = pd.read_sql_table(table_name, self.engine)
            tables[table_name] = df

        primary_keys, foreign_keys, time_columns, type_hints = self._discover_metadata(table_names)

        # Merge with user-provided hints (user hints take precedence)
        primary_keys.update(self.user_primary_keys)
        time_columns.update(self.user_time_columns)
        
        # For foreign keys, we append user-provided ones
        # (In the future we might want to avoid duplicates)
        foreign_keys.extend(self.user_foreign_keys)
        
        # For type hints, merge nested dictionaries
        for table, hints in self.user_type_hints.items():
            if table in type_hints:
                type_hints[table].update(hints)
            else:
                type_hints[table] = hints

        # Create RDB using the API
        rdb = create_rdb(
            name=self.name,
            tables=tables,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            time_columns=time_columns,
            type_hints=type_hints
        )

        return rdb
