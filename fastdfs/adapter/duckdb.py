import pandas as pd
from typing import List, Optional, Union, Tuple, Dict
from pathlib import Path

class DuckDBAdapter:
    """
    Adapter for loading data from DuckDB databases.
    Uses the native duckdb python client for efficient data retrieval.
    """

    def __init__(
        self,
        database_path: Union[str, Path],
        tables: Optional[List[str]] = None,
        name: Optional[str] = None,
        primary_keys: Optional[Dict[str, str]] = None,
        foreign_keys: Optional[List[Tuple[str, str, str, str]]] = None,
        time_columns: Optional[Dict[str, str]] = None,
        type_hints: Optional[Dict[str, Dict[str, str]]] = None
    ):
        """
        Initialize the DuckDB adapter.

        Args:
            database_path: Path to the DuckDB database file.
            tables: Optional list of table names to load.
            name: Optional name for the RDB. Defaults to the database filename.
            primary_keys: Optional dictionary mapping table names to primary key column names.
            foreign_keys: Optional list of foreign key relationships.
            time_columns: Optional dictionary mapping table names to time column names.
            type_hints: Optional dictionary mapping table names to column type hints.
        """
        db_path = Path(database_path)
        # DuckDB can create a new file if it doesn't exist, but for an adapter 
        # we usually expect it to exist.
        if not db_path.exists() and str(database_path) != ":memory:":
            raise FileNotFoundError(f"DuckDB database not found: {db_path}")

        self.db_path = db_path
        self.tables_to_load = tables
        self.name = name or (db_path.stem if str(database_path) != ":memory:" else "duckdb_memory")
        self.user_primary_keys = primary_keys or {}
        self.user_foreign_keys = foreign_keys or []
        self.user_time_columns = time_columns or {}
        self.user_type_hints = type_hints or {}

    def _get_table_names(self) -> List[str]:
        """Get list of tables to load using native duckdb query."""
        if self.tables_to_load:
            return self.tables_to_load
        
        import duckdb
        con = duckdb.connect(str(self.db_path), read_only=True)
        tables = con.execute("PRAGMA show_tables").fetchall()
        con.close()
        return [t[0] for t in tables]

    def _map_duckdb_type(self, duckdb_type: str) -> str:
        """Map DuckDB type string to RDBColumnDType string."""
        duckdb_type = str(duckdb_type).upper()
        if any(t in duckdb_type for t in ["TIMESTAMP", "DATE", "TIME"]):
            return "datetime"
        elif any(t in duckdb_type for t in ["INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "REAL"]):
            return "float"
        elif "BOOL" in duckdb_type:
            return "category"
        else:
            return "text"

    def _discover_metadata(self, table_names: List[str]) -> Tuple[Dict[str, str], List[Tuple[str, str, str, str]], Dict[str, str], Dict[str, Dict[str, str]]]:
        """Override to use native DuckDB PRAGMAs for metadata discovery."""
        primary_keys = {}
        foreign_keys = []
        time_columns = {}
        type_hints = {}

        import duckdb
        con = duckdb.connect(str(self.db_path), read_only=True)
        
        for table_name in table_names:
            # Get columns and PKs
            # cid, name, type, notnull, dflt_value, pk
            try:
                cols_info = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
            except Exception as e:
                from loguru import logger
                logger.warning(f"Failed to get table info for {table_name}: {e}")
                continue

            table_hints = {}
            pk_col = None
            
            for col in cols_info:
                col_name = col[1]
                col_type = col[2]
                is_pk = col[5]
                
                if is_pk:
                    pk_col = col_name
                    primary_keys[table_name] = pk_col
                
                # Map type
                mapped_type = self._map_duckdb_type(col_type)
                if col_name != pk_col:
                    table_hints[col_name] = mapped_type
                
                if mapped_type == "datetime" and table_name not in time_columns:
                    time_columns[table_name] = col_name
            
            if table_hints:
                type_hints[table_name] = table_hints
                
        # Get FKs for all tables at once
        try:
            fks_info = con.execute("""
                SELECT table_name, constraint_column_names, referenced_table, referenced_column_names 
                FROM duckdb_constraints() 
                WHERE constraint_type = 'FOREIGN KEY'
            """).fetchall()
            for child_table, child_cols, parent_table, parent_cols in fks_info:
                if len(child_cols) == 1 and len(parent_cols) == 1:
                    child_col = child_cols[0]
                    parent_col = parent_cols[0]
                    
                    if (parent_table in table_names or self.tables_to_load is None) and (child_table in table_names):
                        foreign_keys.append((child_table, child_col, parent_table, parent_col))
                        if child_table in type_hints and child_col in type_hints[child_table]:
                            del type_hints[child_table][child_col]
        except Exception as e:
            from loguru import logger
            logger.warning(f"Failed to discover foreign keys in DuckDB: {e}")
        
        con.close()
        return primary_keys, foreign_keys, time_columns, type_hints

    def load(self) -> 'RDB': # type: ignore
        """
        Override load to use native duckdb client for faster data retrieval.
        """
        # We still use the base class to discover metadata
        table_names = self._get_table_names()
        
        from loguru import logger
        logger.info(f"Loading {len(table_names)} tables from DuckDB: {table_names}")

        tables = {}
        # Connect natively
        import duckdb
        con = duckdb.connect(str(self.db_path), read_only=True)
        
        for table_name in table_names:
            # 1. Load data natively (much faster than pd.read_sql)
            logger.debug(f"Fetching data for table: {table_name} (native)")
            df = con.execute(f"SELECT * FROM \"{table_name}\"").df()
            tables[table_name] = df
        
        con.close()

        # 2. Discover metadata using native DuckDB queries
        primary_keys, foreign_keys, time_columns, type_hints = self._discover_metadata(table_names)

        # Merge with user-provided hints (user hints take precedence)
        primary_keys.update(self.user_primary_keys)
        time_columns.update(self.user_time_columns)
        foreign_keys.extend(self.user_foreign_keys)
        
        for table, hints in self.user_type_hints.items():
            if table in type_hints:
                type_hints[table].update(hints)
            else:
                type_hints[table] = hints

        # Create RDB using the API
        from ..api import create_rdb
        rdb = create_rdb(
            name=self.name,
            tables=tables,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            time_columns=time_columns,
            type_hints=type_hints
        )

        return rdb
