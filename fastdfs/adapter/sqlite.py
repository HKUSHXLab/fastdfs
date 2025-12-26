from pathlib import Path
from typing import List, Optional, Union
from sqlalchemy import create_engine
from .sql_base import SQLAdapter

class SQLiteAdapter(SQLAdapter):
    """Adapter for loading data from SQLite databases."""

    def __init__(
        self,
        database_path: Union[str, Path],
        tables: Optional[List[str]] = None,
        name: Optional[str] = None,
        primary_keys: Optional[dict] = None,
        foreign_keys: Optional[list] = None,
        time_columns: Optional[dict] = None,
        type_hints: Optional[dict] = None
    ):
        """
        Initialize the SQLite adapter.

        Args:
            database_path: Path to the SQLite database file.
            tables: Optional list of table names to load.
            name: Optional name for the RDB. Defaults to the database filename.
            primary_keys: Optional dictionary mapping table names to primary key column names.
            foreign_keys: Optional list of foreign key relationships.
            time_columns: Optional dictionary mapping table names to time column names.
            type_hints: Optional dictionary mapping table names to column type hints.
        """
        db_path = Path(database_path)
        if not db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {db_path}")
        
        # Create SQLAlchemy engine for SQLite
        # sqlite:////absolute/path/to/db or sqlite:///relative/path/to/db
        engine = create_engine(f"sqlite:///{db_path.absolute()}")
        
        rdb_name = name or db_path.stem
        
        super().__init__(
            engine=engine, 
            tables=tables, 
            name=rdb_name,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            time_columns=time_columns,
            type_hints=type_hints
        )
