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
        name: Optional[str] = None
    ):
        """
        Initialize the SQLite adapter.

        Args:
            database_path: Path to the SQLite database file.
            tables: Optional list of table names to load.
            name: Optional name for the RDB. Defaults to the database filename.
        """
        db_path = Path(database_path)
        if not db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {db_path}")
        
        # Create SQLAlchemy engine for SQLite
        # sqlite:////absolute/path/to/db or sqlite:///relative/path/to/db
        engine = create_engine(f"sqlite:///{db_path.absolute()}")
        
        rdb_name = name or db_path.stem
        
        super().__init__(engine=engine, tables=tables, name=rdb_name)
