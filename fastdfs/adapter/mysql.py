from typing import List, Optional
from sqlalchemy import create_engine
from .sql_base import SQLAdapter

class MySQLAdapter(SQLAdapter):
    """Adapter for loading data from MySQL/MariaDB databases."""

    def __init__(
        self,
        connection_string: str,
        tables: Optional[List[str]] = None,
        name: str = "mysql_rdb"
    ):
        """
        Initialize the MySQL adapter.

        Args:
            connection_string: SQLAlchemy connection string (e.g., 'mysql+pymysql://user:pass@host:port/db').
            tables: Optional list of table names to load.
            name: Name for the RDB.
        """
        engine = create_engine(connection_string)
        super().__init__(engine=engine, tables=tables, name=name)
