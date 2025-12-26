from typing import List, Optional
from sqlalchemy import create_engine
from .sql_base import SQLAdapter

class PostgreSQLAdapter(SQLAdapter):
    """Adapter for loading data from PostgreSQL databases."""

    def __init__(
        self,
        connection_string: str,
        tables: Optional[List[str]] = None,
        name: str = "postgres_rdb",
        schema: Optional[str] = None
    ):
        """
        Initialize the PostgreSQL adapter.

        Args:
            connection_string: SQLAlchemy connection string (e.g., 'postgresql://user:pass@host:port/db').
            tables: Optional list of table names to load.
            name: Name for the RDB.
            schema: Optional database schema to use.
        """
        engine = create_engine(connection_string)
        super().__init__(engine=engine, tables=tables, name=name)
        self.schema = schema

    def _get_table_names(self) -> List[str]:
        """Override to handle schemas in PostgreSQL."""
        all_tables = self.inspector.get_table_names(schema=self.schema)
        if self.tables_to_load:
            valid_tables = [t for t in self.tables_to_load if t in all_tables]
            return valid_tables
        return all_tables
