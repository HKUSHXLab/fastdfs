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
        schema: Optional[str] = None,
        primary_keys: Optional[dict] = None,
        foreign_keys: Optional[list] = None,
        time_columns: Optional[dict] = None,
        type_hints: Optional[dict] = None
    ):
        """
        Initialize the PostgreSQL adapter.

        Args:
            connection_string: SQLAlchemy connection string (e.g., 'postgresql://user:pass@host:port/db').
            tables: Optional list of table names to load.
            name: Name for the RDB.
            schema: Optional database schema to use.
            primary_keys: Optional dictionary mapping table names to primary key column names.
            foreign_keys: Optional list of foreign key relationships.
            time_columns: Optional dictionary mapping table names to time column names.
            type_hints: Optional dictionary mapping table names to column type hints.
        """
        engine = create_engine(connection_string)
        super().__init__(
            engine=engine, 
            tables=tables, 
            name=name,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            time_columns=time_columns,
            type_hints=type_hints
        )
        self.schema = schema

    def _get_table_names(self) -> List[str]:
        """Override to handle schemas in PostgreSQL."""
        all_tables = self.inspector.get_table_names(schema=self.schema)
        if self.tables_to_load:
            valid_tables = [t for t in self.tables_to_load if t in all_tables]
            return valid_tables
        return all_tables
