# Dev Plan: Classical RDB Adapters for FastDFS

## 1. Goal
Implement a set of adapters in `fastdfs/adapter` to allow seamless loading of data from classical Relational Database Management Systems (RDBMS) into FastDFS `RDB` objects. Supported databases will include MySQL, PostgreSQL, SQLite, and DuckDB.

## 2. Architecture

### 2.1 Base SQL Adapter (`SQLAdapter`)
A base class will be implemented in `fastdfs/adapter/sql_base.py` to provide common functionality:
- **Connection Management**: Using SQLAlchemy engines for broad compatibility.
- **Schema Discovery**: Automatically extracting table names, column types, primary keys, and foreign key relationships using SQLAlchemy's `inspect` module.
- **Type Mapping**: Mapping SQL data types (e.g., `VARCHAR`, `INTEGER`, `TIMESTAMP`) to FastDFS `RDBColumnDType`.
- **Data Ingestion**: A standard `load()` method that iterates through tables and fetches data into pandas DataFrames.

### 2.2 Specific Adapters
Each database family will have a dedicated adapter to handle dialect-specific optimizations:
- **`SQLiteAdapter`**: Optimized for local file-based databases.
- **`PostgreSQLAdapter`**: Support for schemas and potentially faster ingestion using `psycopg3` or `adbc`.
- **`MySQLAdapter`**: Support for MySQL/MariaDB specific types.
- **`DuckDBAdapter`**: Optimized using the native `duckdb` python client for high-performance data retrieval.

## 3. Implementation Details

### 3.1 Schema Discovery
The `SQLAdapter` will use `sqlalchemy.inspect(engine)` to:
1. `get_table_names()`: Identify tables to load.
2. `get_columns(table_name)`: Get column names and types.
3. `get_pk_constraint(table_name)`: Identify primary keys.
4. `get_foreign_keys(table_name)`: Identify relationships (child_table, child_col, parent_table, parent_col).

### 3.2 Data Retrieval Logic
- **General**: `pd.read_sql_table(table_name, engine)` or `pd.read_sql_query(query, engine)`.
- **Optimized**:
    - **DuckDB**: `duckdb.connect(database).execute(f"SELECT * FROM {table}").df()`.
    - **PostgreSQL**: Explore `adbc_driver_postgresql` for zero-copy data transfer if available.

### 3.3 Proposed File Structure
```
fastdfs/adapter/
├── __init__.py
├── sql_base.py      # Base SQLAdapter class
├── sqlite.py        # SQLiteAdapter
├── postgres.py      # PostgreSQLAdapter
├── mysql.py         # MySQLAdapter
└── duckdb.py        # DuckDBAdapter
```

## 4. Proposed API Usage
```python
from fastdfs.adapter.sqlite import SQLiteAdapter

# Load from SQLite
adapter = SQLiteAdapter(database_path="data.db")
rdb = adapter.load()

# Load from PostgreSQL with specific tables
from fastdfs.adapter.postgres import PostgreSQLAdapter
adapter = PostgreSQLAdapter(
    connection_string="postgresql://user:pass@localhost/db",
    tables=["users", "transactions"]
)
rdb = adapter.load()
```

## 5. Testing Strategy
- Use `pytest` with a temporary SQLite database for core logic testing.
- Mock database connections for PostgreSQL and MySQL to test schema discovery without requiring live instances in the CI environment.
- Verify that foreign key relationships are correctly captured and mapped to the `RDB` object.

## 6. Timeline
1. **Phase 1**: Implement `SQLAdapter` base class and `SQLiteAdapter`.
2. **Phase 2**: Implement `PostgreSQLAdapter` and `MySQLAdapter`.
3. **Phase 3**: Implement `DuckDBAdapter` with native optimizations.
4. **Phase 4**: Integration tests and documentation updates.
