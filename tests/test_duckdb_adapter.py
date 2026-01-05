
from loguru import logger
logger.enable("fastdfs")
import pytest
import pandas as pd
from pathlib import Path
from fastdfs.adapter.duckdb import DuckDBAdapter
from fastdfs.dataset.rdb import RDB

def test_duckdb_adapter(tmp_path):
    import duckdb
    # 1. Create a dummy DuckDB database
    db_path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    
    # Create tables
    users_df = pd.DataFrame({
        "user_id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35]
    })
    conn.execute("CREATE TABLE users (user_id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    conn.execute("INSERT INTO users SELECT * FROM users_df")
    
    orders_df = pd.DataFrame({
        "order_id": [101, 102, 103],
        "user_id": [1, 1, 2],
        "amount": [10.5, 20.0, 15.0],
        "order_date": ["2024-01-01", "2024-01-02", "2024-01-03"]
    })
    conn.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY, 
            user_id INTEGER, 
            amount REAL, 
            order_date TEXT,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
    """)
    conn.execute("INSERT INTO orders SELECT * FROM orders_df")
    
    conn.close()
    
    # 2. Use the adapter
    adapter = DuckDBAdapter(database_path=db_path)
    rdb = adapter.load()
    
    # 3. Verify
    assert isinstance(rdb, RDB)
    assert set(rdb.table_names) == {"users", "orders"}
    
    # Check primary keys
    assert rdb.get_table_metadata("users").primary_key == "user_id"
    assert rdb.get_table_metadata("orders").primary_key == "order_id"
    
    # Check relationships
    relationships = rdb.get_relationships()
    assert len(relationships) == 1
    assert relationships[0] == ("orders", "user_id", "users", "user_id")
    
    # Check data
    assert len(rdb.get_table("users")) == 3
    assert len(rdb.get_table("orders")) == 3
    assert "amount" in rdb.get_table("orders").columns
