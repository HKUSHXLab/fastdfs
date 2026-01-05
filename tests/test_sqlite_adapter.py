
from loguru import logger
logger.enable("fastdfs")
import pytest
import pandas as pd
import sqlite3
from pathlib import Path
from fastdfs.adapter.sqlite import SQLiteAdapter
from fastdfs.dataset.rdb import RDB

def test_sqlite_adapter(tmp_path):
    # 1. Create a dummy SQLite database
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    
    # Create tables
    users_df = pd.DataFrame({
        "user_id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35]
    })
    users_df.to_sql("users", conn, index=False)
    
    orders_df = pd.DataFrame({
        "order_id": [101, 102, 103],
        "user_id": [1, 1, 2],
        "amount": [10.5, 20.0, 15.0],
        "order_date": ["2024-01-01", "2024-01-02", "2024-01-03"]
    })
    orders_df.to_sql("orders", conn, index=False)
    
    # Add primary keys and foreign keys (SQLite is a bit limited in to_sql for this, 
    # but we can use SQL to add them or just rely on the adapter's discovery)
    # Actually, SQLAlchemy's inspector can find PKs if they are defined.
    # to_sql doesn't define PKs by default.
    
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE users_new (user_id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    cursor.execute("INSERT INTO users_new SELECT * FROM users")
    cursor.execute("DROP TABLE users")
    cursor.execute("ALTER TABLE users_new RENAME TO users")
    
    cursor.execute("""
        CREATE TABLE orders_new (
            order_id INTEGER PRIMARY KEY, 
            user_id INTEGER, 
            amount REAL, 
            order_date TEXT,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
    """)
    cursor.execute("INSERT INTO orders_new SELECT * FROM orders")
    cursor.execute("DROP TABLE orders")
    cursor.execute("ALTER TABLE orders_new RENAME TO orders")
    
    conn.commit()
    conn.close()
    
    # 2. Use the adapter
    adapter = SQLiteAdapter(database_path=db_path)
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
