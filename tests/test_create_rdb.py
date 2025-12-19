import pytest
import pandas as pd
import numpy as np
from fastdfs.api import create_rdb
from fastdfs.dataset.meta import RDBColumnDType, RDBMeta, RDBTableSchema, RDBTableDataFormat
from fastdfs.transform.infer_schema import InferSchemaTransform
from fastdfs.dataset.rdb import RDB

@pytest.fixture
def sample_data():
    users = pd.DataFrame({
        "user_id": [1, 2, 3],
        "age": [25, 30, 35],
        "signup_date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    })
    
    items = pd.DataFrame({
        "item_id": ["a", "b", "c"],
        "price": [10.5, 20.0, 15.75],
        "category": ["cat1", "cat2", "cat1"]
    })
    
    interactions = pd.DataFrame({
        "user_id": [1, 1, 2, 3],
        "item_id": ["a", "b", "a", "c"],
        "timestamp": pd.to_datetime(["2023-01-05", "2023-01-06", "2023-01-07", "2023-01-08"]),
        "rating": [5, 4, 3, 5]
    })
    
    return {
        "users": users,
        "items": items,
        "interactions": interactions
    }

def test_create_rdb_basic(sample_data):
    rdb = create_rdb(
        tables=sample_data,
        name="test_rdb",
        primary_keys={"users": "user_id", "items": "item_id"},
        foreign_keys=[
            ("interactions", "user_id", "users", "user_id"),
            ("interactions", "item_id", "items", "item_id")
        ],
        time_columns={"interactions": "timestamp"}
    )
    
    assert rdb.metadata.name == "test_rdb"
    assert set(rdb.table_names) == {"users", "items", "interactions"}
    
    # Check Users Schema
    users_meta = rdb.get_table_metadata("users")
    assert users_meta.column_dict["user_id"].dtype == RDBColumnDType.primary_key
    assert users_meta.column_dict["age"].dtype == RDBColumnDType.float_t
    assert users_meta.column_dict["signup_date"].dtype == RDBColumnDType.datetime_t
    
    # Check Interactions Schema
    interactions_meta = rdb.get_table_metadata("interactions")
    assert interactions_meta.column_dict["user_id"].dtype == RDBColumnDType.foreign_key
    assert interactions_meta.column_dict["user_id"].link_to == "users.user_id"
    assert interactions_meta.column_dict["timestamp"].dtype == RDBColumnDType.datetime_t
    assert interactions_meta.time_column == "timestamp"

def test_create_rdb_type_hints(sample_data):
    rdb = create_rdb(
        tables=sample_data,
        name="test_rdb_hints",
        type_hints={
            "users": {"age": "category"}
        }
    )
    
    users_meta = rdb.get_table_metadata("users")
    assert users_meta.column_dict["age"].dtype == RDBColumnDType.category_t

@pytest.fixture
def infer_schema_data():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "val": [1.1, 2.2, 3.3],
        "cat": ["a", "b", "a"],
        "ts": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    })
    
    tables = {"table1": df}
    
    # Create initial RDB with empty schema
    schema = RDBTableSchema(
        name="table1",
        source="table1.parquet",
        format=RDBTableDataFormat.PARQUET,
        columns=[]
    )
    metadata = RDBMeta(name="test", tables=[schema])
    return RDB(metadata=metadata, tables=tables)

def test_infer_basic(infer_schema_data):
    transform = InferSchemaTransform(
        primary_keys={"table1": "id"},
        time_columns={"table1": "ts"}
    )
    
    new_rdb = transform(infer_schema_data)
    meta = new_rdb.get_table_metadata("table1")
    
    assert meta.column_dict["id"].dtype == RDBColumnDType.primary_key
    assert meta.column_dict["val"].dtype == RDBColumnDType.float_t
    assert meta.column_dict["cat"].dtype == RDBColumnDType.category_t # Low cardinality
    assert meta.column_dict["ts"].dtype == RDBColumnDType.datetime_t
    assert meta.time_column == "ts"

def test_infer_no_pk_fk_warning(infer_schema_data):
    transform = InferSchemaTransform()
    # Just ensure it runs without error for now
    new_rdb = transform(infer_schema_data)
