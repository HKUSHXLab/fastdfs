import pytest
import pandas as pd
import numpy as np
from fastdfs.dataset.meta import RDBColumnDType, RDBMeta, RDBTableSchema, RDBTableDataFormat, RDBColumnSchema
from fastdfs.transform.infer_schema import InferSchemaTransform
from fastdfs.dataset.rdb import RDB

class TestInferSchemaTransform:
    
    @pytest.fixture
    def basic_rdb(self):
        """Creates a basic RDB with one table and various column types."""
        # Create a dataframe with enough rows to test cardinality thresholds (10)
        df = pd.DataFrame({
            "id": range(15),
            "float_col": [1.1 * i for i in range(15)],
            "bool_col": [True if i % 2 == 0 else False for i in range(15)],
            "date_col": pd.date_range(start="2023-01-01", periods=15),
            "cat_col": ["a", "b", "c"] * 5, # 3 unique values (< 10)
            "text_col": [f"text_{i}" for i in range(15)], # 15 unique values (>= 10)
            "int_as_str": [str(i) for i in range(15)]
        })
        
        tables = {"main_table": df}
        
        # Create initial RDB with empty schema
        schema = RDBTableSchema(
            name="main_table",
            source="main_table.parquet",
            format=RDBTableDataFormat.PARQUET,
            columns=[]
        )
        metadata = RDBMeta(name="test_rdb", tables=[schema])
        return RDB(metadata=metadata, tables=tables)

    def test_infer_basic_types(self, basic_rdb):
        """Test inference of basic pandas types."""
        transform = InferSchemaTransform(
            primary_keys={"main_table": "id"}
        )
        
        new_rdb = transform(basic_rdb)
        meta = new_rdb.get_table_metadata("main_table")
        cols = meta.column_dict
        
        assert cols["id"].dtype == RDBColumnDType.primary_key
        assert cols["float_col"].dtype == RDBColumnDType.float_t
        
        # Integers and Bools are mapped to float_t in current implementation
        assert cols["bool_col"].dtype == RDBColumnDType.float_t 
        
        assert cols["date_col"].dtype == RDBColumnDType.datetime_t
        
        # Low cardinality (< 10) -> category
        assert cols["cat_col"].dtype == RDBColumnDType.category_t
        
        # High cardinality (>= 10) -> text
        assert cols["text_col"].dtype == RDBColumnDType.text_t

    def test_infer_type_hints(self, basic_rdb):
        """Test that type hints override inference."""
        transform = InferSchemaTransform(
            type_hints={
                "main_table": {
                    "float_col": "text", # Force float to text
                    "int_as_str": "float" # Force string to float
                }
            }
        )
        
        new_rdb = transform(basic_rdb)
        meta = new_rdb.get_table_metadata("main_table")
        cols = meta.column_dict
        
        assert cols["float_col"].dtype == RDBColumnDType.text_t
        assert cols["int_as_str"].dtype == RDBColumnDType.float_t

    def test_infer_relationships(self):
        """Test inference of PK and FK relationships."""
        users = pd.DataFrame({"uid": [1, 2], "name": ["a", "b"]})
        logs = pd.DataFrame({"lid": [10, 11], "uid": [1, 2], "val": [0.1, 0.2]})
        
        tables = {"users": users, "logs": logs}
        
        # Setup RDB
        schemas = [
            RDBTableSchema(name="users", source="u.p", format=RDBTableDataFormat.PARQUET, columns=[]),
            RDBTableSchema(name="logs", source="l.p", format=RDBTableDataFormat.PARQUET, columns=[])
        ]
        rdb = RDB(metadata=RDBMeta(name="rel_test", tables=schemas), tables=tables)
        
        transform = InferSchemaTransform(
            primary_keys={"users": "uid", "logs": "lid"},
            foreign_keys=[("logs", "uid", "users", "uid")]
        )
        
        new_rdb = transform(rdb)
        
        users_meta = new_rdb.get_table_metadata("users")
        logs_meta = new_rdb.get_table_metadata("logs")
        
        assert users_meta.column_dict["uid"].dtype == RDBColumnDType.primary_key
        assert logs_meta.column_dict["lid"].dtype == RDBColumnDType.primary_key
        
        assert logs_meta.column_dict["uid"].dtype == RDBColumnDType.foreign_key
        assert logs_meta.column_dict["uid"].link_to == "users.uid"

    def test_infer_time_index(self, basic_rdb):
        """Test inference of time index."""
        transform = InferSchemaTransform(
            time_columns={"main_table": "date_col"}
        )
        
        new_rdb = transform(basic_rdb)
        meta = new_rdb.get_table_metadata("main_table")
        
        assert meta.time_column == "date_col"
        assert meta.column_dict["date_col"].dtype == RDBColumnDType.datetime_t

    def test_empty_dataframe(self):
        """Test handling of empty dataframes."""
        df = pd.DataFrame({"a": [], "b": []})
        tables = {"empty": df}
        
        schema = RDBTableSchema(name="empty", source="e.p", format=RDBTableDataFormat.PARQUET, columns=[])
        rdb = RDB(metadata=RDBMeta(name="empty_test", tables=[schema]), tables=tables)
        
        transform = InferSchemaTransform()
        new_rdb = transform(rdb)
        
        meta = new_rdb.get_table_metadata("empty")
        # Empty dataframe columns are usually float type (all NaNs) in pandas if not specified
        assert meta.column_dict["a"].dtype == RDBColumnDType.float_t
        assert meta.column_dict["b"].dtype == RDBColumnDType.float_t

    def test_all_nan_column(self):
        """Test handling of columns with all NaNs."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "nans": [np.nan, np.nan, np.nan]
        })
        tables = {"nan_table": df}
        
        schema = RDBTableSchema(name="nan_table", source="n.p", format=RDBTableDataFormat.PARQUET, columns=[])
        rdb = RDB(metadata=RDBMeta(name="nan_test", tables=[schema]), tables=tables)
        
        transform = InferSchemaTransform(primary_keys={"nan_table": "id"})
        new_rdb = transform(rdb)
        
        meta = new_rdb.get_table_metadata("nan_table")
        # All NaNs are float in pandas
        assert meta.column_dict["nans"].dtype == RDBColumnDType.float_t

    def test_custom_category_threshold(self):
        """Test custom category threshold."""
        # Create data with 5 unique values
        df = pd.DataFrame({
            "id": range(10),
            "col": ["a", "b", "c", "d", "e"] * 2
        })
        tables = {"t": df}
        schema = RDBTableSchema(name="t", source="t.p", format=RDBTableDataFormat.PARQUET, columns=[])
        rdb = RDB(metadata=RDBMeta(name="test", tables=[schema]), tables=tables)
        
        # Threshold 3 -> Should be text (5 > 3)
        transform_low = InferSchemaTransform(category_threshold=3)
        rdb_low = transform_low(rdb)
        assert rdb_low.get_table_metadata("t").column_dict["col"].dtype == RDBColumnDType.text_t
        
        # Threshold 6 -> Should be category (5 < 6)
        transform_high = InferSchemaTransform(category_threshold=6)
        rdb_high = transform_high(rdb)
        assert rdb_high.get_table_metadata("t").column_dict["col"].dtype == RDBColumnDType.category_t
