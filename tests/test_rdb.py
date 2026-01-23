"""
Tests for the RDB dataset implementation.

This file includes tests for RDB construction, access, mutation, key consistency, and type canonicalization.
"""

from loguru import logger
logger.enable("fastdfs")

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path

from fastdfs.dataset.rdb import RDB
from fastdfs.dataset.meta import RDBColumnDType, RDBTableDataFormat, RDBTableSchema, RDBColumnSchema, RDBMeta

# -----------------------------------------------------------------------------
# 1. RDB Construction & Loading Tests
# -----------------------------------------------------------------------------

class TestRDBConstruction:
    """Test using existing dataset creation and edge cases."""
    
    @pytest.fixture
    def test_data_path(self):
        """Path to existing test dataset."""
        return Path(__file__).parent / "data" / "test_rdb_new"
    
    @pytest.fixture 
    def rdb_dataset(self, test_data_path):
        """Create RDB dataset from existing test data."""
        return RDB(test_data_path)
    
    def test_load_rdb_dataset(self, rdb_dataset):
        """Test that RDB dataset loads correctly."""
        assert rdb_dataset.metadata.name == "sbm_user_item"
        assert len(rdb_dataset.table_names) == 3
        assert set(rdb_dataset.table_names) == {"user", "item", "interaction"}
    
    def test_missing_metadata_file(self):
        """Test error when metadata file is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_path = Path(temp_dir) / "empty"
            empty_path.mkdir()
            
            with pytest.raises(FileNotFoundError, match="Metadata file not found"):
                RDB(empty_path)
    
    def test_missing_data_file(self):
        """Test error when table data file is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "incomplete"
            dataset_path.mkdir()
            
            # Create metadata with reference to non-existent data file
            metadata = {
                "name": "test",
                "tables": [
                    {
                        "name": "test_table",
                        "source": "nonexistent.npz",
                        "format": "numpy",
                        "columns": [
                            {"name": "id", "dtype": "primary_key"}
                        ]
                    }
                ]
            }
            
            import yaml
            with open(dataset_path / "metadata.yaml", "w") as f:
                yaml.dump(metadata, f)
            
            with pytest.raises(FileNotFoundError, match="Table data file not found"):
                RDB(dataset_path)

    def test_missing_column_in_data(self, test_data_path):
        """Test error when metadata references column not in data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "bad_metadata"
            dataset_path.mkdir()
            
            # Copy data files
            data_dir = dataset_path / "data"
            data_dir.mkdir()
            shutil.copy2(test_data_path / "data" / "user.npz", data_dir / "user.npz")
            
            # Create metadata with reference to non-existent column
            metadata = {
                "name": "test",
                "tables": [
                    {
                        "name": "user",
                        "source": "data/user.npz", 
                        "format": "numpy",
                        "columns": [
                            {"name": "user_id", "dtype": "primary_key"},
                            {"name": "nonexistent_column", "dtype": "float"}
                        ]
                    }
                ]
            }
            
            import yaml
            with open(dataset_path / "metadata.yaml", "w") as f:
                yaml.dump(metadata, f)
            
            with pytest.raises(ValueError, match="Column nonexistent_column not found"):
                RDB(dataset_path)


# -----------------------------------------------------------------------------
# 2. RDB Access Tests
# -----------------------------------------------------------------------------

class TestRDBAccess:
    @pytest.fixture
    def test_data_path(self):
        return Path(__file__).parent / "data" / "test_rdb_new"
    
    @pytest.fixture 
    def rdb_dataset(self, test_data_path):
        return RDB(test_data_path)

    def test_get_table(self, rdb_dataset):
        """Test getting individual tables as DataFrames."""
        # Test user table
        user_df = rdb_dataset.get_table("user")
        assert isinstance(user_df, pd.DataFrame)
        assert "user_id" in user_df.columns
        assert "user_feature_0" in user_df.columns
        assert len(user_df) > 0
        
        # Test item table
        item_df = rdb_dataset.get_table("item")
        assert isinstance(item_df, pd.DataFrame)
        
        # Test interaction table
        interaction_df = rdb_dataset.get_table("interaction")
        assert len(interaction_df) > 0
    
    def test_get_table_metadata(self, rdb_dataset):
        """Test getting table metadata."""
        user_meta = rdb_dataset.get_table_metadata("user")
        assert user_meta.name == "user"
        assert user_meta.source == "data/user.npz"
        assert user_meta.format == RDBTableDataFormat.NUMPY
        assert len(user_meta.columns) == 2
        
        # Check column metadata
        user_id_col = None
        for col in user_meta.columns:
            if col.name == "user_id":
                user_id_col = col
                break
        
        assert user_id_col is not None
        assert user_id_col.dtype == RDBColumnDType.primary_key
    
    def test_get_relationships(self, rdb_dataset):
        """Test extracting relationships from foreign keys."""
        relationships = rdb_dataset.get_relationships()
        assert len(relationships) == 2
        expected_relationships = {
            ("interaction", "user_id", "user", "user_id"),
            ("interaction", "item_id", "item", "item_id")
        }
        assert set(relationships) == expected_relationships
    
    def test_table_not_found_error(self, rdb_dataset):
        """Test error handling for non-existent tables."""
        with pytest.raises(ValueError, match="Table nonexistent not found"):
            rdb_dataset.get_table("nonexistent")
        
        with pytest.raises(ValueError, match="Table nonexistent not found"):
            rdb_dataset.get_table_metadata("nonexistent")
    
    def test_sqlalchemy_metadata(self, rdb_dataset):
        """Test SQLAlchemy metadata generation."""
        metadata = rdb_dataset.sqlalchemy_metadata
        table_names = set(metadata.tables.keys())
        assert table_names == {"user", "item", "interaction"}


# -----------------------------------------------------------------------------
# 3. RDB Mutation Tests (Update/Add)
# -----------------------------------------------------------------------------

class TestRDBMutation:
    @pytest.fixture
    def test_data_path(self):
        return Path(__file__).parent / "data" / "test_rdb_new"
    
    @pytest.fixture 
    def rdb_dataset(self, test_data_path):
        return RDB(test_data_path)
    
    @pytest.fixture
    def rdb_simple(self):
        """Create a simple in-memory RDB."""
        user_df = pd.DataFrame({"user_id": [1, 2], "age": [25, 30]})
        user_schema = RDBTableSchema(
            name="user",
            source="user.parquet",
            format=RDBTableDataFormat.PARQUET,
            columns=[
                RDBColumnSchema(name="user_id", dtype=RDBColumnDType.primary_key),
                RDBColumnSchema(name="age", dtype=RDBColumnDType.float_t)
            ]
        )
        metadata = RDBMeta(name="test_rdb", tables=[user_schema])
        tables = {"user": user_df}
        return RDB(metadata=metadata, tables=tables)

    def test_update_tables_basic(self, rdb_dataset):
        """Test creating new dataset with modified tables using update_tables."""
        original_user_df = rdb_dataset.get_table("user")
        
        modified_user_df = original_user_df.copy()
        modified_user_df["new_feature"] = 42.0
        
        new_tables = {
            "user": modified_user_df,
            # Partial update check
            "item": rdb_dataset.get_table("item"),
            "interaction": rdb_dataset.get_table("interaction")
        }

        # Need to provide metadata for strict API
        # Retrieve existing metadata for these tables
        new_metadata = {
            "user": rdb_dataset.get_table_metadata("user"),
            "item": rdb_dataset.get_table_metadata("item"),
            "interaction": rdb_dataset.get_table_metadata("interaction")
        }
        
        # Modify user metadata to include new column
        from fastdfs.dataset.meta import RDBColumnDType
        user_cols = new_metadata["user"].columns
        user_cols.append(RDBColumnSchema(name="new_feature", dtype=RDBColumnDType.float_t))
        
        # Explicit arguments required now and must match keys
        new_dataset = rdb_dataset.update_tables(tables=new_tables, metadata=new_metadata)
        
        new_user_df = new_dataset.get_table("user")
        assert "new_feature" in new_user_df.columns
        assert (new_user_df["new_feature"] == 42.0).all()
        
        original_user_df_after = rdb_dataset.get_table("user")
        assert "new_feature" not in original_user_df_after.columns

    def test_update_tables_strictness(self, rdb_dataset):
        """Test strict validation of update_tables."""
        user_df = rdb_dataset.get_table("user")
        user_meta = rdb_dataset.get_table_metadata("user")
        
        # 1. Test Key Mismatch (Missing Metadata)
        with pytest.raises(ValueError, match="keys must match"):
            rdb_dataset.update_tables(tables={"user": user_df}, metadata={})
            
        # 2. Test Key Mismatch (Missing Table)
        with pytest.raises(ValueError, match="keys must match"):
            rdb_dataset.update_tables(tables={}, metadata={"user": user_meta})
            
        # 3. Test Column Consistency (Column in DF but not in Metadata)
        bad_df = user_df.copy()
        bad_df["ghost_col"] = 1
        with pytest.raises(ValueError, match="Columns .* present in DataFrame but missing"):
            rdb_dataset.update_tables(tables={"user": bad_df}, metadata={"user": user_meta})

    def test_update_table_single(self):
        # Setup simple rdb
        df = pd.DataFrame({'col': [1]})
        col_schema = RDBColumnSchema(name="col", dtype=RDBColumnDType.text_t)
        meta = RDBMeta(name="test", tables=[RDBTableSchema(name="t1", source="s", format="parquet", columns=[col_schema])])
        rdb = RDB(metadata=meta, tables={"t1": df})
        
        # Update table
        new_df = pd.DataFrame({'col': [2]})
        # Schema is now required
        t1_schema = rdb.get_table_metadata("t1")
        new_rdb = rdb.update_table("t1", new_df, t1_schema)
        
        assert new_rdb.get_table("t1")['col'].iloc[0] == 2
        # Ensure immutability
        assert rdb.get_table("t1")['col'].iloc[0] == 1
    
    def test_add_table_basic(self, rdb_simple):
        """Test adding a simple table with type inference."""
        new_df = pd.DataFrame({
            "id": [1, 2],
            "value": [10.5, 20.0],
            "is_valid": [True, False],
            "cat": ["A", "B"]
        })
        
        new_rdb = rdb_simple.add_table(new_df, "new_table")
        
        assert "new_table" in new_rdb.table_names
        assert "user" in new_rdb.table_names
        
        # Verify schema inference
        meta = new_rdb.get_table_metadata("new_table")
        col_types = {col.name: col.dtype for col in meta.columns}
        
        assert col_types["id"] == RDBColumnDType.float_t
        assert col_types["cat"] == RDBColumnDType.category_t

    def test_add_table_with_keys_and_time(self, rdb_simple):
        """Test adding table with PK, FK, Time and manual types."""
        history_df = pd.DataFrame({
            "hist_id": [101, 102],
            "user_id": [1, 2],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "status": ["active", "inactive"]
        })
        
        new_rdb = rdb_simple.add_table(
            history_df,
            "history",
            time_column="timestamp",
            primary_key="hist_id",
            foreign_keys=[("user_id", "user", "user_id")],
            column_types={"status": "category"}
        )
        
        meta = new_rdb.get_table_metadata("history")
        assert meta.time_column == "timestamp"
        rels = new_rdb.get_relationships()
        assert ("history", "user_id", "user", "user_id") in rels

    def test_add_table_duplicate_name(self, rdb_simple):
        """Test error when adding duplicate table name."""
        with pytest.raises(ValueError, match="Table 'user' already exists"):
            rdb_simple.add_table(pd.DataFrame(), "user")
            
    def test_add_table_invalid_fk_parent(self, rdb_simple):
        """Test that adding table with invalid FK parent does NOT raise error."""
        df = pd.DataFrame({"uid": [1]})
        new_rdb = rdb_simple.add_table(df, "test", foreign_keys=[("uid", "nonexistent", "id")])
        assert "test" in new_rdb.table_names


# -----------------------------------------------------------------------------
# 4. Key Consistency Tests
# -----------------------------------------------------------------------------

class TestRDBKeyConsistency:
    def test_validate_key_consistency_relaxed(self):
        """
        Test that validation SKIPS missing parents but checks consistency 
        among children referring to the same parent.
        """
        # Case 1: Parent Missing, Child A (int), Child B (int). Should Pass.
        meta = RDBMeta(
           name="test",
           tables=[
               RDBTableSchema(name="P", source="p", format="parquet", columns=[
                   RDBColumnSchema(name="pid", dtype=RDBColumnDType.primary_key)
               ]),
               RDBTableSchema(name="C1", source="c1", format="parquet", columns=[
                   RDBColumnSchema(name="cid", dtype=RDBColumnDType.primary_key),
                   RDBColumnSchema(name="fk", dtype=RDBColumnDType.foreign_key, link_to="P.pid")
               ]),
               RDBTableSchema(name="C2", source="c2", format="parquet", columns=[
                   RDBColumnSchema(name="cid", dtype=RDBColumnDType.primary_key),
                   RDBColumnSchema(name="fk", dtype=RDBColumnDType.foreign_key, link_to="P.pid")
               ])
           ]
        )
        
        df_c1 = pd.DataFrame({'cid': [1], 'fk': [10]}) # Int
        df_c2 = pd.DataFrame({'cid': [2], 'fk': [20]}) # Int
        
        rdb = RDB(metadata=meta, tables={"C1": df_c1, "C2": df_c2}) # P missing
        
        # Should pass (Consistent Ints among children)
        rdb.validate_key_consistency()
        
    def test_validate_key_consistency_across_children_fail(self):
        """Case 2: Parent Missing, Child A (Int), Child B (Object). Should Fail."""
        meta = RDBMeta(
           name="test",
           tables=[
               RDBTableSchema(name="P", source="p", format="parquet", columns=[
                   RDBColumnSchema(name="pid", dtype=RDBColumnDType.primary_key)
               ]),
               RDBTableSchema(name="C1", source="c1", format="parquet", columns=[
                   RDBColumnSchema(name="cid", dtype=RDBColumnDType.primary_key),
                   RDBColumnSchema(name="fk", dtype=RDBColumnDType.foreign_key, link_to="P.pid")
               ]),
               RDBTableSchema(name="C2", source="c2", format="parquet", columns=[
                   RDBColumnSchema(name="cid", dtype=RDBColumnDType.primary_key),
                   RDBColumnSchema(name="fk", dtype=RDBColumnDType.foreign_key, link_to="P.pid")
               ])
           ]
        )
        
        df_c1 = pd.DataFrame({'cid': [1], 'fk': [10]}) # Int
        df_c2 = pd.DataFrame({'cid': [2], 'fk': ['20']}) # Object
        
        rdb = RDB(metadata=meta, tables={"C1": df_c1, "C2": df_c2}) # P missing
        
        with pytest.raises(TypeError, match="Type mismatch"):
            rdb.validate_key_consistency()


# -----------------------------------------------------------------------------
# 5. Type Canonicalization Tests
# -----------------------------------------------------------------------------

class TestRDBTypeCanonicalization:
    def test_canonicalize_key_types(self):
        # Table A: PK is Int
        df_a = pd.DataFrame({'id': [1, 2], 'val': ['x', 'y']})
        
        # Table B: PK is String, FK to A is String
        # Note: If we use floats here, we expect canonicalize to fail per new reqs
        df_b = pd.DataFrame({'id': ['10', '11'], 'a_id': ['1', '2']})
        
        meta = RDBMeta(
            name="test",
            tables=[
                RDBTableSchema(
                    name="A", source="a.parquet", format="parquet",
                    columns=[
                        RDBColumnSchema(name="id", dtype=RDBColumnDType.primary_key),
                        RDBColumnSchema(name="val", dtype=RDBColumnDType.text_t)
                    ]
                ),
                RDBTableSchema(
                    name="B", source="b.parquet", format="parquet",
                    columns=[
                        RDBColumnSchema(name="id", dtype=RDBColumnDType.primary_key),
                        RDBColumnSchema(name="a_id", dtype=RDBColumnDType.foreign_key, link_to="A.id")
                    ]
                )
            ]
        )
        
        rdb = RDB(metadata=meta, tables={"A": df_a, "B": df_b})
        
        # Before conversion
        assert rdb.get_table("A")['id'].dtype == 'int64'
        assert rdb.get_table("B")['a_id'].dtype == 'object'
        
        # Run Canonicalize
        new_rdb = rdb.canonicalize_key_types()
        
        # Verify
        assert new_rdb.get_table("A")['id'].dtype == 'object'
        assert new_rdb.get_table("B")['a_id'].dtype == 'object'
        
        # Verify values
        assert new_rdb.get_table("A")['id'].tolist() == ['1', '2']
        assert new_rdb.get_table("B")['a_id'].tolist() == ['1', '2']

    def test_canonicalize_key_types_failure(self):
        # Table A: PK is Float
        df_a = pd.DataFrame({'id': [1.0, 2.0], 'val': ['x', 'y']})
        
        meta = RDBMeta(
            name="test",
            tables=[
                RDBTableSchema(
                    name="A", source="a.parquet", format="parquet",
                    columns=[
                        RDBColumnSchema(name="id", dtype=RDBColumnDType.primary_key)
                    ]
                )
            ]
        )
        
        rdb = RDB(metadata=meta, tables={"A": df_a})
        
        # Run Canonicalize - Expect Error
        with pytest.raises(ValueError, match="Cannot safe convert float column"):
             rdb.canonicalize_key_types()
