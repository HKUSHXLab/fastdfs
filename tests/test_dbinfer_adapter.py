import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from pathlib import Path
from fastdfs.adapter.dbinfer import DBInferAdapter
from fastdfs.dataset.meta import RDBColumnDType

class TestDBInferAdapter(unittest.TestCase):
    
    @patch('fastdfs.adapter.dbinfer.dbb')
    def test_load_diginetica(self, mock_dbb):
        # Mock DBInfer Dataset
        mock_dataset = MagicMock()
        mock_dbb.load_rdb_data.return_value = mock_dataset
        
        # Mock Metadata
        # Table 1: Product
        col_item_id = MagicMock()
        col_item_id.name = "itemId"
        col_item_id.dtype = "primary_key"
        
        col_cat_id = MagicMock()
        col_cat_id.name = "categoryId"
        col_cat_id.dtype = "category"
        
        table_product_meta = MagicMock()
        table_product_meta.name = "Product"
        table_product_meta.columns = [col_item_id, col_cat_id]
        table_product_meta.time_column = None
        
        # Table 2: Click
        col_query_id = MagicMock()
        col_query_id.name = "queryId"
        col_query_id.dtype = "foreign_key"
        col_query_id.link_to = "Query.queryId"
        
        col_timestamp = MagicMock()
        col_timestamp.name = "timestamp"
        col_timestamp.dtype = "datetime"
        
        table_click_meta = MagicMock()
        table_click_meta.name = "Click"
        table_click_meta.columns = [col_query_id, col_timestamp]
        table_click_meta.time_column = "timestamp"

        # Table 3: Query (Parent of Click)
        col_q_id = MagicMock()
        col_q_id.name = "queryId"
        col_q_id.dtype = "primary_key"
        
        table_query_meta = MagicMock()
        table_query_meta.name = "Query"
        table_query_meta.columns = [col_q_id]
        table_query_meta.time_column = None
        
        mock_dataset.metadata.tables = [table_product_meta, table_click_meta, table_query_meta]
        
        # Mock Data
        mock_dataset.tables = {
            "Product": {
                "itemId": np.array([1, 2]),
                "categoryId": np.array([10, 20])
            },
            "Click": {
                "queryId": np.array([100, 101]),
                "timestamp": np.array(["2021-01-01", "2021-01-02"])
            },
            "Query": {
                "queryId": np.array([100, 101])
            }
        }
        
        # Initialize Adapter
        adapter = DBInferAdapter(dataset_name="diginetica")
        rdb = adapter.load()
        
        # Verify RDB structure
        self.assertEqual(rdb.metadata.name, "diginetica")
        self.assertIn("Product", rdb.table_names)
        self.assertIn("Click", rdb.table_names)
        
        # Verify PK
        product_meta = rdb.get_table_metadata("Product")
        pk_col = next((col for col in product_meta.columns if col.dtype == RDBColumnDType.primary_key), None)
        self.assertIsNotNone(pk_col)
        self.assertEqual(pk_col.name, "itemId")
        
        # Verify FK
        click_meta = rdb.get_table_metadata("Click")
        self.assertEqual(click_meta.column_dict["queryId"].dtype, RDBColumnDType.foreign_key)
        self.assertEqual(click_meta.column_dict["queryId"].link_to, "Query.queryId")
        
        # Verify Types
        product_meta = rdb.get_table_metadata("Product")
        self.assertEqual(product_meta.column_dict["categoryId"].dtype, RDBColumnDType.category_t)
        
        self.assertEqual(click_meta.column_dict["timestamp"].dtype, RDBColumnDType.datetime_t)
        self.assertEqual(click_meta.time_column, "timestamp")

    @patch('fastdfs.adapter.dbinfer.dbb')
    @patch('fastdfs.dataset.rdb.RDB.save')
    def test_load_with_output_dir(self, mock_save, mock_dbb):
        # Mock DBInfer Dataset
        mock_dataset = MagicMock()
        mock_dbb.load_rdb_data.return_value = mock_dataset
        
        table_meta = MagicMock()
        table_meta.name = "Users"
        table_meta.columns = []
        table_meta.time_column = None
        
        mock_dataset.metadata.tables = [table_meta]
        mock_dataset.tables = {"Users": {"id": [1]}}
        
        # Initialize Adapter
        output_dir = Path("/tmp/dbinfer_out")
        adapter = DBInferAdapter(dataset_name="test", output_dir=output_dir)
        rdb = adapter.load()
        
        # Verify save called
        mock_save.assert_called_once_with(output_dir)

if __name__ == '__main__':
    unittest.main()
