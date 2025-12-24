import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from pathlib import Path
from fastdfs.adapter.relbench import RelBenchAdapter
from fastdfs.dataset.meta import RDBColumnDType

class TestRelBenchAdapter(unittest.TestCase):
    
    @patch('fastdfs.adapter.relbench.relbench', new_callable=MagicMock)
    @patch('fastdfs.adapter.relbench.get_dataset')
    def test_load_rel_trial(self, mock_get_dataset, mock_relbench):
        # Mock RelBench Dataset and Database
        mock_dataset = MagicMock()
        mock_db = MagicMock()
        mock_dataset.get_db.return_value = mock_db
        mock_get_dataset.return_value = mock_dataset
        
        # Create mock tables
        # Table 1: designs (has type hints)
        df_designs = pd.DataFrame({
            "id": [1, 2],
            "intervention_model": ["A", "B"],
            "masking": ["Single", "Double"]
        })
        table_designs = MagicMock()
        table_designs.df = df_designs
        table_designs.pkey_col = "id"
        table_designs.time_col = None
        table_designs.fkey_col_to_pkey_table = {}
        
        # Table 2: outcome_analyses (has filtered columns)
        df_outcomes = pd.DataFrame({
            "id": [1, 2],
            "p_value_raw": [0.01, 0.05], # Should be removed
            "val": [10, 20]
        })
        table_outcomes = MagicMock()
        table_outcomes.df = df_outcomes
        table_outcomes.pkey_col = "id"
        table_outcomes.time_col = None
        table_outcomes.fkey_col_to_pkey_table = {}
        
        mock_db.table_dict = {
            "designs": table_designs,
            "outcome_analyses": table_outcomes
        }
        
        # Initialize Adapter
        adapter = RelBenchAdapter(dataset_name="rel-trial")
        rdb = adapter.load()
        
        # Verify RDB structure
        self.assertEqual(rdb.metadata.name, "rel-trial")
        self.assertIn("designs", rdb.table_names)
        self.assertIn("outcome_analyses", rdb.table_names)
        
        # Verify Type Hints
        designs_meta = rdb.get_table_metadata("designs")
        self.assertEqual(designs_meta.column_dict["intervention_model"].dtype, RDBColumnDType.category_t)
        self.assertEqual(designs_meta.column_dict["masking"].dtype, RDBColumnDType.category_t)
        
        # Verify Column Filtering
        outcomes_df = rdb.tables["outcome_analyses"]
        self.assertNotIn("p_value_raw", outcomes_df.columns)
        self.assertIn("val", outcomes_df.columns)

    @patch('fastdfs.adapter.relbench.relbench', new_callable=MagicMock)
    @patch('fastdfs.adapter.relbench.get_dataset')
    def test_load_rel_stack(self, mock_get_dataset, mock_relbench):
        # Mock RelBench Dataset and Database
        mock_dataset = MagicMock()
        mock_db = MagicMock()
        mock_dataset.get_db.return_value = mock_db
        mock_get_dataset.return_value = mock_dataset
        
        # Table: users (has filtered columns)
        df_users = pd.DataFrame({
            "Id": [1, 2],
            "ProfileImageUrl": ["url1", "url2"], # Should be removed
            "WebsiteUrl": ["web1", "web2"], # Should be removed
            "Reputation": [100, 200]
        })
        table_users = MagicMock()
        table_users.df = df_users
        table_users.pkey_col = "Id"
        table_users.time_col = None
        table_users.fkey_col_to_pkey_table = {}
        
        mock_db.table_dict = {
            "users": table_users
        }
        
        # Initialize Adapter
        adapter = RelBenchAdapter(dataset_name="rel-stack")
        rdb = adapter.load()
        
        # Verify Column Filtering
        users_df = rdb.tables["users"]
        self.assertNotIn("ProfileImageUrl", users_df.columns)
        self.assertNotIn("WebsiteUrl", users_df.columns)
        self.assertIn("Reputation", users_df.columns)

    @patch('fastdfs.adapter.relbench.relbench', new_callable=MagicMock)
    @patch('fastdfs.adapter.relbench.get_dataset')
    def test_load_relationships(self, mock_get_dataset, mock_relbench):
        # Mock RelBench Dataset and Database
        mock_dataset = MagicMock()
        mock_db = MagicMock()
        mock_dataset.get_db.return_value = mock_db
        mock_get_dataset.return_value = mock_dataset
        
        # Parent Table
        df_users = pd.DataFrame({"uid": [1, 2]})
        table_users = MagicMock()
        table_users.df = df_users
        table_users.pkey_col = "uid"
        table_users.time_col = None
        table_users.fkey_col_to_pkey_table = {}
        
        # Child Table
        df_posts = pd.DataFrame({"pid": [10, 11], "owner_uid": [1, 2]})
        table_posts = MagicMock()
        table_posts.df = df_posts
        table_posts.pkey_col = "pid"
        table_posts.time_col = None
        table_posts.fkey_col_to_pkey_table = {"owner_uid": "users"}
        
        mock_db.table_dict = {
            "users": table_users,
            "posts": table_posts
        }
        
        # Initialize Adapter
        adapter = RelBenchAdapter(dataset_name="test-rel")
        rdb = adapter.load()
        
        # Verify FK
        posts_meta = rdb.get_table_metadata("posts")
        self.assertEqual(posts_meta.column_dict["owner_uid"].dtype, RDBColumnDType.foreign_key)
        self.assertEqual(posts_meta.column_dict["owner_uid"].link_to, "users.uid")

    @patch('fastdfs.adapter.relbench.relbench', new_callable=MagicMock)
    @patch('fastdfs.adapter.relbench.get_dataset')
    @patch('fastdfs.dataset.rdb.RDB.save')
    def test_load_with_output_dir(self, mock_save, mock_get_dataset, mock_relbench):
        # Mock RelBench Dataset and Database
        mock_dataset = MagicMock()
        mock_db = MagicMock()
        mock_dataset.get_db.return_value = mock_db
        mock_get_dataset.return_value = mock_dataset
        
        # Simple table
        df_users = pd.DataFrame({"uid": [1, 2]})
        table_users = MagicMock()
        table_users.df = df_users
        table_users.pkey_col = "uid"
        table_users.time_col = None
        table_users.fkey_col_to_pkey_table = {}
        
        mock_db.table_dict = {"users": table_users}
        
        # Initialize Adapter with output_dir
        output_dir = Path("/tmp/test_output")
        adapter = RelBenchAdapter(dataset_name="test-save", output_dir=output_dir)
        rdb = adapter.load()
        
        # Verify save was called
        mock_save.assert_called_once_with(output_dir)

if __name__ == '__main__':
    unittest.main()
