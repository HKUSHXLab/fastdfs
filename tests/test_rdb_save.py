import unittest
import pandas as pd
import shutil
from pathlib import Path
from fastdfs.dataset.rdb import RDB
from fastdfs.dataset.meta import RDBMeta, RDBTableSchema, RDBTableDataFormat, RDBColumnSchema, RDBColumnDType

class TestRDBSave(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/data/test_rdb_save")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True)
        
        self.df = pd.DataFrame({
            "id": [1, 2, 3],
            "val": [1.1, 2.2, 3.3]
        })
        
        self.tables = {"table1": self.df}
        
        self.schema = RDBTableSchema(
            name="table1",
            source="table1.parquet",
            format=RDBTableDataFormat.PARQUET,
            columns=[
                RDBColumnSchema(name="id", dtype=RDBColumnDType.primary_key),
                RDBColumnSchema(name="val", dtype=RDBColumnDType.float_t)
            ]
        )
        
        self.metadata = RDBMeta(name="test_save", tables=[self.schema])
        self.rdb = RDB(metadata=self.metadata, tables=self.tables)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_save_rdb(self):
        save_path = self.test_dir / "saved_rdb"
        self.rdb.save(save_path)
        
        self.assertTrue(save_path.exists())
        self.assertTrue((save_path / "metadata.yaml").exists())
        self.assertTrue((save_path / "table1.parquet").exists())
        
        # Verify we can load it back
        loaded_rdb = RDB(path=save_path)
        self.assertEqual(loaded_rdb.metadata.name, "test_save")
        self.assertIn("table1", loaded_rdb.table_names)
        
        loaded_df = loaded_rdb.get_table_dataframe("table1")
        pd.testing.assert_frame_equal(self.df, loaded_df)

if __name__ == '__main__':
    unittest.main()
