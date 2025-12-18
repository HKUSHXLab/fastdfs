import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from fastdfs.transform.dummy_table_transform import HandleDummyTable
from fastdfs.dataset.rdb import RDBDataset
from fastdfs.dataset.meta import RDBColumnSchema, RDBColumnDType, RDBTableSchema, RDBTableDataFormat

class TestHandleDummyTable:
    
    def setup_method(self):
        self.transform = HandleDummyTable()
        self.mock_dataset = MagicMock(spec=RDBDataset)
        self.mock_dataset.table_names = ['table1', 'table2']

    def test_collect_foreign_key_references_corner_cases(self):
        """
        Test corner cases for relationship parsing in _collect_foreign_key_references.
        Focus on:
        for relationship in relationships:
            if isinstance(relationship, tuple) and len(relationship) == 4:
                ...
        """
        # Setup relationships with various corner cases
        relationships = [
            ('table1', 'fk_col', 'missing_table', 'pk_col'),  # Valid
            ('table1', 'fk_col', 'missing_table'),            # Invalid: len != 4
            ['table1', 'fk_col', 'missing_table', 'pk_col'],  # Invalid: not tuple
            ('table2', 'fk_col2', 'missing_table', 'pk_col'), # Valid: another ref to same missing table
            ('table1', 'fk_col', 'existing_table', 'pk_col')  # Valid: ref to existing table
        ]
        
        # Call the private method directly to test the logic
        fk_refs = self.transform._collect_foreign_key_references(self.mock_dataset, relationships)
        
        # Assertions
        assert 'missing_table' in fk_refs
        assert fk_refs['missing_table']['pk_col'] == 'pk_col'
        assert len(fk_refs['missing_table']['fk_refs']) == 2
        assert ('table1', 'fk_col') in fk_refs['missing_table']['fk_refs']
        assert ('table2', 'fk_col2') in fk_refs['missing_table']['fk_refs']
        
        assert 'existing_table' in fk_refs
        assert len(fk_refs['existing_table']['fk_refs']) == 1

    def test_find_missing_primary_tables(self):
        """Test identification of missing primary tables."""
        # Setup
        fk_refs = {
            'missing_table': {
                'pk_col': 'id',
                'fk_refs': [('table1', 'fk_id')]
            },
            'existing_table': {
                'pk_col': 'id',
                'fk_refs': [('table1', 'other_id')]
            }
        }
        
        self.mock_dataset.table_names = ['table1', 'existing_table']
        
        # Mock get_table to return dataframe with FK values
        mock_df = pd.DataFrame({'fk_id': [1, 2, 3], 'other_id': [4, 5, 6]})
        self.mock_dataset.get_table.return_value = mock_df
        
        # Execute
        missing_tables = self.transform._find_missing_primary_tables(fk_refs, self.mock_dataset)
        
        # Assertions
        assert len(missing_tables) == 1
        table_name, pk_col, fk_values = missing_tables[0]
        assert table_name == 'missing_table'
        assert pk_col == 'id'
        assert fk_values == {1, 2, 3}

    def test_call_integration(self):
        """Test the full __call__ method with a missing table scenario."""
        # Setup relationships
        relationships = [
            ('table1', 'fk_id', 'missing_table', 'id')
        ]
        self.mock_dataset.get_relationships.return_value = relationships
        self.mock_dataset.table_names = ['table1']
        
        # Mock table data
        mock_df = pd.DataFrame({'fk_id': [1, 2, None]}) # Include None to test dropna
        self.mock_dataset.get_table.side_effect = lambda name: mock_df if name == 'table1' else None
        
        # Mock metadata
        mock_metadata = MagicMock(spec=RDBTableSchema)
        self.mock_dataset.get_table_metadata.return_value = mock_metadata
        
        # Execute
        result_dataset = self.transform(self.mock_dataset)
        
        # Verify create_new_with_tables_and_metadata was called
        self.mock_dataset.create_new_with_tables_and_metadata.assert_called_once()
        call_args = self.mock_dataset.create_new_with_tables_and_metadata.call_args
        new_tables = call_args[1]['new_tables']
        new_metadata = call_args[1]['new_metadata']
        
        # Check if dummy table was created
        assert 'missing_table' in new_tables
        dummy_df = new_tables['missing_table']
        assert 'id' in dummy_df.columns
        assert set(dummy_df['id']) == {1.0, 2.0} # 1, 2. None should be dropped
        
        assert 'missing_table' in new_metadata
        dummy_meta = new_metadata['missing_table']
        assert dummy_meta.name == 'missing_table'
        assert dummy_meta.columns[0].name == 'id'
        assert dummy_meta.columns[0].dtype == RDBColumnDType.primary_key

    def test_no_relationships(self):
        """Test early return when no relationships exist."""
        self.mock_dataset.get_relationships.return_value = []
        
        result = self.transform(self.mock_dataset)
        
        assert result == self.mock_dataset
        self.mock_dataset.create_new_with_tables_and_metadata.assert_not_called()

    def test_no_missing_tables(self):
        """Test when all referenced tables exist."""
        relationships = [('table1', 'fk', 'table2', 'pk')]
        self.mock_dataset.get_relationships.return_value = relationships
        self.mock_dataset.table_names = ['table1', 'table2']
        
        result = self.transform(self.mock_dataset)
        
        assert result == self.mock_dataset
        self.mock_dataset.create_new_with_tables_and_metadata.assert_not_called()
