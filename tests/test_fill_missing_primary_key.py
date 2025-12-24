"""
Unit tests for FillMissingPrimaryKey transform.

Tests cover:
1. Basic PK expansion functionality
2. No expansion when PK already contains all FK values
3. Multiple FK tables referencing the same PK
4. Edge cases (empty tables, no relationships)
5. Referential integrity verification
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from fastdfs.transform.fill_missing_pk import FillMissingPrimaryKey
from fastdfs.dataset.rdb import RDB
from fastdfs.dataset.meta import (
    RDBColumnSchema, 
    RDBColumnDType, 
    RDBTableSchema, 
    RDBTableDataFormat
)


class TestFillMissingPrimaryKey:
    """Unit tests for FillMissingPrimaryKey transform."""
    
    def setup_method(self):
        """Setup common test fixtures."""
        self.transform = FillMissingPrimaryKey()
    
    def create_mock_rdb(self, tables_data, relationships):
        """
        Helper to create a mock RDBDataset with specified tables and relationships.
        
        Args:
            tables_data: Dict[str, Dict] with structure:
                {
                    'table_name': {
                        'data': pd.DataFrame,
                        'columns': List[RDBColumnSchema]
                    }
                }
            relationships: List of (child_table, child_col, parent_table, parent_col)
        
        Returns:
            Mock RDBDataset
        """
        mock_rdb = MagicMock(spec=RDB)
        
        # Setup table names
        mock_rdb.table_names = list(tables_data.keys())
        
        # Setup get_table to return appropriate DataFrame
        def get_table(name):
            if name in tables_data:
                return tables_data[name]['data'].copy()
            raise ValueError(f"Table {name} not found")
        
        mock_rdb.get_table.side_effect = get_table
        
        # Setup get_table_metadata to return appropriate schema
        def get_table_metadata(name):
            if name in tables_data:
                return RDBTableSchema(
                    name=name,
                    source=f"{name}.parquet",
                    format=RDBTableDataFormat.PARQUET,
                    columns=tables_data[name]['columns'],
                    time_column=None
                )
            raise ValueError(f"Table {name} not found")
        
        mock_rdb.get_table_metadata.side_effect = get_table_metadata
        
        # Setup relationships
        mock_rdb.get_relationships.return_value = relationships
        
        return mock_rdb
    
    def test_basic_pk_expansion(self):
        """
        Test that PK table expands to include all FK-referenced values.
        
        Setup:
            Product (PK: product_id) has [1, 2, 3]
            Order (FK: product_id) references [1, 2, 3, 4, 5]
        
        Expected:
            Product expands to [1, 2, 3, 4, 5]
        """
        # Create Product table with 3 rows
        product_df = pd.DataFrame({
            'product_id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'price': [10.0, 20.0, 30.0]
        })
        
        # Create Order table referencing products 1-5
        order_df = pd.DataFrame({
            'order_id': [101, 102, 103, 104, 105],
            'product_id': [1, 2, 3, 4, 5],
            'quantity': [1, 2, 1, 3, 2]
        })
        
        tables_data = {
            'Product': {
                'data': product_df,
                'columns': [
                    RDBColumnSchema(name='product_id', dtype=RDBColumnDType.primary_key),
                    RDBColumnSchema(name='name', dtype=RDBColumnDType.text_t),
                    RDBColumnSchema(name='price', dtype=RDBColumnDType.float_t)
                ]
            },
            'Order': {
                'data': order_df,
                'columns': [
                    RDBColumnSchema(name='order_id', dtype=RDBColumnDType.primary_key),
                    RDBColumnSchema(name='product_id', dtype=RDBColumnDType.foreign_key, link_to='Product.product_id'),
                    RDBColumnSchema(name='quantity', dtype=RDBColumnDType.float_t)
                ]
            }
        }
        
        relationships = [('Order', 'product_id', 'Product', 'product_id')]
        
        mock_rdb = self.create_mock_rdb(tables_data, relationships)
        
        # Apply transform
        result_rdb = self.transform(mock_rdb)
        
        # Verify create_new_with_tables_and_metadata was called
        mock_rdb.create_new_with_tables_and_metadata.assert_called_once()
        call_args = mock_rdb.create_new_with_tables_and_metadata.call_args
        new_tables = call_args[0][0]  # First positional argument
        
        # Check Product table was expanded
        expanded_product = new_tables['Product']
        assert len(expanded_product) == 5, "Product table should have 5 rows"
        assert set(expanded_product['product_id']) == {1, 2, 3, 4, 5}, "Should contain all referenced product_ids"
        
        # Check original rows are preserved
        original_products = expanded_product[expanded_product['product_id'].isin([1, 2, 3])]
        assert len(original_products) == 3
        assert original_products['name'].tolist() == ['A', 'B', 'C']
        
        # Check new rows have NaN for non-PK columns
        new_products = expanded_product[expanded_product['product_id'].isin([4, 5])]
        assert len(new_products) == 2
        assert new_products['name'].isna().all() or (new_products['name'] == '').all()
    
    def test_no_expansion_when_complete(self):
        """
        Test that no expansion occurs when PK already contains all FK values.
        
        Setup:
            Product has [1, 2, 3, 4, 5]
            Order references [1, 3, 5]
        
        Expected:
            Product remains unchanged at [1, 2, 3, 4, 5]
        """
        product_df = pd.DataFrame({
            'product_id': [1, 2, 3, 4, 5],
            'name': ['A', 'B', 'C', 'D', 'E']
        })
        
        order_df = pd.DataFrame({
            'order_id': [101, 102, 103],
            'product_id': [1, 3, 5]
        })
        
        tables_data = {
            'Product': {
                'data': product_df,
                'columns': [
                    RDBColumnSchema(name='product_id', dtype=RDBColumnDType.primary_key),
                    RDBColumnSchema(name='name', dtype=RDBColumnDType.text_t)
                ]
            },
            'Order': {
                'data': order_df,
                'columns': [
                    RDBColumnSchema(name='order_id', dtype=RDBColumnDType.primary_key),
                    RDBColumnSchema(name='product_id', dtype=RDBColumnDType.foreign_key, link_to='Product.product_id')
                ]
            }
        }
        
        relationships = [('Order', 'product_id', 'Product', 'product_id')]
        
        mock_rdb = self.create_mock_rdb(tables_data, relationships)
        
        # Apply transform
        result_rdb = self.transform(mock_rdb)
        
        # Get result
        call_args = mock_rdb.create_new_with_tables_and_metadata.call_args
        new_tables = call_args[0][0]
        
        # Check Product table remains unchanged
        result_product = new_tables['Product']
        assert len(result_product) == 5, "Product table should still have 5 rows"
        assert set(result_product['product_id']) == {1, 2, 3, 4, 5}
        assert result_product['name'].tolist() == ['A', 'B', 'C', 'D', 'E']
    
    def test_multiple_fk_references(self):
        """
        Test PK expansion when multiple FK tables reference the same PK.
        
        Setup:
            Product has [1, 2]
            Order references [1, 2, 3]
            Review references [2, 4, 5]
        
        Expected:
            Product expands to [1, 2, 3, 4, 5] (union of all FK values)
        """
        product_df = pd.DataFrame({
            'product_id': [1, 2],
            'name': ['A', 'B']
        })
        
        order_df = pd.DataFrame({
            'order_id': [101, 102, 103],
            'product_id': [1, 2, 3]
        })
        
        review_df = pd.DataFrame({
            'review_id': [201, 202, 203],
            'product_id': [2, 4, 5],
            'rating': [4, 5, 3]
        })
        
        tables_data = {
            'Product': {
                'data': product_df,
                'columns': [
                    RDBColumnSchema(name='product_id', dtype=RDBColumnDType.primary_key),
                    RDBColumnSchema(name='name', dtype=RDBColumnDType.text_t)
                ]
            },
            'Order': {
                'data': order_df,
                'columns': [
                    RDBColumnSchema(name='order_id', dtype=RDBColumnDType.primary_key),
                    RDBColumnSchema(name='product_id', dtype=RDBColumnDType.foreign_key, link_to='Product.product_id')
                ]
            },
            'Review': {
                'data': review_df,
                'columns': [
                    RDBColumnSchema(name='review_id', dtype=RDBColumnDType.primary_key),
                    RDBColumnSchema(name='product_id', dtype=RDBColumnDType.foreign_key, link_to='Product.product_id'),
                    RDBColumnSchema(name='rating', dtype=RDBColumnDType.float_t)
                ]
            }
        }
        
        relationships = [
            ('Order', 'product_id', 'Product', 'product_id'),
            ('Review', 'product_id', 'Product', 'product_id')
        ]
        
        mock_rdb = self.create_mock_rdb(tables_data, relationships)
        
        # Apply transform
        result_rdb = self.transform(mock_rdb)
        
        # Get result
        call_args = mock_rdb.create_new_with_tables_and_metadata.call_args
        new_tables = call_args[0][0]
        
        # Check Product expanded to include all referenced values
        result_product = new_tables['Product']
        assert len(result_product) == 5, "Product should have 5 rows"
        assert set(result_product['product_id']) == {1, 2, 3, 4, 5}, "Should contain union of all FK values"
    
    def test_empty_pk_table(self):
        """
        Test expansion when PK table is initially empty.
        
        Setup:
            Product has 0 rows (empty)
            Order references [1, 2, 3]
        
        Expected:
            Product expands to [1, 2, 3]
        """
        product_df = pd.DataFrame({
            'product_id': pd.Series([], dtype=int),
            'name': pd.Series([], dtype=str)
        })
        
        order_df = pd.DataFrame({
            'order_id': [101, 102, 103],
            'product_id': [1, 2, 3]
        })
        
        tables_data = {
            'Product': {
                'data': product_df,
                'columns': [
                    RDBColumnSchema(name='product_id', dtype=RDBColumnDType.primary_key),
                    RDBColumnSchema(name='name', dtype=RDBColumnDType.text_t)
                ]
            },
            'Order': {
                'data': order_df,
                'columns': [
                    RDBColumnSchema(name='order_id', dtype=RDBColumnDType.primary_key),
                    RDBColumnSchema(name='product_id', dtype=RDBColumnDType.foreign_key, link_to='Product.product_id')
                ]
            }
        }
        
        relationships = [('Order', 'product_id', 'Product', 'product_id')]
        
        mock_rdb = self.create_mock_rdb(tables_data, relationships)
        
        # Apply transform
        result_rdb = self.transform(mock_rdb)
        
        # Get result
        call_args = mock_rdb.create_new_with_tables_and_metadata.call_args
        new_tables = call_args[0][0]
        
        # Check Product was populated
        result_product = new_tables['Product']
        assert len(result_product) == 3, "Product should have 3 rows"
        assert set(result_product['product_id']) == {1, 2, 3}
    
    def test_empty_fk_table(self):
        """
        Test no expansion when FK table is empty.
        
        Setup:
            Product has [1, 2, 3]
            Order has 0 rows (empty)
        
        Expected:
            Product remains unchanged at [1, 2, 3]
        """
        product_df = pd.DataFrame({
            'product_id': [1, 2, 3],
            'name': ['A', 'B', 'C']
        })
        
        order_df = pd.DataFrame({
            'order_id': pd.Series([], dtype=int),
            'product_id': pd.Series([], dtype=int)
        })
        
        tables_data = {
            'Product': {
                'data': product_df,
                'columns': [
                    RDBColumnSchema(name='product_id', dtype=RDBColumnDType.primary_key),
                    RDBColumnSchema(name='name', dtype=RDBColumnDType.text_t)
                ]
            },
            'Order': {
                'data': order_df,
                'columns': [
                    RDBColumnSchema(name='order_id', dtype=RDBColumnDType.primary_key),
                    RDBColumnSchema(name='product_id', dtype=RDBColumnDType.foreign_key, link_to='Product.product_id')
                ]
            }
        }
        
        relationships = [('Order', 'product_id', 'Product', 'product_id')]
        
        mock_rdb = self.create_mock_rdb(tables_data, relationships)
        
        # Apply transform
        result_rdb = self.transform(mock_rdb)
        
        # Get result
        call_args = mock_rdb.create_new_with_tables_and_metadata.call_args
        new_tables = call_args[0][0]
        
        # Check Product remains unchanged
        result_product = new_tables['Product']
        assert len(result_product) == 3, "Product should still have 3 rows"
        assert set(result_product['product_id']) == {1, 2, 3}
    
    def test_no_relationships(self):
        """
        Test early return when no relationships exist.
        
        Expected:
            Original RDB is returned unchanged
        """
        product_df = pd.DataFrame({
            'product_id': [1, 2, 3],
            'name': ['A', 'B', 'C']
        })
        
        tables_data = {
            'Product': {
                'data': product_df,
                'columns': [
                    RDBColumnSchema(name='product_id', dtype=RDBColumnDType.primary_key),
                    RDBColumnSchema(name='name', dtype=RDBColumnDType.text_t)
                ]
            }
        }
        
        relationships = []  # No relationships
        
        mock_rdb = self.create_mock_rdb(tables_data, relationships)
        
        # Apply transform
        result_rdb = self.transform(mock_rdb)
        
        # Verify early return - should return original RDB
        assert result_rdb == mock_rdb
        mock_rdb.create_new_with_tables_and_metadata.assert_not_called()
    