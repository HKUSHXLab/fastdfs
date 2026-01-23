"""
Fill missing primary key transform - expands PK tables to include all FK-referenced values.

This transform:
- Collects all unique values from each PK column + all its referencing FK columns
- Expands PK tables to include all FK-referenced values
- Fills expanded rows with appropriate null values
- Keeps original key values (no integer encoding)
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict

from ..dataset.meta import RDBColumnSchema, RDBColumnDType, RDBTableSchema
from ..dataset.rdb import RDB
from .base import RDBTransform


class FillMissingPrimaryKey(RDBTransform):
    """
    Expand PK tables to include all FK-referenced values (without encoding).
    
    This transform:
    - Collects all unique values from each PK column + all its referencing FK columns
    - Expands PK tables to include all FK-referenced values (with NULL for other fields)
    - Keeps original key values (no integer encoding)
    
    Example:
        Original:
            PromotedContent: ad_id = [128, 444, 636] (3 rows)
            Click: ad_id = [128, 444, 999, 1000] (4 rows)
        
        After FillMissingPrimaryKey:
            PromotedContent: ad_id = [128, 444, 636, 999, 1000] (5 rows, expanded!)
            Click: ad_id = [128, 444, 999, 1000] (same 4 rows, unchanged)
    """
    
    def __call__(self, rdb: RDB) -> RDB:
        """Apply key mapping transformation to RDB."""
        # Get relationships
        relationships = rdb.get_relationships()
        if not relationships:
            return rdb  # No relationships to process
        
        # Build PK → FK mapping
        pk_to_fk_list, fk_to_pk = self._build_key_mappings(relationships)
        
        # Collect unique keys
        unique_keys = self._collect_unique_keys(rdb, pk_to_fk_list)
        
        # Transform tables
        new_tables, new_metadata = self._transform_tables(
            rdb, pk_to_fk_list, fk_to_pk, unique_keys
        )
        
        return rdb.update_tables(tables=new_tables, metadata=new_metadata)
    
    def _build_key_mappings(
        self, 
        relationships: List[Tuple[str, str, str, str]]
    ) -> Tuple[Dict, Dict]:
        """
        Build PK→FK and FK→PK mappings from relationships.
        
        Returns:
            Tuple of (pk_to_fk_list, fk_to_pk) where:
            - pk_to_fk_list[(pk_tbl, pk_col)] = [(fk_tbl1, fk_col1), ...]
            - fk_to_pk[(fk_tbl, fk_col)] = (pk_tbl, pk_col)
        """
        pk_to_fk_list = defaultdict(list)
        fk_to_pk = {}
        
        for fk_tbl, fk_col, pk_tbl, pk_col in relationships:
            pk_to_fk_list[(pk_tbl, pk_col)].append((fk_tbl, fk_col))
            fk_to_pk[(fk_tbl, fk_col)] = (pk_tbl, pk_col)
            fk_to_pk[(pk_tbl, pk_col)] = (pk_tbl, pk_col)  # PK maps to itself
        
        return dict(pk_to_fk_list), fk_to_pk
    
    def _collect_unique_keys(
        self,
        rdb: RDB,
        pk_to_fk_list: Dict[Tuple[str, str], List[Tuple[str, str]]]
    ) -> Dict[Tuple[str, str], np.ndarray]:
        """
        Collect all unique key values without encoding.
        
        For each PK, collects values from:
        - The PK column itself
        - All FK columns that reference it
        
        Returns:
            Dict mapping (table, column) to array of unique values
        """
        unique_keys = {}
        
        for (pk_tbl, pk_col), fk_list in pk_to_fk_list.items():
            # Collect PK values
            key_values = []
            if pk_tbl in rdb.table_names:
                pk_table = rdb.get_table(pk_tbl)
                if pk_col in pk_table.columns:
                    key_values.append(pk_table[pk_col].values)
            
            # Collect FK values from all referencing columns
            for fk_tbl, fk_col in fk_list:
                if fk_tbl in rdb.table_names:
                    fk_table = rdb.get_table(fk_tbl)
                    if fk_col in fk_table.columns:
                        key_values.append(fk_table[fk_col].values)
            
            # Concatenate and get unique values (excluding NaN)
            if key_values:
                all_keys = np.concatenate(key_values)
                # Get unique values, excluding NaN/None
                unique_vals = pd.Series(all_keys).dropna().unique()
                unique_keys[(pk_tbl, pk_col)] = unique_vals
        
        return unique_keys
    
    def _get_null_value(self, dtype: RDBColumnDType) -> object:
        """Get appropriate null value for a dtype."""
        if dtype == RDBColumnDType.datetime_t:
            return pd.NaT
        elif dtype in [RDBColumnDType.text_t, RDBColumnDType.category_t]:
            return ""
        elif dtype == RDBColumnDType.timestamp_t:
            return 0
        else:
            return np.nan
    
    def _transform_tables(
        self,
        rdb: RDB,
        pk_to_fk_list: Dict[Tuple[str, str], List[Tuple[str, str]]],
        fk_to_pk: Dict[Tuple[str, str], Tuple[str, str]],
        unique_keys: Dict[Tuple[str, str], np.ndarray]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, RDBTableSchema]]:
        """
        Transform all tables by expanding PK tables (without encoding).
        
        For each table:
        1. Calculate expansion needed for PK tables
        2. Expand table to include all unique key values
        3. Fill expanded rows with appropriate null values
        """
        new_tables = {}
        new_metadata = {}
        
        for table_name in rdb.table_names:
            table = rdb.get_table(table_name).copy()
            metadata = rdb.get_table_metadata(table_name)
            
            # Find PK column and calculate expansion needed
            num_expansion_rows = 0
            pk_col_name = None
            pk_key = None
            expected_pk_size = None
            
            for col_schema in metadata.columns:
                if col_schema.dtype == RDBColumnDType.primary_key:
                    pk_col_name = col_schema.name
                    pk_key = (table_name, pk_col_name)
                    if pk_key in unique_keys:
                        expected_pk_size = len(unique_keys[pk_key])
                        current_size = len(table)
                        num_expansion_rows = expected_pk_size - current_size
                    break
            
            # Step 1: Expand table if needed (BEFORE transforming columns)
            if num_expansion_rows > 0 and pk_col_name:
                # Build expansion rows with original column types
                expansion_data = {}
                
                for col_schema in metadata.columns:
                    col_name = col_schema.name
                    
                    if col_schema.dtype == RDBColumnDType.primary_key:
                        # Will be filled after expansion
                        expansion_data[col_name] = [None] * num_expansion_rows
                    elif col_schema.dtype == RDBColumnDType.foreign_key:
                        # Temporary placeholder, will be mapped later
                        expansion_data[col_name] = [None] * num_expansion_rows
                    else:
                        # Fill other columns with appropriate NULL values
                        null_val = self._get_null_value(col_schema.dtype)
                        expansion_data[col_name] = [null_val] * num_expansion_rows
                
                # Append expansion rows
                if expansion_data:
                    expansion_df = pd.DataFrame(expansion_data)
                    table = pd.concat([table, expansion_df], ignore_index=True)
            
            # Step 2: Fill PK column with unique values if table was expanded
            if num_expansion_rows > 0 and pk_col_name and pk_key in unique_keys:
                # Get existing PK values
                existing_pk_values = table[pk_col_name].iloc[:len(table)-num_expansion_rows].values
                # Get all unique keys
                all_unique_keys = unique_keys[pk_key]
                # Find missing keys (those not in existing)
                existing_set = set(pd.Series(existing_pk_values).dropna())
                missing_keys = [k for k in all_unique_keys if k not in existing_set]
                # Fill expanded rows with missing keys
                if missing_keys:
                    table.loc[table.index[-num_expansion_rows:], pk_col_name] = missing_keys[:num_expansion_rows]
            
            # Step 3: Copy metadata without modifications
            new_col_schemas = [col_schema.copy() for col_schema in metadata.columns]
            
            new_tables[table_name] = table
            new_metadata[table_name] = RDBTableSchema(
                name=metadata.name,
                source=metadata.source,
                format=metadata.format,
                columns=new_col_schemas,
                time_column=metadata.time_column
            )
        
        return new_tables, new_metadata