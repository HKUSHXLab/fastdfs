"""
Dummy table handling transform.

This module implements HandleDummyTable, an RDBTransform that creates missing
primary key tables for foreign key references based on the original implementation
in fastdfs/preprocess/transform/dummy_table.py.
"""

from typing import Dict, List, Set, Union, Any
import pandas as pd
from ..dataset.meta import RDBColumnSchema, RDBColumnDType, RDBTableSchema, RDBTableDataFormat
from ..dataset.rdb import RDB
from .base import RDBTransform


class HandleDummyTable(RDBTransform):
    """Create dummy tables for missing primary key references."""
    
    def __call__(self, dataset: RDB) -> RDB:
        """Create missing primary key tables from foreign key references."""
        # Get current relationships
        relationships = dataset.get_relationships()
        if not relationships:
            return dataset  # No relationships to process
        
        # Find all foreign key references
        fk_refs = self._collect_foreign_key_references(dataset, relationships)
        
        # Find which primary key tables are missing
        missing_pk_tables = self._find_missing_primary_tables(fk_refs, dataset)
        
        if not missing_pk_tables:
            return dataset  # No missing tables
            
        # Create dummy tables for missing primary keys
        new_tables = {name: dataset.get_table(name) for name in dataset.table_names}
        new_metadata = {name: dataset.get_table_metadata(name) for name in dataset.table_names}
        
        for table_name, pk_column, fk_values in missing_pk_tables:
            dummy_table, dummy_metadata = self._create_dummy_table(
                table_name, pk_column, fk_values
            )
            new_tables[table_name] = dummy_table
            new_metadata[table_name] = dummy_metadata
            
        return dataset.create_new_with_tables_and_metadata(
            new_tables=new_tables,
            new_metadata=new_metadata
        )
    
    def _collect_foreign_key_references(self, dataset: RDB, relationships: List) -> Dict[str, Dict[str, Union[str, List]]]:
        """Collect all foreign key column references and their target tables.
        
        Returns:
            Dict mapping pk_table -> {'pk_col': str, 'fk_refs': [(fk_table, fk_col), ...]}
        """
        fk_refs = {}
        
        # Process relationships to understand foreign key structure
        for relationship in relationships:
            if isinstance(relationship, tuple) and len(relationship) == 4:
                fk_table, fk_col, pk_table, pk_col = relationship
                if pk_table not in fk_refs:
                    fk_refs[pk_table] = {
                        'pk_col': pk_col,
                        'fk_refs': []
                    }
                # Store the FK table and column that references this PK
                fk_refs[pk_table]['fk_refs'].append((fk_table, fk_col))
                        
        return fk_refs
    
    def _find_missing_primary_tables(self, fk_refs: Dict[str, Dict[str, Any]],
                                   dataset: RDB) -> List[tuple]:
        """Find which primary key tables are missing and collect their values."""
        missing_tables = []
        
        for target_table, ref_info in fk_refs.items():
            if target_table not in dataset.table_names:
                # This table is missing - collect FK values from actual FK columns
                fk_values = set()
                pk_column = ref_info['pk_col']  # Use actual PK column from relationships
                
                # Collect foreign key values from all FK columns that reference this table
                for fk_table, fk_col in ref_info['fk_refs']:
                    if fk_table in dataset.table_names:
                        table_df = dataset.get_table(fk_table)
                        if fk_col in table_df.columns:
                            unique_values = table_df[fk_col].dropna().unique()
                            fk_values.update(unique_values)
                
                if fk_values:
                    missing_tables.append((target_table, pk_column, fk_values))
                    
        return missing_tables
    
    def _create_dummy_table(self, table_name: str, pk_column: str, 
                          pk_values: Set) -> tuple:
        """Create a dummy table with just primary key values."""
        # Create DataFrame with primary key values
        pk_values_list = sorted(list(pk_values))
        dummy_df = pd.DataFrame({pk_column: pk_values_list})
        
        # Create metadata schema
        pk_col_schema = RDBColumnSchema(
            name=pk_column,
            dtype=RDBColumnDType.primary_key
        )
        
        dummy_metadata = RDBTableSchema(
            name=table_name,
            source=f"dummy_{table_name}",
            format=RDBTableDataFormat.NUMPY,
            columns=[pk_col_schema],
            time_column=None
        )
        
        return dummy_df, dummy_metadata
