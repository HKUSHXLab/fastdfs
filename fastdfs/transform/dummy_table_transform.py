"""
Dummy table handling transform.

This module implements HandleDummyTable, an RDBTransform that creates missing
primary key tables for foreign key references based on the original implementation
in fastdfs/preprocess/transform/dummy_table.py.
"""

from typing import Dict, List, Set
import pandas as pd
from ..dataset.meta import RDBColumnSchema, RDBColumnDType, RDBTableSchema, RDBTableDataFormat
from ..dataset.rdb import RDBDataset
from .base import RDBTransform


class HandleDummyTable(RDBTransform):
    """Create dummy tables for missing primary key references."""
    
    def __call__(self, dataset: RDBDataset) -> RDBDataset:
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
    
    def _collect_foreign_key_references(self, dataset: RDBDataset, relationships: List) -> Dict[str, Set[str]]:
        """Collect all foreign key column references and their target tables."""
        fk_refs = {}
        
        # Process relationships to understand foreign key structure
        for relationship in relationships:
            if isinstance(relationship, tuple) and len(relationship) == 4:
                fk_table, fk_col, pk_table, pk_col = relationship
                if pk_table not in fk_refs:
                    fk_refs[pk_table] = set()
                        
        return fk_refs
    
    def _find_missing_primary_tables(self, fk_refs: Dict[str, Set[str]], 
                                   dataset: RDBDataset) -> List[tuple]:
        """Find which primary key tables are missing and collect their values."""
        missing_tables = []
        
        for target_table in fk_refs:
            if target_table not in dataset.table_names:
                # Find all foreign key values pointing to this missing table
                fk_values = set()
                pk_column = target_table + '_id'  # Convention assumption
                
                # Collect foreign key values from all tables
                for table_name in dataset.table_names:
                    table_df = dataset.get_table(table_name)
                    for col_name in table_df.columns:
                        if (col_name == pk_column or 
                            col_name.endswith(f'_{target_table}_id') or
                            col_name.startswith(f'{target_table}_')):
                            # This looks like a foreign key to the missing table
                            unique_values = table_df[col_name].dropna().unique()
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
