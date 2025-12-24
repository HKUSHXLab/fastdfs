import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple, Any
from loguru import logger

from ..dataset.meta import (
    RDBColumnDType,
    RDBTableSchema,
    RDBColumnSchema,
    RDBTableDataFormat,
)
from ..dataset.rdb import RDB
from .base import RDBTransform

class InferSchemaTransform(RDBTransform):
    """
    Transform that infers RDBColumnDType from table data types.
    
    It fills in missing `dtype` in RDBColumnSchema based on pandas dtypes
    and provided hints (primary keys, foreign keys, time columns).
    """
    
    def __init__(
        self,
        primary_keys: Optional[Dict[str, str]] = None,
        foreign_keys: Optional[List[Tuple[str, str, str, str]]] = None,
        time_columns: Optional[Dict[str, str]] = None,
        type_hints: Optional[Dict[str, Dict[str, str]]] = None,
        category_threshold: int = 10
    ):
        """
        Args:
            primary_keys: Dict mapping table_name -> primary_key_column
            foreign_keys: List of (child_table, child_col, parent_table, parent_col)
            time_columns: Dict mapping table_name -> time_column
            type_hints: Dict mapping table_name -> {col_name: dtype_str}
            category_threshold: Threshold for unique values to consider a column as category.
        """
        self.primary_keys = primary_keys or {}
        self.foreign_keys = foreign_keys or []
        self.time_columns = time_columns or {}
        self.type_hints = type_hints or {}
        self.category_threshold = category_threshold

    def __call__(self, rdb: RDB) -> RDB:
        new_tables = rdb.tables.copy()
        new_table_schemas = {}
        
        # Build FK map for easier lookup: (table, col) -> (parent_table, parent_col)
        fk_map = {}
        for child_table, child_col, parent_table, parent_col in self.foreign_keys:
            fk_map[(child_table, child_col)] = (parent_table, parent_col)
            
        for table_name, df in new_tables.items():
            # Get existing schema if available, or create a skeleton
            try:
                existing_schema = rdb.get_table_metadata(table_name)
                # Create a map of existing columns
                columns_map = {col.name: col for col in existing_schema.columns}
            except ValueError:
                # Table might not have metadata yet if we are building it
                columns_map = {}
            
            new_columns = []
            has_pk = False
            has_fk = False
            
            # Determine PK and Time Col for this table
            pk_col = self.primary_keys.get(table_name)
            time_col = self.time_columns.get(table_name)
            
            # If not provided in args, check if they exist in current metadata
            if not pk_col and table_name in rdb.table_names:
                try:
                    meta = rdb.get_table_metadata(table_name)
                    for col in meta.columns:
                        if col.dtype == RDBColumnDType.primary_key:
                            pk_col = col.name
                            break
                except ValueError:
                    pass

            if not time_col and table_name in rdb.table_names:
                try:
                    meta = rdb.get_table_metadata(table_name)
                    time_col = meta.time_column
                except ValueError:
                    pass

            for col_name in df.columns:
                # Skip unnamed index columns often created by pandas
                if "Unnamed" in str(col_name):
                    continue
                    
                # Get existing column schema or create new one
                # We copy it to avoid modifying the original if it exists
                if col_name in columns_map:
                    col_schema = columns_map[col_name].model_copy()
                else:
                    col_schema = RDBColumnSchema(name=col_name)
                
                # If dtype is missing, infer it
                if col_schema.dtype is None:
                    inferred_dtype = self._infer_dtype(
                        df, col_name, table_name, pk_col, time_col, fk_map
                    )
                    col_schema.dtype = inferred_dtype
                    
                    # Add link_to for foreign keys
                    if inferred_dtype == RDBColumnDType.foreign_key:
                        parent_table, parent_col = fk_map.get((table_name, col_name))
                        # RDBColumnSchema allows extra fields
                        # We need to construct a new one or use setattr if it wasn't in init
                        # Since we are using Pydantic, we can pass it to constructor if creating new
                        # Or use __dict__ update if modifying?
                        # Safest is to create a new object with all fields
                        
                        # But wait, we already have col_schema.
                        # Let's just set the attribute. Pydantic with extra='allow' should handle it?
                        # Actually, for Pydantic v1/v2 compatibility, it's better to be explicit.
                        # But here we just set it.
                        setattr(col_schema, 'link_to', f"{parent_table}.{parent_col}")

                if col_schema.dtype == RDBColumnDType.primary_key:
                    has_pk = True
                if col_schema.dtype == RDBColumnDType.foreign_key:
                    has_fk = True
                    
                new_columns.append(col_schema)
            
            # Validation
            if not has_pk and not has_fk:
                logger.warning(f"Table {table_name} has no primary key or foreign key columns.")
                
            new_table_schemas[table_name] = RDBTableSchema(
                name=table_name,
                source=f"{table_name}.parquet", # Default source
                format=RDBTableDataFormat.PARQUET, # Default format
                columns=new_columns,
                time_column=time_col
            )
            
        return rdb.create_new_with_tables_and_metadata(new_tables, new_table_schemas)

    def _infer_dtype(self, df, col_name, table_name, pk_col, time_col, fk_map):
        # Check hints first
        if table_name in self.type_hints and col_name in self.type_hints[table_name]:
            return RDBColumnDType(self.type_hints[table_name][col_name])
            
        # Check PK
        if col_name == pk_col:
            return RDBColumnDType.primary_key
            
        # Check Time Col
        if col_name == time_col:
            return RDBColumnDType.datetime_t
            
        # Check FK
        if (table_name, col_name) in fk_map:
            return RDBColumnDType.foreign_key
            
        # Infer from data
        col_data = df[col_name]
        
        if pd.api.types.is_datetime64_any_dtype(col_data):
            return RDBColumnDType.datetime_t
            
        if pd.api.types.is_float_dtype(col_data):
            return RDBColumnDType.float_t
            
        if pd.api.types.is_integer_dtype(col_data) or pd.api.types.is_bool_dtype(col_data):
            # Treat integers as float for feature engineering usually
            return RDBColumnDType.float_t
            
        if pd.api.types.is_object_dtype(col_data) or pd.api.types.is_string_dtype(col_data):
            try:
                n_unique = col_data.nunique()
                if n_unique < self.category_threshold: # Threshold
                    return RDBColumnDType.category_t
                else:
                    return RDBColumnDType.text_t
            except TypeError:
                return RDBColumnDType.text_t
                
        return RDBColumnDType.text_t
