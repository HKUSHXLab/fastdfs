import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union
from loguru import logger

from ..dataset.rdb import RDB
from ..dataset.meta import RDBColumnDType
from ..api import create_rdb

from . import dbinfer_bench as dbb

class DBInferAdapter:
    """Adapter for converting DBInfer Benchmark datasets to FastDFS format."""

    def __init__(self, dataset_name: str, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the DBInfer adapter.

        Args:
            dataset_name: Name of the DBInfer dataset (e.g., "diginetica").
            output_dir: Optional directory path to save the adapted RDB.
                        If provided, the RDB will be saved to this directory after loading.
        """
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir) if output_dir else None
        self.dataset = None

    @staticmethod
    def _safe_convert_float_id(x):
        """
        Convert float with NaN to string (handling integer conversion)
        e.g. 123.0 -> "123", NaN -> None
        """
        if pd.isna(x):
            return None 
        try:
            # strict check: if it's not actually an integer, don't truncate
            if float(x).is_integer():
                return str(int(x))
            else:
                return str(x)
        except (ValueError, TypeError):
            return str(x)

    def _apply_dataset_specific_fixes(self, tables: dict[str, pd.DataFrame]):
        """
        Apply dataset-specific fixes for known issues in raw data.
        """
        # Config: dataset_name -> list of (table_name, column_name)
        float_fix_config = {
            "diginetica": [
                ("View", "userId"),
                ("Purchase", "userId"),
                ("Query", "userId")
            ],
            "retailrocket": [
                ("Category", "parentid")
            ]
        }

        if self.dataset_name in float_fix_config:
            for table_name, col_name in float_fix_config[self.dataset_name]:
                if table_name in tables:
                    df = tables[table_name]
                    if col_name in df.columns:
                        df[col_name] = df[col_name].apply(self._safe_convert_float_id).astype(object)
                        logger.info(f"Fixed {table_name}.{col_name} in {self.dataset_name}: converted float to string IDs.")

    def load(self) -> RDB:
        """
        Load the dataset from DBInfer and convert it to RDB.

        Returns:
            RDB: The loaded RDB object.
        """
        logger.info(f"Loading DBInfer dataset: {self.dataset_name}")
        if self.dataset is None:
            self.dataset = dbb.load_rdb_data(self.dataset_name)
        
        tables = {}
        type_hints = {}
        primary_keys = {}
        foreign_keys = [] # List of (child_table, child_col, parent_table, parent_col)
        time_columns = {}

        # Process metadata
        # dataset.metadata.tables is a list of table metadata objects
        for table_meta in self.dataset.metadata.tables:
            table_name = table_meta.name

            # Get metadata column names for filtering
            metadata_columns = [col.name for col in table_meta.columns]

            # Convert numpy dict to DataFrame
            if table_name in self.dataset.tables:
                # dataset.tables[table_name] is a dict of numpy arrays with ALL columns
                # Filter to only include columns specified in metadata
                table_data = self.dataset.tables[table_name]
                filtered_data = {col_name: table_data[col_name]
                                 for col_name in metadata_columns
                                 if col_name in table_data}

                # Check if any metadata columns are missing
                missing_cols = set(metadata_columns) - set(filtered_data.keys())
                if missing_cols:
                    logger.warning(f"Table {table_name}: columns {missing_cols} in metadata but not in data")

                df = pd.DataFrame(filtered_data)
                tables[table_name] = df
            else:
                logger.warning(f"Table {table_name} found in metadata but not in data.")
                continue

            # Process columns
            table_hints = {}
            for col in table_meta.columns:
                col_name = col.name
                dtype_str = col.dtype
                
                # Map DBInfer types to RDBColumnDType
                if dtype_str == "primary_key":
                    primary_keys[table_name] = col_name
                elif dtype_str == "foreign_key":
                    if hasattr(col, 'link_to') and col.link_to:
                        try:
                            parent_table, parent_col = col.link_to.split('.')
                            foreign_keys.append((table_name, col_name, parent_table, parent_col))
                        except ValueError:
                            logger.warning(f"Invalid link_to format: {col.link_to} in table {table_name}")
                elif dtype_str == "category":
                    table_hints[col_name] = RDBColumnDType.category_t
                elif dtype_str == "float":
                    table_hints[col_name] = RDBColumnDType.float_t
                elif dtype_str == "datetime":
                    table_hints[col_name] = RDBColumnDType.datetime_t
                elif dtype_str == "text":
                    table_hints[col_name] = RDBColumnDType.text_t
                
            if table_hints:
                type_hints[table_name] = table_hints
            
            if table_meta.time_column:
                time_columns[table_name] = table_meta.time_column

        # Apply dataset specific fixes
        self._apply_dataset_specific_fixes(tables)

        # Create RDB
        rdb = create_rdb(
            name=self.dataset_name,
            tables=tables,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            time_columns=time_columns,
            type_hints=type_hints
        )

        if self.output_dir:
            logger.info(f"Saving RDB to {self.output_dir}")
            rdb.save(self.output_dir)

        return rdb
