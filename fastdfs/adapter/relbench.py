import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple
import os
from loguru import logger

from ..dataset.meta import (
    RDBColumnDType,
    RDBTableDataFormat,
    RDBTableSchema,
    RDBColumnSchema,
    RDBMeta,
)
from ..dataset.rdb import RDB
from ..utils import yaml_utils

try:
    import relbench
    from relbench.datasets import get_dataset
    from relbench.base.task_base import TaskType
    from relbench.tasks import get_task_names, get_task
except ImportError:
    relbench = None

class RelBenchAdapter:
    """Adapter for converting RelBench datasets to FastDFS format."""

    def __init__(self, dataset_name: str, output_dir: Optional[Union[str, Path]] = None):
        if relbench is None:
            raise ImportError("relbench is not installed. Please install it with `pip install relbench`.")
        
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir) if output_dir else None
        self.dataset = get_dataset(name=dataset_name, download=True)
        self.db = self.dataset.get_db()

    def load(self) -> RDB:
        """
        Load the RDB in-memory for FastDFS.
        
        Returns:
            rdb: RDB instance
        """
        # 1. Process Data Tables and create RDB
        logger.info("Processing Data Tables...")
        table_schemas = []
        tables = {}
        
        for name, table in self.db.table_dict.items():
            logger.info(f"Processing table: {name}")
            # Apply patches
            self._apply_patches(table, name)
            
            schema = self._generate_table_schema(table, name)
            if schema:
                table_schemas.append(schema)
                tables[name] = table.df

        # 2. Update Foreign Key Links
        logger.info("Updating Foreign Key Links...")
        table_schemas = self._update_foreign_key_links(table_schemas)
        
        # Create RDB in-memory
        dataset_meta = RDBMeta(
            name=self.dataset_name,
            tables=table_schemas,
        )
        rdb = RDB(metadata=dataset_meta, tables=tables)
        
        return rdb

    def convert(self):
        """Convert the dataset and save it to the output directory."""
        if not self.output_dir:
            raise ValueError("output_dir must be provided for conversion.")
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Process Data Tables
        logger.info("Processing Data Tables...")
        table_schemas = []
        for name, table in self.db.table_dict.items():
            logger.info(f"Processing table: {name}")
            schema = self._generate_table_schema(table, name)
            if schema:
                table_schemas.append(schema)
        
        # 2. Update Foreign Key Links
        logger.info("Updating Foreign Key Links...")
        table_schemas = self._update_foreign_key_links(table_schemas)
        
        # 3. Save Data Tables
        logger.info("Saving Data Tables...")
        for table_name, table in self.db.table_dict.items():
            output_path = self.output_dir / f"{table_name}.parquet"
            # Ensure the table schema exists (it might have been skipped)
            if any(s.name == table_name for s in table_schemas):
                self._apply_patches(table, table_name)
                table.df.to_parquet(output_path)
                logger.info(f"Saved table: {table_name}")

        # 4. Create Metadata
        logger.info("Creating RDB Metadata...")
        dataset_meta = RDBMeta(
            name=self.dataset_name,
            tables=table_schemas,
        )
        
        yaml_utils.save_yaml(dataset_meta.model_dump(mode='json'), self.output_dir / "metadata.yaml")
        logger.info("Conversion Complete.")

    def _apply_patches(self, table, name):
        """Apply dataset-specific patches to the dataframe."""
        if self.dataset_name == "rel-f1" and name == "races":
            if "time" in table.df.columns:
                # Check if it's already converted (in case of multiple calls)
                if not pd.api.types.is_float_dtype(table.df["time"]):
                     table.df["time"] = pd.to_timedelta(table.df["time"]).dt.total_seconds()

    def _generate_column_schema(self, column: str, table, name: str) -> Optional[RDBColumnSchema]:
        # Patches
        if self.dataset_name == "rel-f1" and name == "races":
            if column == "time":
                pass # It becomes float/int
        if self.dataset_name == "rel-trial" and name == "designs":
            if column == "intervention_model" or column == "masking":
                return RDBColumnSchema(name=column, dtype=RDBColumnDType.category_t)
        if self.dataset_name == "rel-stack" and name == "users":
            if column == "ProfileImageUrl" or column == "WebsiteUrl":
                return None
        if self.dataset_name == "rel-trial" and name == "outcome_analyses":
            if column in ["ci_upper_limit_raw", "ci_lower_limit_raw", "p_value_raw"]:
                return None
        if self.dataset_name == "rel-trial" and name == "studies":
            if column == "limitations_and_caveats":
                return None

        dtype = None
        link_to = None

        # Primary Key
        if column == table.pkey_col:
            dtype = RDBColumnDType.primary_key
        
        # Timestamp
        elif column == table.time_col:
            dtype = RDBColumnDType.datetime_t
        
        # Foreign Key
        elif column in table.fkey_col_to_pkey_table:
            dtype = RDBColumnDType.foreign_key
            link_to = f"{table.fkey_col_to_pkey_table[column]}.{column}"
        
        # Other types
        elif table.df[column].dtype == "datetime64[ns]":
            dtype = RDBColumnDType.datetime_t
        elif (
            table.df[column].dtype == float
            or table.df[column].dtype == np.float32
            or table.df[column].dtype == np.float64
        ):
            dtype = RDBColumnDType.float_t
        elif (
            table.df[column].dtype == int
            or table.df[column].dtype == np.int32
            or table.df[column].dtype == np.int64
            or table.df[column].dtype == bool
        ):
            dtype = RDBColumnDType.float_t
        elif table.df[column].dtype == object:
            try:
                n_unique = table.df[column].nunique()
                if n_unique < 4: # Threshold from user script
                    dtype = RDBColumnDType.category_t
                else:
                    dtype = RDBColumnDType.text_t
            except TypeError:
                dtype = RDBColumnDType.text_t
        else:
            dtype = RDBColumnDType.text_t
            logger.warning(f"Unknown type for column {column} in table {name}. Treating as text.")

        schema = RDBColumnSchema(name=column, dtype=dtype)
        if dtype == RDBColumnDType.foreign_key and link_to:
            setattr(schema, 'link_to', link_to)
            schema = RDBColumnSchema(name=column, dtype=dtype, link_to=link_to)
        
        return schema

    def _generate_table_schema(self, table, name: str) -> Optional[RDBTableSchema]:
        column_schemas = []
        
        # Note: _apply_patches is called before this in load/convert
        
        for column in table.df.columns:
            if column == "Unnamed: 0":
                continue
            
            col_schema = self._generate_column_schema(column, table, name)
            if col_schema:
                column_schemas.append(col_schema)
        
        if not column_schemas:
            return None

        return RDBTableSchema(
            name=name,
            columns=column_schemas,
            time_column=table.time_col if table.time_col else None,
            format=RDBTableDataFormat.PARQUET, # Default to parquet for schema, though in-memory it's DF
            source=f"{name}.parquet", # Placeholder for in-memory
        )

    def _update_foreign_key_links(self, table_schemas: List[RDBTableSchema]) -> List[RDBTableSchema]:
        def find_pkey_col(table_name):
            for table_schema in table_schemas:
                if table_schema.name == table_name:
                    for column_schema in table_schema.columns:
                        if column_schema.dtype == RDBColumnDType.primary_key:
                            return column_schema.name
            return None

        for table_schema in table_schemas:
            for column_schema in table_schema.columns:
                if column_schema.dtype == RDBColumnDType.foreign_key:
                    if hasattr(column_schema, 'link_to'):
                        target_table_name = column_schema.link_to.split(".")[0]
                        target_table_pkey_col = find_pkey_col(target_table_name)
                        if target_table_pkey_col:
                            new_link_to = f"{target_table_name}.{target_table_pkey_col}"
                            if new_link_to != column_schema.link_to:
                                logger.info(f"Updating link_to from {column_schema.link_to} to {new_link_to}")
                                column_schema.link_to = new_link_to
                        else:
                            logger.warning(f"Could not find primary key for table {target_table_name}")
        return table_schemas
