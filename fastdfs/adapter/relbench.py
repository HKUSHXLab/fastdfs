import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Union
from loguru import logger

from ..dataset.rdb import RDB
from ..api import create_rdb

try:
    import relbench
    from relbench.datasets import get_dataset
except ImportError:
    relbench = None
    get_dataset = None

class RelBenchAdapter:
    """Adapter for converting RelBench datasets to FastDFS format."""

    def __init__(self, dataset_name: str, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the RelBench adapter.

        Args:
            dataset_name: Name of the RelBench dataset (e.g., "rel-trial", "rel-stack").
            output_dir: Optional directory path to save the adapted RDB. 
                        If provided, the RDB will be saved to this directory after loading.
        """
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
        logger.info("Processing Data Tables...")
        
        tables = {}
        primary_keys = {}
        time_columns = {}
        foreign_keys = []
        
        for name, table in self.db.table_dict.items():
            logger.info(f"Processing table: {name}")
            
            # Work on a copy to avoid modifying the original DB in place
            df = table.df.copy()
            
            # Apply patches
            self._apply_patches(df, name)
            
            # Filter columns (Label Leakage)
            df = self._filter_columns(df, name)
            
            tables[name] = df
            
            if table.pkey_col:
                primary_keys[name] = table.pkey_col
                
            if table.time_col:
                time_columns[name] = table.time_col
                
            for fk_col, parent_table_name in table.fkey_col_to_pkey_table.items():
                if parent_table_name in self.db.table_dict:
                    parent_pk = self.db.table_dict[parent_table_name].pkey_col
                    foreign_keys.append((name, fk_col, parent_table_name, parent_pk))
                else:
                    logger.warning(f"Parent table {parent_table_name} not found for FK {fk_col} in {name}")

        type_hints = self._get_type_hints()

        rdb = create_rdb(
            tables=tables,
            name=self.dataset_name,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            time_columns=time_columns,
            type_hints=type_hints
        )
        
        if self.output_dir:
            logger.info(f"Saving RDB to {self.output_dir}...")
            rdb.save(self.output_dir)
        
        return rdb

    def _apply_patches(self, df: pd.DataFrame, name: str):
        """Apply dataset-specific patches to the dataframe."""
        if self.dataset_name == "rel-f1" and name == "races":
            if "time" in df.columns:
                if not pd.api.types.is_float_dtype(df["time"]):
                     df["time"] = pd.to_timedelta(df["time"]).dt.total_seconds()

    def _filter_columns(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Remove columns that cause label leakage or are unwanted."""
        cols_to_drop = []
        
        if self.dataset_name == "rel-stack" and name == "users":
            cols_to_drop.extend(["ProfileImageUrl", "WebsiteUrl"])
            
        if self.dataset_name == "rel-trial":
            if name == "outcome_analyses":
                cols_to_drop.extend(["ci_upper_limit_raw", "ci_lower_limit_raw", "p_value_raw"])
            if name == "studies":
                cols_to_drop.append("limitations_and_caveats")
        
        if self.dataset_name == "rel-event" and name in["event_attendees", "user_friends"]:
            cols_to_drop.append("Unnamed: 0")
                
        # Only drop columns that actually exist
        existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        if existing_cols_to_drop:
            logger.info(f"Dropping columns from {name}: {existing_cols_to_drop}")
            return df.drop(columns=existing_cols_to_drop)
            
        return df

    def _get_type_hints(self) -> Dict[str, Dict[str, str]]:
        """Get dataset-specific type hints."""
        hints = {}
        
        if self.dataset_name == "rel-trial":
            hints["designs"] = {
                "intervention_model": "category",
                "masking": "category"
            }
            
        return hints
