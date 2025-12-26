from typing import Tuple, Dict, Optional, List
from enum import Enum
import pydantic
from pydantic import BaseModel, ConfigDict

__all__ = [
    "RDBColumnDType",
    "RDBColumnSchema",
    "RDBTableDataFormat",
    "RDBTableSchema",
    "RDBMeta",
    "RDBDatasetMeta",
]

class RDBColumnDType(str, Enum):
    """Column data type model."""
    float_t = 'float'            # np.float32
    category_t = 'category'      # str
    datetime_t = 'datetime'      # np.datetime64
    text_t = 'text'              # str
    timestamp_t = 'timestamp'    # pandas.Int64 to allow NaN
    foreign_key = 'foreign_key'  # str
    primary_key = 'primary_key'  # str

class RDBCutoffTime(str, Enum):
    """Column data type model."""
    column_name = "__time_for_cutoff__"
    table_name = "__cutoff_time__"

class RDBColumnSchema(BaseModel):
    """Column schema model.

    Column schema allows extra fields other than the explicitly defined members.
    See `DTYPE_EXTRA_FIELDS` dictionary for more details.
    """
    model_config = ConfigDict(extra='allow', use_enum_values=True)

    # Column name.
    name : str
    # Column data type
    dtype : Optional[RDBColumnDType] = None

class RDBTableDataFormat(str, Enum):
    PARQUET = 'parquet'
    NUMPY = 'numpy'

class RDBTableSchema(BaseModel):
    """Table schema definition."""
    model_config = ConfigDict(use_enum_values=True)
    
    name: str
    source: str
    format: RDBTableDataFormat
    columns: List[RDBColumnSchema]
    time_column: Optional[str] = None
    
    @property
    def column_dict(self) -> Dict[str, RDBColumnSchema]:
        """Get column schemas in a dictionary where the keys are column names."""
        return {col_schema.name: col_schema for col_schema in self.columns}

    @property
    def primary_key(self) -> Optional[str]:
        """Get the name of the primary key column."""
        for col in self.columns:
            if col.dtype == RDBColumnDType.primary_key:
                return col.name
        return None


class RDBMeta(BaseModel):
    """Relational Database metadata."""
    model_config = ConfigDict(use_enum_values=True)
    
    name: str  # Name of the RDB
    tables: List[RDBTableSchema]
    
    @property
    def relationships(self) -> List[Tuple[str, str, str, str]]:
        """Get relationships as (child_table, child_col, parent_table, parent_col)."""
        relationships = []
        for table in self.tables:
            for col in table.columns:
                if col.dtype == RDBColumnDType.foreign_key:
                    parent_table, parent_col = col.link_to.split('.')
                    relationships.append((
                        table.name,    # child table
                        col.name,      # child column
                        parent_table,  # parent table  
                        parent_col     # parent column
                    ))
        return relationships

RDBDatasetMeta = RDBMeta  # Alias for backward compatibility