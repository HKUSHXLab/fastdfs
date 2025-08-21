from typing import Tuple, Dict, Optional, List
from enum import Enum
import pydantic
from dataclasses import dataclass

__all__ = [
    "DBBColumnDType",
    "DBBColumnSchema",
    "DBBTableDataFormat",
    "RDBTableSchema",
    "RDBDatasetMeta"
]

class DBBColumnDType(str, Enum):
    """Column data type model."""
    float_t = 'float'            # np.float32
    category_t = 'category'      # object
    datetime_t = 'datetime'      # np.datetime64
    text_t = 'text'              # str
    timestamp_t = 'timestamp'    # np.int64
    foreign_key = 'foreign_key'  # object
    primary_key = 'primary_key'  # object

class DBBColumnSchema(pydantic.BaseModel):
    """Column schema model.

    Column schema allows extra fields other than the explicitly defined members.
    See `DTYPE_EXTRA_FIELDS` dictionary for more details.
    """
    class Config:
        extra = pydantic.Extra.allow
        use_enum_values = True

    # Column name.
    name : str
    # Column data type
    dtype : DBBColumnDType

class DBBTableDataFormat(str, Enum):
    PARQUET = 'parquet'
    NUMPY = 'numpy'

@dataclass
class RDBTableSchema:
    """Simplified table schema without task-specific metadata."""
    name: str
    source: str
    format: DBBTableDataFormat
    columns: List[DBBColumnSchema]
    time_column: Optional[str] = None
    
    @property
    def column_dict(self) -> Dict[str, DBBColumnSchema]:
        """Get column schemas in a dictionary where the keys are column names."""
        return {col_schema.name: col_schema for col_schema in self.columns}


@dataclass  
class RDBDatasetMeta:
    """Simplified dataset metadata without tasks."""
    dataset_name: str
    tables: List[RDBTableSchema]
    
    @property
    def relationships(self) -> List[Tuple[str, str, str, str]]:
        """Get relationships as (child_table, child_col, parent_table, parent_col)."""
        relationships = []
        for table in self.tables:
            for col in table.columns:
                if col.dtype == DBBColumnDType.foreign_key:
                    parent_table, parent_col = col.link_to.split('.')
                    relationships.append((
                        table.name,    # child table
                        col.name,      # child column
                        parent_table,  # parent table  
                        parent_col     # parent column
                    ))
        return relationships


