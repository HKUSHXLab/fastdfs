from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
import abc
import pydantic
import numpy as np
from ...dataset import DBBColumnDType

from ...utils.device import DeviceInfo

@dataclass
class ColumnData:
    """
    Represents a single column of data with associated metadata.
    
    This class encapsulates both the data values and metadata for a database column,
    providing a unified interface for column-level operations.
    
    Attributes:
        metadata: Dictionary containing column properties (dtype, constraints, etc.)
        data: NumPy array containing the actual column values
    """
    metadata : Dict
    data : np.ndarray

    def __repr__(self):
        return f"Column(dtype: {str(self.metadata['dtype'])}, len: {len(self.data)})"

@dataclass
class RDBData:
    """
    Represents an in-memory relational database structure.
    
    This class provides a complete representation of a relational database with
    tables, optional column groupings, and relationships between tables.
    
    Attributes:
        tables: Dictionary mapping table names to their column collections
        column_groups: Optional list of column groups spanning multiple tables
        relationships: Optional list of foreign key relationships (fk_table, fk_col, pk_table, pk_col)
    """
    tables : Dict[str, Dict[str, ColumnData]]
    column_groups : Optional[List[List[Tuple[str, str]]]] = None
    relationships : Optional[List[Tuple[str, str, str, str]]] = None

    def __repr__(self):
        ret = "{\n"
        for table_name, table in self.tables.items():
            ret += f"  Table(\n"
            ret += f"    name={table_name}\n"
            ret +=  "    columns={\n"
            for col_name, col in table.items():
                ret += f"      {col_name}: {col}\n"
            ret +=  "    })\n"
        ret += f"  column_groups: {self.column_groups}\n"
        ret += f"  relationships: {self.relationships}\n"
        ret += "}"
        return ret

def is_task_table(table_name : str) -> bool:
    """
    Check if a table name refers to a task-specific table.
    
    Task tables are identified by the '__task__:' prefix in their names.
    
    Args:
        table_name: Name of the table to check
        
    Returns:
        True if the table is a task table, False otherwise
    """
    return table_name.startswith('__task__:')

def make_task_table_name(
    task_name : str,
) -> str:
    """
    Generate a task table name from a task identifier.
    
    Args:
        task_name: Name of the task
        
    Returns:
        Formatted task table name with the '__task__:' prefix
    """
    return f'__task__:{task_name}'

def unmake_task_table_name(
    task_table_name : str
) -> str:
    """
    Extract the task name from a task table name.
    
    Args:
        task_table_name: Full task table name with '__task__:' prefix
        
    Returns:
        The original task name without the prefix
        
    Raises:
        AssertionError: If the table name format is invalid
    """
    parts = task_table_name.split(':')
    assert len(parts) == 2
    return parts[1]

class RDBTransform:
    """
    Abstract base class for RDB-level data transformations.
    
    This class provides the interface for transformations that operate on entire
    RDBData objects, implementing the scikit-learn fit/transform pattern.
    RDBTransforms can modify table structures, add/remove columns, and perform
    cross-table operations.
    
    Attributes:
        config_class: Pydantic model class for configuration validation
        name: String identifier for this transform type
    """

    config_class : pydantic.BaseModel = None
    name : str = "base"

    def __init__(self, config):
        """
        Initialize the transform with configuration.
        
        Args:
            config: Configuration object that should be an instance of config_class
        """
        self.config = config

    @abc.abstractmethod
    def fit(
        self,
        rdb_data : RDBData,
        device : DeviceInfo
    ):
        """
        Learn transformation parameters from the data.
        
        This method should analyze the input data and learn any parameters
        needed for the transformation (e.g., normalization statistics,
        encoding mappings, etc.).
        
        Args:
            rdb_data: Input RDB data to learn from
            device: Device information for hardware-specific optimizations
        """
        pass

    @abc.abstractmethod
    def transform(
        self,
        rdb_data : RDBData,
        device : DeviceInfo
    ) -> RDBData:
        """
        Apply the transformation to the data.
        
        This method applies the learned transformation to the input data,
        returning a new RDBData object with the transformed values.
        
        Args:
            rdb_data: Input RDB data to transform
            device: Device information for hardware-specific optimizations
            
        Returns:
            New RDBData object with transformed data
        """
        pass

    def fit_transform(
        self,
        rdb_data : RDBData,
        device : DeviceInfo
    ) -> RDBData:
        """
        Fit the transform and apply it to the data in one step.
        
        This is a convenience method that calls fit() followed by transform().
        
        Args:
            rdb_data: Input RDB data to fit and transform
            device: Device information for hardware-specific optimizations
            
        Returns:
            New RDBData object with transformed data
        """
        self.fit(rdb_data, device)
        return self.transform(rdb_data, device)

_RDB_TRANSFORM_REGISTRY = {}
def rdb_transform(transform_class):
    """
    Decorator to register an RDB transform class.
    
    This decorator adds the transform class to the global registry, making it
    available for lookup by name through get_rdb_transform_class().
    
    Args:
        transform_class: Class that inherits from RDBTransform
        
    Returns:
        The same class, now registered in the global registry
        
    Example:
        @rdb_transform
        class MyTransform(RDBTransform):
            name = "my_transform"
    """
    global _RDB_TRANSFORM_REGISTRY
    _RDB_TRANSFORM_REGISTRY[transform_class.name] = transform_class
    return transform_class

def get_rdb_transform_class(name : str):
    """
    Retrieve a registered RDB transform class by name.
    
    Args:
        name: Name of the transform class to retrieve
        
    Returns:
        The transform class
        
    Raises:
        RuntimeError: If no transform with the given name is registered
    """
    global _RDB_TRANSFORM_REGISTRY
    transform_class =  _RDB_TRANSFORM_REGISTRY.get(name, None)
    if transform_class is None:
        raise RuntimeError(f"Cannot find RDB transform class {name}.")
    return transform_class

class ColumnTransform:
    """
    Abstract base class for column-level data transformations.
    
    This class provides the interface for transformations that operate on individual
    columns. ColumnTransforms can process columns based on their data types and
    potentially output multiple transformed columns from a single input column.
    
    The class supports type-aware transformations through input_dtype and output_dtypes
    specifications, and provides flexible column naming through output_name_formatters.
    
    Attributes:
        config_class: Pydantic model class for configuration validation
        name: String identifier for this transform type
        input_dtype: Expected input column data type
        output_dtypes: List of output column data types  
        output_name_formatters: List of format strings for generating output column names
    """

    # Config class.
    config_class : pydantic.BaseModel = None
    # Name of this transform.
    name : str = "base"
    # Input column data type.
    input_dtype : DBBColumnDType = None
    # Output column data type.
    output_dtypes : List[DBBColumnDType] = None
    # String formatters to generate new column names.
    output_name_formatters : List[str] = ["{name}"]

    def __init__(self, config):
        """
        Initialize the transform with configuration.
        
        Args:
            config: Configuration object that should be an instance of config_class
        """
        self.config = config

    @abc.abstractmethod
    def fit(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> None:
        """
        Learn transformation parameters from the column data.
        
        This method should analyze the input column and learn any parameters
        needed for the transformation (e.g., min/max values for normalization,
        category mappings, etc.).
        
        Args:
            column: Input column data to learn from
            device: Device information for hardware-specific optimizations
        """
        pass

    @abc.abstractmethod
    def transform(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> List[ColumnData]:
        """
        Apply the transformation to the column data.
        
        This method applies the learned transformation to the input column,
        potentially generating multiple output columns (e.g., one-hot encoding
        or datetime feature extraction).
        
        Args:
            column: Input column data to transform
            device: Device information for hardware-specific optimizations
            
        Returns:
            List of transformed ColumnData objects
        """
        pass

_COLUMN_TRANSFORM_REGISTRY = {}
def column_transform(transform_class):
    """
    Decorator to register a column transform class.
    
    This decorator adds the transform class to the global registry, making it
    available for lookup by name through get_column_transform_class().
    
    Args:
        transform_class: Class that inherits from ColumnTransform
        
    Returns:
        The same class, now registered in the global registry
        
    Example:
        @column_transform
        class MyColumnTransform(ColumnTransform):
            name = "my_column_transform"
    """
    global _COLUMN_TRANSFORM_REGISTRY
    _COLUMN_TRANSFORM_REGISTRY[transform_class.name] = transform_class
    return transform_class

def get_column_transform_class(name : str):
    """
    Retrieve a registered column transform class by name.
    
    Args:
        name: Name of the transform class to retrieve
        
    Returns:
        The transform class
        
    Raises:
        RuntimeError: If no transform with the given name is registered
    """
    global _COLUMN_TRANSFORM_REGISTRY
    transform_class =  _COLUMN_TRANSFORM_REGISTRY.get(name, None)
    if transform_class is None:
        raise RuntimeError(f"Cannot find column transform class {name}.")
    return transform_class
