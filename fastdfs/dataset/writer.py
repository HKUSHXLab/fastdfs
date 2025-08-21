from typing import Dict
from pathlib import Path
import numpy as np
import abc
import pandas as pd

from .meta import RDBTableDataFormat

class TableWriter:

    @abc.abstractmethod
    def write(self, path: Path, table_name: str, table_data: Dict[str, np.ndarray]):
        """Write the table data."""
        raise NotImplementedError()

    @abc.abstractmethod
    def filename(self, path: Path, table_name: str) -> Path:
        """Return the on-disk filename that stores the table data."""
        raise NotImplementedError()

class ParquetTableWriter(TableWriter):

    def write(self, path: Path, table_name: str, table_data: Dict[str, np.ndarray]):
        filename = self.filename(path, table_name)
        df = pd.DataFrame(table_data)
        df.to_parquet(filename)

    def filename(self, path: Path, table_name: str) -> Path:
        return path / f"{table_name}.pqt"

class NumpyTableWriter(TableWriter):

    def write(self, path: Path, table_name: str, table_data: Dict[str, np.ndarray]):
        filename = self.filename(path, table_name)
        np.savez(filename, **table_data)

    def filename(self, path: Path, table_name: str) -> Path:
        return path / f"{table_name}.npz"

def get_table_data_writer(format : RDBTableDataFormat) -> TableWriter:
    if format not in WRITER_MAP:
        raise ValueError(f"Unsupported table format: {format}")
    return WRITER_MAP[format]()

WRITER_MAP = {
    RDBTableDataFormat.PARQUET : ParquetTableWriter,
    RDBTableDataFormat.NUMPY : NumpyTableWriter,
}
