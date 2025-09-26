"""
- [ ] enhance expressiveness of the returned file structure.
"""

import json
import math
import os
import pathlib

import pandas as pd

from ..config import ToolkitConfig
from ..utils import SimplifiedAsyncOpenAI, get_logger
from .base import TOOL_PROMPTS, AsyncBaseToolkit, register_tool

logger = get_logger(__name__)


class TabularDataToolkit(AsyncBaseToolkit):
    def __init__(self, config: ToolkitConfig = None):
        super().__init__(config)
        self.llm = SimplifiedAsyncOpenAI(
            **self.config.config_llm.model_provider.model_dump() if self.config.config_llm else {}
        )

    def get_tabular_columns(self, file_path: str, return_feat: list[str] = None) -> str:
        logger.info(f"[tool] get_tabular_columns: {file_path}")
        if not os.path.exists(file_path):
            return self._stringify_column_info([{"error": f"File '{file_path}' does not exist."}])

        try:
            # 1. Load the tabular data using the helper function
            df = self._load_tabular_data(file_path)
            # 2. Build column information
            column_info = []
            for col in df.columns:
                try:
                    # Get data type
                    dtype = str(df[col].dtype)

                    # Get a non-null sample value
                    sample_value = None
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        # Get the first non-null value as sample
                        sample_value = non_null_values.iloc[0]
                        # Convert to string, handling different data types
                        if pd.isna(sample_value):
                            sample_str = "NaN"
                        elif isinstance(sample_value, float):
                            if math.isnan(sample_value):
                                sample_str = "NaN"
                            else:
                                sample_str = str(sample_value)
                        else:
                            sample_str = str(sample_value)
                    else:
                        sample_str = "No data"

                    column_info.append({"column_name": str(col), "type": dtype, "sample": sample_str})

                except Exception as e:  # pylint: disable=broad-except
                    logger.warning(f"Error processing column '{col}': {e}")
                    column_info.append({"column_name": str(col), "type": "unknown", "sample": "Error reading sample"})

            return self._stringify_column_info(column_info, return_feat=return_feat)

        except Exception as e:  # pylint: disable=broad-except
            error_msg = f"Error reading file '{file_path}': {str(e)}"
            logger.error(error_msg)
            return self._stringify_column_info([{"error": error_msg}], return_feat=return_feat)

    @register_tool
    async def get_column_info(self, file_path: str) -> str:
        """Get basic column information from a tabular data file (e.g. csv, xlsx).

        Args:
            file_path (str): Path to the tabular data file.

        Returns:
            str: Basic column information including column name, type, and sample value.
        """
        column_info_str = self.get_tabular_columns(file_path)
        prompt = TOOL_PROMPTS["tabular_column_info"].format(column_info=column_info_str)
        logger.info(f"[tool] get_column_info: {file_path}")

        response = await self.llm.query_one(
            messages=[{"role": "user", "content": prompt}],
            # **self.config.config_llm.model_params.model_dump()
        )
        return response

    def _load_tabular_data(self, file_path: str) -> pd.DataFrame:
        # Get file extension to determine how to read the file
        file_ext = pathlib.Path(file_path).suffix.lower()

        # Read the file based on its extension
        if file_ext == ".csv":
            # Try different encodings for CSV files
            encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                raise Exception("Could not read CSV file with any supported encoding")
        elif file_ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        elif file_ext == ".json":
            # Try to read JSON as tabular data
            df = pd.read_json(file_path)
        elif file_ext == ".parquet":
            df = pd.read_parquet(file_path)
        elif file_ext == ".tsv":
            # Tab-separated values
            encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, sep="\t", encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                raise Exception("Could not read TSV file with any supported encoding")
        else:
            # Try to read as CSV by default
            try:
                df = pd.read_csv(file_path)
            except Exception as e:  # pylint: disable=broad-except
                raise Exception(f"Unsupported file format: {file_ext}") from e

        return df

    def _stringify_column_info(self, column_info: list[dict], return_feat: list[str] = None) -> str:
        """Convert column information to a formatted string."""
        if "error" in column_info[0]:
            return column_info[0]["error"]

        lines = []
        return_keys = ["column_name", "type", "sample"]
        if return_feat:
            return_keys = [key for key in return_keys if key in return_feat]
        for i, col in enumerate(column_info):
            lines.append(
                f"- Column {i + 1}: {json.dumps({k: col[k] for k in return_keys if k in col}, ensure_ascii=False)}"
            )
        return "\n".join(lines)
