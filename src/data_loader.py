"""
Dataset loading
===============

Loads the two cleaned BRFSS survey cycles (2020 and 2022), validates
schema consistency between them, and concatenates them into one
working DataFrame. Cleaning is performed outside this module — the
files arriving here are expected to already be schema-aligned and
deduplicated.
"""

from pathlib import Path
from typing import Tuple
import logging

import pandas as pd

from .config import (
    DATA_DIR, DATASET_2020, DATASET_2022, RANDOM_SEED, TARGET_COLUMN
)


logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Loads, validates, and merges the BRFSS 2020 and 2022 cleaned files.

    Usage
    -----
    >>> loader = DatasetLoader()
    >>> df = loader.load_and_merge()
    >>> print(loader.summary())
    """

    def __init__(self, data_dir: Path = DATA_DIR,
                 file_2020: str = DATASET_2020,
                 file_2022: str = DATASET_2022,
                 random_seed: int = RANDOM_SEED) -> None:
        self.data_dir    = Path(data_dir)
        self.file_2020   = file_2020
        self.file_2022   = file_2022
        self.random_seed = random_seed

        # Filled in by load_and_merge()
        self.df_2020:     pd.DataFrame | None = None
        self.df_2022:     pd.DataFrame | None = None
        self.df_merged:   pd.DataFrame | None = None

    # ─── Public API ────────────────────────────────────────────

    def load_and_merge(self) -> pd.DataFrame:
        """
        Load both files, validate schema compatibility, concatenate them,
        and shuffle so the cycles are interleaved rather than block-ordered.

        Returns the merged DataFrame.

        Raises
        ------
        FileNotFoundError
            If either CSV file is missing.
        ValueError
            If the two files do not share an identical column schema.
        """
        path_2020 = self.data_dir / self.file_2020
        path_2022 = self.data_dir / self.file_2022

        # Two separate try blocks so the error message names the missing file
        if not path_2020.exists():
            raise FileNotFoundError(
                f"Cleaned 2020 dataset not found at {path_2020}. "
                f"Place it in the data folder before running."
            )
        if not path_2022.exists():
            raise FileNotFoundError(
                f"Cleaned 2022 dataset not found at {path_2022}. "
                f"Place it in the data folder before running."
            )

        logger.info("Loading %s …", path_2020.name)
        self.df_2020 = pd.read_csv(path_2020)

        logger.info("Loading %s …", path_2022.name)
        self.df_2022 = pd.read_csv(path_2022)

        self._validate_schemas()

        merged = pd.concat([self.df_2020, self.df_2022], ignore_index=True)
        # Shuffle so the two cycles interleave; otherwise stratified CV
        # would always put one cycle on one side of the split
        merged = merged.sample(frac=1, random_state=self.random_seed)\
                       .reset_index(drop=True)

        self.df_merged = merged
        logger.info("Merged dataset: %d rows × %d columns",
                    len(merged), merged.shape[1])
        return merged

    def summary(self) -> str:
        """Human-readable description of the loaded data."""
        if self.df_merged is None:
            return "DatasetLoader: not yet loaded. Call load_and_merge() first."

        positive_rate = (self.df_merged[TARGET_COLUMN] == "Yes").mean()
        return (
            f"Dataset Summary\n"
            f"  2020 cycle    : {len(self.df_2020):,} records\n"
            f"  2022 cycle    : {len(self.df_2022):,} records\n"
            f"  Combined      : {len(self.df_merged):,} records\n"
            f"  Features      : {self.df_merged.shape[1] - 1} "
            f"(plus 1 target column)\n"
            f"  Positive rate : {positive_rate:.2%}\n"
        )

    def get_partition_sizes(self) -> Tuple[int, int]:
        """Return the row counts from each source cycle."""
        if self.df_2020 is None or self.df_2022 is None:
            raise RuntimeError("Datasets not loaded yet.")
        return len(self.df_2020), len(self.df_2022)

    # ─── Private helpers ───────────────────────────────────────

    def _validate_schemas(self) -> None:
        """
        Confirm both source files share the same column schema. They
        were harmonised during the cleaning step that runs outside this
        codebase, but if the user replaces a file with a non-cleaned
        version we want to fail loudly rather than produce silent
        garbage downstream.
        """
        cols_2020 = set(self.df_2020.columns)
        cols_2022 = set(self.df_2022.columns)

        missing_in_2022 = cols_2020 - cols_2022
        missing_in_2020 = cols_2022 - cols_2020

        if missing_in_2022 or missing_in_2020:
            msg_parts = ["Schema mismatch between cycles."]
            if missing_in_2022:
                msg_parts.append(
                    f"  In 2020 but not 2022: {sorted(missing_in_2022)}"
                )
            if missing_in_2020:
                msg_parts.append(
                    f"  In 2022 but not 2020: {sorted(missing_in_2020)}"
                )
            msg_parts.append(
                "  Re-run the harmonisation cleaning step before loading."
            )
            raise ValueError("\n".join(msg_parts))

        if TARGET_COLUMN not in cols_2020:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' missing from the data. "
                f"Did you load the wrong file?"
            )
