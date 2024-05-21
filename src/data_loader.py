import os

import pandas as pd
import typer

app = typer.Typer()

LABEL_MAP = {
    1: "positive",
    0: "negative"
}


def load_datasets(data_dir):
    """Load the datasets from the given directory."""
    if not data_dir:
        raise ValueError("Data directory not provided.")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} not found.")

    labelled_dev = pd.read_parquet(f"{data_dir}/labelled_dev.parquet")
    unlabelled_dev = pd.read_parquet(f"{data_dir}/unlabelled_dev.parquet")
    validation_set = pd.read_parquet(f"{data_dir}/validation_set.parquet")
    return labelled_dev, unlabelled_dev, validation_set