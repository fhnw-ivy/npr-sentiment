import logging
import os

import pandas as pd
import typer
from datasets import load_dataset
from sklearn.model_selection import train_test_split

app = typer.Typer()

TEXT_COL = "content"
LABEL_COL = "label"
TITLE_COL = "title"

logger = logging.getLogger(__name__)

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


def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')


@app.command()
def prepare_dataset(
        dev_set_fraction: float = typer.Option(0.005,
                                               help="Fraction of the full dataset to be used as the development set"),
        val_set_fraction: float = typer.Option(0.005,
                                               help="Fraction of the test dataset to be used as the validation set"),
        labelled_fraction: float = typer.Option(1 / 6, help="Fraction of the development set that should be labelled"),
        max_text_length: int = typer.Option(None, help="Maximum length of text entries"),
        output_dir: str = typer.Option("./data", help="Directory to save the parquet files"),
        seed: int = typer.Option(1337, help="Seed for random operations for reproducibility"),
        verbose: bool = typer.Option(False, help="Enable verbose logging")
):
    """Prepares labelled, unlabelled, and validation datasets for sentiment analysis and saves them as parquet files."""
    setup_logging(verbose)

    logger.info("Loading Amazon Polarity dataset from Hugging Face Datasets")
    dataset = load_dataset("amazon_polarity")
    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()

    logger.debug(f"Train dataset loaded. Shape: {train_df.shape}")
    logger.debug(f"Test dataset loaded. Shape: {test_df.shape}")

    train_df[TEXT_COL] = train_df[TITLE_COL] + ". " + train_df[TEXT_COL]
    test_df[TEXT_COL] = test_df[TITLE_COL] + ". " + test_df[TEXT_COL]
    train_df.drop(columns=TITLE_COL, inplace=True)
    test_df.drop(columns=TITLE_COL, inplace=True)

    dev_df, _ = train_test_split(train_df, test_size=(1 - dev_set_fraction), stratify=train_df[LABEL_COL],
                                 random_state=seed)
    val_df, _ = train_test_split(test_df, test_size=(1 - val_set_fraction), stratify=test_df[LABEL_COL],
                                 random_state=seed)

    labelled_df, unlabelled_df = train_test_split(dev_df, test_size=(1 - labelled_fraction), stratify=dev_df[LABEL_COL],
                                                  random_state=seed)

    logger.debug(f"Total development set size: {len(dev_df)}, Fraction {dev_set_fraction}")
    logger.debug(f"Total validation set size: {len(val_df)}, Fraction {val_set_fraction}")
    logger.debug(f"Total labelled set size: {len(labelled_df)}, Fraction {labelled_fraction}")
    logger.debug(f"Total unlabelled set size: {len(unlabelled_df)}")

    logger.debug("Label distribution in development set: %s", dev_df[LABEL_COL].value_counts(normalize=True))
    logger.debug("Label distribution in validation set: %s", val_df[LABEL_COL].value_counts(normalize=True))
    logger.debug("Label distribution in labelled set: %s", labelled_df[LABEL_COL].value_counts(normalize=True))
    logger.debug("Label distribution in unlabelled set: %s", unlabelled_df[LABEL_COL].value_counts(normalize=True))

    if max_text_length:
        logger.info(f"Truncating text entries to {max_text_length} characters")
        labelled_df[TEXT_COL] = labelled_df[TEXT_COL].apply(lambda x: x[:max_text_length])
        unlabelled_df[TEXT_COL] = unlabelled_df[TEXT_COL].apply(lambda x: x[:max_text_length])
        val_df[TEXT_COL] = val_df[TEXT_COL].apply(lambda x: x[:max_text_length])

    os.makedirs(output_dir, exist_ok=True)
    labelled_df.to_parquet(f"{output_dir}/labelled_dev.parquet")
    unlabelled_df.to_parquet(f"{output_dir}/unlabelled_dev.parquet")
    val_df.to_parquet(f"{output_dir}/validation_set.parquet")

    logger.info(f"Data sets saved: labelled_dev, unlabelled_dev, and validation_set at {output_dir}")


if __name__ == "__main__":
    app()
