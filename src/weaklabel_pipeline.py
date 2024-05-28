import logging
import os

import pandas as pd
import typer
from dotenv import load_dotenv
from joblib import load
from sentence_transformers import SentenceTransformer

load_dotenv()

app = typer.Typer()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_filename):
    """Load the model from the specified path."""
    models_folder = os.getenv('MODELS_DIR')
    if not models_folder:
        logger.error("MODELS_DIR environment variable not set.")
        raise ValueError("MODELS_DIR environment variable not set.")

    model_path = os.path.join(models_folder, model_filename)
    if not os.path.exists(model_path):
        logger.error(f"No such file or directory: '{model_path}'")
        raise FileNotFoundError(f"No such file or directory: '{model_path}'")

    return load(model_path)


def load_transformer():
    """Load the SentenceTransformer model based on environment settings."""
    transformer_model_name = os.getenv('ST_EMBEDDING_MODEL_NAME')
    if not transformer_model_name:
        logger.error("ST_EMBEDDING_MODEL_NAME environment variable not set.")
        raise ValueError("ST_EMBEDDING_MODEL_NAME environment variable not set.")

    return SentenceTransformer(transformer_model_name)


def ensure_data_directory():
    """Ensure the data directory exists or create it if it does not."""
    data_folder_path = os.getenv('DATA_DIR')
    weak_labelled_path = os.path.join(data_folder_path, 'weak_labelled')
    if not os.path.exists(weak_labelled_path):
        os.makedirs(weak_labelled_path)
    return weak_labelled_path


def predict(df, model, transformer, column_to_embed='content'):
    """Generate predictions using the provided model and transformer."""
    embeddings = transformer.encode(df[column_to_embed].tolist(), show_progress_bar=True)
    df['label'] = model.predict(embeddings)
    df['embedding_vec'] = embeddings.tolist()
    return df


def save_results(df, output_filename):
    """Save the results to a specified file in the data directory."""
    data_folder_path = ensure_data_directory()
    output_path = os.path.join(data_folder_path, output_filename)
    logger.info(f"Saving results to: {output_path}")
    df.to_parquet(output_path, index=False)


@app.command()
def run_pipeline(model_filename: str, unlabelled_parquet_file_path: str, verbose: bool = False):
    """Main command to execute the pipeline."""
    if verbose:
        logger.setLevel(logging.DEBUG)

    logger.debug("Starting pipeline execution...")
    if not os.path.exists(unlabelled_parquet_file_path):
        logger.error(f"No such file or directory: '{unlabelled_parquet_file_path}'")
        raise FileNotFoundError(f"No such file or directory: '{unlabelled_parquet_file_path}'")

    df = pd.read_parquet(unlabelled_parquet_file_path)
    model = load_model(model_filename)
    transformer = load_transformer()

    results = predict(df, model, transformer)

    model_base_name = os.path.splitext(os.path.basename(model_filename))[0]
    output_filename = f"{model_base_name}_weaklabels.parquet"
    save_results(results, output_filename)

    logger.info("Pipeline finished successfully!")


if __name__ == '__main__':
    app()
