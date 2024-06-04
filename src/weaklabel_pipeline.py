import logging
import os

import pandas as pd
import typer
from dotenv import load_dotenv
from joblib import load
from sentence_transformers import SentenceTransformer
from torch import nn

load_dotenv()

app = typer.Typer()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_models(model_filename, transformers: list[str]):
    """Load the model from the specified path."""
    models_folder = os.getenv('MODELS_DIR')
    if not models_folder:
        logger.error("MODELS_DIR environment variable not set.")
        raise ValueError("MODELS_DIR environment variable not set.")

    model_paths = [os.path.join(models_folder, 'weak_labelling', transformer_name, model_filename) for transformer_name in transformers]
    for model_path in model_paths:
        if not os.path.exists(model_path):
            logger.error(f"No such file or directory: '{model_path}'")
            raise FileNotFoundError(f"No such file or directory: '{model_path}'")
    
    models_dict = {transformer_name: load(model_path) for transformer_name, model_path in zip(transformers, model_paths)}

    return models_dict


def load_transformers():
    """Load the SentenceTransformer models based on environment settings."""
    # check if WL_EMBEDDING_MODELS is set
    transformer_names = os.getenv('WL_EMBEDDING_MODELS').split(",")
    if not transformer_names:
        logger.error("WL_EMBEDDING_MODELS environment variable not set.")
        raise ValueError("WL_EMBEDDING_MODELS environment variable not set.")

    transformers = {name: SentenceTransformer(name) for name in transformer_names}

    return transformers


def ensure_data_directory():
    """Ensure the data directory exists or create it if it does not."""
    data_folder_path = os.getenv('DATA_DIR')
    weak_labelled_path = os.path.join(data_folder_path, 'weak_labelled')
    if not os.path.exists(weak_labelled_path):
        os.makedirs(weak_labelled_path)
    return weak_labelled_path


def predict(df, models: dict, transformers: dict[str, SentenceTransformer], column_to_embed='content'):
    """Generate predictions using the provided models and transformers."""
    embeddings_dict = {name: transformer.encode(df[column_to_embed].tolist()) for name, transformer in transformers.items()}
    
    for name, model in models.items():
        embeddings = embeddings_dict[name]
        
        df[f'{name}_label'] = model.predict(embeddings)
        df[f'{name}_embedding_vec'] = embeddings.tolist()
        
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
   
    transformers = load_transformers()
    logger.info(f'Loaded transformer(s): {[name for name, _ in transformers.items()]}')
    
    models = load_models(model_filename, list(transformers.keys()))
    
    results = predict(df, models, transformers)

    model_base_name = os.path.splitext(os.path.basename(model_filename))[0]
    output_filename = f"{model_base_name}_weaklabels.parquet"
    save_results(results, output_filename)

    logger.info("Pipeline finished successfully!")


if __name__ == '__main__':
    app()
