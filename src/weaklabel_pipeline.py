import pandas as pd
from sentence_transformers import SentenceTransformer
import joblib
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()


class WeakLabelingPipeline:
    def __init__(self,
                 model_filename,
                 transformer_model_name='all-MiniLM-L6-v2',
                 column_to_embed='content'):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        models_folder = os.getenv('MODELS_DIR')
        if not models_folder:
            raise ValueError("MODELS_DIR environment variable not set")

        models_folder_path = os.path.join(root_dir, models_folder)

        self.model_path = os.path.join(models_folder_path, model_filename)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No such file or directory: '{self.model_path}'")

        self.model = joblib.load(self.model_path)
        self.transformer = SentenceTransformer(transformer_model_name)
        self.column_to_embed = column_to_embed

        self.data_folder_path = os.path.join(root_dir, 'data')
        if not os.path.exists(self.data_folder_path):
            os.makedirs(self.data_folder_path)

    def predict(self):
        embeddings = self.transformer.encode(self.data[self.column_to_embed].tolist(), show_progress_bar=True)

        self.data['label'] = self.model.predict(embeddings)
        return self.data

    def save_results(self, output_filename):
        output_path = os.path.join(self.data_folder_path, output_filename)
        print(f"Saving results to: {output_path}")
        self.data.to_parquet(output_path, index=False)

    def run(self, df, output_filename=None):
        self.data = df
        self.data.rename(columns={'label': 'ground_truth'}, inplace=True)
        self.predict()
        if output_filename:
            self.save_results(output_filename)
        return self.data
