import phoenix as px
import pandas as pd
import numpy as np

from dotenv import load_dotenv
import os

load_dotenv()

DEFAULT_SCHEMA = px.Schema(
        actual_label_column_name="label",
        prediction_label_column_name="predicted_label",
        embedding_feature_column_names={
                "text_embedding": px.EmbeddingColumnNames(
                vector_column_name="content_vector", raw_data_column_name="content",
                ),
        }
)

def create_dataset(name: str,
                   dataframe: pd.DataFrame, 
                   embedding_vectors: list[np.array], 
                   predicted_labels: np.array):
    dataframe['content_vector'] = embedding_vectors
    dataframe['predicted_label'] = predicted_labels

    return px.Dataset(dataframe=dataframe, schema=DEFAULT_SCHEMA, name=name)

def launch_px(primary_dataset: px.Dataset, reference_dataset: px.Dataset):
    return px.launch_app(primary=primary_dataset, 
                         reference=reference_dataset,
                         host=os.getenv('PHOENIX_HOST'),
                         port=os.getenv('PHOENIX_PORT'))