import os

import numpy as np
import pandas as pd
import phoenix as px
from dotenv import load_dotenv

load_dotenv()

DEFAULT_SCHEMA = px.Schema(
        actual_label_column_name="label",
        embedding_feature_column_names={
                "text_embedding": px.EmbeddingColumnNames(
                vector_column_name="embedding", 
                raw_data_column_name="content",
                ),
        }
)


def create_dataset(name: str,
                   dataframe: pd.DataFrame,
                   embedding_vectors: list[np.array],
                   predicted_labels: np.array = None,
                   content: np.array = None):
    if content is not None:
        dataframe['content'] = content
    dataframe['content_vector'] = embedding_vectors
    if predicted_labels: dataframe['predicted_label'] = predicted_labels

    return px.Dataset(dataframe=dataframe, schema=DEFAULT_SCHEMA, name=name)


def launch_px(primary_dataset: px.Dataset, reference_dataset: px.Dataset):
    return px.launch_app(primary=primary_dataset,
                         reference=reference_dataset,
                         host=os.getenv('PHOENIX_HOST'),
                         port=os.getenv('PHOENIX_PORT'))
