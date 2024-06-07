# NPR MC2: Sentiment Analysis

**Authors**: Dominik Filliger, Nils Fahrni, Noah Leuenberger (2024)

---

This repository contains the implementation for the second mini-challenge of the Natural Language Processing course @
University of Applied Sciences Northwestern Switzerland (FHNW). This challenge focuses on Sentiment Analysis,
emphasizing the development and evaluation of models using both supervised and semi-supervised techniques.

## Table of Contents

<!-- TOC -->

* [NPR MC2: Sentiment Analysis](#npr-mc2-sentiment-analysis)
    * [Table of Contents](#table-of-contents)
    * [Project Overview](#project-overview)
    * [Pre-requisites](#pre-requisites)
    * [Setup](#setup)
        * [Running Locally with a Python Environment (Recommended)](#running-locally-with-a-python-environment-recommended)
        * [Running with Docker](#running-with-docker)
    * [Scripts](#scripts)
        * [Preparation of Datasets](#preparation-of-datasets)
        * [Model Training and Fine-Tuning](#model-training-and-fine-tuning)
        * [Weak Labeling Application](#weak-labeling-application)
    * [Notebooks](#notebooks)
    * [Model Weights from Experiments](#model-weights-from-experiments)

<!-- TOC -->

## Project Overview

```
- data/
    - partitions/
        - labelled_dev.parquet       # Subset treated as labelled for training
        - unlabelled_dev.parquet     # Subset treated as unlabelled for weak labeling
        - validation_set.parquet     # Used for model validation
    - embeddings/
        - mini_lm                    # Embeddings using sentence-transformers/all-MiniLM-L6-v2
        - mpnet_base                 # Embeddings using sentence-transformers/all-mpnet-base-v2
        
- models/
    - weak_labeling/                  # Models and scripts for weak labeling
    - semi-supervised/                # Results for semi-supervised models (safetensors from OneDrive)
    - supervised/                     # Results for supervised models (safetensors from OneDrive)
    - eval/                           # Evaluation results for baseline models
    
- notebooks/
    - weak_labeling.ipynb             # Discusses, optimizes, and compares weak labeling approaches
    - embedding_analysis.ipynb        # Analyzes embeddings and their dimensional reductions
    - main.ipynb                      # Main notebook for EDA, data partitioning, training, and evaluation
    - exports/                        # Exported HTML versions of the notebooks
    
- results/                            # Directory for model training outputs

- src/
    - prep_datasets.py                # Script for dataset preparation and partitioning
    - model_pipeline.py               # Manages model fine-tuning, evaluation, and inference
    - weaklabel_pipeline.py           # Applies weak labels to unlabelled data
    - data_loader.py                  # Functions for loading data
    - utils.py                        # General utility functions for scripts
    - px_utils.py                     # Utility functions for embedding visualisation with Arize Phoenix

- default.env                         # Default environment variables setup
- docker-compose.yml                  # Docker compose configuration for project containerization
- Dockerfile                          # Dockerfile to build the project's Docker container
- README.md                           # Project overview and setup instructions
- requirements.txt                    # Lists Python dependencies for the project
- train.sh                            # Shell script to run model fine-tuning across configurations
- USE-OF-AI.md                        # Document describing the use of AI within the project
```

## Pre-requisites

- Python 3.11
- Docker (if you want to run the project in a container)

## Setup

### Running Locally with a Python Environment (Recommended)

To run the project locally using Python:

1. **Ensure Python 3.11 is installed**: Check your Python version by running `python --version` or `python3 --version`
   in your terminal.

2. **Clone the Repository**: Clone the repository to your local machine.

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Setup**: Set up the necessary environment variables.
   ```bash
   cp default.env .env  # and modify .env as needed according to descriptions in the file
   ```

And you're all set! You should now be able to run the project scripts.

### Running with Docker

To run the project in a Docker container, follow these steps:

1. **Ensure Docker is installed**: Check Docker installation by running `docker --version`.

2. **Build the Docker Image**:
   ```bash
   docker build -t npr-mc2-sentiment-analysis .
   ```

    - If you want to pull the prebuilt `x86` image from GitHub Container Registry, instead use:
      ```bash
      docker pull ghcr.io/fhnw-ivy/npr-sentiment:main
      ```
      Then you can replace `npr-mc2-sentiment-analysis` with `ghcr.io/fhnw-ivy/npr-sentiment:main` for the image tag in
      the `docker-compose.yml` or `docker run` commands.

3. **Run Docker Compose** (defaults to building the image if not already built):
   ```bash
   docker-compose up
   ```

4. **Using the Docker Container**: To interact with the project within the Docker container, use:
   ```bash
   docker run -it --rm --name npr-sentiment-analysis npr-mc2-sentiment-analysis
   ```

    - You can execute Python scripts directly within the container:
      ```bash
      docker exec -it npr-sentiment-analysis python src/model_pipeline.py transfer --nested-splits
      ```

    - Or access the Jupyter notebook server running in the container (defaulting to http://localhost:8888)

- Ensure all paths and environment variables are set correctly before executing scripts.
    - When running Docker, make sure all paths in the `.env` are relative
- The Arize Phoenix visualisation do not work within the Docker container due to networking restrictions.
    - To use the visualisation, run the project locally with a Python environment.
- If GPU support is required, ensure that
  you [passthrough the GPU to the Docker container](https://stackoverflow.com/a/58432877).

## Scripts

### Preparation of Datasets

To prepare your dataset, use the `prep_datasets.py` script. This script organizes the data into
training, validation, and development sets. The already partitioned data is available in the `data/partitions`
directory. The fractions are in relation to
the [Amazon Polarity dataset](https://huggingface.co/datasets/fancyzhx/amazon_polarity) (3.6M training samples, 400k
testing samples). Example usage:

```bash
python src/prep_datasets.py prepare_dataset \
  --dev-set-fraction 0.007 \
  --val-set-fraction 0.003 \
  --labelled-fraction 0.01 \
  --output-dir /path/to/output \
  --verbose
```

**Parameters:**

- **Dev Set Fraction**: The fraction of the full dataset to be used as the development set.
- **Val Set Fraction**: The fraction of the test dataset to be used as the validation set.
- **Labelled Fraction**: The fraction of the development set that should be labelled.
- **Output Dir**: Specifies the directory to save the parquet files, defaulting to the `DATA_DIR` environment variable.
- **Verbose**: Enables verbose logging for detailed output during the script execution.

**Default Settings Explanation:**

- Given a full dataset size of approximately 3,600,000 records:
    - **Development Set**: By default, a small fraction of the dataset (approximately 1/1440, or about 2500 records) is
      designated as the development set.
    - **Validation Set**: Similarly, the validation set uses another 1/1440 of the test dataset, also amounting to about
      2500 records.
    - **Labelled Data**: Within the development set, a default of 10% (about 250 records) is labelled, allowing for a
      feasible amount of data to be manually annotated or reviewed for initial model training.
- The output directory is flexible but defaults to the `DATA_DIR` environment variable, which needs to be set prior to
  execution to ensure smooth operation.
- The verbose flag is false by default, aiming to keep the console output minimal unless detailed feedback is necessary
  for troubleshooting or monitoring purposes.

These settings ensure that the dataset is manageable and optimized for initial phases of model training and evaluation,
balancing between computational efficiency and adequate data representation for model performance.

### Model Training and Fine-Tuning

To train or fine-tune models, execute the `model_pipeline.py` script with the required parameters:

```bash
python src/model_pipeline.py \
  --mode transfer \
  --ckpt-path /path/to/ckpt \
  --nested-splits \
  --weak-label-path /path/to/weak/label \
  --batch-size 32 \
  --learning-rate 2e-5 \
  --num-epochs 10 \
  --warmup-steps 500 \
  --weight-decay 0.01 \
  --optim adamw_torch
```

**Options:**

- **Mode**: Set the operation mode (`transfer`, `finetune` or `eval`).
- **Ckpt Path**: Path to a checkpoint file for model initialization or resumption. If not specified, the script will
  start from scratch.
- **Nested Splits**: Enables nested splits and makes run for each split. If not specified, the script will run on the
  full dataset.
- **Weak Label Path**: Path to the weak label data file. Only required for weak label training, if not specified, the
  script will run in full supervised mode.
- **Batch Size**: Batch size for training.
- **Learning Rate**: Learning rate for the optimizer.
- **Num Epochs**: Number of epochs for the training process.
- **Warmup Steps**: Number of warmup steps before the main optimization phase.
- **Weight Decay**: Regularization parameter.
- **Optim**: Choice of optimizer, default is `adamw_torch`.

### Weak Labeling Application

For applying weak labels using the `weaklabel_pipeline.py` script, specify the model and data paths
correctly:

```bash
python src/weaklabel_pipeline.py \
  --model-filename /path/to/model.pkl \
  --unlabelled-parquet-file-path /path/to/unlabelled/data.parquet \
  --verbose
```

**Parameters:**

- **Model Filename**: Path to the file containing the trained model. The model should be a `scikit-learn` model saved
  in the expected embedding dimension.
- **Unlabelled Parquet File Path**: Path to the unlabelled data file in Parquet format which should be weakly labelled.
- **Verbose**: Enable verbose output for detailed processing logs.

## Notebooks

The `notebooks` directory contains Jupyter notebooks that provide detailed insights into the project's development and
evaluation processes:

- `weak_labeling.ipynb` – Discusses, optimizes, and compares weak labeling approaches.
- `embedding_analysis.ipynb` – Analyzes embeddings and their dimensional reductions.
- `main.ipynb` – Main notebook for EDA, data partitioning, training, and evaluation.

The recommended path is to start with the main notebook and then proceed to the related notebooks for further insights.

## Model Weights from Experiments

The safetensors of each model experiment can be downloaded
from [OneDrive](https://fhnw365-my.sharepoint.com/:f:/g/personal/noah_leuenberger_students_fhnw_ch/Eq9uOhn2iE5PpuqPXzBJqVEBWEMW6QnRbIFM5E05mxf5Hg?e=DDeRd5)
with a FHNW account. The folders are accordingly named and contain the experiment configurations.