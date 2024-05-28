# NPR MC2: Sentiment Analysis

**Authors**: Dominik Filliger, Nils Fahrni, Noah Leuenberger (2024)

---

## Dataset Preparation

The `src/prep_datasets.py` script is used to prepare the dataset for training and weak labeling. You can run the script
with the following command:

```bash
python src/prep_datasets.py --verbose
```

It takes a fraction of the original dataset and splits it into a train and eval set as well as a unlabelled set.

## Training

To begin training models, execute the following command:

```bash
python src/model_training.py "transfer" --nested-splits
```

### Configuration Options

You can customize the training process by using various command-line arguments. Here are the available options:

- **Mode**:
    - `transfer` – Starts training with transfer learning.
    - `finetune` – Starts fine-tuning the model.

- **Nested Splits**:
    - `--nested-splits` – Enables nested splits for the train set to improve model robustness.

- **Learning Rate**:
    - `--lr X` – Sets the learning rate to `X`, where `X` is a float.

- **Epochs**:
    - `--num_epochs X` – Specifies the number of epochs for training, where `X` is an integer.

- **Batch Size**:
    - `--batch_size X` – Defines the batch size for training, where `X` is an integer.

## Weak Labeling

To apply weak labeling to the dataset, you need to have a trained model. If you haven't trained a model yet, please
refer to weak labeling notebook in the `notebooks` directory.

Run the following command to apply weak labeling to the dataset:

```bash
python src/weaklabel_pipeline.py [...]/models/log_reg_weak_labeling.pkl [...]/data/partitions/unlabelled_dev.parquet --verbose
```

Make sure to replace `[...]` with the correct absolute path to the model and the unlabelled dataset.
