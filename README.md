# NPR MC2: Sentiment Analysis

**Authors**: Dominik Filliger, Nils Fahrni, Noah Leuenberger (2024)

---

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