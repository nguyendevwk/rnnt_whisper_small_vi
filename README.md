# RNN Transducer (RNN-T) Training

This repository contains code for training an RNN Transducer model for speech recognition using PyTorch Lightning.

## Requirements

```
requirements.txt
pytorch
pytorch-lightning
torch.utils.data
sentencepiece
```

## Project Structure

```
.
├── models/
│   ├── encoder.py     # Audio encoder model
│   ├── decoder.py     # Text decoder model
│   └── jointer.py     # Joint network
├── utils/
│   ├── dataset.py     # Dataset loading and preprocessing
│   └── scheduler.py   # Learning rate scheduler
├── constants.py       # Configuration constants
├── train.py          # Training loop implementation
└── run.py            # Main training script
```

## Configuration

Key parameters in constants.py:

```python
TRAIN_MANIFEST        # Path to training manifest file
VAL_MANIFEST         # Path to validation manifest file
BG_NOISE_PATH        # Path to background noise files for augmentation
TOKENIZER_MODEL_PATH # Path to SentencePiece tokenizer model
BATCH_SIZE          # Training batch size
NUM_WORKERS         # Number of data loading workers
MAX_EPOCHS          # Maximum training epochs
ATTENTION_CONTEXT_SIZE # Size of attention context window
VOCAB_SIZE          # Vocabulary size
LOG_DIR             # Directory for saving checkpoints and logs
```

## Usage

1. Prepare your data and update paths in constants.py

2. Run training:

```bash
python run.py
```

The script will:

-   Load training and validation datasets
-   Create data loaders with specified batch size and workers
-   Initialize the RNN-T model
-   Configure model checkpointing
-   Train using PyTorch Lightning

## Training Features

-   Mixed precision training (BF16)
-   GPU acceleration when available
-   Automatic model checkpointing
-   TensorBoard logging
-   Learning rate warmup
-   Data augmentation with background noise
-   Supports resuming from checkpoints

## Resume Training

To resume training from a checkpoint:

```python
trainer.fit(model, train_dataloader, val_dataloader,
           ckpt_path="path/to/checkpoint.ckpt")
```

## Monitoring

Training metrics are logged to TensorBoard. View them with:

```bash
tensorboard --logdir=LOG_DIR
```

## Model Outputs

Checkpoints are saved to `LOG_DIR` with naming format:

```
rnnt-{epoch:02d}-{val_loss:.2f}.ckpt
```

The best models are kept based on validation loss.

## License

[Your license information here]
