import os
import torch
import time
import argparse
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import gather_object
from accelerate.logging import get_logger
from jiwer import wer
from loguru import logger

from models.streaming_rnnt import StreamingRNNT
from utils.dataset import AudioDataset, collate_fn
from utils.scheduler import WarmupLR
from constants import (
    ATTENTION_CONTEXT_SIZE, VOCAB_SIZE, TOKENIZER_MODEL_PATH,
    BATCH_SIZE, NUM_WORKERS, MAX_EPOCHS, MAX_SYMBOLS,
    TOTAL_STEPS, WARMUP_STEPS, LR, MIN_LR,
    LOG_DIR
)

def train(args):
    """
    Main training function with Accelerate
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.precision,
        log_with="tensorboard",
        project_dir=args.output_dir
    )

    # Print training information
    logger.info(f"Distributed type: {accelerator.distributed_type}")
    logger.info(f"Mixed precision: {accelerator.mixed_precision}")
    logger.info(f"Number of processes: {accelerator.num_processes}")
    logger.info(f"Device: {accelerator.device}")

    # Convert train_manifest and val_manifest to lists if they're strings
    train_manifest = args.train_manifest
    if isinstance(train_manifest, str):
        train_manifest = [train_manifest]

    val_manifest = args.val_manifest
    if isinstance(val_manifest, str):
        val_manifest = [val_manifest]

    logger.info(f"Train manifest: {train_manifest}")
    logger.info(f"Val manifest: {val_manifest}")

    # Check if manifest files exist
    for manifest_path in train_manifest + val_manifest:
        if not os.path.exists(manifest_path):
            logger.error(f"Manifest file not found: {manifest_path}")
            if accelerator.is_main_process:
                logger.error(f"Current directory: {os.getcwd()}")
                logger.error(f"Directory contents: {os.listdir('.')}")
                # Try to find the file in the working directory structure
                for root, dirs, files in os.walk('.', topdown=True, followlinks=False):
                    if any(f.endswith('.jsonl') for f in files):
                        logger.error(f"Found .jsonl files in: {root}")
                        logger.error(f"Files: {[f for f in files if f.endswith('.jsonl')]}")
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    # Handle bg_noise_path
    bg_noise_path = args.bg_noise_path
    if bg_noise_path and not isinstance(bg_noise_path, list):
        bg_noise_path = [bg_noise_path]

    # Create datasets and dataloaders
    try:
        train_dataset = AudioDataset(
            manifest_files=train_manifest,
            bg_noise_path=bg_noise_path,
            shuffle=True,
            augment=args.augment,
            tokenizer_model_path=args.tokenizer_model_path
        )

        val_dataset = AudioDataset(
            manifest_files=val_manifest,
            shuffle=False,
            tokenizer_model_path=args.tokenizer_model_path
        )

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Val dataset size: {len(val_dataset)}")

    except Exception as e:
        logger.error(f"Error creating datasets: {str(e)}")
        raise

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Create model
    model = StreamingRNNT(
        att_context_size=args.att_context_size,
        vocab_size=args.vocab_size,
        tokenizer_model_path=args.tokenizer_model_path
    )

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupLR(optimizer, args.warmup_steps, args.total_steps, args.min_lr)

    # Prepare all components with accelerator
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    # Initialize tracker for best validation loss
    best_val_loss = float('inf')

    # Load checkpoint if provided
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        accelerator.load_state(args.resume_from_checkpoint)
        logger.info(f"Loaded checkpoint from {args.resume_from_checkpoint}")

    # Initialize tracker for steps
    global_step = 0

    # Training loop
    for epoch in range(args.max_epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        # Training
        for batch_idx, batch in enumerate(train_dataloader):
            x, x_len, y, y_len = model.process_batch(batch)

            # Forward pass
            loss = model(x, x_len, y, y_len)

            # Backward pass
            accelerator.backward(loss)

            # Update parameters
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Update loss
            train_loss += loss.item()
            global_step += 1

            # Log training metrics (every 100 steps as in the original code)
            if batch_idx % 100 == 0:
                lr = optimizer.param_groups[0]['lr']
                accelerator.log({
                    "train_loss": loss.item(),
                    "learning_rate": lr,
                }, step=global_step)

                logger.info(f"Epoch {epoch}, Step {global_step}, Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {lr:.8f}")

            # Display predictions during training (every 2000 steps as in the original code)
            if batch_idx != 0 and batch_idx % 2000 == 0:
                with torch.no_grad():
                    model.eval()
                    all_pred = model.greedy_decoding(x, x_len, max_symbols=MAX_SYMBOLS)
                    all_true = []
                    for i, y_i in enumerate(y):
                        y_i = y_i.cpu().numpy().astype(int).tolist()
                        y_i = y_i[:y_len[i]]
                        all_true.append(model.tokenizer.decode_ids(y_i))

                    # Gather predictions from all processes
                    all_pred = gather_object(all_pred, accelerator)
                    all_true = gather_object(all_true, accelerator)

                    if accelerator.is_main_process:
                        for pred, true in zip(all_pred[:2], all_true[:2]):
                            logger.info(f"Pred: {pred}")
                            logger.info(f"True: {true}")

                        train_wer = wer(all_true, all_pred)
                        accelerator.log({"train_wer": train_wer}, step=global_step)
                        logger.info(f"Training WER: {train_wer:.4f}")
                    model.train()

        # Only run validation every check_val_every_n_epoch epochs
        if epoch % args.check_val_every_n_epoch == 0:
            # Validation
            model.eval()
            val_loss = 0.0
            all_val_pred = []
            all_val_true = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    x, x_len, y, y_len = model.process_batch(batch)

                    # Forward pass
                    loss = model(x, x_len, y, y_len)
                    val_loss += loss.item()

                    # Decoding
                    pred = model.greedy_decoding(x, x_len, max_symbols=MAX_SYMBOLS)
                    true = []
                    for i, y_i in enumerate(y):
                        y_i = y_i.cpu().numpy().astype(int).tolist()
                        y_i = y_i[:y_len[i]]
                        true.append(model.tokenizer.decode_ids(y_i))

                    all_val_pred.extend(pred)
                    all_val_true.extend(true)

                    if batch_idx % 1000 == 0 and accelerator.is_main_process:
                        for p, t in zip(pred[:2], true[:2]):
                            logger.info(f"Val Pred: {p}")
                            logger.info(f"Val True: {t}")

            # Gather all predictions and ground truth from all processes
            all_val_pred = gather_object(all_val_pred, accelerator)
            all_val_true = gather_object(all_val_true, accelerator)

            # Calculate average validation loss and WER
            val_loss /= len(val_dataloader)

            if accelerator.is_main_process:
                val_wer = wer(all_val_true, all_val_pred)

                # Log validation metrics
                accelerator.log({
                    "val_loss": val_loss,
                    "val_wer": val_wer,
                    "epoch": epoch,
                }, step=global_step)

                logger.info(f"Validation - Epoch {epoch}, Loss: {val_loss:.4f}, WER: {val_wer:.4f}")

                # Save checkpoint if better validation loss (matching ModelCheckpoint behavior)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    accelerator.save_state(os.path.join(args.output_dir, f"rnnt-epoch{epoch:02d}-val_loss{val_loss:.2f}.pt"))
                    logger.info(f"Saved new best model with val_loss: {val_loss:.4f}")

        # Calculate average training loss
        train_loss /= len(train_dataloader)

        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch} completed in {time.time() - start_time:.2f}s")
            logger.info(f"Average Train Loss: {train_loss:.4f}")

        # Save latest checkpoint (same as original StreamingRNNT on_train_epoch_end)
        if accelerator.is_main_process:
            accelerator.save_state(os.path.join(args.output_dir, "rnnt-latest.pt"))
            logger.info(f"Saved latest model checkpoint")

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train a StreamingRNNT model with Accelerate")

    # Model configuration
    parser.add_argument("--att_context_size", default=ATTENTION_CONTEXT_SIZE,
                        help="Attention context size")
    parser.add_argument("--vocab_size", type=int, default=VOCAB_SIZE,
                        help="Vocabulary size")
    parser.add_argument("--tokenizer_model_path", type=str, default=TOKENIZER_MODEL_PATH,
                        help="Path to tokenizer model")

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS,
                        help="Maximum number of epochs")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS,
                        help="Number of dataloader workers")
    parser.add_argument("--lr", type=float, default=LR,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS,
                        help="Warmup steps for scheduler")
    parser.add_argument("--total_steps", type=int, default=TOTAL_STEPS,
                        help="Total steps for scheduler")
    parser.add_argument("--min_lr", type=float, default=MIN_LR,
                        help="Minimum learning rate")

    # Data configuration
    parser.add_argument("--train_manifest", default="/kaggle/working/data/train/train_data.jsonl",
                        help="Path to training manifest file")
    parser.add_argument("--val_manifest", default="/kaggle/working/data/test/test_data.jsonl",
                        help="Path to validation manifest file")
    parser.add_argument("--bg_noise_path", default="/kaggle/working/datatest/noise/fsdnoisy18k/",
                        help="Path to background noise for augmentation")
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Apply data augmentation")

    # Output and checkpointing
    parser.add_argument("--output_dir", type=str, default=LOG_DIR,
                        help="Output directory for logs and checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Validation configuration
    parser.add_argument("--check_val_every_n_epoch", type=int, default=2,
                        help="Run validation every N epochs")

    # Mixed precision
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                        choices=["no", "fp16", "bf16", "fp16-mixed", "bf16-mixed"],
                        help="Mixed precision mode")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)