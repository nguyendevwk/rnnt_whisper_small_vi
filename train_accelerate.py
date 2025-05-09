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
    Main training function using Accelerate for multi-GPU training
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize accelerator for distributed training
    accelerator = Accelerator(
        mixed_precision=args.precision,
        log_with="tensorboard",
        project_dir=args.output_dir
    )

    # Log system information
    logger.info(f"Starting training with Accelerate")
    logger.info(f"Distributed type: {accelerator.distributed_type}")
    logger.info(f"Mixed precision: {accelerator.mixed_precision}")
    logger.info(f"Number of processes: {accelerator.num_processes}")
    logger.info(f"Device: {accelerator.device}")
    logger.info(f"Base path: {args.base_path}")

    # Convert manifest paths to lists if they're strings
    train_manifest = args.train_manifest
    if isinstance(train_manifest, str):
        train_manifest = [train_manifest]

    val_manifest = args.val_manifest
    if isinstance(val_manifest, str):
        val_manifest = [val_manifest]

    logger.info(f"Train manifest: {train_manifest}")
    logger.info(f"Val manifest: {val_manifest}")

    # Verify both manifest files exist (QUAN TRỌNG)
    for manifest_path in train_manifest:
        if not os.path.exists(manifest_path):
            logger.error(f"TRAIN manifest file not found: {manifest_path}")
            # Log directory contents to help diagnose issues
            if accelerator.is_main_process:
                logger.error(f"Current directory: {os.getcwd()}")
                logger.error(f"Directory contents: {os.listdir('.')}")
                # Try to find similar files
                for root, dirs, files in os.walk('.', topdown=True, followlinks=False, maxdepth=3):
                    jsonl_files = [f for f in files if f.endswith('.jsonl')]
                    if jsonl_files:
                        logger.error(f"Found .jsonl files in: {root}: {jsonl_files}")
            raise FileNotFoundError(f"TRAIN manifest file not found: {manifest_path}")

    for manifest_path in val_manifest:
        if not os.path.exists(manifest_path):
            logger.error(f"VAL manifest file not found: {manifest_path}")
            # Log directory contents to help diagnose issues
            if accelerator.is_main_process:
                logger.error(f"Current directory: {os.getcwd()}")
                logger.error(f"Directory contents: {os.listdir('.')}")
                # Try to find similar files
                for root, dirs, files in os.walk('.', topdown=True, followlinks=False, maxdepth=3):
                    jsonl_files = [f for f in files if f.endswith('.jsonl')]
                    if jsonl_files:
                        logger.error(f"Found .jsonl files in: {root}: {jsonl_files}")
            raise FileNotFoundError(f"VAL manifest file not found: {manifest_path}")

    # Create train dataset
    try:
        train_dataset = AudioDataset(
            manifest_files=train_manifest,
            tokenizer_model_path=args.tokenizer_model_path,
            base_path=args.base_path,
            bg_noise_path=None,  # Disable background noise for stability
            shuffle=True,
            augment=args.augment
        )
        logger.info(f"Train dataset size: {len(train_dataset)}")

        # Create validation dataset
        val_dataset = AudioDataset(
            manifest_files=val_manifest,
            tokenizer_model_path=args.tokenizer_model_path,
            base_path=args.base_path,
            shuffle=False,
            augment=False  # No augmentation for validation
        )
        logger.info(f"Validation dataset size: {len(val_dataset)}")

        # Validate a few samples to ensure paths are correct
        if accelerator.is_main_process:
            for i in range(min(2, len(train_dataset))):
                sample = train_dataset.samples[i]
                logger.info(f"Train sample {i} path: {sample['audio_filepath']}")
                logger.info(f"Train sample {i} text: {sample['text']}")

            for i in range(min(2, len(val_dataset))):
                sample = val_dataset.samples[i]
                logger.info(f"Val sample {i} path: {sample['audio_filepath']}")
                logger.info(f"Val sample {i} text: {sample['text']}")

    except Exception as e:
        logger.error(f"Error creating datasets: {str(e)}")
        raise

    # Create dataloaders
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
    attention_context_size = args.att_context_size
    if isinstance(attention_context_size, int):
        attention_context_size = [-1, attention_context_size]

    logger.info(f"Using attention context size: {attention_context_size}")
    model = StreamingRNNT(
        att_context_size=attention_context_size,
        vocab_size=args.vocab_size,
        tokenizer_model_path=args.tokenizer_model_path
    )

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupLR(optimizer, args.warmup_steps, args.total_steps, args.min_lr)

    # Prepare for distributed training
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    # Initialize trackers
    global_step = 0
    best_val_loss = float('inf')

    # Load checkpoint if provided
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        accelerator.load_state(args.resume_from_checkpoint)
        logger.info(f"Loaded checkpoint from {args.resume_from_checkpoint}")

    # Training loop
    for epoch in range(args.max_epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        epoch_steps = 0

        # Train epoch
        logger.info(f"Starting epoch {epoch}")
        for batch_idx, batch in enumerate(train_dataloader):
            try:
                # Process batch and compute loss
                x, x_len, y, y_len = model.process_batch(batch)
                loss = model(x, x_len, y, y_len)

                # Backward pass and optimization
                accelerator.backward(loss)

                # Gradient clipping
                if args.grad_clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Update metrics
                train_loss += loss.item()
                global_step += 1
                epoch_steps += 1

                # Log training metrics periodically
                if batch_idx % 100 == 0:
                    lr = optimizer.param_groups[0]['lr']
                    accelerator.log({
                        "train_loss": loss.item(),
                        "learning_rate": lr,
                    }, step=global_step)
                    logger.info(f"Epoch {epoch}, Step {global_step}, Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {lr:.8f}")

                # Evaluate predictions periodically
                if batch_idx != 0 and batch_idx % 2000 == 0:
                    with torch.no_grad():
                        model.eval()
                        all_pred = model.greedy_decoding(x, x_len, max_symbols=MAX_SYMBOLS)
                        all_true = []
                        for i, y_i in enumerate(y):
                            y_i = y_i.cpu().numpy().astype(int).tolist()
                            y_i = y_i[:y_len[i]]
                            all_true.append(model.tokenizer.decode_ids(y_i))

                        # Gather predictions from all processes for accurate WER calculation
                        all_pred = gather_object(all_pred, accelerator)
                        all_true = gather_object(all_true, accelerator)

                        # Log examples and WER
                        if accelerator.is_main_process:
                            for pred, true in zip(all_pred[:2], all_true[:2]):
                                logger.info(f"Pred: {pred}")
                                logger.info(f"True: {true}")

                            train_wer = wer(all_true, all_pred)
                            accelerator.log({"train_wer": train_wer}, step=global_step)
                            logger.info(f"Training WER: {train_wer:.4f}")

                        model.train()

            except Exception as e:
                logger.warning(f"Error processing batch {batch_idx}: {str(e)}. Skipping.")
                continue

        # Run validation after each epoch
        model.eval()
        val_loss = 0.0
        val_steps = 0
        all_val_pred = []
        all_val_true = []

        logger.info(f"Running validation for epoch {epoch}")
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                try:
                    # Process batch
                    x, x_len, y, y_len = model.process_batch(batch)

                    # Compute loss
                    loss = model(x, x_len, y, y_len)
                    val_loss += loss.item()
                    val_steps += 1

                    # Decode predictions
                    pred = model.greedy_decoding(x, x_len, max_symbols=MAX_SYMBOLS)
                    true = []
                    for i, y_i in enumerate(y):
                        y_i = y_i.cpu().numpy().astype(int).tolist()
                        y_i = y_i[:y_len[i]]
                        true.append(model.tokenizer.decode_ids(y_i))

                    all_val_pred.extend(pred)
                    all_val_true.extend(true)

                    # Log examples
                    if batch_idx % 100 == 0 and accelerator.is_main_process:
                        logger.info(f"Val batch {batch_idx}, Loss: {loss.item():.4f}")

                except Exception as e:
                    logger.warning(f"Error in validation batch {batch_idx}: {str(e)}. Skipping.")
                    continue

        # Gather validation results from all processes
        all_val_pred = gather_object(all_val_pred, accelerator)
        all_val_true = gather_object(all_val_true, accelerator)

        # Compute validation metrics
        if val_steps > 0:
            val_loss /= val_steps

            if accelerator.is_main_process and len(all_val_true) > 0:
                val_wer = wer(all_val_true, all_val_pred)

                # Log validation metrics
                accelerator.log({
                    "val_loss": val_loss,
                    "val_wer": val_wer,
                    "epoch": epoch,
                }, step=global_step)

                logger.info(f"Validation - Epoch {epoch}, Loss: {val_loss:.4f}, WER: {val_wer:.4f}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    accelerator.save_state(os.path.join(args.output_dir, f"rnnt-best-val_loss{val_loss:.2f}.pt"))
                    logger.info(f"New best validation loss: {val_loss:.4f}")

        # Compute average training loss
        if epoch_steps > 0:
            train_loss /= epoch_steps
            if accelerator.is_main_process:
                logger.info(f"Epoch {epoch} completed in {time.time() - start_time:.2f}s")
                logger.info(f"Average Train Loss: {train_loss:.4f}")

                # Save epoch checkpoint
                accelerator.save_state(os.path.join(args.output_dir, f"rnnt-epoch{epoch:02d}.pt"))
                accelerator.save_state(os.path.join(args.output_dir, "rnnt-latest.pt"))
                logger.info(f"Saved model checkpoint for epoch {epoch}")

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
    parser.add_argument("--grad_clip", type=float, default=1.0,
                      help="Gradient clipping value")

    # Data configuration - QUAN TRỌNG: Cần cả 2 đường dẫn
    parser.add_argument("--train_manifest", default="/kaggle/working/data/train/train_data.jsonl",
                      help="Path to training manifest file (REQUIRED)")
    parser.add_argument("--val_manifest", default="/kaggle/working/data/test/test_data.jsonl",
                      help="Path to validation manifest file (REQUIRED)")
    parser.add_argument("--base_path", type=str, default="/kaggle/working/",
                      help="Base path for resolving audio file paths")
    parser.add_argument("--augment", action="store_true", default=False,
                      help="Apply data augmentation")

    # Output and checkpointing
    parser.add_argument("--output_dir", type=str, default=LOG_DIR,
                      help="Output directory for logs and checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                      help="Path to checkpoint to resume from")

    # Mixed precision
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                     choices=["no", "fp16", "bf16", "fp16-mixed", "bf16-mixed"],
                     help="Mixed precision mode")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)