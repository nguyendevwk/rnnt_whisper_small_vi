import os
import torch
import time
import argparse
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import gather_object
from accelerate.logging import get_logger
from jiwer import wer
import glob

from models.streaming_rnnt import StreamingRNNT
from utils.dataset import AudioDataset, collate_fn
from utils.scheduler import WarmupLR
from constants import TOTAL_STEPS, WARMUP_STEPS, LR, MIN_LR

# Set up logger
logger = get_logger(__name__)

class AudioDatasetWithBasePath(AudioDataset):
    """
    Extends AudioDataset to handle relative audio file paths by adding a base path
    """
    def __init__(self, base_path=None, **kwargs):
        super().__init__(**kwargs)
        self.base_path = base_path if base_path else ""
        logger.info(f"Using base path for audio files: {self.base_path}")

        # Verify a few audio files exist
        if len(self.samples) > 0:
            for i in range(min(5, len(self.samples))):
                audio_path = self._get_audio_path(self.samples[i]['audio_filepath'])
                if not os.path.exists(audio_path):
                    logger.warning(f"Audio file not found: {audio_path}")
                else:
                    logger.info(f"Audio file verified: {audio_path}")

    def _get_audio_path(self, filepath):
        """Convert relative path to absolute path"""
        if os.path.isabs(filepath):
            return filepath
        else:
            return os.path.join(self.base_path, filepath)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Use the correct absolute path for audio files
        audio_path = self._get_audio_path(sample['audio_filepath'])

        try:
            waveform, sample_rate = librosa.load(
                audio_path,
                sr=SAMPLE_RATE,
                offset=sample['offset'],
                duration=sample['duration']
            )
            waveform = self.augmentation(samples=waveform, sample_rate=sample_rate)
            transcript_ids = self.tokenizer.encode_as_ids(sample['text'])

            waveform, transcript_ids = torch.from_numpy(waveform), torch.tensor(transcript_ids)
            melspec = self.log_mel_spectrogram(waveform, N_MELS, 0, self.device)

            return melspec, transcript_ids
        except Exception as e:
            logger.error(f"Error loading {audio_path}: {str(e)}")
            # Return a fallback sample (better than crashing)
            if idx > 0:
                return self.__getitem__(idx-1)  # Try previous item
            else:
                # Create a dummy sample as last resort
                dummy_waveform = torch.zeros(16000)  # 1 second of silence
                dummy_transcript = torch.tensor([1, 2, 3])  # Some tokens
                dummy_melspec = torch.zeros(N_MELS, 100)  # Dummy mel spectrogram
                logger.error(f"Using dummy sample as fallback")
                return dummy_melspec, dummy_transcript

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

    # Important: Convert train_manifest and val_manifest to lists if they're not already
    train_manifest = [args.train_manifest] if isinstance(args.train_manifest, str) else args.train_manifest
    val_manifest = [args.val_manifest] if isinstance(args.val_manifest, str) else args.val_manifest

    logger.info(f"Train manifest: {train_manifest}")
    logger.info(f"Val manifest: {val_manifest}")

    # Verify manifest files exist
    for manifest in train_manifest + val_manifest:
        if not os.path.isfile(manifest):
            logger.error(f"Manifest file does not exist or is not a file: {manifest}")
            raise FileNotFoundError(f"Manifest file not found: {manifest}")

    # Handle background noise path
    bg_noise_path = args.bg_noise_path if hasattr(args, 'bg_noise_path') and args.bg_noise_path else []

    # Create datasets and dataloaders with the enhanced dataset class
    try:
        train_dataset = AudioDatasetWithBasePath(
            base_path=args.audio_base_path,
            manifest_files=train_manifest,
            tokenizer_model_path=args.tokenizer_model_path,
            bg_noise_path=bg_noise_path,
            shuffle=True,
            augment=args.augment
        )

        val_dataset = AudioDatasetWithBasePath(
            base_path=args.audio_base_path,
            manifest_files=val_manifest,
            tokenizer_model_path=args.tokenizer_model_path,
            shuffle=False
        )

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Val dataset size: {len(val_dataset)}")

    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
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
    att_context_size = args.att_context_size
    if isinstance(att_context_size, int):
        att_context_size = [-1, att_context_size]  # Convert single int to list format

    logger.info(f"Using attention context size: {att_context_size}")

    model = StreamingRNNT(
        att_context_size=att_context_size,
        vocab_size=args.vocab_size if hasattr(args, 'vocab_size') else 1024,  # Default to 1024 if not specified
        tokenizer_model_path=args.tokenizer_model_path
    )

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    total_steps = args.total_steps if hasattr(args, 'total_steps') else TOTAL_STEPS
    warmup_steps = args.warmup_steps if hasattr(args, 'warmup_steps') else WARMUP_STEPS
    min_lr = args.min_lr if hasattr(args, 'min_lr') else MIN_LR

    scheduler = WarmupLR(optimizer, warmup_steps, total_steps, min_lr)

    # Prepare all components with accelerator
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    # Initialize tracker for best validation loss
    best_val_loss = float('inf')

    # Load checkpoint if provided
    if hasattr(args, 'resume_from_checkpoint') and args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
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
            # Skip problematic batches rather than crashing
            try:
                x, x_len, y, y_len = batch

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
                if batch_idx % 100 == 0 and accelerator.is_main_process:
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
                        all_pred = model.greedy_decoding(x, x_len, max_symbols=3)
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
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                continue

            # For debugging/testing - stop after a few steps if max_steps is set
            if hasattr(args, 'max_steps') and args.max_steps > 0 and global_step >= args.max_steps:
                break

        # Only run validation every val_check_interval epochs
        if epoch % args.val_check_interval == 0:
            # Validation
            model.eval()
            val_loss = 0.0
            all_val_pred = []
            all_val_true = []
            valid_batches = 0

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    try:
                        x, x_len, y, y_len = batch

                        # Forward pass
                        loss = model(x, x_len, y, y_len)
                        val_loss += loss.item()
                        valid_batches += 1

                        # Decoding
                        pred = model.greedy_decoding(x, x_len, max_symbols=3)
                        true = []
                        for i, y_i in enumerate(y):
                            y_i = y_i.cpu().numpy().astype(int).tolist()
                            y_i = y_i[:y_len[i]]
                            true.append(model.tokenizer.decode_ids(y_i))

                        all_val_pred.extend(pred)
                        all_val_true.extend(true)
                    except Exception as e:
                        logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                        continue

            # Calculate average validation loss and WER
            if valid_batches > 0:
                val_loss /= valid_batches

                # Gather all predictions and ground truth from all processes
                all_val_pred = gather_object(all_val_pred, accelerator)
                all_val_true = gather_object(all_val_true, accelerator)

                if accelerator.is_main_process and len(all_val_true) > 0:
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
        if global_step > 0:
            train_loss /= global_step

            if accelerator.is_main_process:
                logger.info(f"Epoch {epoch} completed in {time.time() - start_time:.2f}s")
                logger.info(f"Average Train Loss: {train_loss:.4f}")

            # Save latest checkpoint (same as original StreamingRNNT on_train_epoch_end)
            if accelerator.is_main_process:
                accelerator.save_state(os.path.join(args.output_dir, "rnnt-latest.pt"))
                logger.info(f"Saved latest model checkpoint")

        # Stop if max_steps was reached
        if hasattr(args, 'max_steps') and args.max_steps > 0 and global_step >= args.max_steps:
            break

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train a StreamingRNNT model with Accelerate")

    # Model configuration
    parser.add_argument("--att_context_size", type=int, default=80,
                        help="Attention context size")
    parser.add_argument("--vocab_size", type=int, default=1024,
                        help="Vocabulary size")
    parser.add_argument("--tokenizer_model_path", type=str, required=True,
                        help="Path to tokenizer model")

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=10,
                        help="Maximum number of epochs")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Maximum number of steps (0 for no limit)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS,
                        help="Warmup steps for scheduler")
    parser.add_argument("--total_steps", type=int, default=TOTAL_STEPS,
                        help="Total steps for scheduler")
    parser.add_argument("--min_lr", type=float, default=MIN_LR,
                        help="Minimum learning rate")

    # Data configuration
    parser.add_argument("--train_manifest", type=str, required=True,
                        help="Path to training manifest file")
    parser.add_argument("--val_manifest", type=str, required=True,
                        help="Path to validation manifest file")
    parser.add_argument("--bg_noise_path", type=str, default=None,
                        help="Path to background noise for augmentation")
    parser.add_argument("--augment", action="store_true", default=False,
                        help="Apply data augmentation")
    parser.add_argument("--audio_base_path", type=str, default="/kaggle/working/",
                        help="Base path to prepend to audio file paths if they are relative")

    # Output and checkpointing
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Output directory for logs and checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Validation configuration
    parser.add_argument("--val_check_interval", type=int, default=1,
                        help="Run validation every N epochs")

    # Mixed precision
    parser.add_argument("--precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16", "fp16-mixed", "bf16-mixed"],
                        help="Mixed precision mode")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)