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
    os.makedirs(args.output_dir, exist_ok=True)
    accelerator = Accelerator(
        mixed_precision=args.precision,
        log_with="tensorboard",
        project_dir=args.output_dir
    )

    logger.info(f"Distributed type: {accelerator.distributed_type}")
    logger.info(f"Mixed precision: {accelerator.mixed_precision}")
    logger.info(f"Number of processes: {accelerator.num_processes}")
    logger.info(f"Device: {accelerator.device}")
    logger.info(f"Base path: {args.base_path}")

    # Chỉ sử dụng train_manifest
    train_manifest = args.train_manifest
    if isinstance(train_manifest, str):
        train_manifest = [train_manifest]

    logger.info(f"Train manifest: {train_manifest}")

    # Kiểm tra manifest
    for manifest_path in train_manifest:
        if not os.path.exists(manifest_path):
            logger.error(f"Manifest file not found: {manifest_path}")
            logger.error(f"Current directory: {os.getcwd()}")
            logger.error(f"Directory contents: {os.listdir('.')}")
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    # Tạo dataset
    try:
        train_dataset = AudioDataset(
            manifest_files=train_manifest,
            tokenizer_model_path=args.tokenizer_model_path,
            base_path=args.base_path,  # Truyền base_path
            bg_noise_path=None,  # Tắt background noise
            shuffle=True,
            augment=False  # Tắt augmentation
        )
        logger.info(f"Train dataset size: {len(train_dataset)}")
    except Exception as e:
        logger.error(f"Error creating train dataset: {str(e)}")
        raise

    # Kiểm tra thư mục âm thanh
    audio_dir = os.path.join(args.base_path, "data/train/audio_files/")
    if os.path.exists(audio_dir):
        logger.info(f"Found {len(os.listdir(audio_dir))} audio files in {audio_dir}")
    else:
        logger.error(f"Audio directory not found: {audio_dir}")
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=collate_fn,
        pin_memory=True
    )

    model = StreamingRNNT(
        att_context_size=args.att_context_size,
        vocab_size=args.vocab_size,
        tokenizer_model_path=args.tokenizer_model_path
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupLR(optimizer, args.warmup_steps, args.total_steps, args.min_lr)

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    global_step = 0

    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        accelerator.load_state(args.resume_from_checkpoint)
        logger.info(f"Loaded checkpoint from {args.resume_from_checkpoint}")

    # Vòng lặp huấn luyện (bỏ validation)
    for epoch in range(args.max_epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_dataloader):
            try:
                x, x_len, y, y_len = model.process_batch(batch)
                loss = model(x, x_len, y, y_len)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                train_loss += loss.item()
                global_step += 1

                # Log training metrics
                if batch_idx % 100 == 0:
                    lr = optimizer.param_groups[0]['lr']
                    accelerator.log({
                        "train_loss": loss.item(),
                        "learning_rate": lr,
                    }, step=global_step)
                    logger.info(f"Epoch {epoch}, Step {global_step}, Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {lr:.8f}")

                # Display predictions
                if batch_idx != 0 and batch_idx % 2000 == 0:
                    with torch.no_grad():
                        model.eval()
                        all_pred = model.greedy_decoding(x, x_len, max_symbols=MAX_SYMBOLS)
                        all_true = []
                        for i, y_i in enumerate(y):
                            y_i = y_i.cpu().numpy().astype(int).tolist()
                            y_i = y_i[:y_len[i]]
                            all_true.append(model.tokenizer.decode_ids(y_i))

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
                logger.warning(f"Error processing batch {batch_idx}: {str(e)}. Skipping.")
                continue

        # Tính toán và ghi log train loss
        train_loss /= len(train_dataloader)
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch} completed in {time.time() - start_time:.2f}s")
            logger.info(f"Average Train Loss: {train_loss:.4f}")
            accelerator.save_state(os.path.join(args.output_dir, "rnnt-latest.pt"))
            logger.info(f"Saved latest model checkpoint")

def get_args():
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