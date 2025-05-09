#!/usr/bin/env python
import os
import subprocess
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Launch StreamingRNNT training with Accelerate")

    # General options
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers per GPU")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16", "fp16-mixed", "bf16-mixed"],
                        help="Mixed precision mode")

    # Optional overrides
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Resume from checkpoint")
    parser.add_argument("--train_manifest", type=str, default="/kaggle/working/data/train/train_data.jsonl",
                        help="Training manifest path")
    parser.add_argument("--val_manifest", type=str, default="/kaggle/working/data/test/test_data.jsonl",
                        help="Validation manifest path")
    parser.add_argument("--bg_noise_path", type=str, default="/kaggle/working/datatest/noise/fsdnoisy18k/",
                        help="Background noise path")
    parser.add_argument("--tokenizer_model_path", type=str, default="./weights/tokenizer_spe_bpe_v1024_pad/tokenizer.model",
                        help="Path to tokenizer model")
    parser.add_argument("--no_augment", action="store_true", help="Disable augmentation")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=2, help="Run validation every N epochs")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Build command
    cmd = ["accelerate", "launch", "train_accelerate.py"]
    cmd.extend(["--batch_size", str(args.batch_size)])
    cmd.extend(["--max_epochs", str(args.max_epochs)])
    cmd.extend(["--num_workers", str(args.num_workers)])
    cmd.extend(["--output_dir", args.output_dir])
    cmd.extend(["--precision", args.precision])
    cmd.extend(["--lr", str(args.lr)])
    cmd.extend(["--train_manifest", args.train_manifest])
    cmd.extend(["--val_manifest", args.val_manifest])
    cmd.extend(["--bg_noise_path", args.bg_noise_path])
    cmd.extend(["--tokenizer_model_path", args.tokenizer_model_path])
    cmd.extend(["--check_val_every_n_epoch", str(args.check_val_every_n_epoch)])

    if args.resume_from_checkpoint:
        cmd.extend(["--resume_from_checkpoint", args.resume_from_checkpoint])

    if args.no_augment:
        cmd.append("--no_augment")
    else:
        cmd.append("--augment")

    # Log the command
    cmd_str = " ".join(cmd)
    print("Running command:", cmd_str)

    # Write command to a file for reference
    with open("last_command.txt", "w") as f:
        f.write(cmd_str)

    # Execute the command
    subprocess.run(cmd)

if __name__ == "__main__":
    main()